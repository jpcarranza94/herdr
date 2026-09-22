#!/usr/bin/env python3
"""Herdr plugin startup hook: relaunch Muse sessions after a Herdr restart.

When Herdr dies and is relaunched, panes restore as bare shells in their saved
directories. Panes that carried a native session reference (Claude et al.) are
relaunched by Herdr itself; Muse has no native integration, so herdr-muse.py
survives only its bindings file. This hook makes the pane live again:

  - reads ~/.local/share/herdr-muse/bindings.json (session-id -> pane-id, kept
    fresh by the Muse hook on every lifecycle event),
  - drops bindings whose pane no longer exists (stale: pane closed days ago)
    and whose Muse session store is gone,
  - skips panes that already run a muse foreground process (already restarted,
    or Herdr came back via --handoff and never died),
  - waits for each remaining pane's shell to become ready (restored shells can
    take a while to spawn; firing `agent start` before that fails with
    agent_pane_busy and never retries), then relaunches with
    `agent start --kind muse --pane <id> -- resume <session-id>`,
    mirroring what the native resume planner does for Claude.

Runs once at server start. Failures are logged, never fatal: the pane keeps its
shell and the next Muse lifecycle event re-reports state anyway.

Design notes:
  - Must not touch the live server except through the CLI with the env Herdr
    injects (HERDR_SOCKET_PATH / HERDR_ENV / HERDR_BIN_PATH).
  - Dry-run mode (HERDR_MUSE_RESTORE_DRY_RUN=1) classifies every binding and
    prints the plan without launching anything.
  - Knobs (env): MUSE_RESTORE_PANE_TIMEOUT_S (default 600) bounds the
    wait-for-shell per pane; MUSE_RESTORE_POLL_S (default 3) is the poll
    interval; MUSE_RESTORE_BINDINGS overrides the bindings path (tests).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_PARALLEL = 4

BINDINGS_PATH = os.environ.get(
    "MUSE_RESTORE_BINDINGS",
    os.path.join(
        os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
        "herdr-muse",
        "bindings.json",
    ),
)

MUSE_SESSIONS_ROOT = os.path.join(
    os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
    "muse",
    "sessions",
)

DRY_RUN = os.environ.get("HERDR_MUSE_RESTORE_DRY_RUN") == "1"
AGENT_START_TIMEOUT_MS = 90_000
POLL_S = float(os.environ.get("MUSE_RESTORE_POLL_S", "3"))
PANE_TIMEOUT_S = float(os.environ.get("MUSE_RESTORE_PANE_TIMEOUT_S", "600"))

# Pane readiness states from `pane process-info`.
ST_UNKNOWN = "unknown"  # call failed; runtime may still be coming up
ST_NO_SHELL = "no-shell"  # pane up, shell PID not visible yet
ST_MUSE_RUNNING = "muse-running"
ST_SHELL = "shell"  # shell present, no muse foreground process


def find_herdr():
    override = os.environ.get("HERDR_BIN_PATH")
    if override and os.access(override, os.X_OK):
        return override
    found = shutil.which("herdr")
    if found:
        return found
    for candidate in ("/opt/homebrew/bin/herdr", "/usr/local/bin/herdr"):
        if os.access(candidate, os.X_OK):
            return candidate
    return None


def herdr_json(binary, *args):
    try:
        proc = subprocess.run(
            [binary, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout.decode("utf-8", "replace"))
    except ValueError:
        return None


def run_herdr(binary, *args):
    """Run herdr, return (ok, combined_output). Never raises."""
    try:
        proc = subprocess.run(
            [binary, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=AGENT_START_TIMEOUT_MS // 1000 + 30,
        )
    except (OSError, subprocess.SubprocessError) as err:
        return False, f"invocation failed: {err}"
    out = proc.stdout.decode("utf-8", "replace").strip()
    err = proc.stderr.decode("utf-8", "replace").strip()
    combined = f"{out} {err}".strip()
    if proc.returncode != 0:
        return False, f"exit {proc.returncode}: {combined}"
    return True, combined


def live_pane_ids(binary):
    """All pane ids currently on the server, or None when unknown."""
    snapshot = herdr_json(binary, "workspace", "list")
    if not snapshot:
        return None
    try:
        workspaces = snapshot["result"]["workspaces"]
    except (KeyError, TypeError):
        return None
    panes = set()
    for workspace in workspaces:
        wid = workspace.get("workspace_id") if isinstance(workspace, dict) else None
        if not wid:
            continue
        listing = herdr_json(binary, "pane", "list", "--workspace", wid)
        try:
            entries = listing["result"]["panes"]
        except (KeyError, TypeError):
            return None
        for pane in entries:
            if isinstance(pane, dict) and pane.get("pane_id"):
                panes.add(pane["pane_id"])
    return panes


def session_dir_exists(session_id):
    for sub in ("", ".msp-view-v1"):
        candidate = os.path.join(MUSE_SESSIONS_ROOT, sub, session_id)
        try:
            if os.path.isdir(candidate):
                return True
        except OSError:
            continue
    # Date-sharded store: sessions/<shard>/.../<uuid>/. Bounded walk (depth 3)
    # so a permission error or huge tree cannot stall the startup hook.
    try:
        stack = [(MUSE_SESSIONS_ROOT, 0)]
        while stack:
            directory, depth = stack.pop()
            if depth > 3:
                continue
            try:
                with os.scandir(directory) as entries:
                    names = [(entry.name, entry.is_dir(follow_symlinks=False))
                             for entry in entries]
            except OSError:
                continue
            for name, is_dir in names:
                if not is_dir:
                    continue
                if name == session_id:
                    return True
                if depth < 3:
                    stack.append((os.path.join(directory, name), depth + 1))
    except OSError:
        pass
    return False


def pane_state(binary, pane_id):
    """Classify a pane that is known to exist. Never raises."""
    info = herdr_json(binary, "pane", "process-info", "--pane", pane_id)
    if not isinstance(info, dict):
        return ST_UNKNOWN
    try:
        process_info = info["result"]["process_info"]
    except (KeyError, TypeError):
        return ST_UNKNOWN
    if not isinstance(process_info, dict):
        return ST_UNKNOWN
    processes = process_info.get("foreground_processes", [])
    if isinstance(processes, list):
        for proc in processes:
            if not isinstance(proc, dict):
                continue
            for key in ("name", "argv0", "cmdline"):
                value = proc.get(key)
                if isinstance(value, str) and "muse" in value.lower():
                    return ST_MUSE_RUNNING
    if isinstance(process_info.get("shell_pid"), int):
        return ST_SHELL
    return ST_NO_SHELL


def wait_for_shell(binary, pane_id, deadline):
    """Poll until the pane has a shell (or muse), or the deadline passes."""
    while True:
        state = pane_state(binary, pane_id)
        if state in (ST_SHELL, ST_MUSE_RUNNING):
            return state
        if time.monotonic() >= deadline:
            return state
        time.sleep(POLL_S)


def relaunch(binary, session_id, binding, panes_live):
    pane_id = binding.get("pane_id")
    if not pane_id:
        return f"  - {session_id}: no pane_id, skip"
    if panes_live is not None and pane_id not in panes_live:
        return f"  - {session_id} -> {pane_id}: pane gone, skip"
    if not session_dir_exists(session_id):
        return f"  - {session_id} -> {pane_id}: session store missing, skip"
    state = pane_state(binary, pane_id)
    if state == ST_MUSE_RUNNING:
        return f"  - {session_id} -> {pane_id}: muse already running, skip"
    if DRY_RUN:
        if state == ST_SHELL:
            return f"  - {session_id} -> {pane_id}: would launch (shell ready)"
        return f"  - {session_id} -> {pane_id}: would wait for shell (now {state})"
    if state != ST_SHELL:
        print(f"  ~ {session_id} -> {pane_id}: shell not ready ({state}), waiting",
              flush=True)
        state = wait_for_shell(binary, pane_id, time.monotonic() + PANE_TIMEOUT_S)
        if state == ST_MUSE_RUNNING:
            return f"  - {session_id} -> {pane_id}: muse started meanwhile, skip"
        if state != ST_SHELL:
            return (f"  - {session_id} -> {pane_id}: shell never became ready "
                    f"({state}), skip")
    name = f"muse-{session_id[:8]}"
    deadline = time.monotonic() + PANE_TIMEOUT_S
    while True:
        ok, output = run_herdr(
            binary,
            "agent",
            "start",
            name,
            "--kind",
            "muse",
            "--pane",
            pane_id,
            "--timeout",
            str(AGENT_START_TIMEOUT_MS),
            "--",
            "resume",
            session_id,
        )
        if ok:
            return f"  - {session_id} -> {pane_id}: resumed ({output[:200]})"
        if "agent_pane_busy" in output and time.monotonic() < deadline:
            time.sleep(POLL_S)
            continue
        return f"  - {session_id} -> {pane_id}: launch failed ({output[:200]})"


def load_bindings():
    try:
        with open(BINDINGS_PATH) as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def main():
    bindings = load_bindings()
    if not bindings:
        print("muse-restore: no bindings, nothing to do", flush=True)
        return 0

    binary = find_herdr()
    if binary is None:
        print("muse-restore: cannot locate herdr binary", flush=True)
        return 0

    panes_live = live_pane_ids(binary)
    if panes_live is None:
        print("muse-restore: cannot list panes; continuing without gone-check",
              flush=True)

    print(f"muse-restore: {len(bindings)} binding(s) to evaluate", flush=True)
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(bindings))) as pool:
        futures = {
            pool.submit(relaunch, binary, session_id, binding, panes_live): session_id
            for session_id, binding in sorted(bindings.items())
        }
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as err:  # noqa: BLE001 - degrades to a log line
                print(f"  ! {futures[future]}: {err}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
