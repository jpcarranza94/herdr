#!/usr/bin/env python3
"""Herdr plugin startup hook: relaunch Muse sessions after a Herdr restart.

When Herdr dies and is relaunched, panes restore as bare shells in their saved
directories. Panes that carried a native session reference (Claude et al.) are
relaunched by Herdr itself; Muse has no native integration, so herdr-muse.py
survives only its bindings file. This hook makes the pane live again:

  - reads ~/.local/share/herdr-muse/bindings.json (session-id -> pane-id, kept
    fresh by the Muse hook on every lifecycle event),
  - skips panes that already run a muse foreground process (already restarted,
    or Herdr came back via --handoff and never died),
  - relaunches each rest-of-shelf session with
    `agent start --kind muse --pane <id> -- resume <session-id>`,
    mirroring what the native resume planner does for Claude.

Runs once at server start. Failures are logged, never fatal: the pane keeps its
shell and the next Muse lifecycle event re-reports state anyway.

Design notes:
  - Must not touch the live server except through the CLI with the env Herdr
    injects (HERDR_SOCKET_PATH / HERDR_ENV / HERDR_BIN_PATH).
  - Dry-run mode (HERDR_MUSE_RESTORE_DRY_RUN=1) prints the plan without acting.
"""

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_PARALLEL = 4

BINDINGS_PATH = os.path.join(
    os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
    "herdr-muse",
    "bindings.json",
)

MUSE_SESSIONS_ROOT = os.path.join(
    os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
    "muse",
    "sessions",
)

DRY_RUN = os.environ.get("HERDR_MUSE_RESTORE_DRY_RUN") == "1"
AGENT_START_TIMEOUT_MS = 90_000


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
    try:
        proc = subprocess.run(
            [binary, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=AGENT_START_TIMEOUT_MS // 1000 + 30,
        )
    except (OSError, subprocess.SubprocessError) as err:
        print(f"  ! herdr invocation failed: {err}", flush=True)
        return None
    out = proc.stdout.decode("utf-8", "replace").strip()
    err = proc.stderr.decode("utf-8", "replace").strip()
    if proc.returncode != 0:
        print(f"  ! herdr exited {proc.returncode}: {out} {err}".strip(), flush=True)
        return None
    return out


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


def pane_has_muse(binary, pane_id):
    """True when the pane is missing or already runs a muse foreground process."""
    info = herdr_json(binary, "pane", "process-info", "--pane", pane_id)
    if info is None:
        # Pane gone or server unresponsive: safest to skip.
        return True
    try:
        process_info = info["result"]["process_info"]
    except (KeyError, TypeError, AttributeError):
        return True
    if not isinstance(process_info, dict):
        return True
    processes = process_info.get("foreground_processes", [])
    if not isinstance(processes, list):
        return True
    for proc in processes or []:
        if not isinstance(proc, dict):
            continue
        for key in ("name", "argv0", "cmdline"):
            value = proc.get(key)
            if isinstance(value, str) and "muse" in value.lower():
                return True
    return False


def load_bindings():
    try:
        with open(BINDINGS_PATH) as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def relaunch(binary, session_id, binding):
    pane_id = binding.get("pane_id")
    if not pane_id:
        return f"  - {session_id}: no pane_id, skip"
    if DRY_RUN:
        return f"  - {session_id} -> {pane_id}: would launch"
    if pane_has_muse(binary, pane_id):
        return f"  - {session_id} -> {pane_id}: pane missing or muse already running, skip"
    if not session_dir_exists(session_id):
        return f"  - {session_id} -> {pane_id}: session store missing, skip"
    name = f"muse-{session_id[:8]}"
    ret = run_herdr(
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
    if ret is None:
        return f"  - {session_id} -> {pane_id}: launch failed"
    return f"  - {session_id} -> {pane_id}: resumed ({ret[:200]})"


def main():
    bindings = load_bindings()
    if not bindings:
        print("muse-restore: no bindings, nothing to do", flush=True)
        return 0

    binary = find_herdr()
    if binary is None:
        print("muse-restore: cannot locate herdr binary", flush=True)
        return 0

    print(f"muse-restore: {len(bindings)} binding(s) to evaluate", flush=True)
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(bindings))) as pool:
        futures = {
            pool.submit(relaunch, binary, session_id, binding): session_id
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