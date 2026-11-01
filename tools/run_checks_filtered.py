"""Run phase-two checks, optionally filtering names and printing live progress.

No filter runs the complete suite. Instrumentation changes only this in-memory
copy of the decorator; assertions and bootstrap counts remain unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("substring", nargs="?", default="", help="only run matching check names")
parser.add_argument("--progress", action="store_true", help="print each check's start, result and duration")
args = parser.parse_args()
root = Path.cwd()
sys.path.insert(0, str(root))
src = (root / "scripts/check_phase2.py").read_text()
marker = '        if status != "run":'
assert src.count(marker) == 1
src = src.replace(marker, '        if FILTER not in name:\n            return fn\n' + marker, 1)
if args.progress:
    marker = "        try:\n            note = fn()"
    assert src.count(marker) == 1
    src = src.replace(marker,
        '        _progress_started = PROGRESS_CLOCK()\n'
        '        print(f"RUN   {name}", flush=True)\n' + marker, 1)
    marker = "        return fn\n    return wrap"
    assert src.count(marker) == 1
    src = src.replace(marker,
        '        print(f"DONE  {RESULTS[-1][1]} {name} '
        '({PROGRESS_CLOCK() - _progress_started:.2f}s)", flush=True)\n' + marker, 1)
    print(f"Started {datetime.now(timezone.utc).isoformat()} PID {os.getpid()}", flush=True)
    print(f"Filter: {args.substring!r}; empty means the complete suite", flush=True)
    for path in (root / "scripts/check_phase2.py", Path(__file__).resolve()):
        print(f"Source {path.relative_to(root)} sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
g = {"__name__": "__main__", "__file__": str(root / "scripts/check_phase2.py"),
     "FILTER": args.substring, "PROGRESS_CLOCK": time.perf_counter}
exec(compile(src, str(root / "scripts/check_phase2.py"), "exec"), g)
