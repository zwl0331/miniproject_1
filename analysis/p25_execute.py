"""Runs P2.5's sequence (plan section 19.3) as separate processes, stopping at the first failure: the score-free
timing and the reservation (stopping before any score if the initial batch's projection exceeds section 5.5's three
hours), the initialization, wave 1, then plan and run waves 2, 3, ... until a wave plans no batch (allowance,
ceiling, budget or cutoff)."""
import json, os, subprocess, sys, time
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
RX = [".venv/bin/python", "-m", "scripts.run_experiments"]
LOG = "artifacts/phase2/experiments.jsonl"

def run(args):
    t0 = time.time()
    print(f"$ {' '.join(RX + args)}  [{time.strftime('%H:%M:%S')}]", flush=True)
    code = subprocess.run(RX + args, env=ENV).returncode
    print(f"  exit {code} after {time.time() - t0:.0f} s", flush=True)
    if code:
        sys.exit(code)

def records(kind):
    return [r for r in map(json.loads, open(LOG)) if r.get("record") == kind and r.get("task_id") == "P2.5"]

if not records("timing"):
    run(["--timing", "configs/phase2/p2.5/timing.json", "--task", "P2.5"])
if not records("reservation"):
    run(["--reserve", "--task", "P2.5"])
res = records("reservation")[-1]
if res["initial_minutes"] > 180:
    sys.exit(f"the initial batch projects {res['initial_minutes']} min, above section 5.5's 180: reduce it before any score")
run(["--plan-wave", "1", "--task", "P2.5"])
run(["--run-wave", "1", "--anchors-only", "--task", "P2.5"])
run(["--run-wave", "1", "--task", "P2.5"])
w = 2
while True:
    run(["--plan-wave", str(w), "--task", "P2.5"])
    if not [b for b in records("batch") if b["wave"] == w]:
        print(f"wave {w}: nothing planned; the search is complete", flush=True)
        break
    run(["--run-wave", str(w), "--task", "P2.5"])
    w += 1
