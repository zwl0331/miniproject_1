"""Runs P2.3's approved execution sequence (plan section 17.6) as separate processes, stopping at the first failure:
the initialization, wave 1, then plan and run waves 2, 3, ... until a wave plans no batch."""
import json, os, subprocess, sys, time
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
RX = [".venv/bin/python", "-m", "scripts.run_experiments"]

def run(args):
    t0 = time.time()
    print(f"$ {' '.join(RX + args)}  [{time.strftime('%H:%M:%S')}]", flush=True)
    code = subprocess.run(RX + args, env=ENV).returncode
    print(f"  exit {code} after {time.time() - t0:.0f} s", flush=True)
    if code:
        sys.exit(code)

def batches(wave):
    n = 0
    for line in open("artifacts/phase2/experiments.jsonl"):
        r = json.loads(line)
        n += r.get("record") == "batch" and r.get("task_id") == "P2.3" and r.get("wave") == wave
    return n

run(["--run-wave", "1", "--anchors-only", "--task", "P2.3"])
run(["--run-wave", "1", "--task", "P2.3"])
w = 2
while True:
    run(["--plan-wave", str(w), "--task", "P2.3"])
    if not batches(w):
        print(f"wave {w}: nothing planned; the search is complete", flush=True)
        break
    run(["--run-wave", str(w), "--task", "P2.3"])
    w += 1
