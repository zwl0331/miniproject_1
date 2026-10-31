"""Runs P2.4's approved execution sequence (plan section 18.10) as separate processes, stopping at the first
failure: the initial batch, expansions until nothing is due, then both re-tuning rounds."""
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

def planned(stage, index):
    for line in open("artifacts/phase2/experiments.jsonl"):
        r = json.loads(line)
        if r.get("record") == "feature_batch" and r.get("task_id") == "P2.4" and r["stage"] == stage and r["index"] == index:
            return True
    return False

run(["--run-features", "1", "--task", "P2.4"])
n = 2
while True:
    run(["--plan-features", str(n), "--task", "P2.4"])
    if not planned("features", n):
        print(f"features {n}: nothing planned; the feature stage is closed", flush=True)
        break
    run(["--run-features", str(n), "--task", "P2.4"])
    n += 1
for k in (1, 2):
    run(["--plan-retune", str(k), "--task", "P2.4"])
    if not planned("retune", k):
        print(f"retune {k}: nothing planned", flush=True)
        break
    run(["--run-retune", str(k), "--task", "P2.4"])
print("P2.4 sequence complete", flush=True)
