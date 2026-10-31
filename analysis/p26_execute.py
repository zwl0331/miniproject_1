"""Runs P2.6b's sequence (plan sections 20.2 to 20.4) as separate processes, stopping at the first failure.
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p26_execute.py nominees
        the nominations, every nominee's final-policy run, then the pool and the procedure manifest
        (commit configs/phase2/p2.6/procedures.json before the next stage: the manifest is frozen before evaluation)
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p26_execute.py evaluate
        every fixed blend (the within-family average first), the learned procedures, then the menu
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p26_execute.py verify
        the complete replay of the learned procedures (analysis/p26_verify_learned.py: every channel atom refitted,
        every meta-fold fit redone from an empty cache) and the refit of every P2.6b fitted run (--verify)"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
PY = ".venv/bin/python"
RX = [PY, "-m", "scripts.run_experiments"]
OUT = Path("configs/phase2/p2.6")


def run(args):
    t0 = time.time()
    print(f"$ {' '.join(args)}  [{time.strftime('%H:%M:%S')}]", flush=True)
    code = subprocess.run(args, env=ENV).returncode
    print(f"  exit {code} after {time.time() - t0:.0f} s", flush=True)
    if code:
        sys.exit(code)


def nominees():
    run([PY, "-B", "analysis/p26_nominate.py"])
    nom = json.loads((OUT / "nominations.json").read_text())
    for n in nom["nominees"]:
        if n["run_existing"] is None:
            run(RX + ["--recipe", str(OUT / "recipes" / f"{n['recipe']['name']}.json"), "--task", "P2.6b"])
    run([PY, "-B", "analysis/p26_pool.py"])


def evaluate():
    doc = json.loads((OUT / "procedures.json").read_text())
    blends = sorted([p for p in doc["procedures"] if p["kind"] == "blend"], key=lambda p: p["id"] != "family_average")
    for p in blends:
        run(RX + ["--blend", p["id"].replace(".", "_"), "--members"]
            + [f"{m['run_id']}:{m['weight']!r}" for m in p["members"]] + ["--task", "P2.6b"])
    run(RX + ["--run-learned", str(OUT / "procedures.json"), "--task", "P2.6b"])
    run([PY, "-B", "analysis/p26_menu.py"])


def verify():
    with open("artifacts/phase2/verify_learned_p26.txt", "w") as f:
        code = subprocess.run([PY, "-B", "analysis/p26_verify_learned.py"], env=ENV, stdout=f).returncode
    print(f"  replay exit {code}", flush=True)
    fitted = sorted({r["run_id"] for r in map(json.loads, open("artifacts/phase2/experiments.jsonl"))
                     if r.get("record") == "run" and r.get("task_id") == "P2.6b"
                     and r.get("prediction_kind") in ("fitted", "seed_average")})
    with open("artifacts/phase2/verify_p26b_runs.txt", "w") as f:
        code2 = subprocess.run(RX + ["--verify", "--run-id"] + fitted, env=ENV, stdout=f).returncode
    print(f"  refit of {len(fitted)} P2.6b runs exit {code2}", flush=True)
    sys.exit(code or code2)


if __name__ == "__main__":
    {"nominees": nominees, "evaluate": evaluate, "verify": verify}[sys.argv[1]]()
