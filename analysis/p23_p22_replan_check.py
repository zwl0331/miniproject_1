"""P2.2's wave 2 and 3 manifests re-planned under P2.3's protocol code (doc/plans/phase2-models-features.md section
17.6): the refine and protocol changes of P2.3 must leave them unchanged. Run ids come from the saved manifests (the
P2.2 source manifest differs from the running one by design); the recipes, run ids, state and sources of every
re-planned batch must equal the saved ones. Fits nothing; writes nothing but its standard output; exits 1 on a
difference. From the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -I analysis/p23_p22_replan_check.py . > analysis/p23_p22_replan_check_output.txt
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT))

from songpop import artifacts  # noqa: E402
from songpop.phase2 import protocol  # noqa: E402
from songpop.phase2.recipes import recipe_hash, strict_loads  # noqa: E402


def main():
    proto = protocol.load_protocol(ROOT / "configs/phase2/p2.2/protocol.json")
    read = lambda rel: strict_loads((ROOT / rel).read_text())  # noqa: E731
    log = protocol.Log(artifacts.read_experiments(ROOT / "artifacts/phase2/experiments.jsonl"), "P2.2", read)
    saved = {}
    for b in log.batches:
        m = read(b["file"])
        saved[(b["wave"], b["family"])] = m
    ids = {recipe_hash(r): rid for m in saved.values() for r, rid in zip(m["recipes"], m["run_ids"])}
    ok = True
    for wave in sorted({w for w, _ in saved}):
        planned = protocol.plan_wave(proto, log, wave, lambda r: ids.get(recipe_hash(r), "MISSING"))
        fams = sorted(m["family"] for m in planned)
        want = sorted(f for w, f in saved if w == wave)
        good = fams == want
        for m in planned:
            s = saved[(wave, m["family"])]
            same = all(artifacts.dumps(m[k]) == artifacts.dumps(s[k]) for k in ("recipes", "run_ids", "state", "sources"))
            good &= same
            print(f"  wave {wave} {m['family']:14s} {m['kind']:10s} {len(m['recipes']):3d} configurations: "
                  f"{'equal to the saved manifest' if same else 'DIFFERS'}")
        print(f"wave {wave}: families {fams}: {'unchanged' if good else 'CHANGED'}")
        ok &= good
    print("P2.2 re-planning under the current protocol code:", "UNCHANGED" if ok else "CHANGED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
