"""P2.6b menu (plan sections 7.3, 11 and 20.5), read-only: every candidate of the frozen procedure manifest found in
the log with its complete production bank, then at most 12 menu candidates chosen by the frozen selection rule
within each class: the anchor, its three-seed version, the standalone leader, the within-family average, the best
fixed blend, the best constrained-weight procedure and the best stacker, the remaining places filled by repeated
application of the rule. Writes configs/phase2/menu.json. Fits nothing. From the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p26_menu.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np  # noqa: E402

from songpop import artifacts  # noqa: E402
from songpop.phase2 import banks, nested, procedures, runs  # noqa: E402
from songpop.phase2.selection import TIE_BAND, TIE_KEYS, Evidence, select_candidate  # noqa: E402

PROCS = Path("configs/phase2/p2.6/procedures.json")
MENU = Path("configs/phase2/menu.json")
MAX_MENU = 12


def candidate_banks(ctx, doc):
    """Each manifest candidate's production bank, with its run id and bank hash: atoms from their runs, fixed blends
    from the logged P2.6b blend runs with exactly the declared members and weights, learned procedures from their
    procedure records for this manifest."""
    psha = artifacts.sha256_bytes(PROCS.read_bytes())
    blends = [r for r in runs.logged(ctx) if r["prediction_kind"] == "blend" and r["task_id"] == "P2.6b"]
    procs = [r for r in runs.logged(ctx, "procedure") if r["procedures_sha256"] == psha]
    out = {}
    for p in doc["procedures"]:
        if p["kind"] == "atomic":
            rid = p["run_id"]
        elif p["kind"] == "blend":
            want = sorted((m["run_id"], round(m["weight"], 12)) for m in p["members"])
            got = [r for r in blends if sorted((q["run_id"], round(q["weight"], 12)) for q in r["parents"]) == want]
            if len(got) != 1:
                raise ValueError(f"{p['id']}: {len(got)} logged blends with its members and weights")
            rid = got[0]["run_id"]
        else:
            got = [r for r in procs if r["procedure_id"] == p["id"]]
            if len(got) != 1:
                raise ValueError(f"{p['id']}: {len(got)} procedure records for this manifest")
            rec = got[0]
            raw = (ctx.root / rec["bank_path"]).read_bytes()
            if artifacts.sha256_bytes(raw) != rec["bank_sha256"]:
                raise ValueError(f"{p['id']}: its bank differs from its record")
            out[p["id"]] = (rec["run_id"], rec["bank_sha256"], banks.load_bank_v2(raw, validate=False))
            continue
        bank, sha, _ = runs.load_logged(ctx, rid)
        out[p["id"]] = (rid, sha, bank)
    return out


def reservation(ctx, doc, chosen, by_id):
    """P2.7's projected cost from the measured 15-fit times of the menu's atoms at their final seed policies (C_all)
    and of the atoms inside the learned channels (C_L): the plan's projection 3.2 C_all + 16.2 C_L (section 8) and the
    P2.6a pilot's measured multipliers 3.04 and 14.16, doubled for the empty-cache replay, with a 25% margin, against
    the eight hours reserved. The projected path is reported; P2.7 decides the path from its own measurement."""
    elapsed = {r["run_id"]: r["elapsed_seconds"] for r in runs.logged(ctx)}
    atoms = set()
    learned = any(by_id[c]["kind"] == "learned" for c in chosen)
    for c in chosen:
        p = by_id[c]
        if p["kind"] == "atomic":
            atoms.add(p["run_id"])
        elif p["kind"] == "blend":
            atoms |= {m["run_id"] for m in p["members"]}
    lch = [ch for ch in doc["channels"] if ch["id"] in doc["learned_channels"]] if learned else []
    lruns = {r for ch in lch for r in ch["run_ids"]}
    atoms |= lruns
    c_all = sum(elapsed[r] for r in atoms)
    c_l = sum(elapsed[r] for r in lruns)
    plan = 3.2 * c_all + 16.2 * c_l
    pilot = 3.04 * c_all + 14.16 * c_l
    total = 1.25 * 2 * max(plan, pilot)
    return {"c_all_seconds": round(c_all, 1), "c_l_seconds": round(c_l, 1), "menu_atom_runs": sorted(atoms),
            "learned_channel_runs": sorted(lruns), "plan_projection_seconds": round(plan, 1),
            "pilot_projection_seconds": round(pilot, 1),
            "build_and_replay_with_margin_hours": round(total / 3600, 2), "reserved_hours": 8,
            "projected_path": "full" if total <= 8 * 3600 else "reduced",
            "note": "projections from the runs' logged 15-fit times, several measured while other fitting jobs ran "
                    "beside them (P2.4's full refit), so inflated; not runtime bounds. P2.7 decides the path from its "
                    "own measurement, made without concurrent jobs, before any outer score (plan section 8)"}


def evidence(cid, bank):
    return Evidence(cid, bank.train_row_id, bank.y, bank.group_id, tuple(int(s) for s in bank.split_seeds),
                    bank.fold_id, bank.oof, bank.fold_auc)


def main():
    ctx = runs.Context.load(Path("."))
    raw = PROCS.read_bytes()
    doc = json.loads(raw)
    cb = candidate_banks(ctx, doc)
    by_id = {p["id"]: p for p in doc["procedures"]}
    ev = {cid: evidence(cid, b) for cid, (_, _, b) in cb.items()}
    rule_menu = lambda ids: {"candidates": [{"id": i, **{k: by_id[i][k] for k in TIE_KEYS}} for i in ids]}  # noqa: E731
    pick = lambda ids: select_candidate({i: ev[i] for i in ids}, rule_menu(ids))["selected"]  # noqa: E731
    atoms = [i for i, p in by_id.items() if p["kind"] == "atomic"]
    anchor = "atom_p26_anchor"
    anchor_s3 = "atom_p26_anchor_s3"
    chosen, why = [], {}

    def take(cid, reason):
        if cid and cid in by_id and cid not in chosen and len(chosen) < MAX_MENU:
            chosen.append(cid)
            why[cid] = reason
    take(anchor, "the anchor")
    take(anchor_s3, "the anchor's three-seed version")
    take(pick([a for a in atoms if a not in (anchor, anchor_s3)]), "the standalone leader")
    take("family_average" if "family_average" in by_id else None, "the within-family average")
    fixed = [i for i, p in by_id.items() if p["kind"] == "blend" and i != "family_average"]
    take(pick(fixed) if fixed else None, "the best fixed blend")
    cons = [i for i, p in by_id.items() if p["kind"] == "learned" and p["setting"][0] == "constrained"]
    take(pick(cons) if cons else None, "the best constrained-weight procedure")
    stk = [i for i, p in by_id.items() if p["kind"] == "learned" and p["setting"][0] == "stacker"]
    take(pick(stk) if stk else None, "the best stacker")
    rest = [i for i in by_id if i not in chosen]
    while rest and len(chosen) < MAX_MENU:
        c = pick(rest)
        take(c, "filled by the selection rule")
        rest.remove(c)
    sel = select_candidate({i: ev[i] for i in chosen}, rule_menu(chosen))
    atom_runs = sorted({r for c in chosen for r in ([by_id[c]["run_id"]] if by_id[c]["kind"] == "atomic" else
                                                    [m["run_id"] for m in by_id[c].get("members", [])])}
                       | {r for ch in doc["channels"] for r in ch["run_ids"]})
    atom_recipes = {r: runs.load_logged(ctx, r)[0].metadata["recipe"] for r in atom_runs}
    menu = {"schema": "menu_v1", "frozen_by": "P2.6b", "procedures_sha256": artifacts.sha256_bytes(raw),
            "atom_recipes": atom_recipes,
            "nominations_sha256": doc["nominations_sha256"], "channels": doc["channels"],
            "learned_channels": doc["learned_channels"],
            "learned_channel_exclusions": doc["learned_channel_exclusions"],
            "meta_protocol": doc["meta_protocol"],
            "candidates": [dict({k: v for k, v in by_id[c].items()}, run_id=cb[c][0], bank_sha256=cb[c][1],
                                reason=why[c], exploratory_mean_fold_auc=float(cb[c][2].fold_auc.mean()),
                                exploratory_repeat_means=cb[c][2].fold_auc.mean(axis=1).tolist())
                           for c in chosen],
            "selection_rule": {"rank": "unrounded mean fold AUC over the 15 production folds", "tie_band": TIE_BAND,
                               "tie_keys": list(TIE_KEYS) + ["candidate id"],
                               "implementation": "songpop/phase2/selection.py select_candidate"},
            "production_selection": sel,
            "nested_protocol": {"outer_seed": nested.OUTER_SEED, "outer_folds": nested.OUTER_FOLDS,
                                "inner_seeds": list(nested.INNER_SEEDS), "inner_folds": nested.INNER_FOLDS,
                                "meta_folds": procedures.META_FOLDS, "fallback": nested.FALLBACK,
                                "refit_gain": nested.REFIT_GAIN, "anchor": anchor,
                                "implementation": "songpop/phase2/nested.py run and decide",
                                "reduced_path": "if the measured projection of build plus empty-cache replay exceeds "
                                                "eight hours, the fixed-candidate check: the production selection and "
                                                "the anchor each refitted in the three outer training parts; no "
                                                "fallback or deployment decision"},
            "deployment_default": "the equal average of the 15 fold models, or of the 15 fitted procedures",
            "p27_reservation": reservation(ctx, doc, chosen, by_id)}
    nested.executable_menu(menu)  # the frozen menu must be executable by the nested evaluator (R2 round 1, F1)
    MENU.write_text(json.dumps(menu, indent=1) + "\n")
    print(f"menu: {len(chosen)} candidates; production selection {sel['selected']} (maximum {sel['maximum']:.6f}, "
          f"tied {sel['tied']})")
    print(f"P2.7 reservation: {json.dumps(menu['p27_reservation'])}")
    for c in chosen:
        b = cb[c][2]
        print(f"  {c:44s} {b.fold_auc.mean():.6f} {np.round(b.fold_auc.mean(axis=1), 6).tolist()}  {why[c]}")


if __name__ == "__main__":
    main()
