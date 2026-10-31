"""Checks of two claims made by the AUC brainstorm of 2026-10-30/31.

The brainstorm (a read-only Claude session with three sub-agents, recorded under
doc/llm_usage/2026-10-31-auc-brainstorm/) claimed that (1) the test file is an 80/20 row
split from which every test row whose track also occurs in train was removed, so the test
population is singleton tracks, and (2) scoring on singleton validation rows lowers every
score by a family-dependent amount and leaves the ranking intact.

This script checks both on the project's own data, banks and frozen menu. It reads only;
it fits nothing. Run from the project root:

    .venv/bin/python analysis/brainstorm_checks.py | tee analysis/brainstorm_checks_output.txt

Part 1 uses data/*.csv. Part 2 rescores the 12 candidates of configs/phase2/menu.json and
its 8 atom runs from their logged banks on two validation populations: all eligible rows
(the official score) and singleton rows (primary group of size 1 among the eligible rows).
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ["time_signature", "speechiness", "danceability", "duration_ms", "energy", "mode",
         "instrumentalness", "valence", "key", "tempo", "loudness", "acousticness", "liveness"]
TIE_BAND = 0.001


def head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def part1():
    print("## Part 1: how the test file relates to the training file")
    train = pd.read_csv(ROOT / "data/train.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    sample = pd.read_csv(ROOT / "data/sample_submission.csv")
    print(f"train rows {len(train)}; test rows {len(test)}; sample rows {len(sample)}; "
          f"test ids equal the sample's ids in order: {bool((test['ID'].values == sample['ID'].values).all())}")
    ids = np.sort(test["ID"].to_numpy())
    missing = int(ids.max() - ids.min() + 1 - len(np.unique(ids)))
    print(f"test ids: min {ids.min()}, max {ids.max()}, unique {len(np.unique(ids))}, "
          f"ids missing inside the range {missing}")
    total = len(train) + len(test) + missing
    print(f"train + test + missing ids = {total}; train share {len(train) / total:.6f} "
          f"(0.8 exactly would be {0.8 * total:.1f} train rows)")
    labels = set(train["Popularity_Type"].unique())
    if labels != {"High", "Low"}:
        print(f"unexpected labels {labels}; stopping")
        sys.exit(1)
    y = (train["Popularity_Type"] == "High").astype(int)
    tr_uri, te_uri = train["uri"], test["uri"]
    overlap = set(tr_uri) & set(te_uri)
    print(f"test uris that also occur in train: {len(overlap)} uris, {int(te_uri.isin(overlap).sum())} test rows")
    vc_te = te_uri.value_counts()
    rep_te = vc_te[vc_te > 1]
    print(f"within-test repeated uris: {len(rep_te)} uris, {int(rep_te.sum())} rows; "
          f"copies per repeated uri {sorted(rep_te.unique().tolist())}")
    vc_tr = tr_uri.value_counts()
    rep_tr = vc_tr[vc_tr > 1]
    dist = rep_tr.value_counts().sort_index()
    print(f"within-train repeated uris: {len(rep_tr)} uris, {int(rep_tr.sum())} rows; "
          f"copies per repeated uri {dict(zip(dist.index.tolist(), dist.values.tolist()))}")
    is_rep = tr_uri.isin(rep_tr.index)
    print(f"High rate: repeated-uri rows {y[is_rep].mean():.4f} (n {int(is_rep.sum())}); "
          f"singleton-uri rows {y[~is_rep].mean():.4f} (n {int((~is_rep).sum())}); all rows {y.mean():.4f}")
    sig_cols = AUDIO + ["track_album_release_date"]
    tr_sig = set(train[sig_cols].apply(lambda r: "|".join(map(str, r.tolist())), axis=1))
    te_sig = test[sig_cols].apply(lambda r: "|".join(map(str, r.tolist())), axis=1)
    tr_aud = set(train[AUDIO].apply(lambda r: "|".join(map(str, r.tolist())), axis=1))
    te_aud = test[AUDIO].apply(lambda r: "|".join(map(str, r.tolist())), axis=1)
    print(f"test rows whose 13 audio values and date string occur together in train: {int(te_sig.isin(tr_sig).sum())}; "
          f"whose 13 audio values occur in train with any date: {int(te_aud.isin(tr_aud).sum())}")
    print(f"test rows with a repeated uri as a share of test: {int(rep_te.sum()) / len(test):.4f}; "
          f"train rows with a repeated uri as a share of train: {int(rep_tr.sum()) / len(train):.4f}")
    print()


def bank_paths_by_sha():
    out = {}
    with open(ROOT / "artifacts/phase2/experiments.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("bank_sha256") and r.get("bank_path"):
                out[r["bank_sha256"]] = r["bank_path"]
    return out


def bank_paths_by_run_id():
    out = {}
    with open(ROOT / "artifacts/phase2/experiments.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("record") == "run" and r.get("run_id") and r.get("bank_path"):
                out[r["run_id"]] = r["bank_path"]
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bank(path):
    with np.load(ROOT / path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def fold_scores(bank, masks):
    """Mean fold AUC per population mask, with per-seed means. masks: name -> bool array over bank rows."""
    fold_id, oof, y = bank["fold_id"], bank["oof"], bank["y"]
    out = {}
    for name, mask in masks.items():
        per = np.zeros((fold_id.shape[0], 5))
        for s in range(fold_id.shape[0]):
            for k in range(5):
                v = (fold_id[s] == k) & mask
                per[s, k] = roc_auc_score(y[v], oof[s][v])
        out[name] = per
    return out


def select(scores, cands):
    m = max(scores.values())
    tied = [c for c in cands if scores[c["id"]] >= m - TIE_BAND]
    key = lambda c: (c["atomic_recipes"], c["supervised_transformations"], c["learned_combiners"],  # noqa: E731
                     c["complexity_rank"], c["id"])
    return min(tied, key=key)["id"], sorted(c["id"] for c in tied)


def part2():
    print("## Part 2: the frozen menu rescored on singleton validation rows")
    menu = json.loads((ROOT / "configs/phase2/menu.json").read_text())
    by_sha = bank_paths_by_sha()
    by_run = bank_paths_by_run_id()
    splits = pd.read_csv(ROOT / "artifacts/splits/production_v1.csv")
    elig = splits[splits["eligible"] == 1]
    first = None
    rows = []
    for cand in menu["candidates"]:
        path = cand.get("bank_path") or by_sha.get(cand["bank_sha256"])
        if path is None:
            for p in sorted((ROOT / "artifacts/phase2/bank").glob("*.npz")):
                if sha256(p) == cand["bank_sha256"]:
                    path = str(p.relative_to(ROOT))
                    break
        if path is None:
            print(f"no bank found for {cand['id']} (sha {cand['bank_sha256'][:12]}); stopping")
            sys.exit(1)
        if sha256(ROOT / path) != cand["bank_sha256"]:
            print(f"bank hash mismatch for {cand['id']} at {path}; stopping")
            sys.exit(1)
        bank = load_bank(path)
        if first is None:
            first = bank
            assert (bank["train_row_id"] == elig["train_row_id"].to_numpy()).all(), "bank rows differ from the eligible rows"
            assert (bank["group_id"] == elig["primary_group"].to_numpy()).all(), "bank groups differ from the primary groups"
            for i, seed in enumerate([155, 2026, 1029]):
                assert (bank["fold_id"][i] == elig[f"primary_fold_{seed}"].to_numpy()).all(), f"folds differ for seed {seed}"
            sizes = pd.Series(bank["group_id"]).map(pd.Series(bank["group_id"]).value_counts()).to_numpy()
            single = sizes == 1
            print(f"eligible rows {len(single)}; singleton rows (primary group of size 1) {int(single.sum())}; "
                  f"rows in repeated groups {int((~single).sum())}; High rate singleton {bank['y'][single].mean():.4f}, "
                  f"repeated {bank['y'][~single].mean():.4f}, all {bank['y'].mean():.4f}")
            per_fold = [int(((bank['fold_id'][0] == k) & single).sum()) for k in range(5)]
            print(f"singleton validation rows per fold, seed 155: {per_fold}")
            masks = {"all": np.ones(len(single), bool), "singleton": single, "repeated": ~single}
        else:
            assert (bank["train_row_id"] == first["train_row_id"]).all() and (bank["fold_id"] == first["fold_id"]).all()
        sc = fold_scores(bank, masks)
        rec_all = sc["all"].mean()
        if abs(rec_all - cand["exploratory_mean_fold_auc"]) > 1e-9:
            print(f"recomputed all-rows mean {rec_all:.10f} differs from the menu's {cand['exploratory_mean_fold_auc']:.10f} for {cand['id']}; stopping")
            sys.exit(1)
        rows.append({"id": cand["id"], "kind": cand["kind"], "all": rec_all, "singleton": sc["singleton"].mean(),
                     "singleton_by_seed": sc["singleton"].mean(axis=1), "repeated": sc["repeated"].mean(),
                     "gap": rec_all - sc["singleton"].mean(), "cand": cand})
    rows.sort(key=lambda r: -r["all"])
    rank_all = {r["id"]: i + 1 for i, r in enumerate(rows)}
    rank_single = {r["id"]: i + 1 for i, r in enumerate(sorted(rows, key=lambda r: -r["singleton"]))}
    print()
    print("| candidate | kind | all rows (official) | singleton rows | by seed 155, 2026, 1029 | repeated rows | all minus singleton | rank all | rank singleton |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        seeds = ", ".join(f"{v:.4f}" for v in r["singleton_by_seed"])
        print(f"| {r['id']} | {r['kind']} | {r['all']:.4f} | {r['singleton']:.4f} | {seeds} | {r['repeated']:.4f} | "
              f"{r['gap']:+.4f} | {rank_all[r['id']]} | {rank_single[r['id']]} |")
    rho = spearmanr([r["all"] for r in rows], [r["singleton"] for r in rows]).statistic
    print()
    print(f"Spearman correlation of the two scores over the 12 candidates: {rho:.3f}")
    cands = [r["cand"] for r in rows]
    sel_all, tied_all = select({r["id"]: r["all"] for r in rows}, cands)
    sel_single, tied_single = select({r["id"]: r["singleton"] for r in rows}, cands)
    print(f"frozen rule on the official score: selected {sel_all}; tie set {tied_all}")
    print(f"the same rule on the singleton score (a diagnostic, not the rule): selected {sel_single}; tie set {tied_single}")
    print(f"menu production_selection: {menu['production_selection']['selected']}")
    print()
    print("### The menu's 8 atom runs")
    print()
    print("| run | family | blocks | all rows | singleton rows | by seed | all minus singleton |")
    print("|---|---|---|---|---|---|---|")
    atoms = []
    for run_id in menu["p27_reservation"]["menu_atom_runs"]:
        path = by_run.get(run_id)
        if path is None:
            print(f"| {run_id} | no run record with a bank path | | | | | |")
            continue
        bank = load_bank(path)
        assert (bank["fold_id"] == first["fold_id"]).all()
        sc = fold_scores(bank, masks)
        recipe = menu["atom_recipes"][run_id]
        seeds = ", ".join(f"{v:.4f}" for v in sc["singleton"].mean(axis=1))
        atoms.append((recipe["family"], sc["all"].mean() - sc["singleton"].mean()))
        print(f"| {recipe['name']} | {recipe['family']} | {recipe['blocks']} | {sc['all'].mean():.4f} | "
              f"{sc['singleton'].mean():.4f} | {seeds} | {sc['all'].mean() - sc['singleton'].mean():+.4f} |")
    print()


def main():
    print(f"# brainstorm_checks: run {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} at commit {head()}")
    print()
    part1()
    part2()


if __name__ == "__main__":
    main()
