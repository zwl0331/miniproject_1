"""Recipe-aware submission checks (plan section 4.7). Fitted models are not saved. At fit time the engine checks
every estimator seed: identical model inputs get identical test predictions (songpop.phase2.engine). At
submission time each member bank is first matched to the production protocol and to its source files, then each
fold's model inputs are rebuilt (the feature pipeline refitted on the fold's training rows, then the recipe's
preprocessing fitted on that cross-fitted training matrix, both deterministic) and the bank's per-fold test
predictions, the seed average of each recipe, are compared within groups of identical inputs."""
import io
from pathlib import Path

import numpy as np
import pandas as pd

from songpop import artifacts
from songpop.data import AUDIO, DATE
from songpop.phase2 import banks
from songpop.phase2.encoders import FitContext
from songpop.phase2.engine import input_classes
from songpop.phase2.ensembles import blend_banks
from songpop.phase2.features import FeaturePipeline
from songpop.phase2.models import make_preprocessing
from songpop.phase2.runs import check_sources

TOL = banks.TOL


def _spread(p, classes):
    return max([float(np.ptp(p[idx])) for idx in classes], default=0.0)


def repeated_uri_pairs(test):
    """Positions of the test rows sharing a uri, enumerated from the raw file, independent of any features."""
    t = test.reset_index(drop=True)
    return [list(g.index) for _, g in t.groupby("uri") if len(g) > 1]


def check_production(ctx, bank):
    """The bank holds the production evaluation population, labels, groups, test ids, data and saved folds, and
    its source files and dependencies are the ones present now. Raises ValueError or runs.SourceMismatch."""
    md = bank.metadata
    seeds = [int(s) for s in bank.split_seeds]
    same = {"train_row_id": np.array_equal(bank.train_row_id, ctx.data.row_ids),
            "y": np.array_equal(bank.y, ctx.data.y), "group_id": np.array_equal(bank.group_id, ctx.groups),
            "test_id": np.array_equal(bank.test_id, ctx.data.test_id),
            "fold_id": np.array_equal(bank.fold_id, ctx.fold_id(seeds)),
            "data_hashes": md.get("data_hashes") == ctx.data.hashes}
    bad = [k for k, ok in same.items() if not ok]
    if bad:
        raise ValueError(f"bank {md['run_id']} differs from the production protocol in {bad}")
    config = dict(md["config"], **({"recipe": md["recipe"]} if "recipe" in md else {}))
    check_sources(ctx, md["source_manifest"], None, config)


def member_input_classes(ctx, bank):
    """For each repeat and fold of a fitted bank, the classes of test rows with identical model inputs."""
    recipe, rows, y, groups = bank.metadata["recipe"], ctx.rows, ctx.data.y, ctx.groups
    out = []
    for r, s in enumerate(bank.split_seeds):
        for k in range(bank.fold_auc.shape[1]):
            tr = np.flatnonzero(bank.fold_id[r] != k)
            fp = FeaturePipeline(recipe)
            X_train = fp.fit_transform(rows.iloc[tr], y[tr], groups[tr], FitContext(int(s), k))
            if fp.columns_ != bank.metadata["features"]:
                raise ValueError("rebuilt feature columns differ from the bank's")
            X_test = fp.transform(ctx.test_rows)
            prep = make_preprocessing(recipe["preprocessing"], list(X_train.columns))
            Z = X_test.to_numpy(dtype="float64") if prep is None else prep.fit(X_train, y[tr]).transform(X_test)
            out.append(((r, k), input_classes(Z)))
    return out


def members_of(ctx, bank, load):
    """[(fitted bank, sha256)] behind a candidate: its parents for a blend, else itself. load(run_id) returns
    (bank, sha256) from the log."""
    if bank.metadata["prediction_kind"] != "blend":
        return [(bank, None)]
    out = []
    for p in bank.metadata["parents"]:
        b, sha = load(p["run_id"])
        if sha != p["bank_sha256"]:
            raise ValueError(f"parent {p['run_id']} differs from its recorded hash")
        if b.metadata["prediction_kind"] == "blend":
            raise ValueError("a blend of blends is not supported")
        out.append((b, sha))
    return out


def validate_candidate(content, ctx, bank, members):
    """Check submission bytes against the sample, the candidate bank and its fitted members. Raises ValueError
    (or runs.SourceMismatch) on the first failure; returns a report."""
    data = ctx.data
    banks.validate_v2(bank)
    if not bank.metadata["selection_eligible"] or [int(s) for s in bank.split_seeds] != [155, 2026, 1029]:
        raise ValueError("a candidate needs the complete three-repeat bank")
    check_production(ctx, bank)
    for b, _ in members:
        banks.validate_v2(b)
        if not b.metadata["selection_eligible"] or [int(s) for s in b.split_seeds] != [155, 2026, 1029]:
            raise ValueError(f"member {b.metadata['run_id']} lacks the complete three-repeat bank")
        check_production(ctx, b)
    if bank.metadata["prediction_kind"] == "blend":
        rebuilt = blend_banks(members, [p["weight"] for p in bank.metadata["parents"]])
        for f in ("oof", "test_by_fold", "test_mean", "fold_auc"):
            if not np.allclose(getattr(rebuilt, f), getattr(bank, f), rtol=0, atol=TOL):
                raise ValueError(f"the blend's {f} differs from its parents' weighted average")
    df = pd.read_csv(io.BytesIO(content))
    if list(df.columns) != ["ID", "Popularity_Type"]:
        raise ValueError(f"columns are {list(df.columns)}, expected ['ID', 'Popularity_Type']")
    if len(df) != len(data.sample):
        raise ValueError(f"{len(df)} rows, expected {len(data.sample)}")
    if not df["ID"].is_unique:
        raise ValueError("duplicate IDs")
    if list(df["ID"]) != list(data.sample["ID"]):
        raise ValueError("IDs differ from the sample submission's order")
    p = pd.to_numeric(df["Popularity_Type"], errors="coerce").to_numpy(dtype="float64")
    if not np.all(np.isfinite(p)):
        raise ValueError("missing or non-finite probabilities")
    if p.min() < 0 or p.max() > 1:
        raise ValueError("probabilities outside [0, 1]")
    if not np.allclose(p, bank.test_mean, rtol=0, atol=TOL):
        raise ValueError(f"probabilities differ from the deployment output by up to "
                         f"{np.max(np.abs(p - bank.test_mean)):.3g}")
    pairs = repeated_uri_pairs(data.test)
    composite, member_classes = None, 0
    for b, _ in members:
        for (r, k), classes in member_input_classes(ctx, b):
            member_classes += len(classes)
            if _spread(b.test_by_fold[r, k], classes) > TOL:
                raise ValueError(f"member {b.metadata['run_id']} repeat {r} fold {k}: identical model inputs get "
                                 f"different predictions")
            if _spread(b.test_by_fold[r, k], pairs) > TOL:
                raise ValueError(f"member {b.metadata['run_id']} repeat {r} fold {k}: a repeated uri pair gets "
                                 f"different predictions")
            label = np.full(len(p), -1)
            for c, idx in enumerate(classes):
                label[idx] = c
            label[label < 0] = -1 - np.flatnonzero(label < 0)  # singletons get unique labels
            composite = label[:, None] if composite is None else np.hstack([composite, label[:, None]])
    composite_classes = input_classes(composite)
    if _spread(p, composite_classes) > TOL:
        raise ValueError("test rows equivalent for every member get different composite predictions")
    if _spread(p, pairs) > TOL:
        raise ValueError("a repeated uri pair gets different composite predictions")
    raw_classes = input_classes(pd.DataFrame({c: pd.factorize(data.test[c].astype(str))[0] for c in AUDIO + [DATE]}))
    if _spread(p, raw_classes) > TOL:
        raise ValueError("test rows with identical raw audio features and date get different predictions")
    q = [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]
    return {"rows": len(df), "sha256": artifacts.sha256_bytes(content), "run_id": bank.metadata["run_id"],
            "members": [b.metadata["run_id"] for b, _ in members],
            "min": float(p.min()), "max": float(p.max()), "mean": float(p.mean()),
            "quantiles": dict(zip(map(str, q), np.quantile(p, q).round(6).tolist())),
            "unique_values": int(len(np.unique(p))), "member_fold_duplicate_classes": member_classes,
            "composite_duplicate_classes": len(composite_classes), "raw_duplicate_classes": len(raw_classes),
            "repeated_uri_pairs_checked": len(pairs)}


def write_candidate(path, ctx, bank, members):
    """Write the bank's deployment output once, read the bytes back and validate those."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"submission {path} exists; submissions are never overwritten")
    content = artifacts.submission_bytes(bank.test_id, bank.test_mean)
    validate_candidate(content, ctx, bank, members)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(content)
    back = path.read_bytes()
    if back != content:
        raise ValueError("the bytes read back differ from the bytes written")
    report = validate_candidate(back, ctx, bank, members)
    report["path"] = str(path)
    return report
