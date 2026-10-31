"""Learned ensemble procedures (plan section 7.2, task P2.6a): grouped folds of any fitting population, atomic
out-of-fold predictions rebuilt inside it through a content-addressed cache, the constrained-weight combiner and the
regularized stacker, and the evaluation of a whole procedure on held-out groups.

A procedure fitted on rows D: D's five combiner-training folds (StratifiedGroupKFold on D alone, seeded from the
protocol coordinates, never from the candidate or its setting); for each channel, one out-of-fold probability per
row of D, every feature, encoding and stopping stage rebuilt inside each fold's training part, estimator seeds
averaged first; the combiner fitted on that matrix and D's labels; new rows predicted by the channels refitted on
all of D, then the combiner. Production CV runs this on each of the 15 outer training parts; the channels' fits on
an outer training part are exactly the channels' production fold fits, so their outer predictions come from the
channels' production banks, after checking that each bank is that channel's computation on the saved folds. The
global out-of-fold bank is never used as meta-training data."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from songpop import artifacts
from songpop.evaluate import PredictionBank, validate_bank
from songpop.phase2.encoders import FitContext
from songpop.phase2.engine import run_recipe_cv
from songpop.phase2.recipes import recipe_hash, validate_recipe

PROCEDURE_SCHEMA = "procedure_v1"
META_FOLDS = 5
META_PROTOCOL = "meta-v1"
CLIP = 1e-6  # the constrained-weight loss and the stacker's logits clip probabilities to [CLIP, 1 - CLIP]
SLSQP = {"ftol": 1e-12, "maxiter": 2000}
WEIGHT_CLEANUP = 1e-9  # solver weights below this are set to zero and the rest renormalized; larger negatives fail
STACKER = {"solver": "lbfgs", "tol": 1e-10, "max_iter": 5000}
LAMBDAS = (0.0, 0.01, 0.1)
CS = (0.01, 0.1, 1.0)


def coordinate_seed(*coords):
    """A seed from protocol coordinates only."""
    token = "|".join(map(str, coords)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") % (2 ** 31)


def population_folds(y, groups, row_ids, n_splits, seed):
    """Fold ids of a population: StratifiedGroupKFold on its rows in permanent row-id order, shuffled with the given
    seed. Every group lies in one fold, every fold is non-empty, and every training complement holds both classes."""
    y, groups, row_ids = np.asarray(y), np.asarray(groups), np.asarray(row_ids)
    order = np.argsort(row_ids, kind="stable")
    fold = np.full(len(y), -1, dtype=np.int64)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(sgkf.split(np.zeros((len(y), 1)), y[order], groups[order])):
        fold[order[va]] = k
    check_folds(y, groups, fold, n_splits)
    return fold


def check_folds(y, groups, fold, n_splits):
    """Exactly-once coverage with whole groups: every row in one fold in range, every fold non-empty, no group in two
    folds, both classes in every training complement."""
    fold = np.asarray(fold)
    if fold.shape != (len(y),) or fold.min() < 0 or fold.max() >= n_splits \
            or len(np.unique(fold)) != n_splits:
        raise ValueError("fold ids must cover every row once, each of the folds non-empty")
    per_group = {}
    for g, f in zip(np.asarray(groups), fold):
        if per_group.setdefault(g, f) != f:
            raise ValueError(f"group {g} is split across folds")
    for k in range(n_splits):
        if len(set(np.asarray(y)[fold != k])) < 2:
            raise ValueError(f"the training complement of fold {k} lacks a class")


@dataclass(frozen=True)
class Population:
    """A fitting population: raw rows indexed by permanent row id (exactly the recipe columns), labels, primary
    groups, and a name for its place in the protocol."""
    rows: object
    y: np.ndarray
    groups: np.ndarray
    name: str

    @property
    def row_ids(self):
        return np.asarray(self.rows.index, dtype=np.int64)

    def subset(self, positions, name):
        positions = np.asarray(positions)
        return Population(self.rows.iloc[positions], self.y[positions], self.groups[positions], name)


def frame_digest(frame):
    """The ids and raw values of a frame (NaN and strings included), independent of the process."""
    h = hashlib.sha256()
    h.update(np.asarray(frame.index, dtype=np.int64).tobytes())
    h.update(frame.to_csv(index=False, na_rep="NA").encode())
    return h.hexdigest()


class Cache:
    """Content-addressed atomic predictions: one entry per request, keyed by the hash of the whole request (fitting
    rows, labels, groups and raw inputs; the folds and every fit context; the recipe and its estimator seeds; the
    prediction rows and their raw inputs; the source manifest and the package versions). The stored request is
    compared with the caller's on every hit, so an entry from another population, fold assignment, context, query or
    source is refused, never repaired. directory=None keeps entries in memory only; a new namespace starts empty."""

    def __init__(self, directory=None):
        self.directory, self.memory, self.hits, self.misses = (Path(directory) if directory else None), {}, 0, 0

    @staticmethod
    def key(request):
        return artifacts.sha256_bytes(artifacts.dumps(request).encode())

    def get(self, request):
        k = self.key(request)
        entry = self.memory.get(k)
        if entry is None and self.directory is not None and (self.directory / f"{k}.npz").exists():
            with np.load(self.directory / f"{k}.npz", allow_pickle=False) as z:
                entry = {"request": json.loads(str(z["request_json"])), "oof": z["oof"], "pred": z["pred"]}
        if entry is None:
            self.misses += 1
            return None
        if artifacts.dumps(entry["request"]) != artifacts.dumps(request):
            raise ValueError(f"cache entry {k[:12]} holds another request; refused")
        if entry["oof"].shape != (request["folds_shape"][0], request["fitting"]["rows"]) \
                or not np.isfinite(entry["oof"]).all():
            raise ValueError(f"cache entry {k[:12]} is malformed; refused")
        self.hits += 1
        return entry["oof"], entry["pred"]

    def put(self, request, oof, pred):
        k = self.key(request)
        self.memory[k] = {"request": request, "oof": oof, "pred": pred}
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{k}.npz"
            if not path.exists():
                with open(path, "xb") as f:
                    np.savez(f, oof=oof, pred=pred, request_json=np.array(artifacts.dumps(request)))


def atomic_request(recipe, pop, fold_id, contexts, pred_rows, manifest_sha256, packages):
    return {"schema": "atomic_v1", "recipe_sha256": recipe_hash(recipe), "estimator_seeds": list(recipe["estimator_seeds"]),
            "fitting": {"name": pop.name, "rows": int(len(pop.y)), "inputs_sha256": frame_digest(pop.rows),
                        "labels_sha256": artifacts.sha256_bytes(np.asarray(pop.y, dtype=np.int64).tobytes()),
                        "groups_sha256": artifacts.sha256_bytes(np.asarray(pop.groups).astype(str).astype("U").tobytes())},
            "folds_sha256": artifacts.sha256_bytes(np.ascontiguousarray(fold_id, dtype=np.int64).tobytes()),
            "folds_shape": list(np.shape(fold_id)),
            "contexts": [[[str(c.split_seed), str(c.fold)] for c in row] for row in contexts],
            "prediction": {"rows": int(len(pred_rows)), "inputs_sha256": frame_digest(pred_rows)},
            "source_manifest_sha256": manifest_sha256, "packages": packages}


def atomic_cv(recipe, pop, fold_id, context_names, pred_rows, cache, manifest_sha256, packages):
    """The recipe cross-fitted inside a population: fold_id (R, N) over pop's rows, repeat r's fits with
    FitContext(context_names[r], k). Returns the out-of-fold probabilities (R, N), estimator seeds averaged inside
    the recipe, and each fold model's predictions of pred_rows (R, K, len(pred_rows)). Through the cache."""
    validate_recipe(recipe)
    fold_id = np.atleast_2d(np.asarray(fold_id))
    K = int(fold_id.max()) + 1
    contexts = [[FitContext(name, k) for k in range(K)] for name in context_names]
    req = atomic_request(recipe, pop, fold_id, contexts, pred_rows, manifest_sha256, packages)
    got = cache.get(req)
    if got is not None:
        return got
    if len(pred_rows) == 0:
        raise ValueError("at least one prediction row is needed")
    bank = run_recipe_cv(recipe, pop.rows, pop.y, pop.row_ids, pop.groups, fold_id, list(range(len(fold_id))),
                         pred_rows, np.arange(len(pred_rows)), n_splits=K, contexts=lambda r, k: contexts[r][k])
    cache.put(req, bank.oof.copy(), bank.test_by_fold.copy())
    return bank.oof.copy(), bank.test_by_fold.copy()


# ---- combiners -------------------------------------------------------------------------------------------------

def clip(p):
    return np.clip(np.asarray(p, dtype="float64"), CLIP, 1 - CLIP)


def fit_weights(Z, y, lam):
    """Nonnegative weights summing to one minimizing the mean binary log loss of the clipped convex combination
    plus lam times the squared distance to equal weights; SLSQP from equal weights with the analytic gradient.
    Failure (solver status, infeasibility beyond WEIGHT_CLEANUP, non-finite values, a loss above equal weights')
    raises; nothing falls back to equal weights."""
    Z, y = np.asarray(Z, dtype="float64"), np.asarray(y, dtype="float64")
    J = Z.shape[1]
    eq = np.full(J, 1.0 / J)

    def loss(w):
        p = Z @ w
        q = clip(p)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)) + lam * np.sum((w - eq) ** 2))

    def grad(w):
        p = Z @ w
        q = clip(p)
        d = (-(y / q) + (1 - y) / (1 - q)) * ((p > CLIP) & (p < 1 - CLIP))
        return Z.T @ d / len(y) + 2 * lam * (w - eq)

    res = minimize(loss, eq, jac=grad, method="SLSQP", bounds=[(0.0, 1.0)] * J,
                   constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones(J)}],
                   options=SLSQP)
    raw = np.asarray(res.x, dtype="float64")
    if not res.success or not np.isfinite(raw).all() or raw.min() < -WEIGHT_CLEANUP or abs(raw.sum() - 1) > WEIGHT_CLEANUP:
        raise ValueError(f"constrained weights failed: {res.message}, weights {raw.tolist()}")
    w = np.where(raw < WEIGHT_CLEANUP, 0.0, raw)
    w = w / w.sum()
    if loss(w) > loss(eq) + 1e-12:
        raise ValueError("constrained weights end above the equal-weight loss")
    return {"kind": "constrained", "lambda": float(lam), "raw_weights": raw.tolist(), "weights": w.tolist(),
            "loss": loss(w), "equal_weight_loss": loss(eq), "iterations": int(res.nit), "message": str(res.message)}


def fit_stacker(Z, y, C):
    """Clipped probabilities to logits, standardized on the meta-training matrix, then L2 logistic regression with
    an intercept and the declared C; no original features, no class weights. A convergence warning raises."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    L = np.log(clip(Z) / (1 - clip(Z)))
    sc = StandardScaler().fit(L)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        lr = LogisticRegression(C=C, penalty="l2", fit_intercept=True, class_weight=None, **STACKER).fit(
            sc.transform(L), np.asarray(y))
    if list(lr.classes_) != [0, 1]:
        raise ValueError(f"stacker classes {list(lr.classes_)}")
    return {"kind": "stacker", "C": float(C), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "var": sc.var_.tolist(), "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
            "iterations": int(lr.n_iter_[0])}


def apply_combiner(state, Z):
    Z = np.asarray(Z, dtype="float64")
    if state["kind"] == "constrained":
        return Z @ np.asarray(state["weights"])  # the convex combination itself; only the loss clips
    L = np.log(clip(Z) / (1 - clip(Z)))
    s = ((L - np.asarray(state["mean"])) / np.asarray(state["scale"])) @ np.asarray(state["coef"]) + state["intercept"]
    return 1.0 / (1.0 + np.exp(-s))


def fit_combiner(setting, Z, y):
    kind, value = setting
    return fit_weights(Z, y, value) if kind == "constrained" else fit_stacker(Z, y, value)


# ---- channels and whole procedures -----------------------------------------------------------------------------

def channel_cv(channel, pop, fold_id, context_names, pred_rows, cache, manifest_sha256, packages_of):
    """A channel cross-fitted inside a population: an atomic recipe's predictions, or the equal average of a
    within-family average's atoms (each atom's estimator seeds averaged first)."""
    outs = [atomic_cv(r, pop, fold_id, context_names, pred_rows, cache, manifest_sha256, packages_of(r))
            for r in channel["atoms"]]
    return np.mean([o for o, _ in outs], axis=0), np.mean([p for _, p in outs], axis=0)


def check_channel_bank(channel, bank, fold_id, row_ids, recipe_of_run):
    """The production bank that supplies a channel's outer-fold predictions must be that channel's computation on
    the saved folds: an atom's own run, or the equal blend of the atoms' runs (recipe_of_run maps a run id to its
    canonical recipe hash)."""
    if not (np.array_equal(bank.fold_id, fold_id) and np.array_equal(bank.train_row_id, row_ids)):
        raise ValueError("a channel bank has other rows or folds")
    md = bank.metadata
    atoms = sorted(recipe_hash(r) for r in channel["atoms"])
    if len(atoms) == 1:
        ok = md.get("recipe_sha256") == atoms[0]
    else:
        ok = md.get("prediction_kind") == "blend" and len(md["parents"]) == len(atoms) \
            and all(abs(p["weight"] - 1 / len(atoms)) <= 1e-12 for p in md["parents"]) \
            and sorted(recipe_of_run.get(p["run_id"], "") for p in md["parents"]) == atoms
    if not ok:
        raise ValueError(f"the bank {md.get('run_id')} is not the computation of channel {channel['id']}")


def learn_on(D, channels, settings, meta_seed, context_name, cache, manifest_sha256, packages_of):
    """Fit every setting's combiner on population D: D's meta folds from meta_seed, each channel's out-of-fold
    probabilities on D shared by all settings. Returns ({setting: state}, receipt)."""
    meta = population_folds(D.y, D.groups, D.row_ids, META_FOLDS, meta_seed)
    Z = np.column_stack([channel_cv(ch, D, meta, [context_name], D.rows.iloc[:1], cache, manifest_sha256,
                                    packages_of)[0][0] for ch in channels])
    receipt = {"population": D.name, "meta_rows": int(len(D.y)), "meta_folds_sha256": artifacts.sha256_bytes(
        np.column_stack([D.row_ids, meta]).astype(np.int64).tobytes())}
    return {s: fit_combiner(s, Z, D.y) for s in settings}, receipt


def evaluate_learned(channels, settings, channel_outer, pop, fold_id, split_seeds, cache, manifest_sha256,
                     packages_of, protocol=META_PROTOCOL):
    """Every learned setting over the same ordered channels, evaluated as a whole on each fold (r, k) of fold_id:
    D the training part, its meta folds seeded from (protocol, population, split seed, k), each combiner fitted on
    D and applied to the channels' predictions from their fits on D, channel_outer[id] = (out-of-fold (R, N),
    per-fold predictions of the query rows (R, K, T)). Returns {setting: {"oof", "test", "states"}} and the
    meta-fold receipts."""
    R, N = fold_id.shape
    K = int(fold_id.max()) + 1
    T = next(iter(channel_outer.values()))[1].shape[2]
    out = {s: {"oof": np.full((R, N), np.nan), "test": np.full((R, K, T), np.nan), "states": []} for s in settings}
    receipts = []
    for r, seed in enumerate(split_seeds):
        for k in range(K):
            tr, va = np.flatnonzero(fold_id[r] != k), np.flatnonzero(fold_id[r] == k)
            D = pop.subset(tr, f"{pop.name}/{seed}:{k}")
            states, rc = learn_on(D, channels, settings, coordinate_seed(protocol, pop.name, seed, k),
                                  f"{protocol}:{pop.name}:{seed}:{k}", cache, manifest_sha256, packages_of)
            receipts.append(dict(rc, split_seed=str(seed), fold=k))
            Zva = np.column_stack([channel_outer[ch["id"]][0][r, va] for ch in channels])
            Zte = np.column_stack([channel_outer[ch["id"]][1][r, k] for ch in channels])
            for s in settings:
                out[s]["oof"][r, va] = apply_combiner(states[s], Zva)
                out[s]["test"][r, k] = apply_combiner(states[s], Zte)
                out[s]["states"].append(dict(states[s], split_seed=str(seed), fold=k))
    return out, receipts


def procedure_bank(arrays, y, row_ids, groups, fold_id, split_seeds, test_ids, metadata):
    """A prediction bank of a whole procedure: every fold AUC recomputed from its out-of-fold probabilities."""
    R, K = len(split_seeds), arrays["test"].shape[1]
    fold_auc = np.array([[roc_auc_score(y[fold_id[r] == k], arrays["oof"][r, fold_id[r] == k]) for k in range(K)]
                         for r in range(R)])
    bank = PredictionBank(np.asarray(row_ids), np.asarray(y), np.asarray(groups), np.asarray(split_seeds),
                          np.asarray(fold_id), arrays["oof"], np.asarray(test_ids), arrays["test"],
                          arrays["test"].mean(axis=(0, 1)), fold_auc, dict(metadata, schema=PROCEDURE_SCHEMA))
    validate_bank(bank, n_splits=K)
    return bank


def save_procedure_bank(bank, path):
    """Write once (as bank_v2 files are written) and return the sha256."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"bank {path} exists; banks are immutable")
    if bank.metadata.get("schema") != PROCEDURE_SCHEMA:
        raise ValueError("not a procedure bank")
    b = artifacts.bank_bytes(bank)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(b)
    return artifacts.sha256_bytes(b)
