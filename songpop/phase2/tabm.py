"""TabM as a recipe family (plan sections 5.5 and 19): a classifier that owns its preprocessing, chooses its epoch
count on an inner group-safe split of the fitting rows, and is then refitted on all of them.

torch runs only in a child process. LightGBM and XGBoost link Homebrew's OpenMP runtime and torch ships its own; a
process that loads LightGBM or XGBoost before using torch crashes (Finding 024). So the estimator in the fitting
process holds the fitted preprocessing and the network's weights as NumPy arrays, and sends every training and
every forward pass to one persistent worker, `python -m songpop.phase2.tabm`, which imports only NumPy,
scikit-learn, torch and tabm. The worker keeps nothing between requests: a fit seeds torch from the seeds it is
sent, and a prediction rebuilds the network from the weights it is sent. This module must not import LightGBM or
XGBoost, directly or through another project module."""
import atexit
import contextlib
import hashlib
import os
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CATEGORICAL = ["key", "mode", "time_signature"]  # as songpop.phase2.models.CATEGORICAL (checked)
LOG100 = ["instrumentalness", "speechiness", "liveness", "acousticness"]  # as songpop.models.LOG100 (checked)
AUDIO_CONTINUOUS = ["speechiness", "danceability", "duration_ms", "energy", "instrumentalness", "valence", "tempo",
                    "loudness", "acousticness", "liveness"]
MISSING_FLAGS = ["month", "day", "day_of_year", "weekday", "days"]  # F2 fields unobserved for partial dates
SCALERS = ("standard", "quantile")
ITERATIONS = "early_stop_refit_v1"  # the recipe's iteration policy for this family
PREDICTION_BATCH = 256
ADAMW = {"betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False, "foreach": False, "fused": False}


def log_columns(X):
    """log1p(duration_ms / 1000) and log1p(100 x) for the skewed bounded audio features (as phase 1)."""
    X = X.copy()
    if "duration_ms" in X:
        X["duration_ms"] = np.log1p(X["duration_ms"] / 1000)
    for c in LOG100:
        if c in X:
            X[c] = np.log1p(100 * X[c])
    return X


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype="float64"))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class TabMPreprocessor:
    """Fitted on the rows it is given, never on others. "standard": the phase 1 log transforms, then median imputation
    and standardization of every continuous column. "quantile": the ten continuous audio columns median-imputed and
    quantile-transformed to the normal distribution (at most 256 quantiles, no subsampling); the calendar columns
    median-imputed and standardized. Both: a missing indicator for each of the declared F2 fields present
    (MISSING_FLAGS, fixed by the recipe, not by which values happen to be missing in a fold); key, mode and time
    signature imputed by their most frequent value and one-hot encoded with categories from the fitting rows (an
    unseen category is all zero). An all-empty column is imputed as zero and scales to zero."""

    def __init__(self, scaler):
        if scaler not in SCALERS:
            raise ValueError(f"unknown TabM scaler {scaler!r}")
        self.scaler = scaler

    def fit(self, X):
        self.columns_ = list(X.columns)
        self.cat_ = [c for c in self.columns_ if c in CATEGORICAL]
        cont = [c for c in self.columns_ if c not in CATEGORICAL]
        self.quant_ = [c for c in cont if c in AUDIO_CONTINUOUS] if self.scaler == "quantile" else []
        self.std_ = [c for c in cont if c not in self.quant_]
        self.flag_ = [c for c in self.columns_ if c in MISSING_FLAGS]
        L = log_columns(X) if self.scaler == "standard" else X
        self.std_imp_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(L[self.std_].to_numpy("float64"))
        self.std_sc_ = StandardScaler().fit(self.std_imp_.transform(L[self.std_].to_numpy("float64")))
        stats = [self.std_imp_.statistics_, self.std_sc_.mean_, self.std_sc_.scale_]
        if self.quant_:
            Q = X[self.quant_].to_numpy("float64")
            self.q_imp_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(Q)
            self.qt_ = QuantileTransformer(n_quantiles=min(256, len(X)), output_distribution="normal", subsample=None,
                                           random_state=0).fit(self.q_imp_.transform(Q))
            stats += [self.q_imp_.statistics_, self.qt_.quantiles_]
        if self.cat_:
            C = X[self.cat_].to_numpy("float64")
            self.cat_imp_ = SimpleImputer(strategy="most_frequent").fit(C)
            self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype="float64").fit(
                self.cat_imp_.transform(C))
            stats += [self.cat_imp_.statistics_] + list(self.ohe_.categories_)
        self.n_fitted_ = len(X)
        self.fitted_rows_sha256_ = hashlib.sha256(np.sort(np.asarray(X.index)).astype(np.int64).tobytes()).hexdigest()
        self.digest_ = digest(*stats)
        return self

    def output_columns(self):
        out = list(self.std_) + list(self.quant_) + [f"missing_{c}" for c in self.flag_]
        return out + [f"{c}={v:g}" for c, cats in zip(self.cat_, self.ohe_.categories_ if self.cat_ else [])
                      for v in cats]

    def transform(self, X):
        if list(X.columns) != self.columns_:
            raise ValueError(f"columns {list(X.columns)} differ from the fitted {self.columns_}")
        L = log_columns(X) if self.scaler == "standard" else X
        parts = [self.std_sc_.transform(self.std_imp_.transform(L[self.std_].to_numpy("float64")))]
        if self.quant_:
            parts.append(self.qt_.transform(self.q_imp_.transform(X[self.quant_].to_numpy("float64"))))
        if self.flag_:
            parts.append(np.isnan(X[self.flag_].to_numpy("float64")).astype("float64"))
        if self.cat_:
            parts.append(self.ohe_.transform(self.cat_imp_.transform(X[self.cat_].to_numpy("float64"))))
        Z = np.ascontiguousarray(np.hstack(parts), dtype="float32")
        if not np.isfinite(Z).all():
            raise ValueError("non-finite TabM inputs")
        return Z


def inner_split(y, groups, n_folds, seed):
    """The first fold of a seeded StratifiedGroupKFold of the fitting rows: (inner training, inner validation)
    positions; both sides must hold both classes, and no group may be on both sides."""
    y, groups = np.asarray(y), np.asarray(groups)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(np.zeros((len(y), 1)), y, groups))
    if set(groups[tr]) & set(groups[va]) or len(tr) + len(va) != len(y):
        raise AssertionError("the inner split is not a group-safe partition of the fitting rows")
    if len(set(y[tr])) < 2 or len(set(y[va])) < 2:
        raise ValueError("a side of the inner split lacks a class")
    return tr, va


def stop_rule(losses, tolerance, patience):
    """(stop now, chosen epoch so far): patience counts epochs since the loss last fell more than `tolerance` below
    the patience reference; the chosen epoch (1-based) is the earliest whose loss is within `tolerance` of the
    lowest so far."""
    ref, since = np.inf, 0
    for v in losses:
        if not np.isfinite(v):
            raise ValueError("a non-finite inner validation loss")
        ref, since = (v, 0) if v < ref - tolerance else (ref, since + 1)
    low = min(losses)
    return since >= patience, next(e for e, v in enumerate(losses, 1) if v <= low + tolerance)


def rows_sha256(index, positions):
    return hashlib.sha256(np.sort(np.asarray(index)[positions]).astype(np.int64).tobytes()).hexdigest()


# ---- the worker side: everything that touches torch --------------------------------------------------------------

@contextlib.contextmanager
def torch_settings(threads):
    """torch with the thread limit and deterministic algorithms, both restored on exit."""
    import torch
    saved = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield torch
    finally:
        torch.set_num_threads(saved[0])
        torch.use_deterministic_algorithms(saved[1])


def network(torch, p, d_in, init_seed):
    import tabm
    torch.manual_seed(init_seed)
    return tabm.TabM.make(n_num_features=d_in, cat_cardinalities=None, d_out=1, num_embeddings=None,
                          n_blocks=p["n_blocks"], d_block=p["d_block"], dropout=p["dropout"], k=p["k"],
                          arch_type="tabm", activation="ReLU", start_scaling_init="random-signs")


def logits(torch, net, Z):
    """Member logits (rows, k) in evaluation mode, in fixed batches of PREDICTION_BATCH rows, the last one padded with
    zero rows that are discarded, so a row's computation never depends on how many rows are predicted with it."""
    net.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(Z), PREDICTION_BATCH):
            b = Z[i:i + PREDICTION_BATCH]
            pad = np.zeros((PREDICTION_BATCH, Z.shape[1]), dtype="float32")
            pad[:len(b)] = b
            out.append(net(torch.from_numpy(pad)).squeeze(-1)[:len(b)].double())
    return torch.cat(out)


def mean_probability(torch, z):
    """The members' probabilities averaged, not their logits: mean over members of sigmoid(z)."""
    return torch.sigmoid(z).mean(dim=1).numpy()


def ensemble_log_loss(torch, z, y):
    """Mean log loss of the averaged probability, in float64 from the logits: log p = logsumexp(log sigmoid z) - log k."""
    k = z.shape[1]
    lp = torch.logsumexp(torch.nn.functional.logsigmoid(z), dim=1) - np.log(k)
    lq = torch.logsumexp(torch.nn.functional.logsigmoid(-z), dim=1) - np.log(k)
    yt = torch.from_numpy(np.asarray(y, dtype="float64"))
    return float(-(yt * lp + (1 - yt) * lq).mean())


def train(torch, p, Z, y, epochs, init_seed, batch_seed, Zv=None, yv=None):
    """A new network and optimizer; AdamW (ADAMW, weight decay on every parameter), one permutation per epoch from
    its own generator, the short last batch kept, all members on the same batch, the loss the mean of the members'
    binary cross-entropies. With inner validation data, the ensemble log loss after every epoch and a stop by
    stop_rule. Returns (network, losses, optimizer updates)."""
    net = network(torch, p, Z.shape[1], init_seed)
    opt = torch.optim.AdamW(net.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"], **ADAMW)
    gen = torch.Generator().manual_seed(batch_seed)
    Zt, yt = torch.from_numpy(Z), torch.from_numpy(np.asarray(y, dtype="float32"))
    losses, updates = [], 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Zt), generator=gen)
        for i in range(0, len(Zt), p["batch_size"]):
            idx = perm[i:i + p["batch_size"]]
            out = net(Zt[idx]).squeeze(-1)  # (batch, k): each member's logit
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, yt[idx][:, None].expand_as(out))
            if not torch.isfinite(loss):
                raise ValueError("a non-finite training loss")
            opt.zero_grad()
            loss.backward()
            opt.step()
            updates += 1
        if Zv is not None:
            losses.append(ensemble_log_loss(torch, logits(torch, net, Zv), yv))
            if stop_rule(losses, p["tolerance"], p["patience"])[0]:
                break
    return net, losses, updates


def fit_state(p, X, y, groups, seeds):
    """Plan section 5.5, in the worker: the inner split of the fitting rows, preprocessing fitted on its training
    part, the stopping stage, the chosen epoch; then the stopping stage discarded, preprocessing refitted on all
    fitting rows and a new network trained for exactly the chosen number of epochs."""
    y = np.asarray(y)
    with torch_settings(p["threads"]) as torch:
        tr, va = inner_split(y, groups, p["inner_folds"], seeds["inner_split"])
        stop_prep = TabMPreprocessor(p["scaler"]).fit(X.iloc[tr])
        _, losses, stop_updates = train(torch, p, stop_prep.transform(X.iloc[tr]), y[tr], p["max_epochs"],
                                        seeds["stop_init"], seeds["stop_batches"], stop_prep.transform(X.iloc[va]), y[va])
        stopped, epoch = stop_rule(losses, p["tolerance"], p["patience"])
        prep = TabMPreprocessor(p["scaler"]).fit(X)
        Z = prep.transform(X)
        net, _, final_updates = train(torch, p, Z, y, epoch, seeds["final_init"], seeds["final_batches"])
        state = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
        import tabm as tabm_package
        # Plain strings: torch's own version object would import torch into the fitting process when unpickled.
        versions = {"torch": str(torch.__version__), "tabm": str(getattr(tabm_package, "__version__", None))}
    receipt = {"policy": ITERATIONS, "seeds": dict(seeds),
               "inner_train_rows_sha256": rows_sha256(X.index, tr), "inner_validation_rows_sha256": rows_sha256(X.index, va),
               "inner_train_rows": int(len(tr)), "inner_validation_rows": int(len(va)),
               "stopping_preprocessing_rows_sha256": stop_prep.fitted_rows_sha256_,
               "stopping_preprocessing_digest": stop_prep.digest_,
               "final_preprocessing_rows_sha256": prep.fitted_rows_sha256_, "final_preprocessing_digest": prep.digest_,
               "encoded_columns": prep.output_columns(), "epochs_evaluated": len(losses),
               "best_epoch": int(np.argmin(losses)) + 1, "chosen_epoch": epoch,
               "stopped_by": "patience" if stopped else "max_epochs", "validation_losses": [float(v) for v in losses],
               "stopping_updates": stop_updates, "final_updates": final_updates, "versions": versions}
    return {"prep": prep, "state": state, "receipt": receipt, "d_in": int(Z.shape[1])}


def predict_state(p, state, d_in, Z):
    with torch_settings(p["threads"]) as torch:
        net = network(torch, p, d_in, 0)
        net.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
        return mean_probability(torch, logits(torch, net, Z))


def serve(inp, out):
    """The worker loop: one pickled request, one pickled reply ({"ok": value} or {"error": text}, with the warnings
    raised while serving it), until end of input."""
    import torch
    torch.set_num_interop_threads(1)
    while True:
        try:
            req = pickle.load(inp)
        except EOFError:
            return
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if req["op"] == "fit":
                    reply = {"ok": fit_state(req["params"], req["X"], req["y"], req["groups"], req["seeds"])}
                elif req["op"] == "predict":
                    reply = {"ok": predict_state(req["params"], req["state"], req["d_in"], req["Z"])}
                else:
                    raise ValueError(f"unknown request {req['op']!r}")
            except Exception as e:  # sent back and raised in the fitting process
                reply = {"error": f"{type(e).__name__}: {e}"}
        reply["warnings"] = [f"{type(w.message).__name__}: {w.message}" for w in caught]
        pickle.dump(reply, out, protocol=pickle.HIGHEST_PROTOCOL)
        out.flush()


# ---- the fitting-process side ------------------------------------------------------------------------------------

class Worker:
    """The one persistent worker of this process, started on first use and closed at exit."""
    proc = None

    @classmethod
    def call(cls, request):
        if cls.proc is None or cls.proc.poll() is not None:
            cls.proc = subprocess.Popen([sys.executable, "-m", "songpop.phase2.tabm"], cwd=ROOT, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            atexit.register(cls.close)
        pickle.dump(request, cls.proc.stdin, protocol=pickle.HIGHEST_PROTOCOL)
        cls.proc.stdin.flush()
        try:
            reply = pickle.load(cls.proc.stdout)
        except EOFError:
            raise RuntimeError(f"the TabM worker exited with code {cls.proc.wait()}") from None
        for w in reply["warnings"]:
            warnings.warn(f"TabM worker: {w}", RuntimeWarning)
        if "error" in reply:
            raise RuntimeError(f"TabM worker: {reply['error']}")
        if "torch" in sys.modules:
            raise RuntimeError("torch was loaded into the fitting process (Finding 024)")
        return reply["ok"]

    @classmethod
    def close(cls):
        if cls.proc is not None and cls.proc.poll() is None:
            cls.proc.stdin.close()
            cls.proc.wait()
        cls.proc = None


def stage_seeds(context, estimator_seed):
    """Seeds from the fit's context (its split seed and fold) and the estimator seed, never from run order: the inner
    split from the context alone, so every configuration and estimator seed on one fitting population shares it;
    initialization and minibatch order separately for each stage."""
    s = {"inner_split": context.inner_seed("tabm_stop")}
    for stage in ("stop", "final"):
        for kind in ("init", "batches"):
            s[f"{stage}_{kind}"] = context.inner_seed(f"tabm_{stage}_{kind}_{estimator_seed}")
    return s


class TabMClassifier(ClassifierMixin, BaseEstimator):
    """Plan section 5.5. fit(X, y, groups=, context=) runs fit_state in the worker: the stopping stage on an inner
    group-safe split, preprocessing fitted on its training part only; the earliest epoch within `tolerance` of the
    lowest inner validation log loss of the averaged member probabilities, after at most `max_epochs` epochs and a
    stop after `patience` epochs without an improvement larger than `tolerance`; then preprocessing refitted on all
    fitting rows and a new network trained for exactly that many epochs (same batch size, so about a quarter more
    updates). Predictions average the members' probabilities. The constructor does nothing else."""

    def __init__(self, n_blocks, d_block, dropout, learning_rate, weight_decay, batch_size, scaler, random_state,
                 k=16, max_epochs=300, patience=20, tolerance=1e-6, inner_folds=5, threads=4):
        self.n_blocks, self.d_block, self.dropout = n_blocks, d_block, dropout
        self.learning_rate, self.weight_decay, self.batch_size = learning_rate, weight_decay, batch_size
        self.scaler, self.random_state, self.k, self.max_epochs = scaler, random_state, k, max_epochs
        self.patience, self.tolerance, self.inner_folds, self.threads = patience, tolerance, inner_folds, threads

    def fit(self, X, y, groups=None, context=None, sample_weight=None):
        if groups is None or context is None:
            raise ValueError("TabM needs the fitting rows' groups and the fit context for its inner stopping split")
        if sample_weight is not None:
            raise ValueError("TabM takes no training weights (plan section 5.5)")
        seeds = stage_seeds(context, self.random_state)
        got = Worker.call({"op": "fit", "params": self.get_params(), "X": X, "y": np.asarray(y),
                           "groups": np.asarray(groups), "seeds": seeds})
        self.prep_, self.state_, self.receipt_, self.d_in_ = got["prep"], got["state"], got["receipt"], got["d_in"]
        self.receipt_["estimator_seed"] = self.random_state
        self.classes_ = np.array([0, 1])
        return self

    def transform_inputs(self, X):
        """The float32 matrix the final network receives, as float64 (for the identical-input check)."""
        return self.prep_.transform(X).astype("float64")

    def predict_proba(self, X):
        p = Worker.call({"op": "predict", "params": self.get_params(), "state": self.state_, "d_in": self.d_in_,
                         "Z": self.prep_.transform(X)})
        return np.column_stack([1 - p, p])


if __name__ == "__main__":
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # anything printed by the libraries goes to standard error, never into the reply channel
    from songpop.phase2 import tabm  # by its package name, so the classes it pickles back resolve in the parent
    tabm.serve(sys.stdin.buffer, out)
