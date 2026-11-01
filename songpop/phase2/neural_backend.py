"""Raw neural fitting implementation, imported without initializing Torch.

Every fitted transformation belongs to S or D. The stopping model is discarded
before D is fitted. Each architecture uses the same stopping/refit boundary.
"""
import hashlib
import json

import numpy as np

from songpop.phase2 import tabm
from songpop.phase2 import neural_architectures as architectures
from songpop.phase2.encoders import DateCount, DateTargetEncoder, EraRelative, FitContext, splits_sha256
from songpop.phase2.features import RAW_COLUMNS
from songpop.phase2.raw_features import RawFeaturePipeline
from songpop.phase2.raw_recipes import validate
from songpop.phase2.raw_splits import crossfit, partition, population


def array_digest(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def feature_state(features):
    """Canonical inference state, not pickle bytes or object identities."""
    states = {}
    for name, fitted in features.fitted_.items():
        if isinstance(fitted, DateCount):
            state = {"counts": {str(k): int(v) for k, v in fitted.count_.items()}}
        elif isinstance(fitted, DateTargetEncoder):
            enc = fitted.encoder_
            state = {"categories": [list(map(str, a)) for a in enc.categories_],
                     "encodings": [array_digest(a) for a in enc.encodings_],
                     "prior": float(enc.target_mean_), "classes": enc.classes_.tolist(), "m": fitted.m}
        elif isinstance(fitted, EraRelative):
            state = {"global": fitted.global_, "era": fitted.era_mean_}
        else:
            raise ValueError(f"no stable raw feature-state serializer for {type(fitted).__name__}")
        states[name] = state
    return {"columns": features.columns_, "lineage": features.lineage_, "states": states,
            "encoder_receipts": features.receipts_}


def state_digest(state):
    return object_digest({name: array_digest(value) for name, value in sorted(state.items())})


def validate_request(recipe, rows, y, groups, context, diagnostics=None):
    validate(recipe)
    if list(rows.columns) != RAW_COLUMNS:
        raise ValueError("raw neural fitting rows must have exactly the declared raw columns")
    y, groups = population(y, groups, rows.index)
    if not isinstance(context, FitContext):
        raise ValueError("raw neural fitting needs an explicit FitContext")
    p = recipe["params"]
    seed = tabm.stage_seeds(context, 0)["inner_split"]
    diagnostic = {} if diagnostics is None else diagnostics
    if not isinstance(diagnostic, dict) or set(diagnostic) - {"stop_split", "stop_encoder_splits", "final_encoder_splits"}:
        raise ValueError("unknown raw neural diagnostic assignments")
    has_te = any(block.startswith("date_te") for block in recipe["blocks"])
    if "stop_encoder_splits" in diagnostic and "stop_split" not in diagnostic:
        raise ValueError("diagnostic stopping encoder folds require a fixed S/V split")
    if "stop_split" in diagnostic and has_te and "stop_encoder_splits" not in diagnostic:
        raise ValueError("inject S/V and S encoder assignments together for target-encoding diagnostics")
    if not has_te and any(k in diagnostic for k in ("stop_encoder_splits", "final_encoder_splits")):
        raise ValueError("encoder assignments require a target-encoding block")
    split = diagnostic.get("stop_split")
    fit, valid = partition(y, groups, split if split is not None else tabm.inner_split(y, groups, p["inner_folds"], seed))
    enc_s = diagnostic.get("stop_encoder_splits")
    enc_d = diagnostic.get("final_encoder_splits")
    if enc_s is not None:
        enc_s = crossfit(y[fit], groups[fit], enc_s)
    if enc_d is not None:
        enc_d = crossfit(y, groups, enc_d)
    return y, groups, fit, valid, enc_s, enc_d


def prepare(recipe, rows, y, groups, context, diagnostics=None, *, final=True):
    y, groups, fit, valid, enc_s, enc_d = validate_request(recipe, rows, y, groups, context, diagnostics)
    features = RawFeaturePipeline(recipe)
    Xs = features.fit_transform(rows.iloc[fit], y[fit], groups[fit], context, encoder_splits=enc_s)
    Xv = features.transform(rows.iloc[valid])
    prep = tabm.TabMPreprocessor(recipe["preprocessing"]["scaler"]).fit(Xs)
    Zs, Zv = prep.transform(Xs), prep.transform(Xv)
    result = {"fit": fit, "valid": valid, "stop_features": features, "stop_prep": prep,
              "final_encoder_splits": enc_d,
              "Zs": Zs, "Zv": Zv, "Xs": Xs, "Xv": Xv}
    if final:
        d_features = RawFeaturePipeline(recipe)
        Xd = d_features.fit_transform(rows, y, groups, context, encoder_splits=enc_d)
        d_prep = tabm.TabMPreprocessor(recipe["preprocessing"]["scaler"]).fit(Xd)
        result.update(features=d_features, prep=d_prep, Zd=d_prep.transform(Xd), Xd=Xd)
    return result


def configure(device):
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device == "cuda":
        torch.cuda.init()
    elif device != "cpu":
        raise ValueError("unknown neural device")
    return torch


def logits(torch, net, Z, device, family="tabm"):
    net.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(Z), tabm.PREDICTION_BATCH):
            part = Z[start:start + tabm.PREDICTION_BATCH]
            padded = np.zeros((tabm.PREDICTION_BATCH, Z.shape[1]), dtype=np.float32)
            padded[:len(part)] = part
            outputs.append(architectures.forward(net, torch.from_numpy(padded).to(device), family)[:len(part)].cpu().double())
    if not outputs:
        raise ValueError("neural prediction needs at least one query row")
    return torch.cat(outputs)


def numpy_state(net):
    state = {name: value.detach().cpu().numpy().copy() for name, value in net.state_dict().items()}
    if not all(np.isfinite(value).all() for value in state.values()):
        raise ValueError("non-finite neural model state")
    return state


def train(torch, recipe, spec, Z, y, epochs, init_seed, batch_seed, device, Zv=None, yv=None, *, fixed_horizon=False):
    p, family = recipe["params"], recipe["family"]
    net = architectures.build(torch, recipe, spec, init_seed).to(device)
    optimizer = torch.optim.AdamW(architectures.parameter_groups(net, family),
        lr=p["learning_rate"], weight_decay=p["weight_decay"], **tabm.ADAMW)
    generator = torch.Generator(device="cpu").manual_seed(batch_seed)
    inputs = torch.from_numpy(Z).to(device)
    labels = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    losses, trajectory, updates = [], [], 0
    for _ in range(epochs):
        net.train()
        permutation = torch.randperm(len(inputs), generator=generator)
        for start in range(0, len(inputs), p["batch_size"]):
            indices = permutation[start:start + p["batch_size"]].to(device)
            if len(indices) == 1 and family == "resnet":
                # Keep the observation, but use training-derived running moments.
                # BatchNorm cannot estimate variance from a singleton batch.
                for module in net.modules():
                    if isinstance(module, torch.nn.BatchNorm1d):
                        module.eval()
            outputs = architectures.forward(net, inputs[indices], family)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels[indices, None].expand_as(outputs))
            if not torch.isfinite(loss):
                raise ValueError("non-finite neural training loss")
            optimizer.zero_grad()
            loss.backward()
            if not all(torch.isfinite(q.grad).all() for q in net.parameters() if q.grad is not None):
                raise ValueError("non-finite neural training gradient")
            optimizer.step()
            updates += 1
        if fixed_horizon:
            trajectory.append(state_digest(numpy_state(net)))
        if Zv is not None:
            losses.append(tabm.ensemble_log_loss(torch, logits(torch, net, Zv, device, family), yv))
            if not fixed_horizon and tabm.stop_rule(losses, p["tolerance"], p["patience"])[0]:
                break
    return net, losses, updates, trajectory


def fit(torch, recipe, rows, y, groups, context, diagnostics=None):
    """Shared S/D transforms, independent stop/final networks for each seed."""
    prepared = prepare(recipe, rows, y, groups, context, diagnostics, final=False)
    y = np.asarray(y)
    p, device = recipe["params"], recipe["adapter"]["device"]
    fit_rows, valid = prepared["fit"], prepared["valid"]
    stop_spec = architectures.specification(recipe, prepared["Zs"])
    records, states, stopping = [], [], []
    import tabm as tabm_package
    for seed in recipe["estimator_seeds"]:
        seeds = tabm.stage_seeds(context, seed)
        stopping_net, losses, stop_updates, _ = train(torch, recipe, stop_spec, prepared["Zs"], y[fit_rows], p["max_epochs"],
            seeds["stop_init"], seeds["stop_batches"], device, prepared["Zv"], y[valid])
        del stopping_net
        stopped, epoch = tabm.stop_rule(losses, p["tolerance"], p["patience"])
        stopping.append((seed, seeds, losses, stop_updates, stopped, epoch))
    # No D-derived transform exists during stopping. Build D once after every
    # seed has selected its own epoch, then initialize independent final models.
    d_features = RawFeaturePipeline(recipe)
    Xd = d_features.fit_transform(rows, y, groups, context, encoder_splits=prepared["final_encoder_splits"])
    d_prep = tabm.TabMPreprocessor(recipe["preprocessing"]["scaler"]).fit(Xd)
    prepared.update(features=d_features, prep=d_prep, Zd=d_prep.transform(Xd))
    final_spec = architectures.specification(recipe, prepared["Zd"])
    for seed, seeds, losses, stop_updates, stopped, epoch in stopping:
        net, _, final_updates, _ = train(torch, recipe, final_spec, prepared["Zd"], y, epoch,
                                         seeds["final_init"], seeds["final_batches"], device)
        states.append(numpy_state(net))
        del net
        records.append({"policy": tabm.ITERATIONS, "seeds": seeds,
            "inner_train_rows_sha256": tabm.rows_sha256(rows.index, fit_rows),
            "inner_validation_rows_sha256": tabm.rows_sha256(rows.index, valid),
            "inner_train_rows": len(fit_rows), "inner_validation_rows": len(valid),
            "stopping_preprocessing_rows_sha256": prepared["stop_prep"].fitted_rows_sha256_,
            "stopping_preprocessing_digest": prepared["stop_prep"].digest_,
            "final_preprocessing_rows_sha256": prepared["prep"].fitted_rows_sha256_,
            "final_preprocessing_digest": prepared["prep"].digest_,
            "encoded_columns": prepared["prep"].output_columns(), "epochs_evaluated": len(losses),
            "best_epoch": int(np.argmin(losses)) + 1, "chosen_epoch": epoch,
            "stopped_by": "patience" if stopped else "max_epochs", "validation_losses": losses,
            "stopping_updates": stop_updates, "final_updates": final_updates,
            "versions": {"torch": str(torch.__version__), "tabm": str(tabm_package.__version__)}, "estimator_seed": seed})
    raw = {"schema": "raw_neural_receipt_v1", "adapter": recipe["adapter"],
           "stop_split_sha256": splits_sha256(rows.index, [(fit_rows, valid)]),
           "stop_feature_state_sha256": object_digest(feature_state(prepared["stop_features"])),
           "final_feature_state_sha256": object_digest(feature_state(prepared["features"])),
           "stop_training_matrix_sha256": array_digest(prepared["Zs"]),
           "stop_validation_matrix_sha256": array_digest(prepared["Zv"]),
           "final_training_matrix_sha256": array_digest(prepared["Zd"]),
           "state_sha256": [state_digest(s) for s in states], "diagnostic_assignments": diagnostics is not None,
           "stop_model_spec_sha256": object_digest(stop_spec),
           "final_model_spec_sha256": object_digest(final_spec),
           "subfits": {"feature_fits": 2, "preprocessing_fits": 2, "network_trainings": 2 * len(states),
                        "encoder_fits": sum(len(e.splits_) + 1 for fs in (prepared["stop_features"], prepared["features"])
                                            for e in fs.fitted_.values() if isinstance(e, DateTargetEncoder))}}
    receipt = {"split_seed": context.split_seed, "fold": context.fold,
               "fitting_rows_sha256": tabm.rows_sha256(rows.index, np.arange(len(rows))),
               "inner": prepared["features"].receipts_, "tabm": records, "raw_neural": raw}
    return {"features": prepared["features"], "prep": prepared["prep"], "states": states,
            "receipt": receipt, "d_in": prepared["Zd"].shape[1], "model_spec": final_spec}


def predict(torch, recipe, states, Z, model_spec):
    p, device = recipe["params"], recipe["adapter"]["device"]
    predictions = []
    for state in states:
        if model_spec["d_in"] != Z.shape[1]:
            raise ValueError("prediction columns differ from fitted network")
        net = architectures.build(torch, recipe, model_spec, 0).to(device)
        net.load_state_dict({name: torch.from_numpy(value).to(device) for name, value in state.items()})
        predictions.append(tabm.mean_probability(torch, logits(torch, net, Z, device, recipe["family"])))
        del net
    return predictions


def trajectory(torch, recipe, rows, y, groups, context, diagnostics, epochs, seed):
    """Diagnostic only: fixed horizon, no final fit, no production predictions."""
    if not isinstance(epochs, int) or isinstance(epochs, bool) or not 1 <= epochs <= recipe["params"]["max_epochs"]:
        raise ValueError("diagnostic horizon must fit the declared epoch cap")
    if seed not in recipe["estimator_seeds"]:
        raise ValueError("diagnostic estimator seed is absent from the recipe")
    prepared = prepare(recipe, rows, y, groups, context, diagnostics, final=False)
    seeds = tabm.stage_seeds(context, seed)
    fit_rows, valid = prepared["fit"], prepared["valid"]
    spec = architectures.specification(recipe, prepared["Zs"])
    _, losses, updates, states = train(torch, recipe, spec, prepared["Zs"], np.asarray(y)[fit_rows], epochs,
        seeds["stop_init"], seeds["stop_batches"], recipe["adapter"]["device"], prepared["Zv"], np.asarray(y)[valid],
        fixed_horizon=True)
    return {"trajectory": states, "losses": losses, "updates": updates,
            "model_spec_sha256": object_digest(spec),
            "split_sha256": splits_sha256(rows.index, [(fit_rows, valid)]),
            "feature_state_sha256": object_digest(feature_state(prepared["stop_features"])),
            "preprocessing_sha256": prepared["stop_prep"].digest_,
            "Xs_sha256": array_digest(prepared["Xs"].to_numpy()),
            "Xv_sha256": array_digest(prepared["Xv"].to_numpy()),
            "Zs_sha256": array_digest(prepared["Zs"]), "Zv_sha256": array_digest(prepared["Zv"])}
