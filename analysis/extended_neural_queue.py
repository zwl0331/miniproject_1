"""Matched inputs, objective tests and compact neural tuning under fast mode."""
import copy
from pathlib import Path
from analysis.fast_exploration import write_json
from songpop.phase2.raw_recipes import make_recipe


def configurations():
    queue = []
    extras = {"year_relative": True, "year_standardize": True, "date_counts": True, "round": True}
    common = ["date_count", "date_te3", "cyclic"]
    def add(name, family, rep="ple", *, features=True, mixed=True, scaler="standard", params=None,
            stopping="log_loss", pairwise=0., embedding=None):
        p = dict(max_epochs=400, patience=25, d_block=128, n_blocks=2, learning_rate=.001)
        if family == "ft_transformer":
            p.update(d_block=64, learning_rate=.0003, max_epochs=600)
        if family == "retrieval":
            p.update(d_block=64, learning_rate=.0003, max_epochs=400)
        p.update(params or {})
        recipe = make_recipe(family, representation=rep, device="cuda",
            blocks=[b for b in common if family != "retrieval" or not b.startswith("date_te")],
            features=extras if features else {}, layout="mixed_neural_v1" if mixed else "legacy_tabm_v1",
            scaler=scaler, params=p, stopping_metric=stopping, pairwise_weight=pairwise, embedding=embedding)
        queue.append({"id":"x3_"+name.replace(".", "p"), "recipe":recipe, "cutoff_utc":"2026-11-01T20:00:00+00:00"})
    # Match each architecture across feature transfer, input layout and scaling.
    for family in ("tabm", "mlp", "resnet", "ft_transformer"):
        for features in (False, True):
            for mixed in (False, True):
                for scaler in ("standard", "quantile"):
                    add(f"{family}_f{int(features)}_m{int(mixed)}_{scaler}", family,
                        "scalar" if family=="ft_transformer" else "ple", features=features, mixed=mixed, scaler=scaler)
    # Stronger regularization and smaller/larger networks on the richest input.
    for family in ("tabm", "mlp", "resnet"):
        for label, params in (
            ("small_reg", dict(d_block=64, n_blocks=1, dropout=.3, weight_decay=.01)),
            ("slow_reg", dict(d_block=128, dropout=.2, weight_decay=.01, learning_rate=.0003, max_epochs=600)),
            ("wide_reg", dict(d_block=256, n_blocks=3, dropout=.3, weight_decay=.01))):
            add(f"{family}_{label}", family, params=params, scaler="quantile")
    # Matched log-loss -> AUC stopping, then mixed and pure pairwise objectives.
    for family in ("tabm", "mlp", "ft_transformer"):
        rep = "scalar" if family=="ft_transformer" else "ple"
        add(f"{family}_auc",family,rep,stopping="auc")
        for weight in (.5, 1.):
            add(f"{family}_pair{weight:g}",family,rep,stopping="auc",pairwise=weight)
    # Vary numerical embeddings with the new input/features, not just a new seed.
    for family in ("tabm", "mlp"):
        add(f"{family}_periodic_mixed",family,"periodic",scaler="quantile")
        add(f"{family}_ple_bins8",family,embedding={"n_bins":8,"d_embedding":16})
        add(f"{family}_ple_bins32",family,embedding={"n_bins":32,"d_embedding":16})
    # A distinct learned retrieval mechanism, always without supervised inputs.
    for context in (16, 64):
        for width in (64, 128):
            for rep in ("scalar", "ple"):
                add(f"retrieval_c{context}_w{width}_{rep}","retrieval",rep,
                    params={"context_size":context,"d_block":width,"dropout":.2,"weight_decay":.01},
                    stopping="auc",scaler="quantile")
    assert len(queue)==64 and len({c["id"] for c in queue})==64
    # Spread the informative comparisons across independent workers immediately.
    priority = lambda c: (0 if '_f1_m1_' in c['id'] else 1 if c['id'].endswith('_auc') else 2)
    retrieval = [c for c in queue if c['recipe']['family']=='retrieval']
    other = sorted([c for c in queue if c['recipe']['family']!='retrieval'],key=priority)
    result=[]
    for start in range(0,len(other),7):
        result.extend(other[start:start+7])
        if retrieval:
            result.append(retrieval.pop(0))
    return result+retrieval


if __name__ == "__main__":
    queue = configurations()
    out = Path("artifacts/extended")
    write_json(out / "neural_queue.json",queue)
    for worker in range(4):
        write_json(out / f"neural_queue_{worker}.json",queue[worker::4])
    print("Saved 64 neural trials across four GPU workers")
