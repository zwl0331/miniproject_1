"""Small architecture factory. Torch is imported only by the isolated worker.

All inputs are the fitted preprocessor's numeric coordinates, including one-hot
categories and missing flags. PLE bins come from the current training population.
Constant coordinates receive a fixed unit-width interval around their value.
"""
import numpy as np
import warnings


def specification(recipe, Z, prep=None):
    spec = {"d_in": Z.shape[1]}
    if recipe["adapter"]["layout"] == "mixed_neural_v1":
        spec.update(n_cont=prep.n_cont_, n_flags=prep.n_flags_, cat_cardinalities=prep.cat_cardinalities_)
    if "augmentation" in recipe:
        from songpop.phase2.neural_augmentation import AUDIO
        columns = prep.output_columns()
        spec["augmentation_audio"] = [i for i, c in enumerate(columns[:spec["n_cont"]]) if c in AUDIO]
        spec["augmentation_missing"] = {str(i): columns.index("missing_" + columns[i])
            for i in spec["augmentation_audio"] if "missing_" + columns[i] in columns}
    if recipe["family"] == "retrieval":
        spec["memory_rows"] = len(Z)
    if recipe["adapter"]["representation"] == "ple":
        n_bins = recipe["embedding"]["n_bins"]
        bins = []
        for col in Z[:, :spec.get("n_cont", Z.shape[1])].T:
            edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)).astype("float32"))
            if len(edges) == 1:
                edges = np.array([edges[0] - 0.5, edges[0] + 0.5], dtype="float32")
            if len(edges) < 2 or np.any(np.diff(edges) <= 0):
                raise ValueError("invalid training-only embedding bins")
            bins.append(edges.tolist())
        spec["bins"] = bins
    return spec


def mixed_input(torch, embedding, spec, embedding_width):
    nn = torch.nn
    class Input(nn.Module):
        def __init__(self):
            super().__init__()
            self.numerical = embedding
            self.categorical = nn.ModuleList([nn.Embedding(c, min(8, c), padding_idx=0)
                                             for c in spec["cat_cardinalities"]])
        def forward(self, x):
            n, f = spec["n_cont"], spec["n_flags"]
            cont = x[:, :n] if self.numerical is None else self.numerical(x[:, :n]).flatten(1)
            return torch.cat([cont, x[:, n:n+f]] +
                [module(x[:, n+f+i].long()) for i, module in enumerate(self.categorical)], dim=1)
    width = spec["n_cont"] * embedding_width + spec["n_flags"] + sum(min(8, c) for c in spec["cat_cardinalities"])
    return Input(), width


def build(torch, recipe, spec, seed):
    import tabm as tabm_package
    import rtdl_num_embeddings as embeddings
    import rtdl_revisiting_models as rtdl
    from songpop.phase2 import tabm

    p, family = recipe["params"], recipe["family"]
    representation = recipe["adapter"]["representation"]
    mixed = "n_cont" in spec
    n_num = spec.get("n_cont", spec["d_in"])
    if family == "tabm" and representation == "scalar" and not mixed:
        return tabm.network(torch, p, spec["d_in"], seed)
    torch.manual_seed(seed)
    embedding = None
    if representation == "ple":
        # Binary/constant coordinates deliberately have a single interval.
        # Suppress only the library's expected advisory for that exact case.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"The .*feature has just two bin edges.*", category=UserWarning)
            embedding = embeddings.PiecewiseLinearEmbeddings(
                [torch.tensor(b, dtype=torch.float32) for b in spec["bins"]],
                recipe["embedding"]["d_embedding"], activation=False, version="B")
    elif representation == "periodic":
        e = recipe["embedding"]
        embedding = embeddings.PeriodicEmbeddings(n_num, e["d_embedding"],
            n_frequencies=e["n_frequencies"], frequency_init_scale=e["frequency_init_scale"],
            activation=False, lite=False)
    if family == "tabm":
        core = tabm_package.TabM.make(n_num_features=n_num,
            cat_cardinalities=([2] * spec["n_flags"] + spec["cat_cardinalities"]) if mixed else None,
            d_out=1, num_embeddings=embedding, n_blocks=p["n_blocks"], d_block=p["d_block"],
            dropout=p["dropout"], k=p["k"], arch_type="tabm", activation="ReLU",
            start_scaling_init="random-signs" if embedding is None else "normal")
        if not mixed:
            return core
        class MixedTabM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.core = core
            def forward(self, x):
                return self.core(x[:, :n_num], x[:, n_num:].long())
        return MixedTabM()
    d_in = spec["d_in"] * (recipe["embedding"]["d_embedding"] if embedding is not None else 1)
    input_module = torch.nn.Identity() if embedding is None else torch.nn.Sequential(embedding, torch.nn.Flatten(1))
    if mixed and family != "ft_transformer":
        input_module, d_in = mixed_input(torch, embedding, spec,
            recipe["embedding"]["d_embedding"] if embedding is not None else 1)
    if family == "mlp":
        net = rtdl.MLP(d_in=d_in, d_out=1, n_blocks=p["n_blocks"],
                       d_block=p["d_block"], dropout=p["dropout"])
    elif family == "resnet":
        net = rtdl.ResNet(d_in=d_in, d_out=1, n_blocks=p["n_blocks"], d_block=p["d_block"],
                          d_hidden_multiplier=2.0, dropout1=p["dropout"], dropout2=0.0)
    elif family == "ft_transformer":
        core = rtdl.FTTransformer(n_cont_features=n_num + spec["n_flags"] if mixed else spec["d_in"],
            cat_cardinalities=spec["cat_cardinalities"] if mixed else [], d_out=1,
            n_blocks=p["n_blocks"], d_block=p["d_block"], attention_n_heads=p["attention_n_heads"],
            attention_dropout=p["dropout"], ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=p["dropout"], residual_dropout=0.0)
        if not mixed:
            return core
        class MixedFT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.core = core
            def forward(self, x, unused=None):
                n = spec["n_cont"] + spec["n_flags"]
                return self.core(x[:, :n], x[:, n:].long())
            def make_parameter_groups(self):
                return self.core.make_parameter_groups()
        return MixedFT()
    elif family == "retrieval":
        from songpop.phase2.neural_retrieval import build as retrieval
        return retrieval(torch, recipe, spec, input_module, d_in)
    else:
        raise ValueError(f"unimplemented neural family {family}")
    return torch.nn.Sequential(input_module, net) if mixed or embedding is not None else net


def forward(net, inputs, family, groups=None):
    logits = (net(inputs, None) if family == "ft_transformer" else
              net(inputs, query_groups=groups) if family == "retrieval" else net(inputs)).squeeze(-1)
    return logits[:, None] if logits.ndim == 1 else logits


def parameter_groups(net, family):
    if family == "retrieval":
        regular, unregularized = [], []
        for name, p in net.named_parameters():
            (unregularized if p.ndim < 2 or "labels" in name else regular).append(p)
        return [{"params": regular}, {"params": unregularized, "weight_decay": 0.}]
    if family != "ft_transformer":
        return net.parameters()
    # The library builds a set for the no-decay group. Sort for fixed seed runs.
    names = {id(p): name for name, p in net.named_parameters()}
    groups = net.make_parameter_groups()
    for group in groups:
        group["params"] = sorted(group["params"], key=lambda p: names[id(p)])
    return groups
