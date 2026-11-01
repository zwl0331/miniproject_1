"""Small architecture factory. Torch is imported only by the isolated worker.

All inputs are the fitted preprocessor's numeric coordinates, including one-hot
categories and missing flags. PLE bins come from the current training population.
Constant coordinates receive a fixed unit-width interval around their value.
"""
import numpy as np
import warnings


def specification(recipe, Z):
    spec = {"d_in": Z.shape[1]}
    if recipe["adapter"]["representation"] == "ple":
        n_bins = recipe["embedding"]["n_bins"]
        bins = []
        for col in Z.T:
            edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)).astype("float32"))
            if len(edges) == 1:
                edges = np.array([edges[0] - 0.5, edges[0] + 0.5], dtype="float32")
            if len(edges) < 2 or np.any(np.diff(edges) <= 0):
                raise ValueError("invalid training-only embedding bins")
            bins.append(edges.tolist())
        spec["bins"] = bins
    return spec


def build(torch, recipe, spec, seed):
    import tabm as tabm_package
    import rtdl_num_embeddings as embeddings
    import rtdl_revisiting_models as rtdl
    from songpop.phase2 import tabm

    p, family = recipe["params"], recipe["family"]
    representation = recipe["adapter"]["representation"]
    if family == "tabm" and representation == "scalar":
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
        embedding = embeddings.PeriodicEmbeddings(spec["d_in"], e["d_embedding"],
            n_frequencies=e["n_frequencies"], frequency_init_scale=e["frequency_init_scale"],
            activation=False, lite=False)
    if family == "tabm":
        return tabm_package.TabM.make(n_num_features=spec["d_in"], cat_cardinalities=None,
            d_out=1, num_embeddings=embedding, n_blocks=p["n_blocks"], d_block=p["d_block"],
            dropout=p["dropout"], k=p["k"], arch_type="tabm", activation="ReLU",
            start_scaling_init="normal")
    d_in = spec["d_in"] * (recipe["embedding"]["d_embedding"] if embedding is not None else 1)
    if family == "mlp":
        net = rtdl.MLP(d_in=d_in, d_out=1, n_blocks=p["n_blocks"],
                       d_block=p["d_block"], dropout=p["dropout"])
    elif family == "resnet":
        net = rtdl.ResNet(d_in=d_in, d_out=1, n_blocks=p["n_blocks"], d_block=p["d_block"],
                          d_hidden_multiplier=2.0, dropout1=p["dropout"], dropout2=0.0)
    elif family == "ft_transformer":
        return rtdl.FTTransformer(n_cont_features=spec["d_in"], cat_cardinalities=[], d_out=1,
            n_blocks=p["n_blocks"], d_block=p["d_block"], attention_n_heads=p["attention_n_heads"],
            attention_dropout=p["dropout"], ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=p["dropout"], residual_dropout=0.0)
    else:
        raise ValueError(f"unimplemented neural family {family}")
    return net if embedding is None else torch.nn.Sequential(embedding, torch.nn.Flatten(1), net)


def forward(net, inputs, family):
    logits = (net(inputs, None) if family == "ft_transformer" else net(inputs)).squeeze(-1)
    return logits[:, None] if logits.ndim == 1 else logits


def parameter_groups(net, family):
    if family != "ft_transformer":
        return net.parameters()
    # The library builds a set for the no-decay group. Sort for fixed seed runs.
    names = {id(p): name for name, p in net.named_parameters()}
    groups = net.make_parameter_groups()
    for group in groups:
        group["params"] = sorted(group["params"], key=lambda p: names[id(p)])
    return groups
