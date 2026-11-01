"""A compact learned-retrieval network inspired by TabR's key/value mechanism.

This is a project adaptation, not an exact reproduction of published TabR.
Reference: https://github.com/yandex-research/tabular-dl-tabr (Model).
Uses exact Torch neighbor search and excludes the query's entire track group.
Candidate features are unsupervised; candidate labels belong only to S or D.
"""


def build(torch, recipe, spec, input_module, input_width):
    nn = torch.nn
    p = recipe["params"]

    class Retrieval(nn.Module):
        def __init__(self):
            super().__init__()
            width = p["d_block"]
            self.input = input_module
            self.project = nn.Linear(input_width, width)
            def block():
                return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.ReLU(),
                    nn.Dropout(p["dropout"]), nn.Linear(2 * width, width))
            self.encoder = nn.ModuleList([block() for _ in range(p["n_blocks"])])
            self.key = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
            self.labels = nn.Embedding(2, width)
            nn.init.uniform_(self.labels.weight, -1, 1)
            self.difference = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(),
                nn.Dropout(p["dropout"]), nn.Linear(2 * width, width, bias=False))
            self.context_dropout = nn.Dropout(p["context_dropout"])
            self.predictor = nn.ModuleList([block()])
            self.head = nn.Sequential(nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, 1))
            self.register_buffer("memory_x", torch.zeros(spec["memory_rows"], spec["d_in"]))
            self.register_buffer("memory_y", torch.zeros(spec["memory_rows"], dtype=torch.long))
            self.register_buffer("memory_groups", torch.zeros(spec["memory_rows"], dtype=torch.long))

        def encode(self, x):
            h = self.project(self.input(x))
            for block in self.encoder:
                h = h + block(h)
            return h, self.key(h)

        def set_memory(self, x, y, groups):
            with torch.no_grad():
                self.memory_x.copy_(x)
                self.memory_y.copy_(y.long())
                self.memory_groups.copy_(groups)

        def forward(self, x, query_groups=None):
            h, key = self.encode(x)
            _, candidate_keys = self.encode(self.memory_x)
            distances = (key.square().sum(1, keepdim=True) + candidate_keys.square().sum(1)[None]
                         - 2 * key @ candidate_keys.T).clamp_min(0)
            if query_groups is not None:
                distances = distances.masked_fill(query_groups[:, None] == self.memory_groups[None], torch.inf)
            k = min(p["context_size"], len(candidate_keys) - 1)
            indices = distances.detach().topk(k, largest=False).indices
            selected = distances.gather(1, indices)
            if not torch.isfinite(selected).all():
                raise ValueError("retrieval context exceeds available out-of-group candidates")
            self.last_indices_ = indices.detach()
            weights = self.context_dropout(torch.softmax(-selected, dim=1))
            values = self.labels(self.memory_y[indices]) + self.difference(key[:, None] - candidate_keys[indices])
            h = h + (weights[:, :, None] * values).sum(dim=1)
            for block in self.predictor:
                h = h + block(h)
            return self.head(h)

    return Retrieval()
