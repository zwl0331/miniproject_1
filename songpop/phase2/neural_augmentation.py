"""Training-only augmentation of already fold-fitted mixed inputs.

No query population is accepted. Categorical codes and missing indicators stay
exact. Mixup partners share their entire discrete profile; retrieval uses noise
only so its original query-group exclusion remains sufficient.
"""
import numpy as np

AUDIO = {"speechiness", "danceability", "duration_ms", "energy", "instrumentalness",
         "valence", "tempo", "loudness", "acousticness", "liveness"}


class Augmenter:
    def __init__(self, config, spec, Z, seed):
        self.config, self.spec = dict(config), spec
        self.rng = np.random.default_rng(seed ^ 0x41A6C3)
        self.low, self.high = Z.min(axis=0), Z.max(axis=0)
        self.pools = None
        if config["kind"] == "mixup":
            _, self.profile = np.unique(Z[:, spec["n_cont"]:], axis=0, return_inverse=True)
            self.pools = [np.flatnonzero(self.profile == group) for group in range(self.profile.max()+1)]

    def batch(self, torch, inputs, labels, indices):
        device = inputs.device
        ix = torch.as_tensor(indices, device=device, dtype=torch.long)
        batch, targets = inputs[ix].clone(), labels[ix].clone()
        selected = self.rng.random(len(indices)) < self.config["probability"]
        n = self.spec["n_cont"]
        if self.config["kind"] == "mixup":
            partners = np.array([self.rng.choice(self.pools[self.profile[i]]) for i in indices])
            lam = self.rng.beta(self.config["alpha"], self.config["alpha"], len(indices))
            lam = np.where(selected, np.maximum(lam, 1-lam), 1.).astype("float32")
            weight = torch.as_tensor(lam, device=device)
            other = torch.as_tensor(partners, device=device, dtype=torch.long)
            batch[:, :n] = weight[:, None]*batch[:, :n] + (1-weight[:, None])*inputs[other, :n]
            targets = weight*targets + (1-weight)*labels[other]
        else:
            columns = self.spec["augmentation_audio"]
            noise = self.rng.normal(0, self.config["sigma"], (len(indices), len(columns))).astype("float32")
            noise *= selected[:, None]
            for j, column in enumerate(columns):
                delta = torch.as_tensor(noise[:, j], device=device)
                flag = self.spec["augmentation_missing"].get(str(column))
                if flag is not None:
                    delta = delta * (1-batch[:, flag])
                batch[:, column] = (batch[:, column]+delta).clamp(float(self.low[column]), float(self.high[column]))
        return batch, targets
