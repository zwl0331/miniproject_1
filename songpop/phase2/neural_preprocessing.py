"""Mixed continuous, binary and categorical inputs, fitted on S or D only."""
import numpy as np
from songpop.phase2.tabm import TabMPreprocessor


class MixedPreprocessor(TabMPreprocessor):
    def fit(self, X, y=None):
        super().fit(X, y)
        self.n_cont_ = len(self.std_) + len(self.quant_)
        self.n_flags_ = len(self.flag_)
        self.cat_cardinalities_ = [len(c) + 1 for c in self.ohe_.categories_] if self.cat_ else []
        return self

    def output_columns(self):
        return self.std_ + self.quant_ + ["missing_" + c for c in self.flag_] + ["category_" + c for c in self.cat_]

    def transform(self, X):
        Z = super().transform(X)
        start = self.n_cont_ + self.n_flags_
        parts = [Z[:, :start]]
        for cardinality in self.cat_cardinalities_:
            onehot = Z[:, start:start + cardinality - 1]
            # Unknown category is 0, fitted categories are 1..cardinality-1.
            code = np.where(onehot.sum(axis=1) > 0, onehot.argmax(axis=1) + 1, 0)
            parts.append(code[:, None])
            start += cardinality - 1
        return np.ascontiguousarray(np.hstack(parts), dtype="float32")


def make_preprocessor(recipe):
    cls = MixedPreprocessor if recipe["adapter"]["layout"] == "mixed_neural_v1" else TabMPreprocessor
    return cls(recipe["preprocessing"]["scaler"])
