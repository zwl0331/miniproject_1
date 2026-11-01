"""P2.9 feature fitting with explicit, checked encoder partitions for diagnostics.

Call only on the permitted raw fitting population (S, calibration training, or
D). The caller owns those boundaries. No global features or labels are accepted.
This module and its imports are safe in the isolated neural worker.
"""
from songpop.phase2.encoders import DateTargetEncoder, INNER_FOLDS
from songpop.phase2.features import FeaturePipeline, RAW_COLUMNS, STATEFUL
from songpop.phase2.raw_splits import crossfit, population


class RawFeaturePipeline(FeaturePipeline):
    """Same feature definitions as the legacy pipeline, with diagnostic injection.

Production uses the existing context-derived five encoder folds. Diagnostics may
inject all of those folds together. The neural adapter must inject S/V together
with these encoder assignments when a test changes held-out labels.
"""

    def fit_transform(self, rows, y, groups, ctx, *, encoder_splits=None):
        if list(rows.columns) != RAW_COLUMNS:
            raise ValueError(f"feature rows must have exactly the raw columns {RAW_COLUMNS}")
        y, groups = population(y, groups, rows.index)
        self.fitted_ = {b: STATEFUL[b]() for b in self.blocks if b in STATEFUL}
        if encoder_splits is not None:
            if not any(isinstance(enc, DateTargetEncoder) for enc in self.fitted_.values()):
                raise ValueError("injected encoder splits require a target-encoding block")
            encoder_splits = crossfit(y, groups, encoder_splits, n_folds=INNER_FOLDS)
        frames = {}
        for b, enc in self.fitted_.items():
            options = {"splits": encoder_splits} if isinstance(enc, DateTargetEncoder) else {}
            frames[b] = (enc.fit_transform(rows, y, groups, ctx, **options), enc.sources)
        self.receipts_ = [enc.receipt for enc in self.fitted_.values() if enc.receipt is not None]
        X, self.lineage_ = self._assemble(rows, frames)
        self.columns_ = list(X.columns)
        if self.columns_ != self.expected_:
            raise AssertionError("fitted feature columns differ from the declared columns")
        return X
