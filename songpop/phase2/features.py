"""Feature pipelines: a base set from songpop.features plus named blocks (plan section 6.3). Stateless blocks are
functions of the raw rows; fold-fitted blocks come from songpop.phase2.encoders. Every column carries its
lineage to the raw source columns, for the phase 4 permutation analysis (plan section 9)."""
import numpy as np
import pandas as pd

from songpop.data import AUDIO, DATE
from songpop.features import F2_DATE, date_frame, make_features
from songpop.phase2.encoders import DateCount, DateTargetEncoder, EraRelative

RAW_COLUMNS = AUDIO + [DATE]  # the only columns a recipe may read; never labels, ids or uri
YEAR_CENTER = 2000

INTERACTIONS = {
    "ix_year_loudness": (["loudness", DATE], lambda r, d: (d["year"] - YEAR_CENTER) * r["loudness"]),
    "ix_year_acousticness": (["acousticness", DATE], lambda r, d: (d["year"] - YEAR_CENTER) * r["acousticness"]),
    "ix_energy_loudness": (["energy", "loudness"], lambda r, d: r["energy"] * r["loudness"]),
    "ix_energy_danceability": (["energy", "danceability"], lambda r, d: r["energy"] * r["danceability"]),
    "ix_acousticness_instrumentalness": (["acousticness", "instrumentalness"],
                                         lambda r, d: r["acousticness"] * r["instrumentalness"]),
    "beats": (["tempo", "duration_ms"], lambda r, d: r["tempo"] * r["duration_ms"] / 60000),
}


def _interactions(rows):
    d = date_frame(rows[DATE])
    return pd.DataFrame({c: f(rows, d).astype("float64") for c, (_, f) in INTERACTIONS.items()}, index=rows.index)


def _cyclic(rows):
    """sin(2 pi p) and cos(2 pi p) of the month, weekday and day-of-year phases; NaN where unobserved."""
    d = date_frame(rows[DATE])
    leap = d["year"].map(lambda y: 366.0 if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0 else 365.0)
    phases = {"month": (d["month"] - 1) / 12, "weekday": d["weekday"] / 7,
              "doy": (d["day_of_year"] - 1) / leap}
    out = {}
    for name, p in phases.items():
        out[f"{name}_sin"] = np.sin(2 * np.pi * p)
        out[f"{name}_cos"] = np.cos(2 * np.pi * p)
    return pd.DataFrame(out, index=rows.index, dtype="float64")


def _music(rows):
    """The key on the circle of fifths, sin and cos of 2 pi ((7 key) mod 12) / 12, and the tempo octave, sin and
    cos of 2 pi log2(tempo), so tempo T and 2T coincide (plan section 16.3); raw key and tempo stay in the base."""
    fifths = 2 * np.pi * np.mod(7 * rows["key"].to_numpy(dtype="float64"), 12) / 12
    octave = 2 * np.pi * np.log2(rows["tempo"].to_numpy(dtype="float64"))
    return pd.DataFrame({"key_fifths_sin": np.sin(fifths), "key_fifths_cos": np.cos(fifths),
                         "tempo_octave_sin": np.sin(octave), "tempo_octave_cos": np.cos(octave)}, index=rows.index)


STATELESS = {
    "interactions": (_interactions, {c: src for c, (src, _) in INTERACTIONS.items()}),
    "cyclic": (_cyclic, {f"{n}_{t}": [DATE] for n in ("month", "weekday", "doy") for t in ("sin", "cos")}),
    "music": (_music, {"key_fifths_sin": ["key"], "key_fifths_cos": ["key"], "tempo_octave_sin": ["tempo"],
                       "tempo_octave_cos": ["tempo"]}),
}
STATEFUL = {"date_count": DateCount, "date_te3": lambda: DateTargetEncoder(3), "date_te10": lambda: DateTargetEncoder(10),
            "date_te30": lambda: DateTargetEncoder(30), "date_te100": lambda: DateTargetEncoder(100),
            "era_relative": EraRelative}  # date_te3 and date_te100: the smoothing values of plan section 6.1 (P2.4)


BASE_COLUMNS = {"F0": AUDIO, "F1": AUDIO + ["year"], "F2": AUDIO + F2_DATE}
BLOCK_COLUMNS = {"interactions": list(INTERACTIONS), "cyclic": list(STATELESS["cyclic"][1]),
                 "music": list(STATELESS["music"][1]),
                 "date_count": DateCount.columns, "date_te3": ["date_te3"], "date_te10": ["date_te10"],
                 "date_te30": ["date_te30"], "date_te100": ["date_te100"],
                 "era_relative": EraRelative.columns}


def undropped_columns(recipe):
    """The base set's columns, then each block's, in order, before the recipe's drop list."""
    return list(BASE_COLUMNS[recipe["base"]]) + [c for b in recipe["blocks"] for c in BLOCK_COLUMNS[b]]


def feature_columns(recipe):
    """The recipe's feature columns, in order, known without fitting: the base set and the blocks, less the drop
    list (feature-removal checks, plan section 6)."""
    drop = set(recipe["drop"])
    return [c for c in undropped_columns(recipe) if c not in drop]


def base_lineage(columns):
    return {c: [c] if c in AUDIO else [DATE] for c in columns}


class FeaturePipeline:
    """The recipe's base set and blocks, in the recipe's block order; fitted per fitting population."""

    def __init__(self, recipe):
        self.base, self.blocks, self.drop = recipe["base"], list(recipe["blocks"]), list(recipe["drop"])
        self.expected_ = feature_columns(recipe)

    def _assemble(self, rows, stateful_frames):
        if list(rows.columns) != RAW_COLUMNS:
            raise ValueError(f"feature rows must have exactly the raw columns {RAW_COLUMNS}")
        parts = [make_features(rows, self.base)]
        lineage = base_lineage(parts[0].columns)
        for b in self.blocks:
            if b in STATELESS:
                fn, lin = STATELESS[b]
                parts.append(fn(rows))
                lineage.update(lin)
            else:
                frame, lin = stateful_frames[b]
                parts.append(frame)
                lineage.update(lin)
        X = pd.concat(parts, axis=1)
        if X.columns.duplicated().any():
            raise ValueError("duplicate feature columns")
        X = X.drop(columns=self.drop)
        return X.astype("float64"), {c: lineage[c] for c in X.columns}

    def fit_transform(self, rows, y, groups, ctx):
        self.fitted_ = {b: STATEFUL[b]() for b in self.blocks if b in STATEFUL}
        frames = {b: (enc.fit_transform(rows, y, groups, ctx), enc.sources) for b, enc in self.fitted_.items()}
        self.receipts_ = [enc.receipt for enc in self.fitted_.values() if enc.receipt is not None]
        X, self.lineage_ = self._assemble(rows, frames)
        self.columns_ = list(X.columns)
        if self.columns_ != self.expected_:
            raise AssertionError(f"fitted columns {self.columns_} differ from the declared {self.expected_}")
        return X

    def transform(self, rows):
        frames = {b: (enc.transform(rows), enc.sources) for b, enc in self.fitted_.items()}
        X, _ = self._assemble(rows, frames)
        if list(X.columns) != self.columns_:
            raise AssertionError("transformed columns differ from the fitted columns")
        return X
