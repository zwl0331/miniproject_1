"""Small fold-local feature experiments for Sam's 2026-11-01 fast mode."""
import numpy as np
import pandas as pd

from songpop.data import AUDIO, DATE
from songpop.phase2.encoders import inner_group_splits
from songpop.phase2.features import FeaturePipeline


class HierarchicalDate:
    """Grouped cross-fitted year -> year-month -> exact-date posterior means."""

    def __init__(self, smoothing=(30, 10, 3)):
        self.smoothing = tuple(smoothing)

    def fit(self, rows, y):
        self.prior = float(np.mean(y))
        dates = rows[DATE].astype(str)
        self.maps = []
        for length, smooth in zip((4, 7, 10), self.smoothing):
            valid = dates.str.len() >= length
            frame = pd.DataFrame({"key": dates[valid].str[:length].to_numpy(),
                                  "y": np.asarray(y)[valid.to_numpy()]})
            stats = frame.groupby("key").y.agg(["sum", "count"])
            mapping = {}
            for key, row in stats.iterrows():
                parent = self.prior
                if length >= 7:
                    parent = self.maps[0].get(key[:4], parent)
                if length == 10:
                    parent = self.maps[1].get(key[:7], parent)
                mapping[key] = (row["sum"] + smooth * parent) / (row["count"] + smooth)
            self.maps.append(mapping)
        return self

    def transform(self, rows):
        out = []
        for date in rows[DATE].astype(str):
            year = self.maps[0].get(date[:4], self.prior)
            month = self.maps[1].get(date[:7], year) if len(date) >= 7 else year
            day = self.maps[2].get(date, month) if len(date) == 10 else month
            out.append((year, month, day))
        return pd.DataFrame(out, index=rows.index, columns=["hier_year", "hier_month", "hier_date"])

    def fit_transform(self, rows, y, groups, ctx, splits=None):
        if splits is None:
            splits = inner_group_splits(y, groups, ctx.inner_seed("date_te"))
        values = np.full((len(rows), 3), np.nan)
        for tr, va in splits:
            if set(np.asarray(groups)[tr]) & set(np.asarray(groups)[va]):
                raise ValueError("hierarchical encoder split shares track groups")
            enc = HierarchicalDate(self.smoothing).fit(rows.iloc[tr], np.asarray(y)[tr])
            values[va] = enc.transform(rows.iloc[va]).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("hierarchical encoder left rows unencoded")
        self.fit(rows, y)
        return pd.DataFrame(values, index=rows.index, columns=["hier_year", "hier_month", "hier_date"])


def stateless(rows, options):
    out = {}
    duration = rows.duration_ms.astype(float)
    if options.get("round"):
        out["duration_whole_second"] = (duration % 1000 == 0).astype(float)
    if options.get("round_detail"):
        out.update(duration_ms_remainder=duration % 1000,
                   duration_whole_10ms=(duration % 10 == 0).astype(float),
                   duration_whole_100ms=(duration % 100 == 0).astype(float))
    if options.get("audio_extra"):
        for c in ("speechiness", "instrumentalness", "acousticness", "liveness"):
            out[c + "_log"] = np.log1p(rows[c].clip(lower=0) * 1000)
        out["instrumental_near_zero"] = (rows.instrumentalness < 0.001).astype(float)
        out["energy_loudness_gap"] = rows.energy / np.maximum(10 ** (rows.loudness / 20), 0.001)
        out["speech_dance"] = rows.speechiness * rows.danceability
        out["duration_log"] = np.log1p(duration)
    return pd.DataFrame(out, index=rows.index, dtype=float)


class FastFeatures:
    def __init__(self, options):
        self.options = dict(options)
        self.base = FeaturePipeline({"base": "F2", "blocks": options.get("blocks", ["date_count", "date_te3"]),
                                     "drop": options.get("drop", [])})
        self.hier = HierarchicalDate(options["hier"]) if options.get("hier") else None

    def _extra(self, rows, groups=None):
        parts = [stateless(rows, self.options)]
        if self.options.get("date_counts"):
            values = {}
            for length in (4, 7):
                keys = rows[DATE].str[:length]
                a = keys.map(self.counts[length]).fillna(0).to_numpy(float)
                if groups is not None:
                    a -= 1  # every fitting row's own group appears in its key
                values[f"date_count_{length}"] = np.log1p(np.maximum(a, 0))
            parts.append(pd.DataFrame(values, index=rows.index))
        if self.options.get("year_relative"):
            keys = rows[DATE].str[:4]
            values = {}
            for c in self.relative_columns:
                mean = keys.map(self.year_means[c]).fillna(self.global_means[c]).to_numpy()
                values[c + "_year_relative"] = rows[c].to_numpy() - mean
            parts.append(pd.DataFrame(values, index=rows.index))
        return parts

    def fit_transform(self, rows, y, groups, ctx):
        if self.options.get("date_counts"):
            self.counts = {}
            for length in (4, 7):
                frame = pd.DataFrame({"key": rows[DATE].str[:length].to_numpy(), "group": groups})
                self.counts[length] = frame.drop_duplicates().groupby("key").size().to_dict()
        if self.options.get("year_relative"):
            self.relative_columns = [c for c in AUDIO if c not in ("key", "mode", "time_signature")]
            self.global_means = rows[self.relative_columns].mean().to_dict()
            self.year_means = {}
            for c in self.relative_columns:
                stats = rows[c].groupby(rows[DATE].str[:4]).agg(["sum", "count"])
                self.year_means[c] = ((stats["sum"] + 20 * self.global_means[c]) / (stats["count"] + 20)).to_dict()
        parts = [self.base.fit_transform(rows, y, groups, ctx)] + self._extra(rows, groups)
        if self.hier:
            parts.append(self.hier.fit_transform(rows, y, groups, ctx))
        return pd.concat(parts, axis=1).astype(float)

    def transform(self, rows):
        parts = [self.base.transform(rows)] + self._extra(rows)
        if self.hier:
            parts.append(self.hier.transform(rows))
        return pd.concat(parts, axis=1).astype(float)
