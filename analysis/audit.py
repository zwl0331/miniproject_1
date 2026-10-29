"""Data audit for task P1.1: the evidence behind Findings 001 to 005 (doc/findings.md).

Reads only data/train.csv, data/test.csv and data/sample_submission.csv and fits nothing: every
number is descriptive, and test rows are only described (doc/plans/modeling_strategy.md section 3).
Finding 006 rests on analysis/scoping.py and analysis/scoping_output.txt, reviewed in P0.2.

Usage, from the project root: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python analysis/audit.py
It prints to standard output and writes no file. The run of record is analysis/audit_output.txt.
"""
import hashlib
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ["time_signature", "speechiness", "danceability", "duration_ms", "energy", "mode", "instrumentalness",
         "valence", "key", "tempo", "loudness", "acousticness", "liveness"]
CATEGORICAL = ["time_signature", "mode", "key"]
DATE, TARGET = "track_album_release_date", "Popularity_Type"
FILES = ["data/train.csv", "data/test.csv", "data/sample_submission.csv"]
DATE_FORMAT = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")
ERAS = [(-np.inf, 1989, "before 1990"), (1990, 1999, "1990s"), (2000, 2009, "2000s"), (2010, 2019, "2010s"),
        (2020, np.inf, "2020 on")]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    """Raw strings, with only the empty field as missing."""
    return pd.read_csv(ROOT / name, dtype=str, keep_default_na=False, na_values=[""])


def components(edge_keys):
    """Connected components of row positions joined when they share a non-null key in any key list.
    Each component is labeled by its smallest row position."""
    n = len(edge_keys[0])
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for keys in edge_keys:
        first = {}
        for i, k in enumerate(keys):
            if k is None:
                continue
            if k in first:
                a, b = find(i), find(first[k])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                first[k] = i
    return np.array([find(i) for i in range(n)])


def signature(df, cols):
    """Exact key per row: audio columns as floats, the date as its raw string; None if any is missing."""
    parts = [df[c].astype(float) if c in AUDIO else df[c] for c in cols]
    missing = df[cols].isna().any(axis=1).values
    return [None if m else k for k, m in zip(zip(*parts), missing)]


def ks_distance(x, z):
    """Two-sample Kolmogorov-Smirnov distance, descriptive only."""
    x, z = np.sort(x), np.sort(z)
    points = np.union1d(x, z)
    return float(np.max(np.abs(np.searchsorted(x, points, side="right") / len(x)
                               - np.searchsorted(z, points, side="right") / len(z))))


def parse_dates(s):
    """Validate each date string; return year, month (or NaN), day (or NaN), precision, and validity."""
    out = []
    for v in s:
        m = DATE_FORMAT.match(v) if isinstance(v, str) else None
        if not m:
            out.append((np.nan, np.nan, np.nan, len(v) if isinstance(v, str) else 0, False))
            continue
        y, mo, d = int(m.group(1)), m.group(2), m.group(3)
        ok = True
        if mo is not None:
            ok = 1 <= int(mo) <= 12
        if d is not None and ok:
            ok = pd.to_datetime(v, format="%Y-%m-%d", errors="coerce") is not pd.NaT
        out.append((y, float(mo) if mo else np.nan, float(d) if d else np.nan, len(v), ok))
    return pd.DataFrame(out, columns=["year", "month", "day", "precision", "valid"], index=s.index)


def group_summary(name, g, y, uri):
    sizes = pd.Series(g).value_counts()
    hist = {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()}
    d = pd.DataFrame({"g": g, "y": y, "uri": uri})
    mixed = d.groupby("g")["y"].nunique().gt(1)
    multi_uri = d.dropna(subset=["uri"]).groupby("g")["uri"].nunique().gt(1)
    print(f"     {name}: groups {len(sizes)}; group sizes (size: count) {hist}; mixed-label groups {int(mixed.sum())} "
          f"(rows {int(d['g'].isin(mixed[mixed].index).sum())}); groups joining more than one uri {int(multi_uri.sum())}")


def main():
    print("[P] provenance")
    print(f"     command: {' '.join([Path(sys.executable).name] + sys.argv)}")
    print(f"     python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
          f"scikit-learn {sklearn.__version__}; audit.py sha256 {sha256(__file__)}")
    for p in FILES:
        print(f"     {p} sha256 {sha256(ROOT / p)}")

    tr, te, ss = read("data/train.csv"), read("data/test.csv"), read("data/sample_submission.csv")

    print("[A1] dataset contract, target, missingness (Finding 001)")
    print(f"     shapes: train {tr.shape}, test {te.shape}, sample_submission {ss.shape}")
    print(f"     train columns in order: {list(tr.columns)}")
    print(f"     test columns in order: {list(te.columns)}")
    print(f"     columns only in train {sorted(set(tr.columns) - set(te.columns))}; "
          f"only in test {sorted(set(te.columns) - set(tr.columns))}")
    vc = tr[TARGET].value_counts(dropna=False)
    print(f"     {TARGET} values (missing included): { {str(k): int(v) for k, v in vc.items()} }; "
          f"High rate {(tr[TARGET] == 'High').mean():.4f}")
    for name, d in [("train", tr), ("test", te)]:
        miss = d.isna().sum()
        num = d[AUDIO].apply(pd.to_numeric, errors="coerce")
        bad = int((num.isna() & d[AUDIO].notna()).sum().sum())
        nonfinite = int((~np.isfinite(num.values) & num.notna().values).sum())
        print(f"     {name}: missing per column { {k: int(v) for k, v in miss[miss > 0].items()} or 'none'}; "
              f"audio values that fail to parse {bad}; non-finite audio values {nonfinite}")
    inc = tr[tr[AUDIO].isna().all(axis=1)]
    for i, row in inc.iterrows():
        print(f"     incomplete train row (position {i}): non-missing fields "
              f"{ {c: row[c] for c in tr.columns if pd.notna(row[c])} }")
    part = tr[tr[AUDIO].isna().any(axis=1) & ~tr[AUDIO].isna().all(axis=1)]
    print(f"     train rows with some but not all audio features missing: {len(part)}")
    comp = tr[tr[AUDIO].notna().all(axis=1)]
    print(f"     complete train rows {len(comp)}: High {int((comp[TARGET] == 'High').sum())}, "
          f"Low {int((comp[TARGET] == 'Low').sum())}")
    print(f"     test ID unique {te['ID'].is_unique}, range {te['ID'].astype(int).min()} to {te['ID'].astype(int).max()}; "
          f"sample_submission columns {list(ss.columns)}, its IDs equal the test IDs in order "
          f"{list(ss['ID']) == list(te['ID'])}, its values {sorted(ss['Popularity_Type'].unique())}")
    for c in CATEGORICAL:
        a = {str(k): int(v) for k, v in tr[c].value_counts().sort_index().items()}
        b = {str(k): int(v) for k, v in te[c].value_counts().sort_index().items()}
        print(f"     {c} values (train; test): {a}; {b}")
    for c in [c for c in AUDIO if c not in CATEGORICAL]:
        a, b = tr[c].astype(float), te[c].astype(float)
        print(f"     {c:16s} range train {a.min():.6g} to {a.max():.6g}; test {b.min():.6g} to {b.max():.6g}")

    print("[A2] track identity and repeated observations (Finding 002)")
    for name, d in [("train", tr), ("test", te)]:
        ok = d["uri"].notna()
        tok = d.loc[ok, "uri"].str.rsplit(":", n=1).str[-1]
        same = [bool((d.loc[ok, c].str.rsplit("/", n=1).str[-1] == tok).all()) for c in ["track_href", "analysis_url"]]
        print(f"     {name}: uri missing {int((~ok).sum())}; track_href and analysis_url end in the uri token {same}; "
              f"type values {sorted(d['type'].dropna().unique())}")
        n = d.groupby("uri").size()
        rep = n[n > 1]
        vary = d[d["uri"].isin(rep.index)].groupby("uri")[AUDIO + [DATE]].nunique(dropna=False).gt(1).any(axis=1).sum()
        print(f"     {name}: distinct uri {d['uri'].nunique()}; uri group sizes (size: count) "
              f"{ {int(k): int(v) for k, v in n.value_counts().sort_index().items()} }; repeated groups {len(rep)} "
              f"with {int(rep.sum())} rows; groups whose features or date vary {int(vary)}; "
              f"share of rows in repeated uri groups {d['uri'].isin(rep.index).mean():.4f}")
    lab = tr.groupby("uri")[TARGET].nunique()
    conf = lab[lab > 1].index
    print(f"     train uri groups with conflicting labels {len(conf)}, rows {int(tr['uri'].isin(conf).sum())}")
    print(f"     uri shared by train and test {len(set(tr['uri'].dropna()) & set(te['uri'].dropna()))}")
    for label, cols in [("audio + date", AUDIO + [DATE]), ("audio only", AUDIO)]:
        s_tr, s_te = signature(tr, cols), signature(te, cols)
        d_tr = {k for k in s_tr if k is not None}
        d_te = {k for k in s_te if k is not None}
        print(f"     {label} signatures: train complete rows {sum(k is not None for k in s_tr)}, distinct {len(d_tr)}; "
              f"test distinct {len(d_te)}; distinct signatures shared by train and test {len(d_tr & d_te)}")
    uri_key = [None if pd.isna(u) else u for u in tr["uri"]]
    g_uri = components([uri_key])
    g_prod = components([uri_key, signature(tr, AUDIO + [DATE])])
    g_strict = components([uri_key, signature(tr, AUDIO + [DATE]), signature(tr, AUDIO)])
    print("     train groups as connected components (the incomplete row is a singleton in each):")
    group_summary("uri only", g_uri, tr[TARGET].values, tr["uri"].values)
    group_summary("uri or identical audio + date (primary)", g_prod, tr[TARGET].values, tr["uri"].values)
    group_summary("also identical audio only (strict)", g_strict, tr[TARGET].values, tr["uri"].values)
    coarsens = pd.DataFrame({"p": g_prod, "s": g_strict}).groupby("p")["s"].nunique().max() == 1
    print(f"     strict grouping coarsens the primary one: {bool(coarsens)}")

    print("[A3] release-date precision and validity (Finding 003)")
    for name, d in [("train", tr), ("test", te)]:
        p = parse_dates(d[DATE])
        full = p["precision"] == 10
        days = (pd.to_datetime(d.loc[full, DATE], format="%Y-%m-%d") - pd.Timestamp("1970-01-01")).dt.days
        print(f"     {name}: missing dates {int(d[DATE].isna().sum())}; invalid or malformed {int((~p['valid']).sum())}; "
              f"precision (length: count) { {int(k): int(v) for k, v in p['precision'].value_counts().sort_index().items()} }; "
              f"year-month values {d.loc[p['precision'] == 7, DATE].tolist()}")
        print(f"     {name}: year range {int(p['year'].min())} to {int(p['year'].max())}; full dates "
              f"{d.loc[full, DATE].min()} to {d.loc[full, DATE].max()}, days since 1970 {int(days.min())} to {int(days.max())}; "
              f"F2 fields left missing by partial dates: month {int(p['month'].isna().sum())}, "
              f"day, day of year, weekday and days {int((~full).sum())}")

    y = (tr[TARGET] == "High").astype(int).values
    yr = parse_dates(tr[DATE])["year"].values
    tyr = parse_dates(te[DATE])["year"].values

    print("[A4] marginal associations with High (Finding 004)")
    print("     release era: train rows, distinct uri, High count, High rate; test rows")
    for lo, hi, lab_ in ERAS:
        m = (yr >= lo) & (yr <= hi)
        print(f"     {lab_:12s} {int(m.sum()):5d} {tr.loc[m, 'uri'].nunique():5d} {int(y[m].sum()):5d} "
              f"{y[m].mean():.4f} {int(((tyr >= lo) & (tyr <= hi)).sum()):5d}")
    print("     univariate AUC, High positive, increasing raw value as the score")
    num = tr[AUDIO].astype(float)
    res = []
    for c in AUDIO:
        ok = num[c].notna().values
        res.append((c, roc_auc_score(y[ok], num[c].values[ok]), int(ok.sum()), int(y[ok].sum()), int((~ok).sum())))
    res.append(("release year", roc_auc_score(y, yr), len(y), int(y.sum()), 0))
    for c, a, n, npos, nmiss in sorted(res, key=lambda t: -abs(t[1] - 0.5)):
        print(f"     {c:16s} AUC {a:.4f}  n {n} (High {npos}, Low {n - npos}), missing {nmiss}")

    print("[A5] train versus test, descriptive; row order (Finding 005)")
    print(f"     release year percentiles 10/25/50/75/90: train {np.percentile(yr, [10, 25, 50, 75, 90])}, "
          f"test {np.percentile(tyr, [10, 25, 50, 75, 90])}")
    te_num = te[AUDIO].astype(float)
    ks = [("release year", ks_distance(yr, tyr), len(yr), len(tyr))]
    ks += [(c, ks_distance(num[c].dropna().values, te_num[c].values), int(num[c].notna().sum()), len(te_num))
           for c in AUDIO]
    for c, k, n1, n2 in sorted(ks, key=lambda t: -t[1]):
        print(f"     KS distance {c:16s} {k:.4f}  (train n {n1}, test n {n2})")
    share = te[DATE].isin(set(tr[DATE]))
    print(f"     test rows whose exact date string occurs in train: {int(share.sum())} of {len(te)} ({share.mean():.4f})")
    print(f"     AUC of the train row position for High {roc_auc_score(y, np.arange(len(tr))):.4f}; "
          f"correlation of test ID with release year {np.corrcoef(te['ID'].astype(int), tyr)[0, 1]:+.4f}")


if __name__ == "__main__":
    main()
