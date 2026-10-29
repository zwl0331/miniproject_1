"""Feature sets F0, F1 and F2 (doc/plans/phase1-data-baseline.md section 4.4; Finding 003)."""
import re
from datetime import date

import numpy as np
import pandas as pd

from songpop.data import AUDIO, DATE

FEATURE_SETS = ("F0", "F1", "F2")
F2_DATE = ["year", "month", "day", "day_of_year", "weekday", "date_precision", "days"]
FEATURE_VERSION = "features_v1"
FEATURE_DEFINITION = {"version": FEATURE_VERSION, "audio": AUDIO, "F1": ["year"], "F2": F2_DATE,
                      "dates": "observed components only; day, day_of_year, weekday, days for full dates; "
                               "day_of_year January 1 = 1; weekday Monday = 0; days since 1970-01-01"}
_PATTERN = re.compile(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", re.ASCII)
_EPOCH = date(1970, 1, 1)


def parse_date(s):
    """Every observed component of a YYYY, YYYY-MM or YYYY-MM-DD string; NaN for what is not observed.
    Raises ValueError on any other string, a year outside 1 to 9999, or an impossible month or date."""
    m = _PATTERN.fullmatch(s) if isinstance(s, str) else None
    if m is None:
        raise ValueError(f"malformed release date: {s!r}")
    year, month, day = int(m.group(1)), m.group(2), m.group(3)
    if not 1 <= year <= 9999:
        raise ValueError(f"impossible year in release date: {s!r}")
    out = dict.fromkeys(F2_DATE, np.nan)
    out["year"], out["date_precision"] = float(year), float(len(s))
    if month is not None:
        if not 1 <= int(month) <= 12:
            raise ValueError(f"impossible month in release date: {s!r}")
        out["month"] = float(month)
    if day is not None:
        d = date(year, int(month), int(day))  # raises ValueError for an impossible date
        out["day"] = float(d.day)
        out["day_of_year"] = float(d.timetuple().tm_yday)  # January 1 is 1
        out["weekday"] = float(d.weekday())  # Monday 0 to Sunday 6
        out["days"] = float((d - _EPOCH).days)
    return out


def date_frame(dates):
    dates = pd.Series(dates)
    parsed = {s: parse_date(s) for s in dates.unique()}
    return pd.DataFrame([parsed[s] for s in dates], index=dates.index, columns=F2_DATE, dtype="float64")


def make_features(frame, feature_set):
    """A float64 frame with a fixed column order and the frame's index: F0 = audio, F1 = F0 + year,
    F2 = F1 + the other date fields."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}")
    audio = frame[AUDIO].astype("float64")
    if feature_set == "F0":
        return audio
    dates = date_frame(frame[DATE])
    cols = ["year"] if feature_set == "F1" else F2_DATE
    return pd.concat([audio, dates[cols]], axis=1)
