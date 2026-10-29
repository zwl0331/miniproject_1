"""Loading the competition data with a permanent row identity and an explicit target map (Finding 001)."""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

AUDIO = ["time_signature", "speechiness", "danceability", "duration_ms", "energy", "mode", "instrumentalness",
         "valence", "key", "tempo", "loudness", "acousticness", "liveness"]
DATE = "track_album_release_date"
TARGET = "Popularity_Type"
TARGET_MAP = {"Low": 0, "High": 1}
INCOMPLETE_ROW_IDS = (425,)  # Finding 001: the only training row without audio features
FILES = {"train": "data/train.csv", "test": "data/test.csv", "sample": "data/sample_submission.csv"}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode_target(labels):
    """Map "Low" to 0 and "High" to 1; reject a missing or any other label."""
    labels = pd.Series(labels)
    if labels.isna().any():
        raise ValueError("missing target label")
    unknown = set(labels.unique()) - set(TARGET_MAP)
    if unknown:
        raise ValueError(f"unknown target labels: {sorted(map(str, unknown))}")
    return labels.map(TARGET_MAP).astype(int).to_numpy()


@dataclass
class Dataset:
    root: Path
    train: pd.DataFrame  # every training row, with the permanent column train_row_id
    test: pd.DataFrame
    sample: pd.DataFrame
    y_all: np.ndarray  # target of every training row, High = 1
    eligible: np.ndarray  # rows used for training and validation: all audio features present
    hashes: dict

    @property
    def row_ids(self):
        return self.train["train_row_id"].to_numpy()[self.eligible]

    @property
    def y(self):
        return self.y_all[self.eligible]

    @property
    def train_eligible(self):
        return self.train[self.eligible]

    @property
    def test_id(self):
        return self.test["ID"].to_numpy()


def load_dataset(root):
    root = Path(root)
    train = pd.read_csv(root / FILES["train"])
    test = pd.read_csv(root / FILES["test"])
    sample = pd.read_csv(root / FILES["sample"])
    train.insert(0, "train_row_id", np.arange(len(train)))
    y_all = encode_target(train[TARGET])
    eligible = train[AUDIO].notna().all(axis=1).to_numpy()
    excluded = tuple(int(i) for i in train.loc[~eligible, "train_row_id"])
    if excluded != INCOMPLETE_ROW_IDS:
        raise ValueError(f"rows without audio features are {excluded}, expected {INCOMPLETE_ROW_IDS} (Finding 001)")
    missing_test = set(AUDIO + [DATE, "ID"]) - set(test.columns)
    if missing_test:
        raise ValueError(f"test file lacks columns {sorted(missing_test)}")
    if test[AUDIO + [DATE]].isna().any().any():
        raise ValueError("test file has missing audio features or dates")
    if list(sample.columns) != ["ID", TARGET]:
        raise ValueError(f"sample submission columns are {list(sample.columns)}")
    if not test["ID"].is_unique or list(sample["ID"]) != list(test["ID"]):
        raise ValueError("test IDs are not unique or differ from the sample submission's order")
    hashes = {name: sha256_file(root / rel) for name, rel in FILES.items()}
    return Dataset(root, train, test, sample, y_all, eligible, hashes)
