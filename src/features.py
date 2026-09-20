"""
features.py
============
Defines the feature schema used throughout the project and provides
normalization helpers. Every feature listed here is produced causally by
dataset_setup.py (i.e. it only uses information available strictly before
the commit being scored) -- nothing here is invented after the fact.

Two groups are kept explicit because the paper's proposed architecture uses
them differently:

  * SIZE_DIFFUSION_FEATURES  - Kamei-style "what changed" metrics (size of
    the change, how scattered it is). These are cheap process metrics, not
    developer-activity metrics, but the paper's baseline (LR-JIT) and the
    Developer Branch both use the full traditional feature set, matching how
    Kamei et al. and the fusion studies discussed in the review actually set
    up their baselines.
  * DEVELOPER_ACTIVITY_FEATURES - the developer-activity signals explicitly
    requested in the project brief (EXP, REXP, SEXP, NDEV, submission rate,
    contribution interval, recency, ...).

DEVELOPER_FEATURES = SIZE_DIFFUSION_FEATURES + DEVELOPER_ACTIVITY_FEATURES
is what feeds both the Logistic Regression baseline and the Developer
Branch of the hybrid model.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

SIZE_DIFFUSION_FEATURES = [
    "la",       # lines added
    "ld",       # lines deleted
    "lt",       # lines-of-code in touched files before the change (causal)
    "nf",       # number of files touched
    "nd",       # number of directories touched
    "ns",       # number of subsystems (top-level dirs) touched
    "entropy",  # Shannon entropy of change distribution across files
    "fix",      # 1 if the commit message itself looks like a bug fix
]

DEVELOPER_ACTIVITY_FEATURES = [
    "exp",                     # developer experience: # prior commits by author
    "rexp",                    # recency-weighted prior experience
    "sexp",                    # prior experience in the same subsystem(s)
    "ndev",                    # # distinct developers who previously touched these files
    "nuc",                     # # unique prior changes to these files
    "age",                     # avg days since touched files last changed
    "submission_rate",         # author's commits/day over trailing 90 days
    "contribution_interval",   # days since author's previous commit (-1 = new author)
    "recency",                 # exp(-interval/30): bounded "how active right now" score
]

DEVELOPER_FEATURES = SIZE_DIFFUSION_FEATURES + DEVELOPER_ACTIVITY_FEATURES

LABEL_COLUMN = "label"


@dataclass
class FeatureScaler:
    """A minimal, dependency-light standardizer (mean/std) that we can save
    as plain JSON-serializable numbers alongside the model, instead of
    pickling a full sklearn object (keeps model artifacts simple & portable).
    """

    columns: list
    mean_: np.ndarray = None
    scale_: np.ndarray = None

    def fit(self, df: pd.DataFrame):
        X = df[self.columns].astype(float).values
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        self.scale_ = std
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.columns].astype(float).values
        return (X - self.mean_) / self.scale_

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def to_dict(self):
        return {
            "columns": self.columns,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(columns=d["columns"])
        obj.mean_ = np.array(d["mean"])
        obj.scale_ = np.array(d["scale"])
        return obj
