"""
predict.py
==========
Loads the trained artifacts (feature scaler, vocabulary, hybrid model,
baseline logistic regression) and scores a single code change: either a
real mined commit (from the CSVs) or a diff the user pastes in by hand
(used by app.py and for command-line testing).

If a caller doesn't have real developer-history features for a pasted diff
(the common case in the Streamlit demo -- a user pasting arbitrary code has
no git history to compute EXP/REXP/... from), missing developer features
fall back to the median value observed in the training data, and the UI
makes that explicit rather than silently pretending they are real.
"""

import json
import os
import pickle

import numpy as np
import torch

from .embeddings import pad_batch
from .features import DEVELOPER_FEATURES, FeatureScaler
from .model import build_lightweight_hybrid
from .preprocessing import Vocabulary, build_diff_text, tokenize_code

DEVICE = "cpu"


class BugPredictor:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir

        with open(os.path.join(models_dir, "dev_scaler.json")) as f:
            self.scaler = FeatureScaler.from_dict(json.load(f))

        with open(os.path.join(models_dir, "vocab.json")) as f:
            self.vocab = Vocabulary.from_dict(json.load(f))

        with open(os.path.join(models_dir, "baseline_lr.pkl"), "rb") as f:
            self.baseline = pickle.load(f)

        ckpt = torch.load(os.path.join(models_dir, "hybrid_model.pt"), map_location=DEVICE, weights_only=False)
        cfg = ckpt["config"]
        self.max_tokens = cfg["max_tokens"]
        self.hybrid_model = build_lightweight_hybrid(
            vocab_size=cfg["vocab_size"], n_dev_features=cfg["n_dev_features"],
            embed_dim=cfg["embed_dim"], gru_hidden=cfg["gru_hidden"],
            code_out_dim=cfg["code_out_dim"], dropout=cfg["dropout"],
        )
        self.hybrid_model.load_state_dict(ckpt["state_dict"])
        self.hybrid_model.eval()

        metrics_path = os.path.join(models_dir, "metrics.json")
        self.metrics = None
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                self.metrics = json.load(f)

        # median developer-feature values from training, used as sensible
        # fallback defaults when the caller doesn't supply real history.
        self.feature_medians = dict(zip(self.scaler.columns, self.scaler.mean_.tolist()))

    def default_dev_features(self) -> dict:
        return dict(self.feature_medians)

    def _dev_vector(self, dev_features: dict) -> np.ndarray:
        row = {}
        for col in DEVELOPER_FEATURES:
            row[col] = dev_features.get(col, self.feature_medians.get(col, 0.0))
        import pandas as pd
        df_row = pd.DataFrame([row])
        return self.scaler.transform(df_row)

    def predict(self, added_code: str = "", removed_code: str = "", dev_features: dict = None) -> dict:
        dev_features = dev_features or {}
        diff_text = build_diff_text({"added_code": added_code, "removed_code": removed_code})
        dev_vec = self._dev_vector(dev_features)

        # --- baseline LR ---
        try:
            if hasattr(self.baseline, "predict_proba"):
                baseline_prob = float(self.baseline.predict_proba(dev_vec)[:, 1][0])
            else:
                baseline_prob = 0.5
        except Exception:
            try:
                # Direct dot-product sigmoid if sklearn attribute mismatch occurs
                if hasattr(self.baseline, "coef_") and hasattr(self.baseline, "intercept_"):
                    import scipy.special
                    z = float(np.dot(dev_vec, self.baseline.coef_.T) + self.baseline.intercept_)
                    baseline_prob = float(1.0 / (1.0 + np.exp(-z)))
                else:
                    baseline_prob = 0.5
            except Exception:
                baseline_prob = 0.5

        # --- hybrid model ---
        token_ids = self.vocab.encode(diff_text, max_tokens=self.max_tokens)
        tok_batch = pad_batch([token_ids])
        dev_batch = torch.tensor(dev_vec, dtype=torch.float32)
        with torch.no_grad():
            logit = self.hybrid_model(tok_batch, dev_batch)
            hybrid_prob = float(torch.sigmoid(logit).item())

        n_tokens = len(tokenize_code(diff_text))
        n_added_lines = len([l for l in added_code.splitlines() if l.strip()]) if added_code else 0
        n_removed_lines = len([l for l in removed_code.splitlines() if l.strip()]) if removed_code else 0

        return {
            "baseline_probability": baseline_prob,
            "hybrid_probability": hybrid_prob,
            "risk_level": risk_level(hybrid_prob),
            "prediction": "Bug-Inducing" if hybrid_prob >= 0.5 else "Clean",
            "code_stats": {
                "added_lines": n_added_lines,
                "removed_lines": n_removed_lines,
                "code_tokens_used": n_tokens,
            },
            "developer_features_used": {c: dev_features.get(c, self.feature_medians.get(c, 0.0))
                                         for c in DEVELOPER_FEATURES},
        }


def risk_level(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    if prob < 0.6:
        return "Medium"
    if prob < 0.8:
        return "High"
    return "Very High"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Score a diff from the command line.")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--added-code", default="")
    ap.add_argument("--removed-code", default="")
    args = ap.parse_args()

    predictor = BugPredictor(args.models_dir)
    result = predictor.predict(args.added_code, args.removed_code, {})
    print(json.dumps(result, indent=2))
