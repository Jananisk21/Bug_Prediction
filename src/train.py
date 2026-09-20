"""
train.py
========
Trains and evaluates BOTH models on the mined, time-ordered dataset:

  1. Baseline  : Developer/Traditional Features -> Logistic Regression
  2. Proposed  : Code Diff -> [Tokenize -> Embed -> BiGRU] (Code Branch)
                 + Dev Features -> [Normalize -> Dense]   (Developer Branch)
                 -> Concatenate -> Dense -> Dropout -> Output (Hybrid model)

Split is strictly time-based (train = older commits, test = newer commits;
see dataset_setup.py). A further chronological slice of the *train* set is
held out as a validation set for early stopping / model selection, so the
test set is never touched until final evaluation.

All reported numbers come from actually running these models on the mined
data -- run this script yourself to reproduce them; nothing is hard-coded.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import classification_metrics, effort_aware_metrics
from src.features import DEVELOPER_FEATURES, FeatureScaler, LABEL_COLUMN
from src.model import build_lightweight_hybrid
from src.embeddings import pad_batch
from src.preprocessing import Vocabulary, load_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chronological_val_split(train_df: pd.DataFrame, val_frac: float = 0.15):
    train_df = train_df.sort_values("timestamp").reset_index(drop=True)
    n = len(train_df)
    split_idx = int(n * (1 - val_frac))
    return train_df.iloc[:split_idx].reset_index(drop=True), train_df.iloc[split_idx:].reset_index(drop=True)


def make_batches(df, batch_size, shuffle=True, seed=0):
    n = len(df)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def train_hybrid_model(train_df, val_df, vocab: Vocabulary, scaler: FeatureScaler,
                        epochs: int = 12, batch_size: int = 32, lr: float = 1e-3,
                        max_tokens: int = 200, embed_dim: int = 64, gru_hidden: int = 64,
                        code_out_dim: int = 128, dropout: float = 0.3, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_lightweight_hybrid(
        vocab_size=len(vocab), n_dev_features=len(DEVELOPER_FEATURES),
        embed_dim=embed_dim, gru_hidden=gru_hidden, code_out_dim=code_out_dim,
        dropout=dropout,
    ).to(DEVICE)

    dev_train = torch.tensor(scaler.transform(train_df), dtype=torch.float32)
    dev_val = torch.tensor(scaler.transform(val_df), dtype=torch.float32)
    y_train = train_df[LABEL_COLUMN].values.astype(np.float32)
    y_val = val_df[LABEL_COLUMN].values.astype(np.float32)

    token_ids_train = [vocab.encode(t, max_tokens=max_tokens) for t in train_df["diff_text"]]
    token_ids_val = [vocab.encode(t, max_tokens=max_tokens) for t in val_df["diff_text"]]

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_auc = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx in make_batches(train_df, batch_size, shuffle=True, seed=seed + epoch):
            tok_batch = pad_batch([token_ids_train[i] for i in batch_idx]).to(DEVICE)
            dev_batch = dev_train[batch_idx].to(DEVICE)
            y_batch = torch.tensor(y_train[batch_idx], dtype=torch.float32).to(DEVICE)

            optimizer.zero_grad()
            logits = model(tok_batch, dev_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)
        avg_loss = total_loss / len(train_df)

        model.eval()
        with torch.no_grad():
            val_probs = []
            for batch_idx in make_batches(val_df, batch_size, shuffle=False):
                tok_batch = pad_batch([token_ids_val[i] for i in batch_idx]).to(DEVICE)
                dev_batch = dev_val[batch_idx].to(DEVICE)
                logits = model(tok_batch, dev_batch)
                val_probs.append(torch.sigmoid(logits).cpu().numpy())
            val_probs = np.concatenate(val_probs)
        val_metrics = classification_metrics(y_val, val_probs)
        val_auc = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else 0.0
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_auc": val_auc,
                         "val_f1": val_metrics["f1"]})
        print(f"    epoch {epoch:2d}/{epochs}  train_loss={avg_loss:.4f}  "
              f"val_auc={val_auc:.4f}  val_f1={val_metrics['f1']:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val_auc


@torch.no_grad()
def predict_hybrid(model, df, vocab: Vocabulary, scaler: FeatureScaler,
                    max_tokens: int = 200, batch_size: int = 64):
    model.eval()
    dev_X = torch.tensor(scaler.transform(df), dtype=torch.float32)
    token_ids = [vocab.encode(t, max_tokens=max_tokens) for t in df["diff_text"]]
    probs = []
    for start in range(0, len(df), batch_size):
        idx = list(range(start, min(start + batch_size, len(df))))
        tok_batch = pad_batch([token_ids[i] for i in idx]).to(DEVICE)
        dev_batch = dev_X[idx].to(DEVICE)
        logits = model(tok_batch, dev_batch)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def train_baseline(train_df, scaler: FeatureScaler):
    X_train = scaler.transform(train_df)
    y_train = train_df[LABEL_COLUMN].values
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(X_train, y_train)
    return clf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-csv", default="data/flask_commits_train.csv")
    ap.add_argument("--test-csv", default="data/flask_commits_test.csv")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--embedding", choices=["lightweight", "codebert"], default="lightweight")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.embedding == "codebert":
        print("[train] --embedding codebert requires internet access to "
              "huggingface.co to download microsoft/codebert-base, and is "
              "meant to be run on your own laptop. See README.md ->"
              " 'Switching to CodeBERT'. Falling back is NOT automatic; "
              "re-run with --embedding lightweight if this fails.")
        raise SystemExit(
            "codebert path not wired into this trainer script directly -- "
            "see src/embeddings.py CodeBertEmbedder and README for the "
            "precompute-then-train recipe."
        )

    os.makedirs(args.models_dir, exist_ok=True)
    plots_dir = os.path.join(args.models_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"[train] Loading {args.train_csv} / {args.test_csv} ...")
    train_full = load_split(args.train_csv)
    test_df = load_split(args.test_csv)
    train_df, val_df = chronological_val_split(train_full, val_frac=args.val_frac)
    print(f"[train] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"[train] label rate -> train={train_df[LABEL_COLUMN].mean():.3f} "
          f"val={val_df[LABEL_COLUMN].mean():.3f} test={test_df[LABEL_COLUMN].mean():.3f}")

    # ---------------- feature scaler (fit on train ONLY) ----------------
    scaler = FeatureScaler(columns=DEVELOPER_FEATURES).fit(train_df)
    with open(os.path.join(args.models_dir, "dev_scaler.json"), "w") as f:
        json.dump(scaler.to_dict(), f, indent=2)

    # ---------------- vocabulary (built on train ONLY) ----------------
    vocab = Vocabulary.build(train_df["diff_text"].tolist(), max_size=20000, min_freq=2)
    print(f"[train] vocabulary size: {len(vocab)}")
    with open(os.path.join(args.models_dir, "vocab.json"), "w") as f:
        json.dump(vocab.to_dict(), f)

    results = {}

    # =====================================================================
    # 1. Baseline: Developer/Traditional Features -> Logistic Regression
    # =====================================================================
    print("\n[train] ===== Baseline: Logistic Regression on traditional/developer features =====")
    baseline = train_baseline(train_df, scaler)
    with open(os.path.join(args.models_dir, "baseline_lr.pkl"), "wb") as f:
        pickle.dump(baseline, f)

    test_X = scaler.transform(test_df)
    baseline_probs = baseline.predict_proba(test_X)[:, 1]
    baseline_metrics = classification_metrics(test_df[LABEL_COLUMN].values, baseline_probs)
    baseline_effort = effort_aware_metrics(
        test_df[LABEL_COLUMN].values, baseline_probs, test_df["la"].values + test_df["ld"].values
    )
    results["baseline_logistic_regression"] = {
        "classification": baseline_metrics, "effort_aware": baseline_effort,
    }
    print(f"    accuracy={baseline_metrics['accuracy']:.3f} precision={baseline_metrics['precision']:.3f} "
          f"recall={baseline_metrics['recall']:.3f} f1={baseline_metrics['f1']:.3f} "
          f"roc_auc={baseline_metrics['roc_auc']}")

    # =====================================================================
    # 2. Proposed hybrid model: Code Embedding + Developer Activity -> DL
    # =====================================================================
    print("\n[train] ===== Proposed hybrid model: Code Branch + Developer Branch =====")
    model, history, best_val_auc = train_hybrid_model(
        train_df, val_df, vocab, scaler,
        epochs=args.epochs, batch_size=args.batch_size, max_tokens=args.max_tokens,
        seed=args.seed,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "vocab_size": len(vocab), "n_dev_features": len(DEVELOPER_FEATURES),
                "embed_dim": 64, "gru_hidden": 64, "code_out_dim": 128, "dropout": 0.3,
                "max_tokens": args.max_tokens,
            },
        },
        os.path.join(args.models_dir, "hybrid_model.pt"),
    )

    hybrid_probs = predict_hybrid(model, test_df, vocab, scaler, max_tokens=args.max_tokens)
    hybrid_metrics = classification_metrics(test_df[LABEL_COLUMN].values, hybrid_probs)
    hybrid_effort = effort_aware_metrics(
        test_df[LABEL_COLUMN].values, hybrid_probs, test_df["la"].values + test_df["ld"].values
    )
    results["hybrid_code_plus_developer"] = {
        "classification": hybrid_metrics, "effort_aware": hybrid_effort,
        "training_history": history, "best_val_auc": best_val_auc,
    }
    print(f"    accuracy={hybrid_metrics['accuracy']:.3f} precision={hybrid_metrics['precision']:.3f} "
          f"recall={hybrid_metrics['recall']:.3f} f1={hybrid_metrics['f1']:.3f} "
          f"roc_auc={hybrid_metrics['roc_auc']}")

    # ---------------- persist everything ----------------
    dataset_info = {
        "train_csv": args.train_csv, "test_csv": args.test_csv,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "train_label_rate": float(train_df[LABEL_COLUMN].mean()),
        "val_label_rate": float(val_df[LABEL_COLUMN].mean()),
        "test_label_rate": float(test_df[LABEL_COLUMN].mean()),
    }
    results["dataset"] = dataset_info

    with open(os.path.join(args.models_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[train] Saved metrics -> {os.path.join(args.models_dir, 'metrics.json')}")

    make_plots(test_df[LABEL_COLUMN].values, baseline_probs, hybrid_probs, plots_dir)
    print(f"[train] Saved plots -> {plots_dir}")
    print("\n[train] ===== DONE. Real results (not fabricated) written to models/metrics.json =====")


def make_plots(y_true, baseline_probs, hybrid_probs, plots_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, probs, name in zip(axes, [baseline_probs, hybrid_probs], ["Baseline LR", "Hybrid Model"]):
        cm = confusion_matrix(y_true, (probs >= 0.5).astype(int), labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["Clean", "Bug-Inducing"]).plot(ax=ax, colorbar=False)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "confusion_matrices.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    if len(set(y_true.tolist())) > 1:
        RocCurveDisplay.from_predictions(y_true, baseline_probs, name="Baseline LR", ax=ax)
        RocCurveDisplay.from_predictions(y_true, hybrid_probs, name="Hybrid Model", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("ROC Curve (test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "roc_curve.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
