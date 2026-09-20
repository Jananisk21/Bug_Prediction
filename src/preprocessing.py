"""
preprocessing.py
=================
Loading and cleaning utilities shared by training, prediction and the
Streamlit app. Handles:
  * reading the CSVs produced by dataset_setup.py
  * turning (added_code, removed_code) into a single diff string with
    change-type markers, the same representation used for both the
    lightweight tokenizer and the CodeBERT path
  * a small, dependency-free code tokenizer + vocabulary builder used by the
    lightweight embedding (src/embeddings.py -> LightweightCodeEncoder)
"""

import re
from collections import Counter

import pandas as pd

from .features import DEVELOPER_FEATURES, LABEL_COLUMN

PAD, UNK = "<pad>", "<unk>"

# Splits on whitespace and on common code punctuation while keeping
# identifiers, numbers and operators as separate tokens. Deliberately simple
# -- this is the "lightweight alternative" tokenizer, not a real code parser.
_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|==|!=|<=|>=|->|::|[{}()\[\];,.:+\-*/%<>=!&|^~]"
)


def build_diff_text(row) -> str:
    """Combine added/removed code into one marked-up diff string."""
    added = row.get("added_code", "") or ""
    removed = row.get("removed_code", "") or ""
    if not isinstance(added, str):
        added = ""
    if not isinstance(removed, str):
        removed = ""
    return f"[ADD] {added} [DEL] {removed}".strip()


def tokenize_code(text: str, max_tokens: int = 256):
    if not isinstance(text, str) or not text:
        return []
    tokens = _TOKEN_RE.findall(text)
    return tokens[:max_tokens]


def load_split(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in DEVELOPER_FEATURES:
        df[col] = df[col].fillna(0.0)
    df["added_code"] = df["added_code"].fillna("")
    df["removed_code"] = df["removed_code"].fillna("")
    df["diff_text"] = df.apply(build_diff_text, axis=1)
    return df


class Vocabulary:
    """A tiny word2idx vocabulary for the lightweight trainable code
    embedding (no external downloads required)."""

    def __init__(self, tokens_to_id=None):
        self.token_to_id = tokens_to_id or {PAD: 0, UNK: 1}

    @classmethod
    def build(cls, texts, max_size: int = 20000, min_freq: int = 2):
        counter = Counter()
        for t in texts:
            counter.update(tokenize_code(t))
        vocab = cls()
        for tok, freq in counter.most_common():
            if freq < min_freq:
                continue
            if len(vocab.token_to_id) >= max_size:
                break
            if tok not in vocab.token_to_id:
                vocab.token_to_id[tok] = len(vocab.token_to_id)
        return vocab

    def encode(self, text, max_tokens: int = 256):
        ids = [
            self.token_to_id.get(tok, self.token_to_id[UNK])
            for tok in tokenize_code(text, max_tokens=max_tokens)
        ]
        if not ids:
            ids = [self.token_to_id[PAD]]
        return ids

    def __len__(self):
        return len(self.token_to_id)

    def to_dict(self):
        return self.token_to_id

    @classmethod
    def from_dict(cls, d):
        return cls(tokens_to_id=d)
