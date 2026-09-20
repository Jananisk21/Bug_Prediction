"""
model.py
========
The architecture requested in the project brief:

    Code Diff  -> Tokenization -> Code Embedding -> Neural Network   (Code Branch)
    Dev Feats  -> Normalization -> Dense Layers                      (Developer Branch)
    Code Representation + Developer Representation
        -> Concatenate -> Dense -> Dropout -> Output -> Bug Probability

Two code-branch variants share the same DeveloperBranch / FusionHead:
  * CodeBertProjectionEncoder - takes a precomputed 768-d CodeBERT vector
    (CodeBERT itself is frozen; see src/embeddings.py) and projects it down.
  * LightweightCodeEncoder (src/embeddings.py) - takes raw token ids and
    learns an embedding + BiGRU encoder from scratch. This is the variant
    actually trained in this project (no internet / GPU required).

The plain Logistic-Regression baseline (Developer/Traditional Features ->
Logistic Regression) intentionally has NO deep code branch at all -- it is
implemented directly with scikit-learn in src/train.py, so that the
comparison between "hand-crafted metrics only" and "metrics + learned code
representation" mirrors exactly what the reviewed literature (Kamei's
LR-JIT vs. DeepJIT/CC2Vec/hybrid-fusion models) compares.
"""

import torch
import torch.nn as nn

from .embeddings import LightweightCodeEncoder


class CodeBertProjectionEncoder(nn.Module):
    """Projects a frozen, precomputed CodeBERT embedding down to out_dim."""

    def __init__(self, in_dim: int = 768, out_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


class DeveloperBranch(nn.Module):
    """Normalized developer/process features -> dense layers."""

    def __init__(self, n_features: int, hidden_dim: int = 32, out_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


class HybridBugPredictor(nn.Module):
    """Code representation + Developer representation -> Concatenate ->
    Dense -> Dropout -> Output -> bug probability (logit)."""

    def __init__(self, code_encoder: nn.Module, n_dev_features: int,
                 dev_hidden: int = 32, dev_out: int = 32,
                 fusion_hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.code_encoder = code_encoder
        self.dev_branch = DeveloperBranch(n_dev_features, dev_hidden, dev_out, dropout=0.2)
        fused_dim = code_encoder.out_dim + self.dev_branch.out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, code_input, dev_features):
        code_repr = self.code_encoder(code_input)
        dev_repr = self.dev_branch(dev_features)
        fused = torch.cat([code_repr, dev_repr], dim=-1)
        logit = self.fusion(fused).squeeze(-1)
        return logit  # raw logit; apply sigmoid outside (BCEWithLogitsLoss during training)


def build_lightweight_hybrid(vocab_size: int, n_dev_features: int,
                              embed_dim: int = 64, gru_hidden: int = 64,
                              code_out_dim: int = 128, dropout: float = 0.3) -> HybridBugPredictor:
    code_encoder = LightweightCodeEncoder(
        vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=gru_hidden,
        out_dim=code_out_dim, pad_idx=0, dropout=0.2,
    )
    return HybridBugPredictor(code_encoder, n_dev_features, dropout=dropout)


def build_codebert_hybrid(n_dev_features: int, codebert_dim: int = 768,
                           code_out_dim: int = 128, dropout: float = 0.3) -> HybridBugPredictor:
    code_encoder = CodeBertProjectionEncoder(in_dim=codebert_dim, out_dim=code_out_dim)
    return HybridBugPredictor(code_encoder, n_dev_features, dropout=dropout)
