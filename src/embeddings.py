"""
embeddings.py
=============
Two interchangeable ways to turn a code diff into a fixed-size vector for
the Code Branch of the hybrid model.

1. CodeBertEmbedder
   Uses a pretrained code language model (default: microsoft/codebert-base)
   through Hugging Face Transformers to produce a 768-d mean-pooled
   embedding. This is the representation the base paper points to
   ("a transformer-based code representation in the spirit of
   CodeBERT/UniXCoder"). It requires downloading ~500MB of weights the
   first time it runs, so it needs a working internet connection to
   huggingface.co. It is used in *feature-extraction* mode (frozen encoder)
   so it stays usable on a CPU-only laptop -- no fine-tuning of CodeBERT's
   125M parameters is required or attempted.

2. LightweightCodeEncoder
   A small trainable Embedding + BiGRU encoder (a few hundred thousand
   parameters) trained from scratch on the diff tokens, in the same spirit
   as the original DeepJIT / Kim et al. token-based encoders discussed in
   the review. It needs no internet access and no GPU, so it is what this
   project actually trains and evaluates on this machine.
   Set --embedding codebert in train.py to switch, once you have a
   laptop/environment with internet access to huggingface.co.
"""

import os

import numpy as np
import torch
import torch.nn as nn

CODEBERT_MODEL_NAME = "microsoft/codebert-base"
CODEBERT_DIM = 768


class CodeBertEmbedder:
    """Frozen CodeBERT feature extractor. Only imports `transformers`/loads
    weights when actually instantiated, so the rest of the project works
    fine even where internet access (and therefore CodeBERT) is unavailable.
    """

    def __init__(self, model_name: str = CODEBERT_MODEL_NAME, device: str = "cpu", max_length: int = 256):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is not installed. Run: pip install transformers"
            ) from e
        self.device = device
        self.max_length = max_length
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
        except Exception as e:
            raise RuntimeError(
                f"Could not load '{model_name}' from Hugging Face. This "
                "requires an internet connection to huggingface.co (works "
                "on a normal laptop, but is blocked in this sandbox). Use "
                "the lightweight embedding instead (--embedding lightweight)."
            ) from e
        self.model.eval()

    @torch.no_grad()
    def embed(self, texts):
        """texts: list[str] -> np.ndarray [N, 768] (mean-pooled last hidden state)."""
        all_vecs = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-6)
            mean_pooled = summed / counts
            all_vecs.append(mean_pooled.cpu().numpy())
        return np.concatenate(all_vecs, axis=0)


class LightweightCodeEncoder(nn.Module):
    """Embedding -> BiGRU -> mean+max pooling -> linear projection.
    Trained end-to-end together with the rest of the hybrid model
    (see src/model.py). No pretrained weights, no internet required.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 64,
                 out_dim: int = 128, pad_idx: int = 0, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim * 2 * 2, out_dim)  # mean+max pooling, both directions
        self.out_dim = out_dim

    def forward(self, token_ids, lengths=None):
        # token_ids: [B, T]
        emb = self.embedding(token_ids)              # [B, T, E]
        outputs, _ = self.gru(emb)                    # [B, T, 2H]
        mask = (token_ids != 0).unsqueeze(-1).float()  # [B, T, 1]
        masked = outputs * mask
        summed = masked.sum(1)
        counts = mask.sum(1).clamp(min=1e-6)
        mean_pool = summed / counts
        max_pool = masked.masked_fill(mask == 0, -1e9).max(1).values
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        pooled = self.dropout(pooled)
        return torch.relu(self.proj(pooled))          # [B, out_dim]


def pad_batch(id_lists, pad_value: int = 0):
    max_len = max((len(x) for x in id_lists), default=1)
    max_len = max(max_len, 1)
    out = torch.full((len(id_lists), max_len), pad_value, dtype=torch.long)
    for i, ids in enumerate(id_lists):
        if len(ids) > 0:
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out
