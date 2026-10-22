"""Prediction-head / pooling variants for the ESM-C 6B LoRA ablation.

Five architectures, all deltaG-free.  H = model hidden size.

    mean_linear          A  masked-mean over residues -> Linear(H, 1)
    mean_mlp128          B  masked-mean -> Linear(H,128) -> ReLU -> Linear(128,1)
    residue_mlp128_mean  C  per-residue Linear(H,128) -> ReLU -> Linear(128,1),
                            then masked mean of the scalars
    attention_linear     D  attention pooling -> Linear(H, 1)
    attention_mlp128     E  attention pooling -> Linear(H,128) -> ReLU -> Linear(128,1)

B and C share the head shape and parameter count; they differ only in whether
the nonlinearity is applied before or after pooling.  The per-residue MLP in C
is shared across all residues.  A is mathematically identical to a per-residue
Linear followed by a mean, because an affine map commutes with averaging --
that is the head used by the existing Track A baseline.

No Dropout, no LayerNorm anywhere, by design.
"""
import torch
from torch import nn

ARCHITECTURES = (
    "mean_linear",
    "mean_mlp128",
    "residue_mlp128_mean",
    "attention_linear",
    "attention_mlp128",
)

_MEAN_POOL = ("mean_linear", "mean_mlp128")
_ATTN_POOL = ("attention_linear", "attention_mlp128")
_MLP_HEAD = ("mean_mlp128", "residue_mlp128_mean", "attention_mlp128")


def _check_any_valid(residue_mask):
    """Never let an all-masked sequence turn into a silent NaN or a zero vector."""
    if not torch.all(residue_mask.any(dim=1)):
        bad = (~residue_mask.any(dim=1)).nonzero().flatten().tolist()
        raise ValueError(f"sequences with zero valid residues at batch positions {bad}")


def masked_mean(hidden, residue_mask):
    _check_any_valid(residue_mask)
    m = residue_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1)


class MaskedAttentionPool(nn.Module):
    """e_i = Linear(attn_width,1,bias=False)(tanh(Linear(H,attn_width)(h_i)))

    alpha = softmax over the residue dimension with invalid positions set to
    -inf *before* the softmax, so the valid weights sum to exactly 1 and no
    probability mass is thrown away after normalisation.  z = sum_i alpha_i h_i,
    a weighted average of feature vectors -- not of per-residue predictions.
    Never materialises an L x L attention matrix.
    """

    def __init__(self, hidden_size, attn_width=64):
        super().__init__()
        self.attn_width = int(attn_width)
        self.proj = nn.Linear(hidden_size, self.attn_width)
        self.score = nn.Linear(self.attn_width, 1, bias=False)

    def weights(self, hidden, residue_mask):
        _check_any_valid(residue_mask)
        e = self.score(torch.tanh(self.proj(hidden))).squeeze(-1)   # [B, L]
        # softmax in float32 for numerical stability, then back to model dtype
        e = e.float().masked_fill(~residue_mask.bool(), float("-inf"))
        alpha = torch.softmax(e, dim=1)
        if not torch.isfinite(alpha).all():
            raise ValueError("non-finite attention weights")
        return alpha.to(hidden.dtype)

    def forward(self, hidden, residue_mask):
        alpha = self.weights(hidden, residue_mask)
        pooled = torch.bmm(alpha.unsqueeze(1), hidden).squeeze(1)   # [B, H]
        return pooled, alpha


class PoolingHead(nn.Module):
    """One module holding both the pooling rule and the scalar head."""

    def __init__(self, architecture, hidden_size, mlp_width=128, attn_width=64):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {architecture!r}; expected one of {ARCHITECTURES}")
        self.architecture = architecture
        self.hidden_size = int(hidden_size)
        self.mlp_width = int(mlp_width)
        self.attn_width = int(attn_width) if architecture in _ATTN_POOL else None

        if architecture in _MLP_HEAD:
            self.net = nn.Sequential(
                nn.Linear(self.hidden_size, self.mlp_width),
                nn.ReLU(),
                nn.Linear(self.mlp_width, 1),
            )
        else:
            self.net = nn.Linear(self.hidden_size, 1)

        self.pool = MaskedAttentionPool(self.hidden_size, self.attn_width) \
            if architecture in _ATTN_POOL else None

    def config(self):
        return {
            "architecture": self.architecture,
            "hidden_size": self.hidden_size,
            "mlp_width": self.mlp_width if self.architecture in _MLP_HEAD else None,
            "attn_width": self.attn_width,
        }

    def forward(self, hidden, residue_mask):
        """hidden [B, L, H], residue_mask [B, L] bool -> (pred [B], residue_logits or None)"""
        if self.architecture == "residue_mlp128_mean":
            per_residue = self.net(hidden).squeeze(-1)              # [B, L]
            _check_any_valid(residue_mask)
            m = residue_mask.to(hidden.dtype)
            pred = (per_residue * m).sum(dim=1) / m.sum(dim=1)
            return pred, per_residue

        if self.architecture in _MEAN_POOL:
            pooled = masked_mean(hidden, residue_mask)
        else:
            pooled, _ = self.pool(hidden, residue_mask)

        return self.net(pooled).squeeze(-1), None
