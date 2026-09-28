"""Pure-PyTorch replica of the scGPT pretraining transformer (Cui et al., Nat. Methods 2024).

The whole-human checkpoint (``args.json``: ``fast_transformer=true``,
``USE_GENERATIVE_TRAINING=true``) was trained with the flash-attn 1.0.1 layers in
``bowang-lab/scGPT@dev-temp`` (4068d67), ``scgpt/model/flash_layers.py``:

=======================  ===========================
upstream                 here
=======================  ===========================
``FlashscGPTMHA``        ``ScGPTMultiheadAttention``
``FlashscGPTLayer``      ``ScGPTLayer``
``FlashscGPTGenerator``  ``ScGPTGenerator``
``GeneEncoder`` etc.     same names (``model.py``)
=======================  ===========================

Module and parameter names match the checkpoint exactly, so weights load with a
strict key check (the stock ``nn.TransformerEncoderLayer`` fallback names the QKV
projection ``in_proj_weight`` and silently drops the pretrained ``Wqkv``).

flash-attn only runs fp16/bf16 on CUDA, so the fused kernel is replaced by the
computation it performs (paper Eq. 10):

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d) + A_mask) V

with ``A_mask`` from paper Eq. 11 on the generative (unknown-gene) path.
"""

from __future__ import annotations

import copy
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

PAPER_CONFIG = {"embsize": 512, "nlayers": 12, "nheads": 8, "d_hid": 512}
SPECIAL_TOKENS = ("<pad>", "<cls>", "<eoc>")


# --------------------------------------------------------------------------- #
# Input embeddings (paper Eq. 5, without condition tokens)
# --------------------------------------------------------------------------- #


class GeneEncoder(nn.Module):
    """emb_g: gene token → D-dim vector, then LayerNorm."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.enc_norm(self.embedding(x))


class ContinuousValueEncoder(nn.Module):
    """emb_x: binned expression value → D-dim vector via a 2-layer MLP."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        x = torch.clamp(x.unsqueeze(-1), max=self.max_value)
        x = self.linear2(self.activation(self.linear1(x)))
        return self.dropout(self.norm(x))


# --------------------------------------------------------------------------- #
# Attention kernels
# --------------------------------------------------------------------------- #


def flash_self_attention(
    qkv: Tensor,
    key_padding_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
) -> Tensor:
    """``flash_attn.flash_attention.FlashAttention`` (v1.0.1) without the fused kernel.

    Args:
        qkv: ``(batch, seq, 3, heads, head_dim)``.
        key_padding_mask: ``(batch, seq)`` bool, **True = valid token** (flash-attn
            convention), or ``None`` for no padding.

    Returns:
        Context ``(batch, seq, heads, head_dim)``. Rows of padded queries are zero,
        as flash-attn's ``pad_input`` leaves them.
    """
    q, k, v = qkv.unbind(dim=2)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bshd,bthd->bhst", q, k).float() * scale
    if key_padding_mask is not None:
        scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1).to(v.dtype)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    context = torch.einsum("bhst,bthd->bshd", attn, v)
    if key_padding_mask is not None:
        context = context.masked_fill(~key_padding_mask[:, :, None, None], 0.0)
    return context


def generative_attention_mask(n_gen: int, n_total: int, device: torch.device) -> Tensor:
    """Paper Eq. 11 for unknown-gene queries: True = attention allowed.

    Rows are the ``n_gen`` unknown genes; columns are ``[known genes | unknown genes]``.
    An unknown gene may attend to every known gene and to itself only.
    """
    allowed = torch.ones((n_gen, n_total), dtype=torch.bool, device=device)
    allowed[:, -n_gen:] = torch.eye(n_gen, dtype=torch.bool, device=device)
    return allowed


def generative_cross_attention(
    gen_qkv: Tensor,
    pcpt_qkv: Tensor,
    pcpt_key_padding_mask: Tensor | None = None,
    gen_key_padding_mask: Tensor | None = None,
    dropout_p: float = 0.0,
) -> Tensor:
    """Unknown-gene queries attend to ``[known K/V ; unknown K/V]`` under Eq. 11.

    Masks follow the flash-attn convention (True = valid). Returns
    ``(batch, gen_len, heads * head_dim)``.
    """
    q = gen_qkv[:, :, 0]
    k, v = torch.cat([pcpt_qkv[:, :, 1:], gen_qkv[:, :, 1:]], dim=1).unbind(dim=2)
    n_gen, n_total = q.shape[1], k.shape[1]

    scores = torch.einsum("bshd,bthd->bhst", q, k).float() / math.sqrt(q.shape[-1])
    allowed = generative_attention_mask(n_gen, n_total, q.device)
    scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    if pcpt_key_padding_mask is not None or gen_key_padding_mask is not None:
        b = q.shape[0]
        if pcpt_key_padding_mask is None:
            pcpt_key_padding_mask = torch.ones((b, pcpt_qkv.shape[1]), dtype=torch.bool, device=q.device)
        if gen_key_padding_mask is None:
            gen_key_padding_mask = torch.ones((b, n_gen), dtype=torch.bool, device=q.device)
        keep = torch.cat([pcpt_key_padding_mask, gen_key_padding_mask], dim=1)
        scores = scores.masked_fill(~keep[:, None, None, :], float("-inf"))

    attn = torch.softmax(scores, dim=-1).to(v.dtype)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    return rearrange(torch.einsum("bhst,bthd->bshd", attn, v), "b s h d -> b s (h d)")


# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #


class ScGPTMultiheadAttention(nn.Module):
    """Replica of ``FlashscGPTMHA``: packed ``Wqkv`` projection + ``out_proj``.

    Known ("perceptual", ``pcpt``) genes get full self-attention. Unknown
    ("generative", ``gen``) genes are queries against known genes + themselves.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if not (self.head_dim % 8 == 0 and self.head_dim <= 128):
            raise ValueError("flash-attn requires head_dim <= 128 and divisible by 8")
        self.attention_dropout = attention_dropout
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _qkv(self, x: Tensor) -> Tensor:
        return rearrange(self.Wqkv(x), "b s (three h d) -> b s three h d", three=3, h=self.num_heads)

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor | None = None,
        pcpt_key_padding_mask: Tensor | None = None,
        gen_key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Masks use the flash-attn convention: True = valid token."""
        dropout_p = self.attention_dropout if self.training else 0.0

        pcpt_qkv = self._qkv(pcpt_total_embs)
        pcpt_context = flash_self_attention(pcpt_qkv, pcpt_key_padding_mask, dropout_p)
        pcpt_out = self.out_proj(rearrange(pcpt_context, "b s h d -> b s (h d)"))
        if gen_total_embs is None:
            return pcpt_out, None

        # Upstream routes this path through a parameter-free MHA with identity
        # in/out projections, so the pretrained weights never apply out_proj here.
        gen_out = generative_cross_attention(
            self._qkv(gen_total_embs),
            pcpt_qkv,
            pcpt_key_padding_mask,
            gen_key_padding_mask,
            dropout_p,
        )
        return pcpt_out, gen_out

    def attention_logits(self, x: Tensor) -> Tensor:
        """Raw per-head scores ``Q K^T`` of shape ``(batch, heads, seq, seq)``.

        Unscaled, as in scGPT's attention-GRN workflow; the ``1/sqrt(d)`` factor
        is a positive constant and does not change the downstream rank normalization.
        """
        q, k, _ = self._qkv(x).unbind(dim=2)
        return torch.einsum("bshd,bthd->bhst", q, k)


class ScGPTLayer(nn.Module):
    """Replica of ``FlashscGPTLayer``: attention + ReLU FFN, post-norm by default."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        norm_scheme: str = "post",
    ):
        super().__init__()
        if norm_scheme not in ("pre", "post"):
            raise ValueError("norm_scheme must be either pre or post")
        self.self_attn = ScGPTMultiheadAttention(d_model, nhead, attention_dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_scheme = norm_scheme

    @staticmethod
    def _reverse_key_padding_mask(src_key_padding_mask: Tensor | None) -> Tensor | None:
        """PyTorch convention (True = pad) → flash-attn convention (True = valid)."""
        if src_key_padding_mask is None or not src_key_padding_mask.any().item():
            return None
        return ~src_key_padding_mask.bool()

    def _ffn(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor | None = None,
        pcpt_key_padding_mask: Tensor | None = None,
        gen_key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Masks use the PyTorch convention: True = padding."""
        pcpt_mask = self._reverse_key_padding_mask(pcpt_key_padding_mask)
        gen_mask = self._reverse_key_padding_mask(gen_key_padding_mask)

        if self.norm_scheme == "pre":
            pcpt_total_embs = self.norm1(pcpt_total_embs)
            if gen_total_embs is not None:
                gen_total_embs = self.norm1(gen_total_embs)
            pcpt2, gen2 = self.self_attn(pcpt_total_embs, gen_total_embs, pcpt_mask, gen_mask)
            pcpt_total_embs = self.norm2(pcpt_total_embs + self.dropout1(pcpt2))
            pcpt_total_embs = pcpt_total_embs + self.dropout2(self._ffn(pcpt_total_embs))
            if gen_total_embs is not None:
                gen_total_embs = self.norm2(gen_total_embs + self.dropout1(gen2))
                gen_total_embs = gen_total_embs + self.dropout2(self._ffn(gen_total_embs))
        else:
            pcpt2, gen2 = self.self_attn(pcpt_total_embs, gen_total_embs, pcpt_mask, gen_mask)
            pcpt_total_embs = self.norm1(pcpt_total_embs + self.dropout1(pcpt2))
            pcpt_total_embs = self.norm2(pcpt_total_embs + self.dropout2(self._ffn(pcpt_total_embs)))
            if gen_total_embs is not None:
                gen_total_embs = self.norm1(gen_total_embs + self.dropout1(gen2))
                gen_total_embs = self.norm2(gen_total_embs + self.dropout2(self._ffn(gen_total_embs)))

        return pcpt_total_embs, gen_total_embs


class ScGPTGenerator(nn.Module):
    """Replica of ``FlashscGPTGenerator``: ``num_layers`` independent copies of one layer."""

    def __init__(self, encoder_layer: ScGPTLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor | None = None,
        pcpt_key_padding_mask: Tensor | None = None,
        gen_key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        for layer in self.layers:
            pcpt_total_embs, gen_total_embs = layer(
                pcpt_total_embs, gen_total_embs, pcpt_key_padding_mask, gen_key_padding_mask
            )
        return pcpt_total_embs, gen_total_embs


class ScGPTPretrainedModel(nn.Module):
    """Input embeddings + transformer stack of pretrained scGPT (decoders omitted)."""

    def __init__(
        self,
        ntoken: int,
        d_model: int = 512,
        nhead: int = 8,
        d_hid: int = 512,
        nlayers: int = 12,
        dropout: float = 0.2,
        pad_token_id: int | None = None,
        pre_norm: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=pad_token_id)
        self.flag_encoder = nn.Embedding(2, d_model)
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        layer = ScGPTLayer(
            d_model, nhead, d_hid, dropout, norm_scheme="pre" if pre_norm else "post"
        )
        self.transformer_encoder = ScGPTGenerator(layer, nlayers)

    def embed(
        self,
        pcpt_genes: Tensor,
        pcpt_values: Tensor,
        gen_genes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Known genes: gene + value embedding. Unknown genes: gene + generative flag."""
        pcpt_total_embs = self.encoder(pcpt_genes) + self.value_encoder(pcpt_values)
        if gen_genes is None:
            return pcpt_total_embs, None
        gen_flags = self.flag_encoder(torch.tensor(1, device=gen_genes.device)).expand(
            gen_genes.shape[0], gen_genes.shape[1], -1
        )
        return pcpt_total_embs, self.encoder(gen_genes) + gen_flags

    def forward(
        self,
        pcpt_genes: Tensor,
        pcpt_values: Tensor,
        pcpt_key_padding_mask: Tensor | None = None,
        gen_genes: Tensor | None = None,
        gen_key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        pcpt_total_embs, gen_total_embs = self.embed(pcpt_genes, pcpt_values, gen_genes)
        return self.transformer_encoder(
            pcpt_total_embs, gen_total_embs, pcpt_key_padding_mask, gen_key_padding_mask
        )

    def attention_logits(
        self,
        genes: Tensor,
        values: Tensor,
        key_padding_mask: Tensor | None = None,
        layer_index: int = -1,
    ) -> Tensor:
        """Raw ``Q K^T`` of transformer block ``layer_index`` for known-gene inputs.

        Returns ``(batch, heads, seq, seq)``. ``key_padding_mask`` uses the PyTorch
        convention (True = padding).
        """
        layer_index = layer_index % self.nlayers
        h, _ = self.embed(genes, values)
        for layer in self.transformer_encoder.layers[:layer_index]:
            h, _ = layer(h, None, key_padding_mask)
        return self.transformer_encoder.layers[layer_index].self_attn.attention_logits(h)


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #


@dataclass
class LoadReport:
    """Which checkpoint tensors were loaded into the replica and which were skipped."""

    n_loaded: int
    ignored_checkpoint_keys: list[str] = field(default_factory=list)
    config_deviations_from_paper: dict[str, tuple[int, int]] = field(default_factory=dict)

    def summary(self) -> str:
        prefixes = sorted({k.split(".")[0] for k in self.ignored_checkpoint_keys})
        dev = self.config_deviations_from_paper or "none"
        return (
            f"loaded {self.n_loaded} tensors (strict); "
            f"skipped decoder heads {prefixes}; config deviations from paper: {dev}"
        )


def load_vocab(model_dir: Path | str) -> dict[str, int]:
    """``vocab.json`` token→id, appending missing special tokens like ``GeneVocab``."""
    with open(Path(model_dir) / "vocab.json") as f:
        vocab = json.load(f)
    for tok in SPECIAL_TOKENS:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def load_pretrained_scgpt(
    model_dir: Path | str,
    device: str | torch.device | None = None,
) -> tuple[ScGPTPretrainedModel, dict[str, int], dict, LoadReport]:
    """Build the replica from ``args.json`` and load ``best_model.pt`` strictly.

    Every replica parameter must exist in the checkpoint with the same shape;
    otherwise this raises instead of leaving randomly initialized weights behind.
    Checkpoint tensors the replica does not need (expression / MVC decoders) are
    listed in the returned :class:`LoadReport`.
    """
    model_dir = Path(model_dir)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = load_vocab(model_dir)
    with open(model_dir / "args.json") as f:
        cfg = json.load(f)

    deviations = {k: (cfg.get(k), v) for k, v in PAPER_CONFIG.items() if cfg.get(k) != v}
    if deviations:
        warnings.warn(f"checkpoint config differs from the paper's pretrained model: {deviations}")

    model = ScGPTPretrainedModel(
        ntoken=len(vocab),
        d_model=cfg["embsize"],
        nhead=cfg["nheads"],
        d_hid=cfg["d_hid"],
        nlayers=cfg["nlayers"],
        dropout=cfg.get("dropout", 0.2),
        pad_token_id=vocab["<pad>"],
        pre_norm=bool(cfg.get("pre_norm", False)),
    )

    ckpt = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
    own = model.state_dict()
    missing = [k for k in own if k not in ckpt]
    mismatched = {
        k: (tuple(ckpt[k].shape), tuple(v.shape))
        for k, v in own.items()
        if k in ckpt and ckpt[k].shape != v.shape
    }
    if missing or mismatched:
        raise RuntimeError(
            "checkpoint does not match the scGPT pretraining architecture: "
            f"missing={missing[:8]}{'...' if len(missing) > 8 else ''} "
            f"shape_mismatch={dict(list(mismatched.items())[:8])}"
        )
    model.load_state_dict({k: ckpt[k] for k in own}, strict=True)
    model.to(device_t).eval()

    report = LoadReport(
        n_loaded=len(own),
        ignored_checkpoint_keys=sorted(k for k in ckpt if k not in own),
        config_deviations_from_paper=deviations,
    )
    return model, vocab, cfg, report


# --------------------------------------------------------------------------- #
# Paper preprocessing + attention-map recipe
# --------------------------------------------------------------------------- #


def bin_expression(row: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-cell value binning used in scGPT pretraining (paper Eq. 2).

    Non-zero values are digitized against ``n_bins - 1`` quantile edges computed
    within this cell; zeros stay 0. Deterministic, like the pretraining collator.
    """
    row = np.asarray(row, dtype=np.float64)
    if row.max() == 0:
        return np.zeros(row.shape, dtype=np.int64)
    if row.min() <= 0:
        nz = row.nonzero()
        bins = np.quantile(row[nz], np.linspace(0, 1, n_bins - 1))
        binned = np.zeros(row.shape, dtype=np.int64)
        binned[nz] = np.digitize(row[nz], bins)
        return binned
    bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
    return np.digitize(row, bins).astype(np.int64)


def rank_normalize_attention(logits: Tensor) -> Tensor:
    """Rank-normalize by row, then by column, then average heads (paper Methods, Fig. 6).

    ``(batch, heads, M, M)`` → ``(batch, M, M)`` with entries in ``[0, 1)``.
    """
    b, h, m, _ = logits.shape
    x = logits.reshape(-1, m)
    x = torch.argsort(torch.argsort(x, dim=1), dim=1).reshape(b, h, m, m).float() / m

    x = x.permute(0, 1, 3, 2).reshape(-1, m)
    x = torch.argsort(torch.argsort(x, dim=1), dim=1).reshape(b, h, m, m).float() / m
    return x.permute(0, 1, 3, 2).mean(dim=1)
