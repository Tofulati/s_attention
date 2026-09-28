"""Pretrained scGPT in scGPT's own ``TransformerModel`` + the Attention-GRN recipe.

Everything architectural comes from the vendored package (``third_party/scGPT``):

- model: ``scgpt.model.TransformerModel``, built with the arguments used by
  DeltaNMF's ``create_transformer_similarity_matrix_scgpt.py`` and scGPT's
  ``Tutorial_Attention_GRN.ipynb``
- vocab: ``scgpt.tokenizer.gene_tokenizer.GeneVocab`` + appended special tokens
- attention logits / rank normalization: the tutorial's extraction cell

The checkpoint was trained with flash-attn layers, whose packed QKV projection is
``self_attn.Wqkv``. Without flash-attn (CPU / macOS) ``TransformerModel`` builds
``nn.TransformerEncoderLayer``, which holds the same packed ``[q; k; v]`` projection
as ``self_attn.in_proj_weight``. The upstream loaders keep only name-matched keys,
so they would silently leave every attention block randomly initialized. Renaming
``Wqkv`` → ``in_proj`` is the only project-specific step.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
SCGPT_ROOT = ROOT / "third_party" / "scGPT"
SPECIAL_TOKENS = ("<pad>", "<cls>", "<eoc>")
PAD_TOKEN = "<pad>"
USED_PREFIXES = ("encoder.", "value_encoder.", "transformer_encoder.")


def ensure_scgpt_on_path() -> None:
    if str(SCGPT_ROOT) not in sys.path:
        sys.path.insert(0, str(SCGPT_ROOT))


def _flash_to_torch_key(key: str) -> str:
    return re.sub(r"self_attn\.Wqkv\.(weight|bias)$", r"self_attn.in_proj_\1", key)


@dataclass
class LoadReport:
    """Checkpoint tensors loaded, skipped, and model tensors left at initialization."""

    n_loaded: int
    skipped_checkpoint_keys: list[str] = field(default_factory=list)
    uninitialized_model_keys: list[str] = field(default_factory=list)

    def summary(self) -> str:
        def prefixes(keys: list[str]) -> list[str]:
            return sorted({k.split(".")[0] for k in keys})

        return (
            f"loaded {self.n_loaded} tensors (all of {list(USED_PREFIXES)}); "
            f"checkpoint-only {prefixes(self.skipped_checkpoint_keys)}; "
            f"unused & uninitialized {prefixes(self.uninitialized_model_keys)}"
        )


def load_vocab(model_dir: Path | str):
    """``GeneVocab.from_file`` + missing special tokens, as in the DeltaNMF script."""
    ensure_scgpt_on_path()
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    vocab = GeneVocab.from_file(Path(model_dir) / "vocab.json")
    for tok in SPECIAL_TOKENS:
        if tok not in vocab:
            vocab.append_token(tok)
    return vocab


def build_scgpt_model(vocab, cfg: dict):
    """``TransformerModel`` with the DeltaNMF-script / tutorial constructor arguments."""
    ensure_scgpt_on_path()
    from scgpt.model import TransformerModel

    if cfg.get("pre_norm", False):
        raise ValueError("non-flash TransformerModel ignores pre_norm; checkpoint needs post-norm")
    return TransformerModel(
        len(vocab),
        cfg["embsize"],
        cfg["nheads"],
        cfg["d_hid"],
        cfg["nlayers"],
        vocab=vocab,
        pad_value=cfg.get("pad_value", -2),
        n_input_bins=cfg.get("n_bins", 51),
        use_fast_transformer=False,
    )


def load_pretrained_scgpt(model_dir: Path | str, device: str | torch.device | None = None):
    """Return ``(model, vocab, cfg, report)`` with every used weight from the checkpoint.

    Raises if any ``encoder`` / ``value_encoder`` / ``transformer_encoder`` tensor is
    missing from the checkpoint or has a different shape.
    """
    model_dir = Path(model_dir)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = load_vocab(model_dir)
    with open(model_dir / "args.json") as f:
        cfg = json.load(f)
    model = build_scgpt_model(vocab, cfg)

    raw = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
    ckpt = {_flash_to_torch_key(k): v for k, v in raw.items()}
    own = model.state_dict()
    matched = {k: ckpt[k] for k, v in own.items() if k in ckpt and ckpt[k].shape == v.shape}
    required = [k for k in own if k.startswith(USED_PREFIXES)]
    missing = [k for k in required if k not in matched]
    if missing:
        raise RuntimeError(
            f"checkpoint lacks {len(missing)} scGPT weights, e.g. {missing[:5]}"
        )
    model.load_state_dict(matched, strict=False)
    model.to(device_t).eval()

    report = LoadReport(
        n_loaded=len(matched),
        skipped_checkpoint_keys=sorted(k for k in ckpt if k not in matched),
        uninitialized_model_keys=sorted(k for k in own if k not in matched),
    )
    return model, vocab, cfg, report


def attention_logits(
    model,
    gene_ids: Tensor,
    values: Tensor,
    src_key_padding_mask: Tensor,
    layer_index: int,
) -> Tensor:
    """Raw per-head ``Q K^T`` of block ``layer_index``: ``(batch, heads, seq, seq)``.

    Tutorial_Attention_GRN extraction cell, with ``Wqkv`` read as ``in_proj``.
    """
    total_embs = model.encoder(gene_ids) + model.value_encoder(values)
    for layer in model.transformer_encoder.layers[:layer_index]:
        total_embs = layer(total_embs, src_key_padding_mask=src_key_padding_mask)
    attn = model.transformer_encoder.layers[layer_index].self_attn
    qkv = F.linear(total_embs, attn.in_proj_weight, attn.in_proj_bias)
    qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=attn.num_heads)
    q, k = qkv[:, :, 0], qkv[:, :, 1]
    return q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)


def rank_normalize_attention(attn_scores: Tensor) -> Tensor:
    """Tutorial rank normalization: by row, then by column, then mean over heads.

    ``(batch, heads, M, M)`` → ``(batch, M, M)`` with entries in ``[0, 1)``.
    """
    b, h, m, _ = attn_scores.shape
    x = attn_scores.reshape((-1, m))
    rank = torch.argsort(torch.argsort(x, dim=1), dim=1)
    x = rank.reshape((-1, h, m, m)) / m

    x = x.permute(0, 1, 3, 2).reshape((-1, m))
    rank = torch.argsort(torch.argsort(x, dim=1), dim=1)
    x = (rank.reshape((-1, h, m, m)) / m).permute(0, 1, 3, 2)
    return x.mean(1)
