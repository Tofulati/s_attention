"""Extract global (S_emb) and contextual (S_att) priors from frozen scGPT.

Embedding path (DeltaNMF default):
  ``model.encoder(gene_ids)`` → corrcoef → ReLU.

Attention path (scGPT ``Tutorial_Attention_GRN.ipynb``):
  per-cell ``scgpt.preprocess.binning`` → ``tokenize_and_pad_batch`` (``<cls>``
  prepended, zero genes kept) → pretrained ``TransformerModel`` → raw ``Q K^T`` of
  all heads in the chosen block → rank-normalize by row, then by column → mean over
  heads → mean over cells → drop ``<cls>``.
  Then (this project, not the tutorial) symmetrize + kNN sparsify.

Protocol: do **not** fine-tune scGPT for the first experiment.

``layer_index`` is a scientific design choice (default: last layer, as in the
tutorial). Treat it as an ablation knob.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .deltanmf_bridge import build_S_E_scgpt
from .gene_graph import affinity_from_attention
from .prior import GenePrior, ScgptPriors
from .scgpt_attention import (
    PAD_TOKEN,
    attention_logits,
    ensure_scgpt_on_path,
    load_pretrained_scgpt,
    rank_normalize_attention,
)

SEED = 42


def load_scgpt_model(model_dir: Path | str, device: str | None = None):
    """Return ``(model, vocab, device, cfg, report)`` for the pretrained checkpoint."""
    model, vocab, cfg, report = load_pretrained_scgpt(model_dir, device=device)
    return model, vocab, next(model.parameters()).device, cfg, report


def tokenize_cells(
    X_cells_by_genes: np.ndarray,
    gene_names: list[str],
    vocab,
    *,
    n_bins: int,
    pad_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tutorial preprocessing: per-cell binning, then ``tokenize_and_pad_batch``.

    ``binning`` ranks values within each cell, so it gives the same bins for raw
    counts and for the tutorial's ``normalize_total`` + ``log1p`` input.
    Returns ``(gene_ids, values, src_key_padding_mask)``, each ``(cells, G + 1)``.
    """
    ensure_scgpt_on_path()
    from scgpt.preprocess import binning
    from scgpt.tokenizer import tokenize_and_pad_batch

    binned = np.stack([binning(row, n_bins) for row in X_cells_by_genes])
    tokenized = tokenize_and_pad_batch(
        binned,
        np.array([vocab[g] for g in gene_names]),
        max_len=len(gene_names) + 1,
        vocab=vocab,
        pad_token=PAD_TOKEN,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
    )
    gene_ids, values = tokenized["genes"], tokenized["values"]
    return gene_ids, values, gene_ids.eq(vocab[PAD_TOKEN])


def paper_attention_map(
    model,
    vocab,
    gene_names: list[str],
    X_cells_by_genes: np.ndarray,
    *,
    layer_index: int,
    n_bins: int = 51,
    pad_value: float = -2,
    batch_size: int = 4,
) -> np.ndarray:
    """Cell-averaged, rank-normalized scGPT attention over ``gene_names`` (``<cls>`` removed).

    Rank normalization runs over the full ``G + 1`` sequence, including ``<cls>``.
    """
    device = next(model.parameters()).device
    gene_ids, values, padding = tokenize_cells(
        X_cells_by_genes, gene_names, vocab, n_bins=n_bins, pad_value=pad_value
    )
    n_cells, n_genes = X_cells_by_genes.shape
    use_amp = device.type == "cuda"

    running = np.zeros((n_genes, n_genes), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            sl = slice(start, start + batch_size)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = attention_logits(
                    model,
                    gene_ids[sl].to(device),
                    values[sl].to(device),
                    padding[sl].to(device),
                    layer_index,
                )
            maps = rank_normalize_attention(logits.float())
            running += maps[:, 1:, 1:].sum(dim=0).cpu().numpy()
    return running / max(n_cells, 1)


def extract_scgpt_priors(
    X_cells_by_genes: np.ndarray,
    gene_names: list[str],
    model_dir: Path | str,
    *,
    layer_index: int | None = None,
    max_cells: int | None = 64,
    batch_size: int = 4,
    device: str | None = None,
    dry_run: bool = False,
    attention_knn: int | None = 8,
) -> ScgptPriors:
    """Extract DeltaNMF ``S_E`` (embedding) + novel contextual ``S_att``.

    - **Embedding / ``S_E``**: DeltaNMF recipe only
      (``model.encoder`` → corrcoef → ReLU). See
      ``third_party/deltanmf/resources/scgpt/create_transformer_similarity_matrix_scgpt.py``.
    - **Attention / ``S_att``** (this project's addition): the tutorial's
      attention-map recipe (:func:`paper_attention_map`) → symmetrize → kNN.

    ``X_cells_by_genes`` should be non-negative expression (raw counts or
    normalized/log1p).
    """
    ensure_scgpt_on_path()
    from scgpt.utils import set_seed

    set_seed(SEED)
    X = np.asarray(X_cells_by_genes, dtype=np.float32)
    n_cells, n_genes = X.shape
    if len(gene_names) != n_genes:
        raise ValueError("gene_names must match X columns")
    if (X < 0).any():
        raise ValueError("X must be non-negative expression (scGPT bins per-cell values)")

    if dry_run:
        return _dry_run_scgpt_priors(X, gene_names, device=device, attention_knn=attention_knn)

    model_dir = Path(model_dir)
    if not (model_dir / "best_model.pt").exists():
        raise FileNotFoundError(
            f"scGPT checkpoint not found at {model_dir}/best_model.pt. "
            "Download a pretrained model into checkpoints/scgpt/ "
            "(see checkpoints/scgpt/README.md)."
        )

    model, vocab, device_t, cfg, load_report = load_scgpt_model(model_dir, device=device)

    keep = [i for i, g in enumerate(gene_names) if g in vocab]
    if len(keep) < 2:
        raise ValueError(
            "Fewer than 2 genes matched scGPT vocab. Use gene symbols present in vocab.json."
        )
    X = X[:, keep]
    gene_names = [gene_names[i] for i in keep]
    max_genes = int(cfg.get("max_seq_len", 1200))
    if len(gene_names) > max_genes:
        raise ValueError(f"{len(gene_names)} genes exceeds scGPT's {max_genes}-gene input")

    nlayers = int(cfg["nlayers"])
    if layer_index is None:
        layer_index = nlayers - 1
    if not (0 <= layer_index < nlayers):
        raise ValueError(f"layer_index must be in [0, {nlayers})")

    if max_cells is not None:
        X = X[:max_cells]
        n_cells = X.shape[0]

    S_emb, emb = build_S_E_scgpt(model, vocab, gene_names, device=device_t)

    attention_raw = paper_attention_map(
        model,
        vocab,
        gene_names,
        X,
        layer_index=layer_index,
        n_bins=int(cfg.get("n_bins", 51)),
        pad_value=float(cfg.get("pad_value", -2)),
        batch_size=batch_size,
    )
    S_attn = affinity_from_attention(attention_raw, knn=attention_knn)

    shared_meta = {
        "layer_index": layer_index,
        "n_cells_aggregated": int(n_cells),
        "nheads": int(cfg["nheads"]),
        "pretrained": True,
        "model_dir": str(model_dir),
        "weights": load_report.summary(),
    }
    attention_prior = GenePrior(
        kind="attention",
        gene_names=list(gene_names),
        similarity=S_attn,
        gene_embeddings=emb,
        meta={
            **shared_meta,
            "raw_attention": attention_raw,
            "novel": True,
            "extraction": (
                "scGPT Tutorial_Attention_GRN: binning + tokenize_and_pad_batch (<cls>) -> "
                "TransformerModel -> raw QK^T (all heads) -> rank-norm row, col -> "
                "mean heads -> mean cells"
            ),
            "affinity": f"symmetrize + knn={attention_knn}",
        },
    )
    embedding_prior = GenePrior(
        kind="embedding",
        gene_names=list(gene_names),
        similarity=S_emb,
        gene_embeddings=emb,
        meta={
            **shared_meta,
            "novel": False,
            "extraction": "DeltaNMF create_transformer_similarity_matrix_scgpt.py",
            "affinity": "corrcoef -> ReLU, dense (DeltaNMF S_E)",
        },
    )
    return ScgptPriors(
        gene_names=list(gene_names),
        attention=attention_prior,
        embedding=embedding_prior,
        gene_embeddings=emb,
        meta=shared_meta,
    )


def _dry_run_scgpt_priors(
    X: np.ndarray,
    gene_names: list[str],
    device: str | None = None,
    attention_knn: int | None = 8,
) -> ScgptPriors:
    """Plumbing test only — same code path as the real model, small random ``TransformerModel``."""
    from .scgpt_attention import build_scgpt_model

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = {g: i for i, g in enumerate(gene_names)}
    vocab.update({"<pad>": len(vocab), "<cls>": len(vocab) + 1, "<eoc>": len(vocab) + 2})
    cfg = {"embsize": 64, "nheads": 4, "d_hid": 64, "nlayers": 2}
    model = build_scgpt_model(vocab, cfg).to(device_t).eval()

    attention_raw = paper_attention_map(
        model, vocab, gene_names, X[: min(16, X.shape[0])], layer_index=1
    )
    S_emb, emb = build_S_E_scgpt(model, vocab, gene_names, device=device_t)
    S_attn = affinity_from_attention(attention_raw, knn=attention_knn)

    meta = {
        "pretrained": False,
        "dry_run": True,
        "warning": "Not a scientific prior. Use a real scGPT checkpoint.",
    }
    attention_prior = GenePrior(
        kind="attention",
        gene_names=list(gene_names),
        similarity=S_attn,
        gene_embeddings=emb,
        meta={**meta, "extraction": "tutorial recipe (random-weight TransformerModel)"},
    )
    embedding_prior = GenePrior(
        kind="embedding",
        gene_names=list(gene_names),
        similarity=S_emb,
        gene_embeddings=emb,
        meta={**meta, "extraction": "DeltaNMF S_E recipe (random weights)"},
    )
    return ScgptPriors(
        gene_names=list(gene_names),
        attention=attention_prior,
        embedding=embedding_prior,
        gene_embeddings=emb,
        meta=meta,
    )
