"""Extract global (S_emb) and contextual (S_att) priors from frozen scGPT.

Embedding path (DeltaNMF default):
  ``model.encoder(gene_ids)`` → corrcoef → ReLU.

Attention path (scGPT paper, Methods "attention-based target gene selection"):
  per-cell binned input with ``<cls>`` prepended → pretrained transformer
  (:mod:`.scgpt_attention` replica of the flash-attn pretraining layers) →
  raw ``Q K^T`` of all 8 heads in the chosen layer → rank-normalize by row, then
  by column → mean over heads → mean over cells → drop ``<cls>``.
  Then (this project, not the paper) symmetrize + kNN sparsify.

Protocol: do **not** fine-tune scGPT for the first experiment.

``layer_index`` is a scientific design choice (default: last layer, as in the
paper). Treat it as an ablation knob.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .gene_graph import affinity_from_attention
from .prior import GenePrior, ScgptPriors
from .deltanmf_bridge import build_S_E_scgpt
from .scgpt_attention import (
    LoadReport,
    ScGPTPretrainedModel,
    bin_expression,
    load_pretrained_scgpt,
    rank_normalize_attention,
)

PRETRAIN_MAX_SEQ_LEN = 1200


def load_scgpt_model(
    model_dir: Path | str,
    device: str | None = None,
) -> tuple[ScGPTPretrainedModel, dict[str, int], torch.device, dict, LoadReport]:
    """Load pretrained scGPT into the paper-exact replica (strict weight loading)."""
    model, vocab, cfg, report = load_pretrained_scgpt(model_dir, device=device)
    return model, vocab, next(model.parameters()).device, cfg, report


def paper_attention_map(
    model: ScGPTPretrainedModel,
    gene_ids: torch.Tensor,
    X_cells_by_genes: np.ndarray,
    *,
    cls_id: int,
    cls_value: float,
    n_bins: int,
    layer_index: int,
    batch_size: int = 4,
) -> np.ndarray:
    """Cell-averaged scGPT attention map over ``gene_ids`` (``<cls>`` removed).

    Each cell is binned independently, prefixed with ``<cls>``, and fed with all
    genes (zeros included, bin 0) so every cell yields the same ``G × G`` map.
    Rank normalization runs over the full ``(G + 1)`` sequence, including ``<cls>``.
    """
    device = gene_ids.device
    n_cells, n_genes = X_cells_by_genes.shape
    seq = torch.cat([torch.tensor([cls_id], device=device), gene_ids])
    use_amp = device.type == "cuda"

    running = np.zeros((n_genes, n_genes), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            chunk = X_cells_by_genes[start : start + batch_size]
            b = chunk.shape[0]
            binned = np.stack([bin_expression(row, n_bins) for row in chunk])
            values = np.concatenate([np.full((b, 1), cls_value), binned], axis=1)
            values_t = torch.as_tensor(values, dtype=torch.float32, device=device)
            ids = seq.unsqueeze(0).expand(b, -1)
            padding = torch.zeros_like(ids, dtype=torch.bool)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model.attention_logits(ids, values_t, padding, layer_index)
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
    embedding_knn: int | None = None,
) -> ScgptPriors:
    """Extract DeltaNMF ``S_E`` (embedding) + novel contextual ``S_att``.

    - **Embedding / ``S_E``**: DeltaNMF recipe only
      (``model.encoder`` → corrcoef → ReLU). See
      ``third_party/deltanmf/resources/scgpt/create_transformer_similarity_matrix_scgpt.py``.
    - **Attention / ``S_att``** (this project's addition): the scGPT paper's
      attention-map recipe (:func:`paper_attention_map`) → symmetrize → kNN.

    ``X_cells_by_genes`` should be non-negative expression (raw counts or
    normalized/log1p); it is binned per cell exactly as in pretraining.
    ``embedding_knn`` is ignored (DeltaNMF keeps dense ``S_E``).
    """
    del embedding_knn  # DeltaNMF S_E is dense; do not alter their recipe.
    X = np.asarray(X_cells_by_genes, dtype=np.float32)
    n_cells, n_genes = X.shape
    if len(gene_names) != n_genes:
        raise ValueError("gene_names must match X columns")
    if (X < 0).any():
        raise ValueError("X must be non-negative expression (scGPT bins per-cell values)")

    if dry_run:
        return _dry_run_scgpt_priors(
            X,
            gene_names,
            device=device,
            attention_knn=attention_knn,
        )

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
    n_genes = len(gene_names)
    if n_genes > PRETRAIN_MAX_SEQ_LEN:
        raise ValueError(
            f"{n_genes} genes exceeds the paper's 1,200-gene input (select HVGs first)"
        )
    gene_ids_t = torch.tensor([vocab[g] for g in gene_names], dtype=torch.long, device=device_t)

    nlayers = int(cfg["nlayers"])
    nheads = int(cfg["nheads"])
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
        gene_ids_t,
        X,
        cls_id=vocab["<cls>"],
        cls_value=float(cfg.get("pad_value", -2)),
        n_bins=int(cfg.get("n_bins", 51)),
        layer_index=layer_index,
        batch_size=batch_size,
    )
    S_attn = affinity_from_attention(attention_raw, knn=attention_knn)

    shared_meta = {
        "layer_index": layer_index,
        "n_cells_aggregated": int(n_cells),
        "nheads": nheads,
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
                "scGPT paper recipe: binned input + <cls> -> replica flash layers -> "
                "raw QK^T (all heads) -> rank-norm row, col -> mean heads -> mean cells"
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


# Back-compat alias used by older call sites / smoke tests.
def extract_scgpt_attention(
    X_cells_by_genes: np.ndarray,
    gene_names: list[str],
    model_dir: Path | str,
    **kwargs,
) -> GenePrior:
    """Return only the attention prior from ``extract_scgpt_priors``."""
    return extract_scgpt_priors(X_cells_by_genes, gene_names, model_dir, **kwargs).attention


def extract_scgpt_embeddings(
    X_cells_by_genes: np.ndarray,
    gene_names: list[str],
    model_dir: Path | str,
    **kwargs,
) -> GenePrior:
    """Return only the DeltaNMF embedding prior from ``extract_scgpt_priors``."""
    return extract_scgpt_priors(X_cells_by_genes, gene_names, model_dir, **kwargs).embedding


def _dry_run_scgpt_priors(
    X: np.ndarray,
    gene_names: list[str],
    device: str | None = None,
    attention_knn: int | None = 8,
) -> ScgptPriors:
    """Plumbing test only — same attention path as the real model, random weights."""
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n_cells, n_genes = X.shape
    pad_id, cls_id = n_genes, n_genes + 1

    torch.manual_seed(0)
    model = ScGPTPretrainedModel(
        ntoken=n_genes + 2, d_model=64, nhead=4, d_hid=64, nlayers=2, pad_token_id=pad_id
    ).to(device_t).eval()
    gene_ids = torch.arange(n_genes, device=device_t)

    attention_raw = paper_attention_map(
        model,
        gene_ids,
        X[: min(16, n_cells)],
        cls_id=cls_id,
        cls_value=-2.0,
        n_bins=51,
        layer_index=1,
    )
    with torch.no_grad():
        emb = model.encoder(gene_ids).cpu().numpy()

    S_attn = affinity_from_attention(attention_raw, knn=attention_knn)
    S_emb = np.maximum(0.0, np.corrcoef(emb))
    np.fill_diagonal(S_emb, 0.0)

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
        meta={**meta, "extraction": "scGPT paper recipe (random-weight replica)"},
    )
    embedding_prior = GenePrior(
        kind="embedding",
        gene_names=list(gene_names),
        similarity=S_emb,
        gene_embeddings=emb,
        meta={**meta, "extraction": "DeltaNMF-style corrcoef->ReLU (random weights)"},
    )
    return ScgptPriors(
        gene_names=list(gene_names),
        attention=attention_prior,
        embedding=embedding_prior,
        gene_embeddings=emb,
        meta=meta,
    )
