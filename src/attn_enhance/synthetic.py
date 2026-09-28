"""Synthetic cell x gene expression (stand-in until bone-marrow data arrives)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Immune / myeloid / interferon markers commonly present in scGPT_human vocab.
# Used so a *pretrained* checkpoint can tokenize the toy matrix.
PREFERRED_MARKER_GENES: tuple[str, ...] = (
    "CD3D",
    "CD3E",
    "CD8A",
    "CD4",
    "MS4A1",
    "CD79A",
    "NKG7",
    "GNLY",
    "LYZ",
    "S100A8",
    "S100A9",
    "CST3",
    "FCGR3A",
    "MS4A7",
    "FCER1A",
    "CSF3R",
    "PPBP",
    "PF4",
    "IL7R",
    "CCR7",
    "GZMB",
    "PRF1",
    "CCL5",
    "CXCL10",
    "IFI6",
    "ISG15",
    "MX1",
    "OAS1",
    "STAT1",
    "IRF7",
    "ACTB",
    "GAPDH",
    "B2M",
    "HLA-A",
    "HLA-B",
    "HLA-DRA",
    "CD14",
    "FCN1",
    "VCAN",
    "APOE",
    "TREM2",
    "C1QA",
    "C1QB",
    "NEAT1",
    "MALAT1",
    "MT-CO1",
    "MT-CO2",
    "LDHB",
)


@dataclass
class SyntheticAtlas:
    """Toy atlas with planted gene programs.

    Attributes
    ----------
    X :
        Non-negative expression matrix of shape (n_cells, n_genes).
    gene_names :
        Length-n_genes labels. Use real symbols (e.g. ``CD3D``) when feeding a
        pretrained scGPT vocab; ``G000``-style placeholders only work with
        ``dry_run=True``.
    program_W :
        Ground-truth gene loadings, shape (n_genes, n_programs).
    program_H :
        Ground-truth cell usages, shape (n_cells, n_programs).
    program_gene_sets :
        For each program, the gene indices that load on it.
    """

    X: np.ndarray
    gene_names: list[str]
    program_W: np.ndarray
    program_H: np.ndarray
    program_gene_sets: list[np.ndarray]


def genes_from_scgpt_vocab(
    model_dir: Path | str,
    n_genes: int = 48,
    preferred: tuple[str, ...] | list[str] | None = None,
    seed: int = 0,
) -> list[str]:
    """Pick ``n_genes`` symbols that exist in ``model_dir/vocab.json``.

    Prefers biologically familiar markers, then fills the remainder from the
    vocab (skipping special tokens) so pretrained scGPT can embed / attend.
    """
    model_dir = Path(model_dir)
    with open(model_dir / "vocab.json") as f:
        vocab = json.load(f)
    if not isinstance(vocab, dict):
        raise TypeError("expected scGPT vocab.json to be a token→id dict")

    preferred = list(preferred) if preferred is not None else list(PREFERRED_MARKER_GENES)
    chosen: list[str] = [g for g in preferred if g in vocab]
    if len(chosen) >= n_genes:
        return chosen[:n_genes]

    rng = np.random.default_rng(seed)
    special = {"<pad>", "<cls>", "<eoc>", "<eos>", "<unk>"}
    pool = [g for g in vocab if g not in special and g not in chosen]
    need = n_genes - len(chosen)
    if need > len(pool):
        raise ValueError(f"vocab only has {len(chosen) + len(pool)} usable genes")
    extra = rng.choice(pool, size=need, replace=False).tolist()
    return chosen + extra


def make_synthetic_expression(
    n_cells: int = 240,
    n_genes: int = 64,
    n_programs: int = 4,
    genes_per_program: int = 10,
    noise_scale: float = 0.15,
    seed: int = 0,
    gene_names: list[str] | None = None,
) -> SyntheticAtlas:
    """Plant overlapping non-negative programs, then add light noise.

    Mental model
    ------------
    Classical NMF says ``X ~= H @ W.T`` (cells x genes) with H, W >= 0.
    Here we *know* W and H, so later we can check whether attention- or
    embedding-regularized NMF recovers the planted gene blocks more cleanly
    than plain NMF.

    Parameters
    ----------
    gene_names :
        Optional length-``n_genes`` labels. Pass symbols from
        ``genes_from_scgpt_vocab(...)`` when using a real scGPT checkpoint.
        If omitted, placeholder names ``G000``, ``G001``, … are used (dry-run only).
    """
    if gene_names is not None:
        if len(gene_names) != n_genes:
            raise ValueError(
                f"gene_names length {len(gene_names)} != n_genes={n_genes}"
            )
        names = list(gene_names)
    else:
        names = [f"G{g:03d}" for g in range(n_genes)]

    rng = np.random.default_rng(seed)

    # Assign contiguous blocks of genes to each program (with mild overlap).
    program_gene_sets: list[np.ndarray] = []
    W = np.zeros((n_genes, n_programs), dtype=np.float64)
    cursor = 0
    for p in range(n_programs):
        start = cursor
        end = min(n_genes, start + genes_per_program)
        # Overlap: pull 2 genes from the previous block when possible.
        idxs = list(range(start, end))
        if p > 0 and start >= 2:
            idxs = list(range(start - 2, end))
        idxs = np.array(sorted(set(i for i in idxs if 0 <= i < n_genes)))
        program_gene_sets.append(idxs)
        W[idxs, p] = rng.uniform(0.6, 1.4, size=len(idxs))
        cursor = end

    # Soft cell-type mixture: each cell prefers 1-2 programs.
    H = np.zeros((n_cells, n_programs), dtype=np.float64)
    for i in range(n_cells):
        primary = i % n_programs
        H[i, primary] = rng.uniform(0.8, 1.5)
        if rng.random() < 0.35:
            secondary = (primary + 1) % n_programs
            H[i, secondary] = rng.uniform(0.2, 0.6)

    X = H @ W.T
    X = X + noise_scale * rng.random(X.shape)
    X = np.clip(X, 0.0, None)

    return SyntheticAtlas(
        X=X,
        gene_names=names,
        program_W=W,
        program_H=H,
        program_gene_sets=program_gene_sets,
    )
