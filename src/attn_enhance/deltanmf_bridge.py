"""Thin bridge to vendored DeltaNMF — Embedding-NMF is DeltaNMF, unchanged.

Embedding prior ``S_E`` follows the official DeltaNMF scGPT resource script:

  ``third_party/deltanmf/resources/scgpt/create_transformer_similarity_matrix_scgpt.py``

NMF with that prior uses DeltaNMF's neural solver:

  ``deltanmf.models.solve_ntc_regularized``

The *only* project-specific swap is feeding an attention-derived similarity
in place of ``S_E`` (Attention-NMF / Combined-NMF). Do not reimplement DeltaNMF.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DELTANMF_ROOT = ROOT / "third_party" / "deltanmf"


def _ensure_deltanmf_on_path() -> None:
    if str(DELTANMF_ROOT) not in sys.path:
        sys.path.insert(0, str(DELTANMF_ROOT))


def build_S_E_scgpt(
    model,
    vocab,
    gene_names: list[str],
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """DeltaNMF scGPT embedding similarity (exact recipe from their resource script).

    ``gene_embeddings = model.encoder(gene_ids)``
    ``S_E = max(0, corrcoef(embeddings)); diag = 0``
    """
    device_t = device or next(model.parameters()).device
    ids = torch.tensor([vocab[g] for g in gene_names], dtype=torch.long, device=device_t)
    with torch.no_grad():
        emb = model.encoder(ids).detach().cpu().numpy()
    S_E = np.maximum(0.0, np.corrcoef(emb))
    if not np.isfinite(S_E).all():
        S_E = np.nan_to_num(S_E, nan=0.0, posinf=0.0, neginf=0.0)
        S_E = np.maximum(0.0, S_E)
    np.fill_diagonal(S_E, 0.0)
    return S_E.astype(np.float64), emb


@dataclass
class DeltaNMFFit:
    """Result of one DeltaNMF one-stage solve (genes × K, K × cells)."""

    W: np.ndarray
    H: np.ndarray
    loss_history: object
    S_E: np.ndarray | None
    name: str
    meta: dict


def fit_deltanmf(
    X_genes_by_cells: np.ndarray,
    S_E: np.ndarray | None,
    *,
    K: int,
    name: str = "DeltaNMF",
    alpha_ntc: float = 0.05,
    max_iter: int = 400,
    lr: float = 0.01,
    seed: int = 0,
    nonneg: str = "softplus",
    softplus_beta: float = 5.0,
) -> DeltaNMFFit:
    """Run DeltaNMF's ``solve_ntc_regularized`` (Embedding-NMF when ``S_E`` is theirs).

    Parameters
    ----------
    X_genes_by_cells :
        Expression in DeltaNMF orientation (**genes × cells**).
    S_E :
        Gene–gene similarity prior, or ``None`` for plain NMF (``use_fm=False``).
        Attention-NMF = pass attention similarity here instead of embedding ``S_E``.
    alpha_ntc :
        Strength of the FM Laplacian term (DeltaNMF's ``alpha_ntc``).
    """
    _ensure_deltanmf_on_path()
    from deltanmf.models import solve_ntc_regularized

    X = np.asarray(X_genes_by_cells, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D genes×cells; got {X.shape}")

    use_fm = S_E is not None and alpha_ntc > 0
    S = None if not use_fm else np.asarray(S_E, dtype=np.float64)
    if S is not None and S.shape != (X.shape[0], X.shape[0]):
        raise ValueError(f"S_E shape {S.shape} != ({X.shape[0]}, {X.shape[0]})")

    W, H, hist = solve_ntc_regularized(
        X,
        k=K,
        S_E=S,
        alpha_ntc=float(alpha_ntc) if use_fm else 0.0,
        max_iter=int(max_iter),
        tol=0,
        nonneg=nonneg,
        softplus_beta=softplus_beta,
        normalize_W=False,
        init_fix_scale=False,
        seed=int(seed),
        lr_start=lr,
        fm_target_ratio=None,
        fm_apply_late=False,
    )
    return DeltaNMFFit(
        W=np.asarray(W),
        H=np.asarray(H),
        loss_history=hist,
        S_E=S,
        name=name,
        meta={"alpha_ntc": float(alpha_ntc) if use_fm else 0.0, "max_iter": max_iter, "K": K},
    )


def write_S_E_files(
    S_E: np.ndarray,
    gene_names: list[str],
    out_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Write ``S_E_relu.npy`` + ``genes_order.json`` for DeltaNMF's path-based API."""
    out = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="deltanmf_S_E_"))
    out.mkdir(parents=True, exist_ok=True)
    se_path = out / "S_E_relu.npy"
    genes_path = out / "genes_order.json"
    np.save(se_path, np.asarray(S_E, dtype=np.float32))
    import json

    with open(genes_path, "w") as f:
        json.dump(list(gene_names), f)
    return se_path, genes_path
