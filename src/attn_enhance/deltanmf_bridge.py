"""Thin bridge to vendored DeltaNMF — Embedding-NMF is DeltaNMF, unchanged.

Embedding prior ``S_E`` follows the official DeltaNMF scGPT resource script:

  ``third_party/deltanmf/resources/scgpt/create_transformer_similarity_matrix_scgpt.py``

Every NMF arm runs DeltaNMF's one-stage pipeline, as in
``example_scripts/run_onestage.py``:

  ``deltanmf.api.run_onestage_deltanmf`` (min-cells filter → unit-variance genes →
  consensus-NMF init → FM Laplacian at relative strength ``rel_alpha`` over the last
  ``FM_LAST_ITERS`` iterations)

The *only* project-specific swap is the similarity file handed to it as ``S_E``
(Attention-NMF / Combined-NMF). Do not reimplement DeltaNMF.
"""

from __future__ import annotations

import json
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

    Pearson correlation is pairwise, so computing it on ``gene_names`` equals
    computing it on the whole vocab and subsetting.
    """
    device_t = device or next(model.parameters()).device
    ids = torch.tensor([vocab[g] for g in gene_names], dtype=torch.long, device=device_t)
    with torch.no_grad():
        emb = model.encoder(ids).detach().cpu().numpy()
    S_E = np.maximum(0, np.corrcoef(emb))
    np.fill_diagonal(S_E, 0)
    return S_E.astype(np.float64), emb


@dataclass
class DeltaNMFFit:
    """Result of one ``run_onestage_deltanmf`` call.

    ``X`` is the genes × cells matrix DeltaNMF actually factorized (after its gene
    filter and normalization), so ``X ≈ W @ H``.
    """

    W: np.ndarray
    H: np.ndarray
    X: np.ndarray
    gene_names: list[str]
    name: str
    meta: dict

    @property
    def recon(self) -> float:
        return float(np.mean((self.X - self.W @ self.H) ** 2))


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
    with open(genes_path, "w") as f:
        json.dump(list(gene_names), f)
    return se_path, genes_path


def fit_deltanmf(
    X_genes_by_cells: np.ndarray,
    gene_names: list[str],
    S: np.ndarray | None,
    *,
    K: int,
    name: str = "DeltaNMF",
    rel_alpha: float = 0.05,
    min_cells: int = 10,
    max_iter: int = 10000,
    seed: int = 1337,
    use_tpm: bool = False,
    use_median: bool = False,
    use_unitvar: bool = True,
) -> DeltaNMFFit:
    """Run ``deltanmf.api.run_onestage_deltanmf`` with ``S`` as its ``S_E``.

    Parameters
    ----------
    X_genes_by_cells :
        Expression in DeltaNMF orientation (**genes × cells**).
    S :
        Gene–gene similarity over ``gene_names``, or ``None`` for DeltaNMF's
        ``use_fm=False`` baseline. Attention-NMF passes ``S_att`` here.
    rel_alpha :
        DeltaNMF's relative FM strength: the Laplacian term is scaled to
        ``rel_alpha × recon_loss`` when it switches on, so priors with different
        edge-weight scales get the same strength.
    Defaults for ``min_cells``, ``rel_alpha`` and ``use_minibatch_ntc=False`` follow
    ``example_scripts/run_onestage.py``; the rest are ``run_onestage_deltanmf`` defaults.
    """
    _ensure_deltanmf_on_path()
    from deltanmf.api import run_onestage_deltanmf
    from deltanmf.pipeline_utils import apply_normalization

    X = np.asarray(X_genes_by_cells, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] != len(gene_names):
        raise ValueError(f"X must be genes×cells with {len(gene_names)} rows; got {X.shape}")
    use_fm = S is not None and rel_alpha > 0
    if use_fm and np.shape(S) != (len(gene_names), len(gene_names)):
        raise ValueError(f"S shape {np.shape(S)} != ({len(gene_names)}, {len(gene_names)})")

    with tempfile.TemporaryDirectory(prefix="deltanmf_S_E_") as tmp:
        se_path, genes_path = write_S_E_files(S, gene_names, tmp) if use_fm else (None, None)
        res = run_onestage_deltanmf(
            X_control=X,
            gene_names=list(gene_names),
            S_E_PATH=se_path,
            S_E_GENES_PATH=genes_path,
            K=K,
            MIN_CELLS=min_cells,
            USE_TPM=use_tpm,
            USE_MEDIAN=use_median,
            USE_UNITVAR=use_unitvar,
            rel_alpha=rel_alpha if use_fm else 0.0,
            max_iter=max_iter,
            use_minibatch_ntc=False,
            use_fm=use_fm,
            BASE_SEED=seed,
        )

    genes = [str(g) for g in res["gene_names_aligned"]]
    row = {g: i for i, g in enumerate(gene_names)}
    X_fit = apply_normalization(
        X[[row[g] for g in genes]], use_tpm, use_median, use_unitvar, 1e6
    ).astype(np.float32, copy=False)
    return DeltaNMFFit(
        W=np.asarray(res["W"]),
        H=np.asarray(res["H"]),
        X=X_fit,
        gene_names=genes,
        name=name,
        meta={
            "rel_alpha": float(rel_alpha) if use_fm else 0.0,
            "use_fm": use_fm,
            "max_iter": max_iter,
            "K": K,
            "gene_filter_info": res["gene_filter_info"],
        },
    )
