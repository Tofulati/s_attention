"""Build gene similarity graphs and Laplacians for FM-regularized NMF."""

from __future__ import annotations

import numpy as np


def affinity_from_attention(
    attention: np.ndarray,
    knn: int | None = 8,
    symmetrize: bool = True,
    row_normalize: bool = False,
) -> np.ndarray:
    """Contextual prior S_att from aggregated per-cell attention.

    Steps
    -----
    1. Optionally symmetrize: treat attention as association, not directed regulation.
    2. Clip negatives, zero diagonal.
    3. Optional kNN sparsify (avoids an overly smooth dense prior).
    4. Optional row-normalize (down-weights hub genes).
    """
    A = np.asarray(attention, dtype=np.float64)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"attention must be square; got {A.shape}")

    S = 0.5 * (A + A.T) if symmetrize else A.copy()
    np.fill_diagonal(S, 0.0)
    S = np.clip(S, 0.0, None)

    if knn is not None and knn < n - 1:
        S = _knn_sparsify(S, knn=knn)
        if symmetrize:
            S = 0.5 * (S + S.T)
            np.fill_diagonal(S, 0.0)

    if row_normalize:
        S = normalize_graph(S, mode="row")
    return S


def affinity_from_embeddings(
    gene_embeddings: np.ndarray,
    *,
    method: str = "corrcoef_relu",
    knn: int | None = None,
    row_normalize: bool = False,
) -> np.ndarray:
    """Global prior S_emb from scGPT gene embeddings (DeltaNMF default).

    DeltaNMF script: ``corrcoef(embeddings)`` → ReLU → zero diagonal.
    """
    E = np.asarray(gene_embeddings, dtype=np.float64)
    if E.ndim != 2:
        raise ValueError(f"gene_embeddings must be 2D; got {E.shape}")
    n = E.shape[0]
    if n < 2:
        raise ValueError("need at least 2 genes")

    if method == "corrcoef_relu":
        cts = np.corrcoef(E)
        if not np.isfinite(cts).all():
            cts = np.nan_to_num(cts, nan=0.0, posinf=0.0, neginf=0.0)
        S = np.maximum(0.0, cts)
    elif method == "cosine":
        norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
        E_n = E / norms
        cos = E_n @ E_n.T
        S = np.clip(cos, 0.0, None)
    else:
        raise ValueError(f"unknown method: {method!r}")

    np.fill_diagonal(S, 0.0)

    if knn is not None and knn < n - 1:
        S = _knn_sparsify(S, knn=knn)
        S = 0.5 * (S + S.T)
        np.fill_diagonal(S, 0.0)

    if row_normalize:
        S = normalize_graph(S, mode="row")
    return S


def affinity_from_expression(
    X_cells_by_genes: np.ndarray,
    *,
    knn: int | None = None,
    row_normalize: bool = False,
) -> np.ndarray:
    """Expression-correlation control graph S^expr (idea.md §3 / §19 Control 2).

    Asks: which genes *vary together in the observed data*?
    Built as ReLU(corrcoef) over gene columns of X — not an FM prior.
    """
    X = np.asarray(X_cells_by_genes, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError(f"X must be cells×genes with ≥2 genes; got {X.shape}")
    cts = np.corrcoef(X.T)
    if not np.isfinite(cts).all():
        cts = np.nan_to_num(cts, nan=0.0, posinf=0.0, neginf=0.0)
    S = np.maximum(0.0, cts)
    np.fill_diagonal(S, 0.0)
    if knn is not None and knn < S.shape[0] - 1:
        S = _knn_sparsify(S, knn=knn)
        S = 0.5 * (S + S.T)
        np.fill_diagonal(S, 0.0)
    if row_normalize:
        S = normalize_graph(S, mode="row")
    return S


def random_affinity(
    n_genes: int,
    *,
    knn: int = 8,
    seed: int = 0,
    row_normalize: bool = False,
) -> np.ndarray:
    """Density-matched random graph control (idea.md §19 Control 1)."""
    rng = np.random.default_rng(seed)
    S = rng.random((n_genes, n_genes))
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 0.0)
    if knn is not None and knn < n_genes - 1:
        S = _knn_sparsify(S, knn=knn)
        S = 0.5 * (S + S.T)
        np.fill_diagonal(S, 0.0)
    if row_normalize:
        S = normalize_graph(S, mode="row")
    return S


def upper_triangle_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """Pearson correlation of upper-triangular entries (matrix-vs-matrix)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"need matching square matrices; got {A.shape}, {B.shape}")
    iu = np.triu_indices(A.shape[0], k=1)
    a, b = A[iu], B[iu]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def combine_similarities(
    S_emb: np.ndarray,
    S_att: np.ndarray,
    alpha: float,
    *,
    renorm: bool = True,
) -> np.ndarray:
    """Combined prior S = alpha * S_emb + (1 - alpha) * S_att.

    ``alpha=1`` → embedding only; ``alpha=0`` → attention only.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    S_emb = np.asarray(S_emb, dtype=np.float64)
    S_att = np.asarray(S_att, dtype=np.float64)
    if S_emb.shape != S_att.shape:
        raise ValueError(f"shape mismatch: S_emb {S_emb.shape} vs S_att {S_att.shape}")
    S = alpha * S_emb + (1.0 - alpha) * S_att
    np.fill_diagonal(S, 0.0)
    S = np.clip(S, 0.0, None)
    if renorm:
        S = normalize_graph(S, mode="row")
    return S


def normalize_graph(S: np.ndarray, mode: str = "row", eps: float = 1e-12) -> np.ndarray:
    """Normalize a non-negative similarity graph."""
    S = np.asarray(S, dtype=np.float64).copy()
    np.fill_diagonal(S, 0.0)
    if mode == "row":
        denom = S.sum(axis=1, keepdims=True) + eps
        return S / denom
    if mode == "symmetric":
        d = S.sum(axis=1)
        d_inv_sqrt = np.zeros_like(d)
        mask = d > 0
        d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        Dmh = np.diag(d_inv_sqrt)
        return Dmh @ S @ Dmh
    if mode == "none":
        return S
    raise ValueError(f"unknown normalize mode: {mode!r}")


def differential_attention_similarity(
    S_att_treat: np.ndarray,
    S_att_ctrl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stage-2 helper: DeltaS = S_treat - S_ctrl, split into positive/negative parts.

    Returns ``(delta, delta_pos, delta_neg)``. Do not feed raw ``delta`` into a
    standard graph regularizer — it can be negative.
    """
    delta = np.asarray(S_att_treat, dtype=np.float64) - np.asarray(S_att_ctrl, dtype=np.float64)
    return delta, np.maximum(delta, 0.0), np.maximum(-delta, 0.0)


def _knn_sparsify(S: np.ndarray, knn: int) -> np.ndarray:
    n = S.shape[0]
    out = np.zeros_like(S)
    for i in range(n):
        row = S[i]
        if knn >= n - 1:
            out[i] = row
            continue
        idx = np.argpartition(row, -knn)[-knn:]
        out[i, idx] = row[idx]
    return out


def graph_laplacian(S: np.ndarray, normalized: bool = True) -> np.ndarray:
    """Graph Laplacian for lam * Tr(W.T @ L @ W).

    Unnormalized: L = D - S.
    Normalized: L = I - D^{-1/2} S D^{-1/2}.

    Equivalent view: Tr(W.T L W) = 0.5 * sum_ij S_ij ||W_i - W_j||^2
    for the unnormalized Laplacian (up to scaling conventions).
    """
    S = np.asarray(S, dtype=np.float64)
    d = S.sum(axis=1)
    if not normalized:
        return np.diag(d) - S

    d_inv_sqrt = np.zeros_like(d)
    mask = d > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    Dmh = np.diag(d_inv_sqrt)
    I = np.eye(S.shape[0])
    return I - Dmh @ S @ Dmh


def graph_smoothness(W: np.ndarray, S: np.ndarray) -> float:
    """Mean Rayleigh quotient ``w^T L w / w^T w`` of W's columns on ``L = D - S``.

    Lower = gene loadings vary less across edges of S. Columns are mean-centered
    first: a constant vector has zero energy on every Laplacian, so without
    centering any strong regularizer (even on a random graph) looks "smooth".
    Scale-invariant, so arms with different W magnitudes are comparable.
    """
    L = graph_laplacian(S, normalized=False)
    Wc = np.asarray(W, dtype=np.float64)
    Wc = Wc - Wc.mean(axis=0, keepdims=True)
    num = np.einsum("gk,gh,hk->k", Wc, L, Wc)
    den = (Wc**2).sum(axis=0) + 1e-12
    return float(np.mean(num / den))


def top_edges(S: np.ndarray, gene_names: list[str], k: int = 10) -> list[tuple[str, str, float]]:
    n = S.shape[0]
    tri = [(S[i, j], i, j) for i in range(n) for j in range(i + 1, n)]
    tri.sort(reverse=True)
    return [(gene_names[i], gene_names[j], float(w)) for w, i, j in tri[:k]]
