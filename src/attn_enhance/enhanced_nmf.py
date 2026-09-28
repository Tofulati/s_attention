"""Legacy pedagogical MU-NMF (not used by the notebook).

Prefer ``deltanmf_bridge.fit_deltanmf`` → ``deltanmf.models.solve_ntc_regularized``.
Kept only for older experiments / unit smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NMFResult:
    W: np.ndarray  # genes x rank  (program loadings)
    H: np.ndarray  # rank x cells  (program usages)
    losses: list[float]
    recon_losses: list[float]
    graph_losses: list[float]


def fit_graph_regularized_nmf(
    X_cells_by_genes: np.ndarray,
    L: np.ndarray,
    n_components: int = 4,
    lam: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-5,
    seed: int = 0,
) -> NMFResult:
    """Fit a genes x cells factorization with graph regularization on W.

    Notation
    --------
    Let ``V`` be genes x cells (we transpose the incoming cells x genes matrix).
    Factor ``V ~= W @ H`` with W >= 0 (genes x k), H >= 0 (k x cells).

    Objective
    ---------
    ``||V - W H||_F^2 + lam * Tr(W.T @ L @ W)``

    With the unnormalized Laplacian this is equivalent (up to scaling) to

    ``||V - WH||_F^2 + (lam/2) * sum_{i,j} S_ij ||W_i - W_j||^2``

    so genes linked by the foundation-model prior S (global embedding,
    contextual attention, or their convex combination) are pushed toward
    similar program memberships.

    Updates
    -------
    Multiplicative updates (Lee-Seung style) with an extra graph term on W.
    This is a simplified pedagogical cousin of DeltaNMF's neural formulation —
    enough to feel the regularization, not a full reimplementation.
    """
    rng = np.random.default_rng(seed)
    V = np.asarray(X_cells_by_genes, dtype=np.float64).T  # genes x cells
    n_genes, n_cells = V.shape
    if L.shape != (n_genes, n_genes):
        raise ValueError(f"L shape {L.shape} != ({n_genes}, {n_genes})")

    W = rng.random((n_genes, n_components)) + 0.1
    H = rng.random((n_components, n_cells)) + 0.1

    # Split L = L_pos - L_neg for multiplicative safety (L may have negatives).
    L_pos = np.maximum(L, 0.0)
    L_neg = np.maximum(-L, 0.0)

    losses: list[float] = []
    recon_losses: list[float] = []
    graph_losses: list[float] = []
    eps = 1e-10

    for _ in range(max_iter):
        WH = W @ H
        recon = np.linalg.norm(V - WH) ** 2
        graph = float(np.trace(W.T @ L @ W))
        total = recon + lam * graph
        recon_losses.append(float(recon))
        graph_losses.append(graph)
        losses.append(float(total))

        # H update (standard NMF)
        H *= (W.T @ V) / (W.T @ W @ H + eps)

        # W update with graph regularization.
        num = V @ H.T + lam * (L_neg @ W)
        den = W @ H @ H.T + lam * (L_pos @ W) + eps
        W *= num / den

        if len(losses) > 1 and abs(losses[-2] - losses[-1]) / (abs(losses[-2]) + eps) < tol:
            break

    return NMFResult(W=W, H=H, losses=losses, recon_losses=recon_losses, graph_losses=graph_losses)


def program_gene_overlap(W: np.ndarray, true_sets: list[np.ndarray], top_n: int = 10) -> list[dict]:
    """Quick recovery check: do top-loaded genes match planted sets?"""
    reports = []
    for p, true_idx in enumerate(true_sets):
        if p >= W.shape[1]:
            break
        ranked = np.argsort(W[:, p])[::-1][:top_n]
        overlap = len(set(ranked.tolist()) & set(true_idx.tolist()))
        reports.append(
            {
                "program": p,
                "top_genes": ranked.tolist(),
                "overlap_with_planted": overlap,
                "planted_size": int(len(true_idx)),
            }
        )
    return reports
