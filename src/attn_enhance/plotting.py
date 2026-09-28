"""Visual comparisons for DeltaNMF vs attention-prior arms.

Keeps plotting next to the experiment — no new scientific logic.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .compare import ArmResult, ExperimentResult
from .gene_graph import upper_triangle_correlation
from .prior import ScgptPriors


def _align_programs(W_ref: np.ndarray, W_other: np.ndarray) -> np.ndarray:
    """Greedy column permutation of ``W_other`` to match ``W_ref`` by abs corr."""
    K = W_ref.shape[1]
    remaining = set(range(K))
    order: list[int] = []
    for k in range(K):
        best_j, best_c = None, -np.inf
        for j in remaining:
            a, b = W_ref[:, k], W_other[:, j]
            if a.std() < 1e-12 or b.std() < 1e-12:
                c = 0.0
            else:
                c = abs(float(np.corrcoef(a, b)[0, 1]))
            if c > best_c:
                best_c, best_j = c, j
        assert best_j is not None
        order.append(best_j)
        remaining.remove(best_j)
    return W_other[:, order]


def plot_synthetic_atlas(
    X: np.ndarray,
    program_W: np.ndarray,
    program_H: np.ndarray,
    *,
    gene_names: Sequence[str] | None = None,
    max_cells: int | None = None,
    figsize: tuple[float, float] = (12, 6.2),
) -> plt.Figure:
    """Show the toy expression matrix and the planted factors that generated it.

    Layout
    ------
    - $X$ (cells × genes): what NMF sees
    - planted $W^\\top$ (programs × genes): which genes belong to which program
    - planted $H^\\top$ (cells × programs): which cells use which program
    - per-gene mean expression: quick scale check
    """
    n_cells = X.shape[0] if max_cells is None else min(max_cells, X.shape[0])
    X_show = X[:n_cells]
    H_show = program_H[:n_cells]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, :])
    im0 = ax0.imshow(X_show, aspect="auto", cmap="viridis")
    ax0.set_title(r"Expression $X$ (cells × genes) — input to scGPT + DeltaNMF")
    ax0.set_xlabel("gene index")
    ax0.set_ylabel("cell")
    fig.colorbar(im0, ax=ax0, fraction=0.015, pad=0.01, label="expression")

    ax1 = fig.add_subplot(gs[1, 0])
    im1 = ax1.imshow(program_W.T, aspect="auto", cmap="Blues")
    ax1.set_title(r"Planted $W^\top$ (ground truth programs)")
    ax1.set_xlabel("gene index")
    ax1.set_ylabel("program")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[1, 1])
    im2 = ax2.imshow(H_show.T, aspect="auto", cmap="YlOrBr")
    ax2.set_title(r"Planted $H^\top$ (cell × program usage)")
    ax2.set_xlabel("cell")
    ax2.set_ylabel("program")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    if gene_names is not None and len(gene_names) == X.shape[1]:
        # Tiny caption under X with a few gene symbols for orientation.
        sample = ", ".join(list(gene_names)[:6])
        ax0.text(
            0.0,
            -0.22,
            f"genes[0:6] = {sample}, …",
            transform=ax0.transAxes,
            fontsize=8,
            color="0.35",
            clip_on=False,
        )
    return fig


def plot_prior_edge_diagnostics(
    priors: ScgptPriors,
    *,
    figsize: tuple[float, float] = (11, 3.6),
) -> plt.Figure:
    """Histograms + degree profiles that explain *how* $S_E$ and $S^{att}$ differ.

    Useful when heatmaps alone look abstract: this shows weight scale and
    whether attention collapses onto hub genes after $k$NN sparsification.
    """
    S_E = priors.embedding.similarity
    S_att = priors.attention.similarity
    iu = np.triu_indices(S_E.shape[0], k=1)
    e_w, a_w = S_E[iu], S_att[iu]
    # Ignore exact zeros so the attention histogram is not dominated by sparsity.
    e_pos = e_w[e_w > 0]
    a_pos = a_w[a_w > 0]
    deg_E = S_E.sum(axis=1)
    deg_A = S_att.sum(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].hist(e_pos, bins=30, color="#3d5a80", alpha=0.85, label=r"$S_E>0$")
    axes[0].hist(a_pos, bins=30, color="#ee6c4d", alpha=0.65, label=r"$S^{att}>0$")
    axes[0].set_xlabel("edge weight")
    axes[0].set_ylabel("count")
    axes[0].set_title("Positive edge weights")
    axes[0].legend(fontsize=8)

    axes[1].bar(np.arange(len(deg_E)), deg_E, color="#3d5a80", alpha=0.7, label=r"$S_E$")
    axes[1].set_xlabel("gene index")
    axes[1].set_ylabel(r"$\sum_j S_{ij}$")
    axes[1].set_title(r"$S_E$ degree (row sum)")

    axes[2].bar(np.arange(len(deg_A)), deg_A, color="#ee6c4d", alpha=0.85, label=r"$S^{att}$")
    axes[2].set_xlabel("gene index")
    axes[2].set_ylabel(r"$\sum_j S_{ij}$")
    axes[2].set_title(r"$S^{att}$ degree (row sum)")

    nnz_E = float(np.mean(S_E > 0))
    nnz_A = float(np.mean(S_att > 0))
    fig.suptitle(
        f"Prior diagnostics  |  density $S_E$={nnz_E:.2f}, $S^{{att}}$={nnz_A:.2f}",
        y=1.03,
    )
    fig.tight_layout()
    return fig


def plot_prior_comparison(
    priors: ScgptPriors,
    *,
    figsize: tuple[float, float] = (11, 3.4),
) -> plt.Figure:
    """Side-by-side ``S_E``, ``S_att``, absolute difference, and edge scatter."""
    S_E = priors.embedding.similarity
    S_att = priors.attention.similarity
    iu = np.triu_indices(S_E.shape[0], k=1)

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    axes[0].imshow(S_E, cmap="viridis")
    axes[0].set_title(r"$S_E$ (DeltaNMF)")
    axes[1].imshow(S_att, cmap="magma")
    axes[1].set_title(r"$S^{att}$ (novel)")
    axes[2].imshow(np.abs(S_E - S_att), cmap="cividis")
    axes[2].set_title(r"$|S_E - S^{att}|$")
    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[3].scatter(S_E[iu], S_att[iu], s=6, alpha=0.35, c="#3d5a80")
    lo = min(S_E[iu].min(), S_att[iu].min())
    hi = max(S_E[iu].max(), S_att[iu].max())
    axes[3].plot([lo, hi], [lo, hi], ls="--", c="0.4", lw=1)
    axes[3].set_xlabel(r"$S_E$ edge weight")
    axes[3].set_ylabel(r"$S^{att}$ edge weight")
    corr = upper_triangle_correlation(S_E, S_att)
    axes[3].set_title(f"edges (corr={corr:.3f})")
    fig.suptitle("Prior comparison", y=1.02)
    fig.tight_layout()
    return fig


def plot_recon_comparison(
    result: ExperimentResult,
    *,
    figsize: tuple[float, float] = (10, 3.6),
) -> plt.Figure:
    """Bar chart of primary arms + α-sweep recon MSE."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    arms = [result.nmf, result.embedding, result.attention, result.combined]
    labels = ["NMF", "Embedding\n(=DeltaNMF)", "Attention\n(novel)", "Combined"]
    colors = ["#98c1d9", "#3d5a80", "#ee6c4d", "#293241"]
    axes[0].bar(range(4), [a.recon for a in arms], color=colors)
    axes[0].set_xticks(range(4), labels, fontsize=8)
    axes[0].set_ylabel("recon MSE")
    axes[0].set_title("Primary arms")

    if result.alpha_sweep:
        xs = [a.alpha for a in result.alpha_sweep]
        ys = [a.recon for a in result.alpha_sweep]
        axes[1].plot(xs, ys, marker="o", color="#ee6c4d")
        axes[1].axvline(0.0, color="#ee6c4d", ls=":", lw=1, label=r"$\alpha=0$ att")
        axes[1].axvline(1.0, color="#3d5a80", ls=":", lw=1, label=r"$\alpha=1$ $S_E$")
        axes[1].set_xlabel(r"$\alpha$ (weight on $S_E$)")
        axes[1].set_ylabel("recon MSE")
        axes[1].set_title(r"Combined $\alpha$ sweep")
        axes[1].legend(fontsize=7)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    return fig


def plot_program_loadings(
    result: ExperimentResult,
    gene_names: Sequence[str],
    *,
    align_to: str = "embedding",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Heatmaps of $W$ (genes × programs) for all four arms.

    Columns of non-reference arms are greedily aligned to ``align_to``
    so visual comparison is not scrambled by NMF permutation ambiguity.
    """
    ref_map = {
        "nmf": result.nmf,
        "embedding": result.embedding,
        "attention": result.attention,
        "combined": result.combined,
    }
    ref = ref_map[align_to].fit.W
    panels = [
        ("NMF", result.nmf.fit.W),
        ("Embedding-NMF", result.embedding.fit.W),
        ("Attention-NMF", result.attention.fit.W),
        ("Combined-NMF", result.combined.fit.W),
    ]
    aligned = []
    for name, W in panels:
        if name == {"nmf": "NMF", "embedding": "Embedding-NMF", "attention": "Attention-NMF", "combined": "Combined-NMF"}[align_to]:
            aligned.append((name, W))
        else:
            aligned.append((name, _align_programs(ref, W)))

    K = ref.shape[1]
    if figsize is None:
        figsize = (11, 2.4 + 0.55 * K)
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    vmax = max(float(W.max()) for _, W in aligned)
    for ax, (name, W) in zip(axes, aligned):
        im = ax.imshow(W, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("program")
        ax.set_xticks(range(K))
    axes[0].set_ylabel("gene index")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="loading")
    fig.suptitle(f"Program loadings $W$ (aligned to {align_to})", y=1.02)
    fig.subplots_adjust(right=0.88)
    return fig


def plot_top_genes_comparison(
    result: ExperimentResult,
    gene_names: Sequence[str],
    *,
    top_n: int = 8,
    program: int = 0,
    figsize: tuple[float, float] = (10, 3.5),
) -> plt.Figure:
    """Top genes for one program: Embedding-NMF vs Attention-NMF."""
    W_emb = result.embedding.fit.W
    W_att = _align_programs(W_emb, result.attention.fit.W)
    names = list(gene_names)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=False)
    for ax, W, title, color in [
        (axes[0], W_emb, "Embedding-NMF (=DeltaNMF)", "#3d5a80"),
        (axes[1], W_att, "Attention-NMF (novel)", "#ee6c4d"),
    ]:
        scores = W[:, program]
        top = np.argsort(scores)[::-1][:top_n]
        ax.barh([names[i] for i in top][::-1], scores[top][::-1], color=color)
        ax.set_title(f"{title}\nprogram {program}")
        ax.set_xlabel("loading")
    fig.tight_layout()
    return fig


def plot_cell_program_usage(
    result: ExperimentResult,
    *,
    max_cells: int = 80,
    figsize: tuple[float, float] = (11, 4.5),
) -> plt.Figure:
    """Heatmaps of $H$ (programs × cells) for Embedding vs Attention."""
    H_emb = result.embedding.fit.H[:, :max_cells]
    # align Attention programs to Embedding via W, then apply same order to H rows
    W_emb = result.embedding.fit.W
    W_att = result.attention.fit.W
    order = []
    remaining = set(range(W_emb.shape[1]))
    for k in range(W_emb.shape[1]):
        best_j, best_c = None, -np.inf
        for j in remaining:
            a, b = W_emb[:, k], W_att[:, j]
            c = 0.0 if a.std() < 1e-12 or b.std() < 1e-12 else abs(float(np.corrcoef(a, b)[0, 1]))
            if c > best_c:
                best_c, best_j = c, j
        order.append(best_j)
        remaining.remove(best_j)
    H_att = result.attention.fit.H[order, :max_cells]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    vmax = max(float(H_emb.max()), float(H_att.max()))
    im0 = axes[0].imshow(H_emb, aspect="auto", cmap="YlOrBr", vmin=0, vmax=vmax)
    axes[0].set_ylabel("program")
    axes[0].set_title("Embedding-NMF  $H$ (program activity)")
    im1 = axes[1].imshow(H_att, aspect="auto", cmap="YlOrBr", vmin=0, vmax=vmax)
    axes[1].set_ylabel("program")
    axes[1].set_xlabel("cell")
    axes[1].set_title("Attention-NMF  $H$ (aligned programs)")
    fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="usage")
    fig.subplots_adjust(hspace=0.35)
    return fig


def plot_arm_W_similarity(
    result: ExperimentResult,
    *,
    figsize: tuple[float, float] = (4.2, 3.6),
) -> plt.Figure:
    """Mean abs column-corr between arms after aligning each to Embedding-NMF."""
    arms: list[tuple[str, ArmResult]] = [
        ("NMF", result.nmf),
        ("Emb", result.embedding),
        ("Att", result.attention),
        ("Comb", result.combined),
    ]
    W_ref = result.embedding.fit.W
    Ws = []
    for name, arm in arms:
        W = arm.fit.W if name == "Emb" else _align_programs(W_ref, arm.fit.W)
        Ws.append(W)

    n = len(Ws)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cors = []
            for k in range(W_ref.shape[1]):
                a, b = Ws[i][:, k], Ws[j][:, k]
                if a.std() < 1e-12 or b.std() < 1e-12:
                    cors.append(0.0)
                else:
                    cors.append(abs(float(np.corrcoef(a, b)[0, 1])))
            M[i, j] = float(np.mean(cors))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1)
    labels = [a[0] for a in arms]
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(r"Mean $|$corr$|$ of aligned $W$ columns")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
