"""Experimental matrix via DeltaNMF: plain / Embedding (S_E) / Attention / Combined.

Embedding-NMF **is** DeltaNMF with their scGPT ``S_E``.
Attention-NMF and Combined-NMF reuse the same DeltaNMF solver and only
change the similarity matrix fed in as ``S_E``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .deltanmf_bridge import fit_deltanmf, DeltaNMFFit
from .gene_graph import combine_similarities
from .prior import ScgptPriors

DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class ArmResult:
    """One row of the experimental matrix (DeltaNMF solve)."""

    name: str
    fit: DeltaNMFFit
    recon: float
    alpha: float | None = None


@dataclass
class ExperimentResult:
    nmf: ArmResult
    embedding: ArmResult
    attention: ArmResult
    combined: ArmResult
    alpha_sweep: list[ArmResult] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def baseline(self) -> ArmResult:
        return self.nmf


def align_expression_to_genes(
    X_cells_by_genes: np.ndarray,
    gene_names: list[str],
    target_genes: list[str],
) -> tuple[np.ndarray, list[str]]:
    idx = {g: i for i, g in enumerate(gene_names)}
    missing = [g for g in target_genes if g not in idx]
    if missing:
        raise ValueError(
            f"Expression missing {len(missing)} prior genes (e.g. {missing[:3]})."
        )
    cols = [idx[g] for g in target_genes]
    return X_cells_by_genes[:, cols], list(target_genes)


def _recon(fit: DeltaNMFFit, X_gc: np.ndarray) -> float:
    return float(np.mean((X_gc - fit.W @ fit.H) ** 2))


def _fit_arm(
    name: str,
    X_gc: np.ndarray,
    S: np.ndarray | None,
    *,
    K: int,
    alpha_ntc: float,
    max_iter: int,
    seed: int,
    alpha: float | None = None,
) -> ArmResult:
    fit = fit_deltanmf(
        X_gc,
        S,
        K=K,
        name=name,
        alpha_ntc=0.0 if S is None else alpha_ntc,
        max_iter=max_iter,
        seed=seed,
    )
    return ArmResult(name=name, fit=fit, recon=_recon(fit, X_gc), alpha=alpha)


def run_program_discovery_experiment(
    X_cells_by_genes: np.ndarray,
    gene_names: list[str],
    priors: ScgptPriors,
    *,
    n_components: int = 4,
    alpha_ntc: float = 0.05,
    nmf_iters: int = 400,
    primary_alpha: float = 0.5,
    alphas: tuple[float, ...] | list[float] | None = DEFAULT_ALPHAS,
    seed: int = 0,
    # kept for call-site compat; DeltaNMF path does not use planted overlap here
    planted_sets=None,
    lam: float | None = None,
) -> ExperimentResult:
    """Four-arm matrix on DeltaNMF's solver.

    - **NMF** — DeltaNMF with ``S_E=None``
    - **Embedding-NMF** — DeltaNMF with DeltaNMF ``S_E`` (static embeddings)
    - **Attention-NMF** — same solver, ``S_E`` ← novel ``S_att``
    - **Combined-NMF** — same solver, ``S_E`` ← ``α S_emb + (1-α) S_att``
    """
    del planted_sets, lam  # pedagogical MU / overlap removed; use DeltaNMF only
    X_cg, names = align_expression_to_genes(
        X_cells_by_genes, gene_names, priors.gene_names
    )
    # DeltaNMF expects genes × cells
    X_gc = np.asarray(X_cg.T, dtype=np.float32)

    S_emb = priors.embedding.similarity
    S_att = priors.attention.similarity
    S_comb = combine_similarities(S_emb, S_att, primary_alpha, renorm=False)

    common = dict(K=n_components, alpha_ntc=alpha_ntc, max_iter=nmf_iters, seed=seed)

    nmf_arm = _fit_arm("NMF (DeltaNMF, no FM)", X_gc, None, **common)
    emb_arm = _fit_arm("Embedding-NMF (= DeltaNMF S_E)", X_gc, S_emb, alpha=1.0, **common)
    att_arm = _fit_arm("Attention-NMF (novel S_att → DeltaNMF)", X_gc, S_att, alpha=0.0, **common)
    comb_arm = _fit_arm(
        "Combined-NMF", X_gc, S_comb, alpha=primary_alpha, **common
    )

    sweep: list[ArmResult] = []
    if alphas is not None:
        for a in alphas:
            S = combine_similarities(S_emb, S_att, float(a), renorm=False)
            sweep.append(
                _fit_arm(
                    f"Combined-NMF(alpha={a})",
                    X_gc,
                    S,
                    alpha=float(a),
                    **common,
                )
            )

    return ExperimentResult(
        nmf=nmf_arm,
        embedding=emb_arm,
        attention=att_arm,
        combined=comb_arm,
        alpha_sweep=sweep,
        meta={
            "solver": "deltanmf.models.solve_ntc_regularized",
            "S_E_source": "DeltaNMF scGPT resource script",
            "S_att_source": "scGPT Tutorial_Attention_GRN (Wqkv)",
            "alpha_ntc": alpha_ntc,
            "n_components": n_components,
            "n_genes": len(names),
            "n_cells": int(X_cg.shape[0]),
            "primary_alpha": primary_alpha,
            "reading": {
                "fm_helps": "NMF → Embedding-NMF (DeltaNMF)",
                "context_helps": "Embedding-NMF → Attention-NMF (novel prior)",
                "complementary": "Attention-NMF → Combined-NMF (α sweep)",
            },
        },
    )


def run_attention_vs_embedding_nmf(*args, **kwargs) -> ExperimentResult:
    return run_program_discovery_experiment(*args, **kwargs)


def format_comparison(result: ExperimentResult) -> str:
    lines = [
        "DeltaNMF solver × {no FM | S_E | S_att | α-mix}",
        f"  {'model':<42} {'recon_mse':>12}",
    ]
    for arm in (result.nmf, result.embedding, result.attention, result.combined):
        lines.append(f"  {arm.name:<42} {arm.recon:12.6f}")
    if result.alpha_sweep:
        lines.append("  α sweep (Combined; S = α S_E + (1-α) S_att):")
        for arm in result.alpha_sweep:
            a = arm.alpha if arm.alpha is not None else float("nan")
            lines.append(f"    α={a:<6} recon_mse={arm.recon:.6f}")
    ranked = sorted(
        [result.embedding, result.attention, result.combined], key=lambda a: a.recon
    )
    lines.append(f"  lowest recon among FM arms: {ranked[0].name}")
    lines.append(
        "  read: NMF→Embedding (=DeltaNMF); Embedding→Attention (novel); "
        "Attention→Combined (complementary?)"
    )
    return "\n".join(lines)


ComparisonResult = ExperimentResult
