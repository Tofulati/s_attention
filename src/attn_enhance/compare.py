"""Experimental matrix via DeltaNMF: plain / Embedding (S_E) / Attention / Combined.

Embedding-NMF **is** DeltaNMF with their scGPT ``S_E``.
Attention-NMF and Combined-NMF run the same ``run_onestage_deltanmf`` pipeline and
only change the similarity matrix fed in as ``S_E``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .deltanmf_bridge import DeltaNMFFit, fit_deltanmf
from .gene_graph import combine_similarities
from .prior import ScgptPriors

DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class ArmResult:
    """One row of the experimental matrix (one DeltaNMF run)."""

    name: str
    fit: DeltaNMFFit
    alpha: float | None = None

    @property
    def recon(self) -> float:
        return self.fit.recon


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


def run_program_discovery_experiment(
    X_cells_by_genes: np.ndarray,
    gene_names: list[str],
    priors: ScgptPriors,
    *,
    n_components: int = 4,
    rel_alpha: float = 0.05,
    min_cells: int = 10,
    max_iter: int = 10000,
    primary_alpha: float = 0.5,
    alphas: tuple[float, ...] | list[float] | None = DEFAULT_ALPHAS,
    seed: int = 1337,
) -> ExperimentResult:
    """Four-arm matrix on DeltaNMF's one-stage pipeline.

    - **NMF** — ``run_onestage_deltanmf(use_fm=False)``
    - **Embedding-NMF** — DeltaNMF with DeltaNMF ``S_E`` (static embeddings)
    - **Attention-NMF** — same pipeline, ``S_E`` ← novel ``S_att``
    - **Combined-NMF** — same pipeline, ``S_E`` ← ``α S_emb + (1-α) S_att``
    """
    X_cg, names = align_expression_to_genes(X_cells_by_genes, gene_names, priors.gene_names)
    X_gc = np.asarray(X_cg.T, dtype=np.float32)

    S_emb = priors.embedding.similarity
    S_att = priors.attention.similarity

    def arm(name: str, S: np.ndarray | None, alpha: float | None = None) -> ArmResult:
        fit = fit_deltanmf(
            X_gc,
            names,
            S,
            K=n_components,
            name=name,
            rel_alpha=rel_alpha,
            min_cells=min_cells,
            max_iter=max_iter,
            seed=seed,
        )
        return ArmResult(name=name, fit=fit, alpha=alpha)

    def mix(a: float) -> np.ndarray:
        return combine_similarities(S_emb, S_att, a, renorm=False)

    nmf_arm = arm("NMF (DeltaNMF, use_fm=False)", None)
    emb_arm = arm("Embedding-NMF (= DeltaNMF S_E)", S_emb, alpha=1.0)
    att_arm = arm("Attention-NMF (S_att as S_E)", S_att, alpha=0.0)
    comb_arm = arm("Combined-NMF", mix(primary_alpha), alpha=primary_alpha)
    sweep = [arm(f"Combined-NMF(alpha={a})", mix(float(a)), float(a)) for a in (alphas or ())]

    return ExperimentResult(
        nmf=nmf_arm,
        embedding=emb_arm,
        attention=att_arm,
        combined=comb_arm,
        alpha_sweep=sweep,
        meta={
            "pipeline": "deltanmf.api.run_onestage_deltanmf",
            "S_E_source": "DeltaNMF scGPT resource script",
            "S_att_source": "scGPT Tutorial_Attention_GRN recipe (+ symmetrize, kNN)",
            "rel_alpha": rel_alpha,
            "n_components": n_components,
            "n_genes": len(nmf_arm.fit.gene_names),
            "n_cells": int(X_cg.shape[0]),
            "primary_alpha": primary_alpha,
            "reading": {
                "fm_helps": "NMF → Embedding-NMF (DeltaNMF)",
                "context_helps": "Embedding-NMF → Attention-NMF (novel prior)",
                "complementary": "Attention-NMF → Combined-NMF (α sweep)",
            },
        },
    )


def format_comparison(result: ExperimentResult) -> str:
    lines = [
        "DeltaNMF run_onestage_deltanmf × {no FM | S_E | S_att | α-mix}",
        f"  {'model':<42} {'recon_mse':>12}",
    ]
    for arm in (result.nmf, result.embedding, result.attention, result.combined):
        lines.append(f"  {arm.name:<42} {arm.recon:12.6f}")
    if result.alpha_sweep:
        lines.append("  α sweep (Combined; S = α S_E + (1-α) S_att):")
        for arm in result.alpha_sweep:
            lines.append(f"    α={arm.alpha:<6} recon_mse={arm.recon:.6f}")
    lines.append(
        "  recon_mse is on DeltaNMF's normalized X; a graph prior raises it by design, "
        "so it does not rank the priors."
    )
    return "\n".join(lines)
