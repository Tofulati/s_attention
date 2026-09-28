"""Containers for global, contextual, and combined scGPT gene priors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GenePrior:
    """Gene × gene similarity prior from one construction path.

    ``kind`` is one of ``embedding`` (global), ``attention`` (contextual),
    or ``combined`` (S = alpha * S_emb + (1 - alpha) * S_att).
    """

    kind: str
    gene_names: list[str]
    similarity: np.ndarray
    gene_embeddings: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"attention", "embedding", "combined"}:
            raise ValueError(
                f"kind must be 'attention', 'embedding', or 'combined'; got {self.kind!r}"
            )
        S = np.asarray(self.similarity, dtype=np.float64)
        if S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError(f"similarity must be square; got {S.shape}")
        if len(self.gene_names) != S.shape[0]:
            raise ValueError("gene_names length must match similarity size")
        self.similarity = S


@dataclass
class ScgptPriors:
    """Paired global (S_emb) and contextual (S_att) priors, aligned genes.

    Use ``combined(alpha)`` to build S = alpha * S_emb + (1 - alpha) * S_att.
    """

    gene_names: list[str]
    attention: GenePrior
    embedding: GenePrior
    gene_embeddings: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.attention.kind != "attention":
            raise ValueError("attention prior must have kind='attention'")
        if self.embedding.kind != "embedding":
            raise ValueError("embedding prior must have kind='embedding'")
        if list(self.attention.gene_names) != list(self.gene_names):
            raise ValueError("attention.gene_names must match ScgptPriors.gene_names")
        if list(self.embedding.gene_names) != list(self.gene_names):
            raise ValueError("embedding.gene_names must match ScgptPriors.gene_names")

    def combined(self, alpha: float, *, renorm: bool = True) -> GenePrior:
        """Return combined prior at a fixed alpha."""
        from .gene_graph import combine_similarities

        S = combine_similarities(
            self.embedding.similarity,
            self.attention.similarity,
            alpha,
            renorm=renorm,
        )
        return GenePrior(
            kind="combined",
            gene_names=list(self.gene_names),
            similarity=S,
            gene_embeddings=self.gene_embeddings,
            meta={
                **self.meta,
                "alpha": float(alpha),
                "formula": "S = alpha * S_emb + (1 - alpha) * S_att",
            },
        )
