"""End-to-end plumbing on a random-weight scGPT and DeltaNMF's one-stage pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attn_enhance import (  # noqa: E402
    extract_scgpt_priors,
    make_synthetic_expression,
    run_program_discovery_experiment,
)


def test_dry_run_priors_feed_every_deltanmf_arm():
    atlas = make_synthetic_expression(n_cells=30, n_genes=20, seed=0)
    priors = extract_scgpt_priors(atlas.X, atlas.gene_names, ROOT / "checkpoints" / "scgpt", dry_run=True)
    result = run_program_discovery_experiment(
        atlas.X, atlas.gene_names, priors, n_components=3, max_iter=300, alphas=None
    )

    for arm in (result.nmf, result.embedding, result.attention, result.combined):
        assert arm.fit.W.shape == (20, 3)
        assert arm.fit.H.shape == (3, 30)
        assert arm.fit.gene_names == priors.gene_names
    assert not result.nmf.fit.meta["use_fm"]
    assert result.attention.fit.meta["use_fm"]
    assert not np.allclose(result.attention.fit.W, result.nmf.fit.W)
