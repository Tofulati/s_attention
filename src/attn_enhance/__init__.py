"""Context-aware scGPT attention prior plugged into DeltaNMF.

Embedding-NMF = DeltaNMF (their ``S_E`` + their solver).
Attention-NMF = same DeltaNMF solver with attention similarity as ``S_E``.
"""

from .prior import GenePrior, ScgptPriors
from .extract_scgpt import (
    extract_scgpt_priors,
    load_scgpt_model,
    paper_attention_map,
    tokenize_cells,
)
from .scgpt_attention import (
    LoadReport,
    attention_logits,
    load_pretrained_scgpt,
    rank_normalize_attention,
)
from .deltanmf_bridge import build_S_E_scgpt, fit_deltanmf, DeltaNMFFit, write_S_E_files
from .gene_graph import (
    affinity_from_attention,
    combine_similarities,
    normalize_graph,
    differential_attention_similarity,
    graph_laplacian,
    graph_smoothness,
    top_edges,
    upper_triangle_correlation,
)
from .compare import (
    run_program_discovery_experiment,
    format_comparison,
    ExperimentResult,
    ArmResult,
    DEFAULT_ALPHAS,
)
from .synthetic import (
    SyntheticAtlas,
    make_synthetic_expression,
    genes_from_scgpt_vocab,
    PREFERRED_MARKER_GENES,
)
from .plotting import (
    plot_synthetic_atlas,
    plot_prior_edge_diagnostics,
    plot_prior_comparison,
    plot_recon_comparison,
    plot_program_loadings,
    plot_top_genes_comparison,
    plot_cell_program_usage,
    plot_arm_W_similarity,
)

__all__ = [
    "GenePrior",
    "ScgptPriors",
    "extract_scgpt_priors",
    "load_scgpt_model",
    "paper_attention_map",
    "tokenize_cells",
    "LoadReport",
    "attention_logits",
    "load_pretrained_scgpt",
    "rank_normalize_attention",
    "build_S_E_scgpt",
    "fit_deltanmf",
    "DeltaNMFFit",
    "write_S_E_files",
    "affinity_from_attention",
    "combine_similarities",
    "normalize_graph",
    "differential_attention_similarity",
    "graph_laplacian",
    "graph_smoothness",
    "top_edges",
    "upper_triangle_correlation",
    "run_program_discovery_experiment",
    "format_comparison",
    "ExperimentResult",
    "ArmResult",
    "DEFAULT_ALPHAS",
    "SyntheticAtlas",
    "make_synthetic_expression",
    "genes_from_scgpt_vocab",
    "PREFERRED_MARKER_GENES",
    "plot_synthetic_atlas",
    "plot_prior_edge_diagnostics",
    "plot_prior_comparison",
    "plot_recon_comparison",
    "plot_program_loadings",
    "plot_top_genes_comparison",
    "plot_cell_program_usage",
    "plot_arm_W_similarity",
]
