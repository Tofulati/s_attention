# Context-aware foundation-model priors for gene-program discovery

Convert **scGPT** gene relationships into graph priors for **NMF**, and ask whether a **contextual attention prior** improves program discovery beyond a **static embedding prior** (DeltaNMF-style).

Concept writeup: [`docs/CONCEPT.md`](docs/CONCEPT.md)

## Scientific question

> Can contextual information from a frozen single-cell foundation model improve biologically coherent gene-program discovery by NMF, compared with a static FM gene-embedding prior?

## Experimental matrix

| Model | Expression | $S^{emb}$ | $S^{att}$ |
|---|---|---|---|
| NMF | ✓ | — | — |
| Embedding-NMF | ✓ | ✓ | — |
| Attention-NMF | ✓ | — | ✓ |
| Combined-NMF | ✓ | $\alpha$ | $1-\alpha$ |

Sweep $\alpha\in\{0,0.25,0.5,0.75,1\}$.

## Contained environment

```bash
cd /Users/albert/Documents/attention
bash scripts/setup_env.sh          # creates .venv, installs deps, registers kernel
source .venv/bin/activate
# or, if the venv already exists:
pip install -e ".[notebooks]" -r requirements.txt
pip install -e "third_party/scGPT" --no-deps
```

Select Jupyter kernel **Python (attention)**.

Open [`notebooks/01_scgpt_attention_vs_embeddings_nmf.ipynb`](notebooks/01_scgpt_attention_vs_embeddings_nmf.ipynb) for the taught end-to-end sample.

## Layout

| Path | Role |
|---|---|
| `idea.md` | Scientific design (source of truth) |
| `third_party/deltanmf` | **Embedding-NMF** = DeltaNMF (`S_E` + solver) |
| `third_party/scGPT` | scGPT + Attention-GRN tutorial (`Wqkv`) |
| `src/attn_enhance/deltanmf_bridge.py` | Thin wrapper: call DeltaNMF, build `S_E` their way |
| `src/attn_enhance/scgpt_attention.py` | Checkpoint → scGPT's own `TransformerModel` (flash-attn `Wqkv` renamed to `in_proj`); tutorial `Q K^T` extraction |
| `src/attn_enhance/extract_scgpt.py` | **Novel:** `S_att` via the Attention-GRN tutorial recipe (+ DeltaNMF `S_E` for alignment) |
| `tests/` | Weights vs checkpoint, torch attention vs flash `Wqkv` formula, rank-norm vs tutorial code, pipeline smoke (`pytest tests`) |
| `src/attn_enhance/compare.py` | Same `run_onestage_deltanmf` pipeline for every arm; swap / mix priors |
| `notebooks/01_scgpt_attention_vs_embeddings_nmf.ipynb` | Dry-run workbook |
| `checkpoints/scgpt/` | Pretrained scGPT folder |

**Division of labor:** Embedding-NMF is DeltaNMF. The only new content is the attention-derived graph plugged into that same solver.

## Smoke test

**Dry-run** (no biology; random weights):

```bash
source .venv/bin/activate
PYTHONPATH=src python -c "
from attn_enhance import (
    make_synthetic_expression, extract_scgpt_priors,
    run_program_discovery_experiment, format_comparison,
)
a = make_synthetic_expression(n_cells=20, n_genes=24, seed=0)
priors = extract_scgpt_priors(a.X, a.gene_names, 'checkpoints/scgpt', dry_run=True)
r = run_program_discovery_experiment(a.X, a.gene_names, priors, n_components=4, max_iter=300)
print(format_comparison(r))
"
```

**With checkpoint** (gene symbols must be in `vocab.json`):

```bash
PYTHONPATH=src python -c "
from attn_enhance import (
    genes_from_scgpt_vocab, make_synthetic_expression,
    extract_scgpt_priors, run_program_discovery_experiment, format_comparison,
)
genes = genes_from_scgpt_vocab('checkpoints/scgpt', n_genes=32, seed=0)
a = make_synthetic_expression(n_cells=40, n_genes=len(genes), gene_names=genes, seed=0)
priors = extract_scgpt_priors(a.X, a.gene_names, 'checkpoints/scgpt', max_cells=16, dry_run=False)
print(format_comparison(run_program_discovery_experiment(a.X, a.gene_names, priors, max_iter=300)))
"
```
