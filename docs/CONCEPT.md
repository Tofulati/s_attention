# Concept: Context-aware foundation-model priors for gene-program discovery

**Canonical design:** [`idea.md`](../idea.md).

**Implementation rule:** Embedding-NMF **is** [DeltaNMF](../third_party/deltanmf) (their scGPT `S_E` + their solver). The only new content in this repo is the **attention-derived** similarity `S_att`, plugged into that same DeltaNMF path. Do not reimplement Embedding-NMF.

We investigate whether **contextual gene relationships** encoded by a frozen single-cell foundation model (scGPT) can be converted into a **dataset-specific graph prior** that improves non-negative matrix factorization for discovering biologically coherent gene programs—beyond a **static** gene-embedding prior (DeltaNMF’s default).

## One-sentence story

> Static gene representation → contextual gene relationship → NMF structural prior.

## Core scientific question

> Can contextual information encoded by a single-cell foundation model improve the discovery of biologically coherent and condition-specific gene programs by NMF, compared with a static foundation-model gene-embedding prior?

**Hypothesis.** Static scGPT gene embeddings provide a **global gene-level prior**. scGPT attention provides a **context-dependent prior** that reflects relationships the model represents in the cellular population being analyzed. Incorporating that contextual prior into NMF may improve identification of coherent / condition-specific programs.

## Progressive models

| Model | Expression | Embedding prior $S^{emb}$ | Attention prior $S^{att}$ |
|---|---|---|---|
| NMF | ✓ | — | — |
| Embedding-NMF | ✓ | ✓ | — |
| Attention-NMF | ✓ | — | ✓ |
| Combined-NMF | ✓ | ✓ | ✓ |

Combined prior:

$$
S = \alpha\, S^{emb} + (1-\alpha)\, S^{att},\qquad 0\le\alpha\le 1
$$

| $\alpha$ | Meaning |
|---:|---|
| 1 | Static embedding only (DeltaNMF default) |
| 0.75 | Mostly global |
| 0.5 | Equal global / contextual |
| 0.25 | Mostly contextual |
| 0 | Attention only |

## Biological problem

Given $X\in\mathbb{R}_+^{N\times G}$ (cells × genes), discover programs

$$
X \approx WH,\quad W\in\mathbb{R}_+^{G\times K},\quad H\in\mathbb{R}_+^{K\times N}.
$$

Expression reconstruction alone need not yield biologically coherent programs: correlated expression can be statistical; biologically related genes can be split by sparsity / dropout. A foundation-model graph prior regularizes program memberships $W$.

## Global prior (DeltaNMF / scGPT embeddings)

From frozen scGPT:

$$
e_i = \texttt{model.encoder}(\text{gene}_i),\qquad
S^{emb}_{ij} = \max\bigl(0,\operatorname{corr}(e_i,e_j)\bigr).
$$

Interpretation: *genes $i,j$ have similar pretrained representations* (static; no expression context).

## Contextual prior (scGPT attention)

Per cell $c$, layer $l$, head $h$:

$$
A^{(c,l,h)}_{ij}
=
\text{attention of query gene } i \text{ to key gene } j.
$$

scGPT’s FlashAttention path does **not** return scores, and flash-attn does not run on CPU. `src/attn_enhance/scgpt_attention.py` is a pure-PyTorch replica of the flash-attn layers scGPT was pretrained with (`FlashscGPTMHA`, from the `dev-temp` branch), with identical parameter names. From it we take the raw $Q K^\top$ of `Wqkv` in the chosen layer, then rank-normalize by row and then by column, and average over heads. This is the paper's Methods recipe and the one in the [Attention GRN tutorial](https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_Attention_GRN.ipynb). Inputs are binned per cell and prefixed with `<cls>`, as in pretraining.

**Design choice (important ablation):** which layers / heads to keep.

- Default (first experiments): last layer, mean over heads (tutorial default).
- Alternatives: layer-specific $A^{(c,l)}$, head-specific, or weighted $\sum_{l,h}\alpha_{lh} A^{(c,l,h)}$.

Aggregate over cells, then treat attention as **association** (not regulatory causality):

$$
\bar A_{ij} = \frac1N\sum_c A^{(c)}_{ij},\qquad
S^{att}_{ij} = \tfrac12(\bar A_{ij}+\bar A_{ji}).
$$

Interpretation: *across cells in this dataset, the model repeatedly represents gene $i$ as attending to gene $j$.*

Optional: $k$NN sparsify and/or row-normalize so hub genes do not dominate.

**Protocol:** frozen pretrained scGPT → dataset-specific attention → NMF. Do **not** fine-tune scGPT in the first experiment (avoids confounding adaptation with the prior itself).

## Central objective

$$
\min_{W,H\ge 0}
\tfrac12\|X-WH\|_F^2
+
\lambda\,\operatorname{Tr}(W^\top L W)
=
\tfrac12\|X-WH\|_F^2
+
\tfrac{\lambda}{2}\sum_{i,j} S_{ij}\,\|W_i-W_j\|^2
$$

with $L = D-S$ (or the normalized Laplacian). Genes the prior links are pushed toward similar program memberships.

## Hypotheses

1. **H1 (contextual signal).** $S^{att}$ contains condition-relevant structure not captured by $S^{emb}$.
2. **H2 (coherence).** Adding $S^{att}$ improves biological coherence of programs vs expression-only NMF.
3. **H3 (specificity).** Condition-specific attention yields more state-specific programs than a static embedding prior.
4. **H4 (complementarity).** Some $\alpha\in(0,1)$ can outperform either prior alone.

## Experimental reading

- Does the FM help? NMF → Embedding-NMF
- Does context help? Embedding-NMF → Attention-NMF
- Are they complementary? Attention-NMF → Combined-NMF ($\alpha$ sweep)

## Stage 2 (later): condition-specific $\Delta S$

For control vs treatment, build $S^{att}_{ctrl}$ and $S^{att}_{treat}$, then $\Delta S = S^{att}_{treat}-S^{att}_{ctrl}$. Split $\Delta S^\pm = \max(\pm\Delta S,0)$ before any regularizer—raw $\Delta S$ is not a valid similarity graph. First ship the simpler contextual prior above.

## Paper anchors

- **scGPT** — [Nature Methods](https://www.nature.com/articles/s41592-024-02201-0); attention extraction via `Wqkv`.
- **DeltaNMF** — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.22.701049v1); embedding → $S_E$ → graph-regularized NMF.

## What this repo implements

1. Frozen scGPT (or dry-run) → $S^{emb}$ and $S^{att}$ on an aligned gene set.
2. Experimental matrix: NMF / Embedding-NMF / Attention-NMF / Combined-NMF.
3. $\alpha$ continuum and layer-index choice for attention.
4. Stage-2 helper for $\Delta S^\pm$ (not yet the full differential NMF loss).
