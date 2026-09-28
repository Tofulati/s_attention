# Project Title

**Context-Aware Gene Program Discovery Using scGPT Attention-Derived Gene Relationships and Non-Negative Matrix Factorization**

---

# 1. Project Overview

## 1.1 Background

Single-cell RNA-sequencing data contain complex patterns of gene expression that can be difficult to organize into biologically coherent gene programs. Non-negative matrix factorization (NMF) provides a way to identify latent gene programs from an expression matrix, but expression reconstruction alone does not necessarily ensure that genes belonging to the same biological program are grouped together.

Single-cell foundation models such as scGPT provide additional information about genes beyond their observed expression in a particular dataset. In particular, pretrained gene embeddings provide a static representation of gene relationships, while transformer attention provides contextual information that depends on the cellular input.

DeltaNMF provides a framework for incorporating gene relationships derived from scGPT gene embeddings into NMF through graph-based regularization.

This project investigates whether **contextual gene relationships derived from scGPT attention can provide an additional, dataset-specific prior for NMF**.

---

# 2. Central Research Question

> **Can contextual gene relationships encoded by a single-cell foundation model improve the discovery of biologically coherent and condition-specific gene programs by NMF, compared with a static foundation-model gene-embedding prior?**

---

# 3. Scientific Motivation

The project is based on the distinction between three types of information:

### Expression-based information

$$
S^{expr}_{ij}
$$

represents relationships observed directly in the dataset, such as expression correlation.

This asks:

> **Which genes vary together in the observed data?**

### Static foundation-model information

$$
S^{emb}_{ij}
$$

is derived from the pretrained scGPT gene embeddings.

This asks:

> **Which genes are represented as similar by the pretrained foundation model?**

### Contextual foundation-model information

$$
S^{att}_{ij}(D)
$$

is derived from scGPT attention and aggregated over cells in dataset/context \(D\).

This asks:

> **Which gene relationships does the foundation model represent in this particular cellular context?**

The central hypothesis is that the contextual representation contains information that is not completely captured by either expression relationships or static gene embeddings.

---

# 4. Main Hypothesis

> **Static scGPT gene embeddings provide a global gene-level prior, whereas scGPT attention provides a context-dependent prior reflecting relationships represented by the model in a specific cellular population or biological condition. Incorporating this contextual prior into NMF may improve the identification of biologically coherent and condition-specific gene programs.**

---

# 5. Conceptual Model

The project compares four increasingly informed models:

$$
\boxed{\text{NMF}}
$$

$$
\boxed{\text{NMF}+\text{static scGPT embedding prior}}
$$

$$
\boxed{\text{NMF}+\text{contextual scGPT attention prior}}
$$

$$
\boxed{\text{NMF}+\text{static}+\text{contextual prior}}
$$

These comparisons allow the project to ask three separate questions:

### Question 1 — Does foundation-model information help?

$$
NMF
\rightarrow
Embedding\text{-}NMF
$$

### Question 2 — Does contextual information add information?

$$
Embedding\text{-}NMF
\rightarrow
Attention\text{-}NMF
$$

### Question 3 — Are static and contextual information complementary?

$$
Attention\text{-}NMF
\rightarrow
Combined\text{-}NMF
$$

---

# 6. Mathematical Framework

## 6.1 Expression matrix

Let

$$
X\in\mathbb{R}_+^{N\times G}
$$

represent the single-cell expression matrix, where:

* \(N\) = number of cells
* \(G\) = number of genes.

NMF seeks:

$$
X\approx WH
$$

where

$$
W\in\mathbb{R}_+^{G\times K}
$$

contains gene-program memberships and

$$
H\in\mathbb{R}_+^{K\times N}
$$

contains program activity across cells.

---

# 7. Static scGPT Prior

For each gene \(i\), obtain its pretrained scGPT embedding:

$$
e_i\in\mathbb{R}^{d}.
$$

Construct a gene similarity matrix:

$$
S^{emb}_{ij}
=
\operatorname{sim}(e_i,e_j).
$$

Following the DeltaNMF-style construction:

$$
S^{emb}_{ij}
=
\max(0,\operatorname{corr}(e_i,e_j)).
$$

Construct the graph Laplacian:

$$
D_{ii}=\sum_jS_{ij}
$$

and

$$
L=D-S.
$$

The embedding-informed objective becomes:

$$
\mathcal{L}_{emb}
=
\frac12\|X-WH\|_F^2
+
\lambda_{emb}
\operatorname{Tr}(W^TL_{emb}W).
$$

The graph regularization encourages genes considered similar by scGPT to have similar NMF program memberships.

---

# 8. Contextual scGPT Attention Prior

For each cell \(c\), transformer layer \(l\), and attention head \(h\), obtain:

$$
A^{(c,l,h)}_{ij}.
$$

Here:

* \(c\) = cell
* \(l\) = transformer layer
* \(h\) = attention head
* \(i\) = query gene
* \(j\) = key gene.

The attention matrix represents how information is routed between genes within a particular cellular input.

Unlike the static embedding:

$$
S^{emb}_{ij}
$$

the attention relationship can vary with the input cell.

---

# 9. Attention Aggregation

Define a cell-level attention matrix:

$$
A^{(c)}
=
\operatorname{Aggregate}_{l,h}
A^{(c,l,h)}.
$$

The initial baseline can use:

$$
A^{(c)}
=
\frac{1}{LH}
\sum_l\sum_h
A^{(c,l,h)}.
$$

However, layer-specific, head-specific, or weighted aggregation should be treated as experimental variables rather than assumed to be optimal.

---

# 10. Dataset-Specific Attention Graph

For a dataset \(D\) containing \(N_D\) cells:

$$
\bar A_{ij}
=
\frac{1}{N_D}
\sum_{c\in D}A^{(c)}_{ij}.
$$

Because NMF graph regularization naturally treats relationships symmetrically, define:

$$
S^{att}_{ij}
=
\frac12
\left(
\bar A_{ij}
+
\bar A_{ji}
\right).
$$

This should be interpreted as a **contextual gene association**, rather than as evidence of causal regulation.

The resulting matrix is:

$$
S^{att}\in\mathbb{R}^{G\times G}.
$$

---

# 11. Attention Graph Processing

Investigate appropriate normalization and sparsification strategies.

Potential normalization:

$$
\tilde S_{ij}
=
\frac{S_{ij}}
{\sum_jS_{ij}+\epsilon}.
$$

Potential graph sparsification:

$$
S^{att}_{ij}
=
\begin{cases}
S^{att}_{ij}, & j\in kNN(i)\\
0, & \text{otherwise}.
\end{cases}
$$

The dry run should test whether graph density and normalization substantially affect NMF results.

---

# 12. Combined Prior

The simplest combined model is:

$$
S^{combined}
=
\alpha S^{emb}
+
(1-\alpha)S^{att}
$$

where

$$
0\leq\alpha\leq1.
$$

Possible values:

$$
\alpha\in
\{0,0.25,0.5,0.75,1\}.
$$

Interpretation:

| \(\alpha\) | Prior                   |
| ---------: | ----------------------- |
|          1 | Static embedding only   |
|       0.75 | Mostly static           |
|       0.50 | Equal static/contextual |
|       0.25 | Mostly contextual       |
|          0 | Attention only          |

---

# 13. Final NMF Objective

Construct:

$$
D_{ii}=\sum_jS^{combined}_{ij}
$$

and

$$
L=D-S^{combined}.
$$

Then optimize:

$$
\boxed{
\min_{W,H\geq0}
\frac12\|X-WH\|_F^2
+
\lambda
\operatorname{Tr}(W^TLW)
}
$$

or equivalently:

$$
\boxed{
\min_{W,H\geq0}
\frac12\|X-WH\|_F^2
+
\frac{\lambda}{2}
\sum_{i,j}
S_{ij}
\|W_i-W_j\|^2
}
$$

---

# 14. Condition-Specific Extension

If the dataset contains two biological states, such as control and treatment, construct separate contextual graphs:

$$
S^{att}_{control}
$$

and

$$
S^{att}_{treatment}.
$$

Then calculate:

$$
\Delta S^{att}
=
S^{att}_{treatment}
-
S^{att}_{control}.
$$

This asks:

> **Which gene relationships does the foundation model represent differently between biological states?**

Potential downstream analysis:

$$
\Delta S^+_{ij}
=
\max(\Delta S_{ij},0)
$$

and

$$
\Delta S^-_{ij}
=
\max(-\Delta S_{ij},0).
$$

The initial experiment should preferably use the simpler condition-specific graphs rather than directly inserting \(\Delta S\) into a standard positive similarity graph.

---

# 15. Experimental Design

## Dataset

Specify:

* Dataset:
* Number of cells:
* Number of genes:
* Species:
* Tissue:
* Cell types:
* Conditions:
* Control:
* Treatment/perturbation:
* Relevant biological question:

---

# 16. Models to Compare

### Model 1 — Expression-only NMF

$$
X\approx WH
$$

No foundation-model prior.

### Model 2 — Embedding-NMF

$$
X\approx WH
$$

with:

$$
S=S^{emb}.
$$

### Model 3 — Attention-NMF

$$
X\approx WH
$$

with:

$$
S=S^{att}.
$$

### Model 4 — Combined-NMF

$$
X\approx WH
$$

with:

$$
S=
\alpha S^{emb}
+
(1-\alpha)S^{att}.
$$

---

# 17. Primary Evaluation Questions

The dry run should explicitly answer:

### A. Does scGPT attention contain context-dependent information?

Test whether:

$$
S^{att}_{context1}
\neq
S^{att}_{context2}.
$$

For example:

$$
S^{att}_{Tcell}
\neq
S^{att}_{Macrophage}.
$$

Or:

$$
S^{att}_{resting}
\neq
S^{att}_{activated}.
$$

### B. Does attention add information beyond static embeddings?

Compare:

$$
S^{att}
\quad\text{vs.}\quad
S^{emb}.
$$

Measure their similarity/correlation and identify relationships captured by one but not the other.

### C. Does the attention prior improve NMF?

Compare:

$$
NMF
$$

against

$$
Embedding\text{-}NMF
$$

against

$$
Attention\text{-}NMF
$$

against

$$
Combined\text{-}NMF.
$$

### D. Are the resulting programs biologically coherent?

Evaluate gene programs using independent biological evidence rather than only reconstruction loss.

Potential evaluation categories:

* known marker genes
* pathway enrichment
* gene ontology enrichment
* known biological programs
* program specificity
* reproducibility across cells/datasets
* robustness across NMF initializations

### E. Does contextual attention improve condition specificity?

Determine whether genes/programs associated with different conditions become more distinguishable under the contextual prior.

---

# 18. Critical Ablations

The dry run should include:

### Prior type

* No prior
* Static embedding
* Attention
* Combined

### Attention aggregation

* Last layer
* Intermediate layers
* All layers
* Individual heads
* Averaged heads
* Weighted layers/heads

### Graph construction

* Dense
* kNN
* Different \(k\)

### Prior strength

$$
\lambda
$$

and

$$
\alpha.
$$

### NMF rank

$$
K.
$$

### Random initialization

Run multiple NMF initializations to determine whether the observed programs are stable.

---

# 19. Key Controls

The dry run should explicitly include controls that determine whether the attention prior is genuinely useful.

### Control 1 — Random graph

Replace the attention graph with a random graph having comparable density.

### Control 2 — Expression correlation graph

Construct:

$$
S^{corr}_{ij}
=
\operatorname{corr}(X_{\cdot i},X_{\cdot j}).
$$

Compare:

$$
S^{corr}
$$

with:

$$
S^{emb}
$$

and

$$
S^{att}.
$$

### Control 3 — Static versus contextual

Directly compare:

$$
S^{emb}
$$

and

$$
S^{att}.
$$

This is particularly important because the main scientific claim is that contextual information adds something beyond the static foundation-model representation.

---

# 20. Important Methodological Caveat

The attention graph is derived from the same dataset that is subsequently factorized.

Therefore, the attention prior should not automatically be described as a completely independent external prior.

A more precise description is:

> **a foundation-model-derived, dataset-informed contextual prior.**

The dry run should consider whether this creates information leakage or circularity.

Potential mitigation strategies include:

* deriving the attention graph from held-out cells;
* cross-fitting;
* separating cells used to construct the graph from cells used to evaluate NMF;
* testing whether the attention prior provides information beyond expression correlation.

---

# 21. Initial Implementation Strategy

Do not initially fine-tune scGPT.

Start with:

$$
\boxed{
\text{Frozen pretrained scGPT}
\rightarrow
\text{attention extraction}
\rightarrow
\text{contextual graph}
\rightarrow
\text{NMF}
}
$$

This isolates the effect of the contextual attention prior.

---

# 22. High-Level Algorithm

```python
# 1. Load expression data
X = expression_matrix

# 2. Load frozen pretrained scGPT
model = load_scGPT()

# 3. Construct static embedding prior
E = model.gene_embedding_matrix()

S_emb = compute_embedding_similarity(E)
S_emb = normalize_graph(S_emb)

# 4. Construct contextual attention prior
attention_matrices = []

for cell in cells:

    tokens = tokenize_genes(cell)
    expression = preprocess_expression(cell)

    outputs = model(
        tokens,
        expression,
        return_attention=True
    )

    A_cell = aggregate_attention(outputs)
    A_cell = symmetrize(A_cell)

    attention_matrices.append(A_cell)

# 5. Aggregate across cells
S_att = mean(attention_matrices)

# 6. Normalize/sparsify
S_att = normalize_graph(S_att)

# 7. Combine priors
S = alpha * S_emb + (1 - alpha) * S_att

# 8. Construct graph Laplacian
D = diag(sum(S, axis=1))
L = D - S

# 9. Run graph-regularized NMF
W, H = graph_regularized_NMF(
    X,
    L,
    lambda_=lambda_
)

# 10. Evaluate programs
evaluate_gene_programs(W, H)
```

---

# 23. Dry-Run Objective

The first dry run should **not** attempt to prove the complete biological hypothesis.

Instead, establish the pipeline:

$$
\boxed{
X
\rightarrow
scGPT
\rightarrow
A
\rightarrow
S^{att}
\rightarrow
L
\rightarrow
NMF
\rightarrow
W,H
}
$$

and answer the following feasibility questions:

1. Can attention be extracted from the chosen scGPT model?
2. Can attention be mapped consistently back to gene identities?
3. Can attention be aggregated into a meaningful \(G\times G\) graph?
4. Does the graph differ between biological contexts?
5. Can the graph be incorporated into the NMF objective?
6. Does the resulting NMF converge?
7. Are the resulting programs stable?
8. Does the contextual prior produce programs that differ from the static-prior baseline?
9. Are those differences biologically interpretable?

---

# 24. Expected Contribution

The proposed contribution is a modification of the DeltaNMF framework in which the foundation-model prior is no longer restricted to static gene embeddings.

Conceptually:

$$
\boxed{
\text{Static gene representation}
\rightarrow
\text{Contextual gene relationship}
}
$$

followed by:

$$
\boxed{
\text{Contextual gene relationship}
\rightarrow
\text{NMF structural prior}
}
$$

The resulting framework tests whether information encoded dynamically by a single-cell foundation model can improve latent gene-program discovery.

---

# 25. One-Sentence Project Statement

> **We investigate whether contextual gene relationships encoded in a single-cell foundation model can be converted into a dataset-specific graph prior that improves non-negative matrix factorization for discovering biologically coherent and condition-specific gene programs.**

---

# 26. Open Questions for the Agent

The writing/research agent should **not assume these answers**. It should explicitly investigate them during the dry run:

1. Which scGPT model/checkpoint should be used?
2. How exactly are attention matrices exposed by the implementation?
3. Which layers contain the most useful information?
4. Which attention heads should be used?
5. Should attention be averaged, weighted, or selected?
6. How should genes absent from a cell's input sequence be handled?
7. Should attention be symmetrized?
8. How should attention be normalized?
9. Should the graph be dense or sparse?
10. How should attention-derived similarity be compared with expression correlation?
11. How should the graph regularization parameter \(\lambda\) be selected?
12. How should \(\alpha\) be selected?
13. What biological benchmarks should define program quality?
14. How should information leakage from using the same cells for attention extraction and NMF be addressed?
15. Does contextual attention actually contain information beyond static scGPT embeddings?
16. Does that additional information improve gene-program discovery?

---

# 27. Final Research Logic

The entire project should follow this progression:

$$
\boxed{
\text{Can scGPT attention vary with cellular context?}
}
$$

↓

$$
\boxed{
\text{Does that contextual information contain information beyond static embeddings?}
}
$$

↓

$$
\boxed{
\text{Can it be represented as a useful gene-gene graph?}
}
$$

↓

$$
\boxed{
\text{Can that graph improve NMF gene-program discovery?}
}
$$

↓

$$
\boxed{
\text{Do changes in contextual relationships correspond to biological-state-specific programs?}
}
$$

This progression should be preserved throughout the dry run so that the project does not jump directly from “attention exists” to “attention improves biology.”
