# scGPT checkpoints

Place a pretrained scGPT model directory here, for example:

```
checkpoints/scgpt/
  args.json
  best_model.pt
  vocab.json
```

## Where to get weights

Official pretrained models are listed in the scGPT repo:

- https://github.com/bowang-lab/scGPT#pretrained-scgpt-models

Common choices:

- `scGPT_human` (whole-body)
- blood / organ-specific fine-tunes for attention-GRN style analyses

After download, point the notebook at this folder:

```python
SCGPT_DIR = ROOT / "checkpoints" / "scgpt"
priors = extract_scgpt_priors(X, gene_names, SCGPT_DIR, layer_index=11)
```

Gene names must be **symbols** present in `vocab.json`.

## Without weights

```python
priors = extract_scgpt_priors(X, gene_names, "unused", dry_run=True)
```

runs the same Q@K.T + encoder extraction math with random weights (plumbing only).
