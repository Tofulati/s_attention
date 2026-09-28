"""scGPT path vs the vendored package, checkpoint, and tutorial notebook."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attn_enhance.scgpt_attention import (  # noqa: E402
    load_pretrained_scgpt,
    rank_normalize_attention,
)

CKPT_DIR = ROOT / "checkpoints" / "scgpt"
TUTORIAL = ROOT / "third_party" / "scGPT" / "tutorials" / "Tutorial_Attention_GRN.ipynb"
needs_ckpt = pytest.mark.skipif(
    not (CKPT_DIR / "best_model.pt").exists(), reason="no scGPT checkpoint"
)


@pytest.fixture(scope="module")
def pretrained():
    model, vocab, cfg, report = load_pretrained_scgpt(CKPT_DIR, device="cpu")
    ckpt = torch.load(CKPT_DIR / "best_model.pt", map_location="cpu", weights_only=True)
    return model, cfg, report, ckpt


@needs_ckpt
def test_every_used_weight_is_the_checkpoint_tensor(pretrained):
    model, _, report, ckpt = pretrained
    for key, value in model.state_dict().items():
        if not key.startswith(("encoder.", "value_encoder.", "transformer_encoder.")):
            continue
        src = key.replace("self_attn.in_proj_weight", "self_attn.Wqkv.weight").replace(
            "self_attn.in_proj_bias", "self_attn.Wqkv.bias"
        )
        assert torch.equal(value, ckpt[src]), key
    assert {k.split(".")[0] for k in report.skipped_checkpoint_keys} <= {"flag_encoder", "mvc_decoder"}
    assert {k.split(".")[0] for k in report.uninitialized_model_keys} <= {"cls_decoder"}


@needs_ckpt
def test_torch_attention_equals_flash_wqkv_formula(pretrained):
    model, cfg, _, ckpt = pretrained
    i = cfg["nlayers"] - 1
    attn = model.transformer_encoder.layers[i].self_attn
    prefix = f"transformer_encoder.layers.{i}.self_attn."
    torch.manual_seed(0)
    h = torch.randn(2, 9, cfg["embsize"])

    qkv = F.linear(h, ckpt[prefix + "Wqkv.weight"], ckpt[prefix + "Wqkv.bias"])
    q, k, v = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=cfg["nheads"]).unbind(2)
    probs = torch.softmax(torch.einsum("bshd,bthd->bhst", q, k) / math.sqrt(q.shape[-1]), dim=-1)
    context = rearrange(torch.einsum("bhst,bthd->bshd", probs, v), "b s h d -> b s (h d)")
    flash = F.linear(context, ckpt[prefix + "out_proj.weight"], ckpt[prefix + "out_proj.bias"])

    with torch.no_grad():
        torch_out = attn(h, h, h, need_weights=False)[0]
    torch.testing.assert_close(torch_out, flash, atol=1e-4, rtol=1e-4)


def _tutorial_rank_normalization_source() -> str:
    cells = json.loads(TUTORIAL.read_text())["cells"]
    src = next("".join(c["source"]) for c in cells if "# Rank normalization by row" in "".join(c["source"]))
    start = src.index("        # Rank normalization by row")
    end = src.index("        attn_scores = attn_scores.mean(1)")
    body = src[start : end + len("        attn_scores = attn_scores.mean(1)")]
    return "\n".join(line[8:] for line in body.splitlines())


def test_rank_normalization_is_the_tutorial_code():
    torch.manual_seed(4)
    m = 11
    scores = torch.randn(2, 8, m, m)
    namespace = {"torch": torch, "attn_scores": scores.clone(), "M": m}
    exec(_tutorial_rank_normalization_source(), namespace)
    torch.testing.assert_close(rank_normalize_attention(scores), namespace["attn_scores"])
