"""Equivalence tests: scGPT replica vs independent reference implementations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attn_enhance.scgpt_attention import (  # noqa: E402
    ScGPTGenerator,
    ScGPTLayer,
    ScGPTMultiheadAttention,
    bin_expression,
    load_pretrained_scgpt,
    rank_normalize_attention,
)

CKPT_DIR = ROOT / "checkpoints" / "scgpt"
D, H, FF = 32, 4, 48


def _torch_reference(layer: ScGPTLayer) -> nn.TransformerEncoderLayer:
    """Stock PyTorch post-norm layer carrying the replica's weights (Wqkv → in_proj)."""
    ref = nn.TransformerEncoderLayer(
        layer.self_attn.embed_dim,
        layer.self_attn.num_heads,
        layer.linear1.out_features,
        dropout=0.2,
        batch_first=True,
    )
    sd = {
        k.replace("self_attn.Wqkv.weight", "self_attn.in_proj_weight").replace(
            "self_attn.Wqkv.bias", "self_attn.in_proj_bias"
        ): v
        for k, v in layer.state_dict().items()
    }
    ref.load_state_dict(sd, strict=True)
    return ref.eval()


def _upstream_cross_attention(gen_qkv, pcpt_qkv, pcpt_keep, gen_keep, num_heads):
    """``FlashscGPTMHA`` generative path as written upstream (identity-projection MHA)."""
    cross_q = rearrange(gen_qkv[:, :, 0], "b s h d -> b s (h d)")
    cross_kv = rearrange(
        torch.cat([pcpt_qkv[:, :, 1:], gen_qkv[:, :, 1:]], dim=1), "b s two h d -> b s two (h d)"
    )
    q_len, k_len = cross_q.shape[1], cross_kv.shape[1]
    attn_mask = torch.zeros((q_len, k_len), dtype=torch.bool)
    attn_mask[:, -q_len:] = ~torch.eye(q_len, dtype=torch.bool)
    key_padding_mask = ~torch.cat([pcpt_keep, gen_keep], dim=1)
    e = cross_q.shape[-1]
    eye = torch.eye(e)
    out, _ = F.multi_head_attention_forward(
        cross_q.transpose(0, 1),
        cross_kv[:, :, 0].transpose(0, 1),
        cross_kv[:, :, 1].transpose(0, 1),
        e,
        num_heads,
        in_proj_weight=None,
        in_proj_bias=None,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=eye,
        out_proj_bias=None,
        training=False,
        key_padding_mask=key_padding_mask,
        need_weights=False,
        attn_mask=attn_mask,
        use_separate_proj_weight=True,
        q_proj_weight=eye,
        k_proj_weight=eye,
        v_proj_weight=eye,
    )
    return out.transpose(0, 1)


def test_known_gene_layer_matches_torch_encoder_layer():
    torch.manual_seed(0)
    layer = ScGPTLayer(D, H, FF, dropout=0.2).eval()
    ref = _torch_reference(layer)
    x = torch.randn(3, 10, D)

    with torch.no_grad():
        out, gen = layer(x)
        expected = ref(x)
    assert gen is None
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_padding_matches_torch_on_valid_tokens():
    torch.manual_seed(1)
    layer = ScGPTLayer(D, H, FF, dropout=0.2).eval()
    ref = _torch_reference(layer)
    x = torch.randn(2, 9, D)
    pad = torch.zeros(2, 9, dtype=torch.bool)
    pad[0, 6:] = True

    with torch.no_grad():
        out, _ = layer(x, None, pad)
        expected = ref(x, src_key_padding_mask=pad)
    valid = ~pad
    torch.testing.assert_close(out[valid], expected[valid], atol=1e-5, rtol=1e-5)


def test_generative_path_matches_upstream_cross_attention():
    torch.manual_seed(2)
    mha = ScGPTMultiheadAttention(D, H).eval()
    pcpt, gen = torch.randn(2, 7, D), torch.randn(2, 5, D)
    pcpt_keep = torch.ones(2, 7, dtype=torch.bool)
    pcpt_keep[1, 5:] = False
    gen_keep = torch.ones(2, 5, dtype=torch.bool)

    with torch.no_grad():
        _, gen_out = mha(pcpt, gen, pcpt_keep, gen_keep)
        expected = _upstream_cross_attention(
            mha._qkv(gen), mha._qkv(pcpt), pcpt_keep, gen_keep, H
        )
    torch.testing.assert_close(gen_out, expected, atol=1e-5, rtol=1e-5)


def test_eq11_mask_isolates_unknown_genes():
    torch.manual_seed(3)
    stack = ScGPTGenerator(ScGPTLayer(D, H, FF, dropout=0.2), 3).eval()
    pcpt, gen = torch.randn(1, 6, D), torch.randn(1, 4, D)
    gen_perturbed = gen.clone()
    gen_perturbed[:, 2] += 5.0

    with torch.no_grad():
        p1, g1 = stack(pcpt, gen)
        p2, g2 = stack(pcpt, gen_perturbed)
    torch.testing.assert_close(p1, p2)
    others = [0, 1, 3]
    torch.testing.assert_close(g1[:, others], g2[:, others])
    assert not torch.allclose(g1[:, 2], g2[:, 2])


def test_rank_normalization_matches_tutorial():
    torch.manual_seed(4)
    b, m = 2, 11
    attn_scores = torch.randn(b, 8, m, m)

    x = attn_scores.reshape((-1, m))
    order = torch.argsort(x, dim=1)
    rank = torch.argsort(order, dim=1)
    x = rank.reshape((-1, 8, m, m)) / m
    x = x.permute(0, 1, 3, 2).reshape((-1, m))
    order = torch.argsort(x, dim=1)
    rank = torch.argsort(order, dim=1)
    expected = (rank.reshape((-1, 8, m, m)) / m).permute(0, 1, 3, 2).mean(1)

    torch.testing.assert_close(rank_normalize_attention(attn_scores), expected)


def test_binning_matches_pretraining_collator():
    rng = np.random.default_rng(0)
    row = rng.poisson(2.0, size=200).astype(float)
    n_bins = 51

    nz = row.nonzero()
    edges = np.quantile(row[nz], np.linspace(0, 1, n_bins - 1))
    expected = np.zeros_like(row, dtype=np.int64)
    expected[nz] = np.digitize(row[nz], edges)

    got = bin_expression(row, n_bins)
    np.testing.assert_array_equal(got, expected)
    assert (got[row == 0] == 0).all()
    assert got.max() <= n_bins - 1
    np.testing.assert_array_equal(bin_expression(np.zeros(5), n_bins), np.zeros(5))


@pytest.mark.skipif(not (CKPT_DIR / "best_model.pt").exists(), reason="no scGPT checkpoint")
def test_checkpoint_loads_strictly_and_matches_torch_stack():
    model, vocab, cfg, report = load_pretrained_scgpt(CKPT_DIR, device="cpu")
    ckpt = torch.load(CKPT_DIR / "best_model.pt", map_location="cpu", weights_only=True)

    assert (cfg["embsize"], cfg["nlayers"], cfg["nheads"], cfg["d_hid"]) == (512, 12, 8, 512)
    assert not report.config_deviations_from_paper
    for i, layer in enumerate(model.transformer_encoder.layers):
        assert torch.equal(
            layer.self_attn.Wqkv.weight, ckpt[f"transformer_encoder.layers.{i}.self_attn.Wqkv.weight"]
        )
    assert all(k.startswith(("decoder.", "mvc_decoder.")) for k in report.ignored_checkpoint_keys)

    genes = ["<cls>", "CD3E", "CD19", "LYZ", "MS4A1", "NKG7", "GNLY"]
    ids = torch.tensor([[vocab[g] for g in genes]])
    values = torch.tensor([[-2.0, 3.0, 0.0, 12.0, 7.0, 1.0, 50.0]])
    ref_stack = nn.ModuleList(_torch_reference(layer) for layer in model.transformer_encoder.layers)

    with torch.no_grad():
        out, _ = model(ids, values)
        h, _ = model.embed(ids, values)
        for ref in ref_stack:
            h = ref(h)
    torch.testing.assert_close(out, h, atol=1e-4, rtol=1e-4)
