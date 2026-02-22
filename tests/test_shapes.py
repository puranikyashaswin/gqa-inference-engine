"""
Sanity checks for GQA implementation.

Run: python tests/test_shapes.py
"""

import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.attention import GroupedQueryAttention, expand_kv_to_query_heads


def check_expand_kv():
    print("-- expand_kv_to_query_heads --")
    kv = torch.randn(2, 2, 4, 64)
    out = expand_kv_to_query_heads(kv, n_rep=1)
    assert torch.equal(out, kv)
    out = expand_kv_to_query_heads(kv, n_rep=4)
    assert out.shape == (2, 8, 4, 64)
    for h in range(4):
        assert torch.equal(out[:, h], kv[:, 0])
    for h in range(4, 8):
        assert torch.equal(out[:, h], kv[:, 1])
    print("  ok")


def check_shapes_across_configs():
    print("-- output shapes --")
    x = torch.randn(2, 128, 512)
    for H, G, tag in [
        (8, 8, "mha"), (8, 1, "mqa"), (8, 2, "gqa-2"), (8, 4, "gqa-4"), (16, 4, "gqa-16h")
    ]:
        out, _ = GroupedQueryAttention(512, H, G, causal=True)(x)
        assert out.shape == (2, 128, 512), f"{tag}: got {out.shape}"
        print(f"  {tag}: ok")


def check_gradients_flow():
    print("-- gradient flow --")
    x = torch.randn(2, 32, 256, requires_grad=True)
    model = GroupedQueryAttention(256, 8, 2, causal=True)
    out, _ = model(x)
    out.mean().backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"dead gradient: {name}"
        assert not torch.isnan(p.grad).any(), f"nan gradient: {name}"
    print("  ok -- all projections receive gradients")


def check_no_nans():
    print("-- numerical stability --")
    model = GroupedQueryAttention(512, 8, 2, causal=True)
    with torch.no_grad():
        out, _ = model(torch.randn(4, 256, 512))
    assert not torch.isnan(out).any() and not torch.isinf(out).any()
    assert out.abs().max() < 100
    print(f"  ok -- range [{out.min():.3f}, {out.max():.3f}]")


def check_causal_mask():
    print("-- causal masking --")
    torch.manual_seed(42)
    model = GroupedQueryAttention(64, 4, 2, causal=True, dropout=0.0)
    model.eval()
    x = torch.randn(1, 8, 64)
    out_original, _ = model(x)
    x_perturbed = x.clone()
    x_perturbed[:, 5, :] = torch.randn(64)
    out_perturbed, _ = model(x_perturbed)
    for t in range(5):
        delta = (out_original[:, t] - out_perturbed[:, t]).abs().max().item()
        assert delta < 1e-6, f"causal leak at position {t}, delta={delta}"
    delta_5 = (out_original[:, 5] - out_perturbed[:, 5]).abs().max().item()
    assert delta_5 > 1e-4
    print("  ok -- past positions isolated from future modifications")


def check_param_counts():
    print("-- parameter counts --")
    mha_params = sum(p.numel() for p in GroupedQueryAttention(512, 8, 8).parameters())
    gqa_params = sum(p.numel() for p in GroupedQueryAttention(512, 8, 2).parameters())
    savings_pct = (1 - gqa_params / mha_params) * 100
    print(f"  ok -- GQA-2 saves {savings_pct:.1f}% params vs MHA")


def check_mha_fallback():
    print("-- MHA fallback (G=H) --")
    model = GroupedQueryAttention(128, 4, 4, causal=False, dropout=0.0)
    assert model.queries_per_kv == 1
    out, _ = model(torch.randn(2, 32, 128))
    assert out.shape == (2, 32, 128) and not torch.isnan(out).any()
    print("  ok")


if __name__ == "__main__":
    print("GQA test suite\n")
    check_expand_kv()
    check_shapes_across_configs()
    check_gradients_flow()
    check_no_nans()
    check_causal_mask()
    check_param_counts()
    check_mha_fallback()
    print("\nall passed")
