"""
Numerical gradient verification for GQA.

Uses torch.autograd.gradcheck to compare analytical gradients (autograd)
against finite-difference approximations. float64 for precision.
"""

import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.attention import GroupedQueryAttention


def _forward_output_only(model, x):
    out, _ = model(x)
    return out


def _run_gradcheck(label, num_kv_heads, causal=True):
    model = GroupedQueryAttention(
        d_model=32, num_heads=4, num_kv_heads=num_kv_heads,
        causal=causal, dropout=0.0,
    ).double()
    x = torch.randn(1, 4, 32, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(
        lambda inp: _forward_output_only(model, inp),
        (x,), eps=1e-6, atol=1e-4, rtol=1e-3,
    )
    assert ok, f"gradcheck failed for {label}"
    print(f"  {label}: ok")


if __name__ == "__main__":
    print("gradient correctness (finite differences)\n")
    _run_gradcheck("gqa (G=2, H=4)", num_kv_heads=2)
    _run_gradcheck("mha (G=H=4)",    num_kv_heads=4)
    _run_gradcheck("mqa (G=1)",       num_kv_heads=1)
    _run_gradcheck("non-causal gqa",  num_kv_heads=2, causal=False)
    print("\nall passed")
