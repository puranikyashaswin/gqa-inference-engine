"""
Equivalence tests: cached vs uncached attention.

The whole point of KV caching is that it produces IDENTICAL output
to the uncached path -- it's a pure performance optimization, not an
approximation. If these tests fail, the cache is corrupting attention.
"""

import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.attention import GroupedQueryAttention
from src.kv_cache import KVCache


def check_prefill_matches_uncached():
    print("-- prefill == uncached --")
    torch.manual_seed(42)
    model = GroupedQueryAttention(128, 4, 2, causal=True, dropout=0.0)
    model.eval()
    x = torch.randn(2, 16, 128)
    with torch.no_grad():
        out_nocache, _ = model(x, use_cache=False)
        out_cached, cache = model(x, use_cache=True)
    diff = (out_nocache - out_cached).abs().max().item()
    assert diff < 1e-6
    assert cache.seq_len == 16
    print(f"  ok -- max diff: {diff:.2e}, cache holds {cache.seq_len} positions")


def check_incremental_matches_full_recompute():
    print("-- incremental decode == full recompute --")
    torch.manual_seed(123)
    d_model, H, G = 128, 4, 2
    model = GroupedQueryAttention(d_model, H, G, causal=True, dropout=0.0)
    model.eval()
    prompt = torch.randn(1, 8, d_model)
    n_decode_steps = 6
    with torch.no_grad():
        cache = KVCache()
        out_prefill, cache = model(prompt, past_kv=cache, use_cache=True)
        cached_outputs = [out_prefill[:, -1:, :]]
        current = out_prefill[:, -1:, :]
        for _ in range(n_decode_steps - 1):
            out_step, cache = model(current, past_kv=cache, use_cache=True)
            cached_outputs.append(out_step)
            current = out_step
        full_seq = prompt
        uncached_outputs = []
        for i in range(n_decode_steps):
            out_full, _ = model(full_seq, use_cache=False)
            uncached_outputs.append(out_full[:, -1:, :])
            full_seq = torch.cat([full_seq, cached_outputs[i]], dim=1)
    max_diff = 0
    for step in range(n_decode_steps):
        diff = (cached_outputs[step] - uncached_outputs[step]).abs().max().item()
        max_diff = max(max_diff, diff)
        assert diff < 1e-5, f"step {step}: diff={diff}"
    print(f"  ok -- {n_decode_steps} decode steps match (max diff: {max_diff:.2e})")


def check_cache_with_mha():
    print("-- cached decode with MHA (G=H) --")
    torch.manual_seed(99)
    model = GroupedQueryAttention(64, 4, 4, causal=True, dropout=0.0)
    model.eval()
    prompt = torch.randn(1, 6, 64)
    with torch.no_grad():
        cache = KVCache()
        out, cache = model(prompt, past_kv=cache, use_cache=True)
        current = out[:, -1:, :]
        for _ in range(3):
            out, cache = model(current, past_kv=cache, use_cache=True)
            current = out
    assert cache.seq_len == 9 and not torch.isnan(current).any()
    print(f"  ok -- cache.seq_len={cache.seq_len}")


def check_cache_with_mqa():
    print("-- cached decode with MQA (G=1) --")
    torch.manual_seed(77)
    model = GroupedQueryAttention(64, 4, 1, causal=True, dropout=0.0)
    model.eval()
    prompt = torch.randn(1, 6, 64)
    with torch.no_grad():
        cache = KVCache()
        out, cache = model(prompt, past_kv=cache, use_cache=True)
        assert cache.k.shape[1] == 1
        current = out[:, -1:, :]
        for _ in range(4):
            out, cache = model(current, past_kv=cache, use_cache=True)
            current = out
    assert cache.seq_len == 10
    print(f"  ok -- G=1 cache: {list(cache.k.shape)}, {cache.memory_bytes()} bytes")


if __name__ == "__main__":
    print("cached vs uncached equivalence\n")
    check_prefill_matches_uncached()
    check_incremental_matches_full_recompute()
    check_cache_with_mha()
    check_cache_with_mqa()
    print("\nall passed")
