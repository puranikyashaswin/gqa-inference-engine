"""
Tests for KV cache correctness.
"""

import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.kv_cache import KVCache, LayeredKVCache


def check_cache_grows():
    print("-- cache grows with each update --")
    cache = KVCache()
    assert cache.seq_len == 0
    k0 = torch.randn(1, 2, 10, 64)
    v0 = torch.randn(1, 2, 10, 64)
    k_out, v_out = cache.update(k0, v0)
    assert cache.seq_len == 10
    for step in range(5):
        k_out, v_out = cache.update(torch.randn(1, 2, 1, 64), torch.randn(1, 2, 1, 64))
        assert cache.seq_len == 11 + step
    assert cache.seq_len == 15 and k_out.shape == (1, 2, 15, 64)
    print(f"  ok -- grew from 0 to 15")


def check_values_preserved():
    print("-- cached values are preserved --")
    cache = KVCache()
    sentinel = torch.ones(1, 2, 1, 64) * 42.0
    cache.update(sentinel, sentinel.clone())
    for _ in range(4):
        cache.update(torch.randn(1, 2, 1, 64), torch.randn(1, 2, 1, 64))
    assert torch.equal(cache.k[:, :, 0:1, :], sentinel)
    print("  ok")


def check_memory():
    print("-- memory accounting --")
    cache = KVCache()
    assert cache.memory_bytes() == 0
    cache.update(torch.randn(1, 2, 10, 64), torch.randn(1, 2, 10, 64))
    expected = 2 * 1 * 2 * 10 * 64 * 4
    assert cache.memory_bytes() == expected
    print(f"  ok -- {cache.memory_bytes()} bytes")


def check_reset():
    print("-- reset --")
    cache = KVCache()
    cache.update(torch.randn(1, 2, 10, 64), torch.randn(1, 2, 10, 64))
    cache.reset()
    assert cache.seq_len == 0 and cache.k is None
    print("  ok")


def check_layered():
    print("-- layered cache --")
    lc = LayeredKVCache(4)
    for i in range(4):
        lc[i].update(torch.randn(1, 2, 10, 64), torch.randn(1, 2, 10, 64))
    assert lc.total_seq_len == 10
    assert lc.total_memory_bytes() == 4 * 2 * 1 * 2 * 10 * 64 * 4
    lc.reset()
    assert lc.total_seq_len == 0
    print(f"  ok")


if __name__ == "__main__":
    print("KV cache tests\n")
    check_cache_grows()
    check_values_preserved()
    check_memory()
    check_reset()
    check_layered()
    print("\nall passed")
