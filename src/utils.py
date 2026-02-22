"""Misc helpers."""
import time
import torch


def time_forward_ms(model, x, n_iter=100):
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(x)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        return (time.perf_counter() - t0) / n_iter * 1000


def kv_cache_bytes(num_kv_heads, seq_len, head_dim, n_layers=1, dtype_bytes=4):
    return 2 * num_kv_heads * seq_len * head_dim * n_layers * dtype_bytes
