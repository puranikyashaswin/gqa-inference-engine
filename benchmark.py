"""
Latency and memory comparison: cached vs uncached generation.

Without cache: O(N^2) because you reprocess the full sequence each step.
With cache: O(N) because you only process the new token + cache lookup.
"""

import torch
import time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.attention import GroupedQueryAttention
from src.kv_cache import KVCache


def time_generation_no_cache(model, prompt, n_tokens):
    model.eval()
    seq = prompt
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n_tokens):
            out, _ = model(seq, use_cache=False)
            seq = torch.cat([seq, out[:, -1:, :]], dim=1)
        return (time.perf_counter() - t0) * 1000


def time_generation_with_cache(model, prompt, n_tokens):
    model.eval()
    cache = KVCache()
    with torch.no_grad():
        t0 = time.perf_counter()
        out, cache = model(prompt, past_kv=cache, use_cache=True)
        current = out[:, -1:, :]
        for _ in range(n_tokens - 1):
            out, cache = model(current, past_kv=cache, use_cache=True)
            current = out
        return (time.perf_counter() - t0) * 1000, cache.memory_bytes()


def main():
    d_model, H, G = 512, 8, 2
    prompt_len = 64
    print(f"GQA config: d_model={d_model}, H={H} query heads, G={G} KV heads")
    print(f"prompt: {prompt_len} tokens\n")

    model = GroupedQueryAttention(d_model, H, G, causal=True, dropout=0.0)
    prompt = torch.randn(1, prompt_len, d_model)
    with torch.no_grad():
        model(prompt)  # warmup

    print(f"{'Tokens':>8} {'No Cache':>12} {'Cached':>12} {'Speedup':>10} {'Cache MB':>10}")
    print("-" * 56)
    for n in [16, 32, 64, 128]:
        t1 = time_generation_no_cache(model, prompt, n)
        t2, cb = time_generation_with_cache(model, prompt, n)
        print(f"{n:>8} {t1:>10.1f}ms {t2:>10.1f}ms {t1/t2 if t2>0 else 0:>9.1f}x {cb/1024/1024:>8.2f}")

    print("\nKV cache size (per batch, 256 tokens, float32):")
    dk = d_model // H
    for label, g in [("MHA (G=8)", 8), ("GQA (G=2)", 2), ("MQA (G=1)", 1)]:
        print(f"  {label}: {2*g*256*dk*4:>10,} bytes ({2*g*256*dk*4/1024:.1f} KB)")


if __name__ == "__main__":
    main()
