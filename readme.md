# Grouped Query Attention + KV Cache Inference Engine

From-scratch PyTorch implementation of GQA ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))
with a KV cache system for autoregressive token generation.

No `nn.MultiheadAttention`, no `F.scaled_dot_product_attention` -- just
matmuls, reshapes, and softmax.

## What this is

Two things built on top of each other:

**1. GQA attention** -- shares K/V heads across groups of query heads
(used in Llama 2/3, Mistral, Gemma)

**2. KV cache** -- stores past K/V tensors during generation so they
don't need to be recomputed every step. Per decode step, attention cost drops
from O(N^2) (recompute everything) to O(N) (one query against N cached keys).

The cache stores K/V in the compact G-head layout (not expanded to H
heads), so GQA's memory savings compound with caching.

## Numbers

All numbers below come from `benchmarks/results_2026-10-12.json`, produced by
`python benchmarks/verify_and_bench.py` (Apple M2, 8 GB, torch 2.14.1, CPU,
float32, 4 torch threads). That run happened on a busy machine (load average
about 7 on 8 cores), so absolute milliseconds are inflated. The speedup ratio is
measured within each run and holds up better. Re-run on an idle machine before
quoting absolute latency.

**Correctness** (max absolute difference, float32):

| Check | Max abs diff |
|---|---|
| GQA with G=H vs `F.scaled_dot_product_attention` MHA (3 seeds, 2x96x512) | 1.8e-7 |
| GQA G=2 / MQA G=1 vs SDPA with repeat-interleaved K/V | 2.7e-7 / 2.2e-7 |
| GQA G=2 vs MHA with K/V weights tiled per group | 0 |
| Cached vs uncached, 64-token prompt + 128 decode steps, batch 2 (G=8, 2, 1) | 3.7e-8 |
| Chunked prefill (40 + 24 tokens) vs one 64-token pass | 0 |

All six test files (`main.py` + `tests/*.py`) exit 0.

**Latency: cached vs uncached generation.** This times one GQA attention layer
(no FFN or LayerNorm, output fed back as the next input), so it measures the
attention/projection cost a KV cache removes, not a full model. Config: d_model=512,
H=8 query heads, G=2 KV heads, prompt=64 tokens, batch 1. Median and [IQR] over 7
interleaved runs:

| Generated tokens | No cache | Cached | Speedup |
|-----------------|----------|--------|---------|
| 16              | 13.8 ms  | 4.3 ms  | 3.3x [3.1–3.5] |
| 32              | 27.2 ms  | 8.3 ms  | 3.5x [3.2–3.6] |
| 64              | 59.0 ms  | 16.7 ms | 3.5x [3.3–3.7] |
| 128             | 172.0 ms | 29.7 ms | 5.6x [5.3–5.7] |

The speedup grows with length because the uncached step re-projects and re-attends
the whole sequence (O(N) projections and O(N²) attention per step). The cached step
projects one token and attends over N cached keys (O(N) per step).

**KV cache per batch element** (256 tokens, float32, measured with `KVCache.memory_bytes()`):

| Config | Bytes | KiB | kB (decimal) |
|--------|-------|-----|--------------|
| MHA (G=8) | 1,048,576 | 1024 | 1,048.6 |
| GQA (G=2) | 262,144   | 256  | 262.1   |
| MQA (G=1) | 131,072   | 128  | 131.1   |

Attention parameters at d_model=512, H=8: MHA 1,048,576, GQA G=2 655,360
(37.5% fewer), MQA 589,824.

*`benchmark.py` prints binary KiB but labels them "KB".*

## Usage

```bash
pip install torch
python main.py
python benchmark.py
python tests/test_shapes.py
python tests/test_gradients.py
python tests/test_against_mha.py
python tests/test_cache.py
python tests/test_equivalence.py
python benchmarks/verify_and_bench.py   # all checks + timings -> benchmarks/results_<date>.json
```

## Files

```
src/
  attention.py          GQA with KV cache support
  kv_cache.py           cache data structure (per-layer + multi-layer)
  inference.py          cached vs uncached generation loops
  utils.py              timing, memory helpers

tests/
  test_shapes.py        shape, causality, gradient flow
  test_gradients.py     finite-difference gradient verification
  test_against_mha.py   GQA(G=H) == reference MHA
  test_cache.py         cache grow/reset/memory accounting
  test_equivalence.py   cached output == uncached output (critical)

benchmark.py            latency comparison: cached vs uncached (single run)
benchmarks/
  verify_and_bench.py   tests + SDPA equivalence + repeated timings, saves JSON
  results_<date>.json   saved results with provenance
docs/
  paper_notes.md        GQA paper breakdown
  algorithm_steps.md    equation to step mapping with tensor shapes
  insights.md           observations from building this
  kv_cache_notes.md     how caching changes the attention computation
  cache_insights.md     debugging notes from cache implementation
```

## What I learned

- The cache itself is trivial -- it's just a concat along dim 2. The hard
  part is getting the causal mask right when Q is (1, S_total) during
  decode instead of (S, S) during training.

- Caching in G-head layout (before expand) instead of H-head layout
  is important. With G=2 and H=8, you get 4x cache reduction on top
  of whatever sequence-length savings the cache provides.

- The equivalence test (cached == uncached for every decode step) is
  the only test that matters. If that passes, everything else is just
  performance.

- `torch.cat` on the cache is fine for prototyping but in production
  you'd pre-allocate a max-length buffer and slice into it. The
  repeated allocation/copy is wasteful for long sequences.

## References

1. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer
   Models from Multi-Head Checkpoints", 2023
2. Pope et al., "Efficiently Scaling Transformer Inference", 2022
3. Kwon et al., "Efficient Memory Management for LLM Serving with
   PagedAttention", 2023
4. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", 2019
