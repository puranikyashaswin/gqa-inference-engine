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
don't need to be recomputed every step. Turns O(N^2) inference into O(N).

The cache stores K/V in the compact G-head layout (not expanded to H
heads), so GQA's memory savings compound with caching.

## Numbers

d_model=512, H=8 query heads, G=2 KV heads, prompt=64 tokens, CPU (Apple Silicon), average of 5 runs:

| Generated tokens | No cache | Cached | Speedup |
|-----------------|----------|--------|---------| 
| 16              | 7.5ms    | 2.8ms  | 2.5–3.0x |
| 32              | 14.1ms   | 5.1ms  | 2.8x     |
| 64              | 30.0ms   | 9.9ms  | 3.0x     |
| 128             | 79.9ms   | 20.7ms | 3.4–5.2x |

KV cache per batch element (256 tokens, float32):

| Config | Cache size (Binary KiB) | Cache size (Decimal KB) |
|--------|-------------------------|-------------------------|
| MHA (G=8) | 1024 KiB                | 1,048 KB                |
| GQA (G=2) | 256 KiB                 | 262 KB                  |
| MQA (G=1) | 128 KiB                 | 131 KB                  |

*Note: The benchmark script `benchmark.py` outputs binary KiB (labeled as KB for simplicity).*

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

benchmark.py            latency comparison: cached vs uncached
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
