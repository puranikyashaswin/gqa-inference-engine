# KV Cache -- Implementation Observations

Notes from building the cache on top of the existing GQA module.
Most of the "hard" stuff wasn't in the cache data structure -- it
was in how the attention mask changes when Q and K have different
lengths.

## The Mask Is Where All The Bugs Are

During training (no cache), the causal mask is simple: a square
(S x S) upper-triangular matrix. Every implementation does this.

During cached decode, Q is (1) and K is (S_total). The mask should
be a single row of all-zeros -- the new token can attend to everything
in the cache because it's the latest position. Most tutorials skip
this detail and you end up with a square mask that doesn't match
the score dimensions.

I initially had a bug where the mask was still (S x S) but the
scores were (1 x S_total). PyTorch doesn't error on this -- it
broadcasts the mask in a way that produces garbage attention
patterns without any NaN or shape errors. The only way I caught
it was through the equivalence test (cached output != uncached
output at step 3).

## torch.cat vs Pre-allocated Buffer

The current cache uses torch.cat to grow K/V at each step:

```python
self.k = torch.cat([self.k, k_new], dim=2)
```

This is fine for prototyping (and correct), but for production
inference you'd pre-allocate a buffer of max_seq_len and maintain
a write pointer:

```python
# production pattern (not implemented)
self.k_buffer[:, :, self.pos:self.pos+1, :] = k_new
self.pos += 1
```

The difference matters at long sequences. Each cat() allocates a
new tensor and copies everything, so the total cost over N steps
is O(N^2) in memory copies. Pre-allocation makes it O(N).

I didn't implement this because it adds complexity (you need
max_seq_len upfront, and slicing logic) without changing the
attention computation. But it's worth noting -- this is literally
what vLLM's PagedAttention optimizes.

## Cache In G-Head Layout, Not H-Head

Early version cached the *expanded* K/V (after broadcast to H heads).
This worked but wasted memory -- with G=2 and H=8, you're storing
4 redundant copies of each KV head.

The fix was simple: cache before expand, expand after retrieval.
The cache stores (bs, G, seq, d_k) and expand_kv_to_query_heads
runs on the full cached tensor each step.

This does mean we re-expand the entire cache every step. For very
long sequences, it might be worth caching the expanded version
to avoid the repeated expand. But the expand is a stride trick
(essentially free until the contiguous() copy), so in practice
it's negligible compared to the Q*K matmul.

## Prefill vs Decode: Two Very Different Workloads

Prefill (processing the prompt) is compute-bound: you're doing a
full (S x S) attention computation. This parallelizes well on GPU.

Decode (generating tokens) is memory-bound: the computation is
tiny (1 x S_total), but you need to load the entire KV cache
from memory for each layer at each step. The bottleneck is
memory bandwidth, not FLOPS.

This is why KV cache size matters so much. A smaller cache (GQA
vs MHA) means less data to load per decode step, which directly
translates to tokens/second.

This also explains why the CPU benchmark shows modest speedups
(3-6x). On CPU, the bottleneck is matmul, not bandwidth. On GPU
with long sequences, the speedup from caching would be 20-50x
because you're eliminating redundant computation AND the cache
fits in faster memory tiers.

## The Equivalence Test Is Everything

I have 23 tests but only one actually matters: check_incremental_
matches_full_recompute. This test runs the cached path and the
uncached path side-by-side and checks that they produce identical
output at every decode step.

If the equivalence test passes, the cache is correct. By definition.
Everything else (shape checks, gradient flow, parameter counts)
is just making sure the base attention works.

The tricky part of the equivalence test is that you need to feed
the *same input* to both paths. During cached decode, token N's
input is the output of step N-1. So you build the uncached
sequence incrementally using the cached path's outputs. Any
divergence at step K propagates to all later steps.

## What I'd Add Next

1. Pre-allocated buffer with max_seq_len (the vLLM pattern)
2. Multi-layer generation (stack SimpleTransformerLayer instances)
3. Actual tokenizer integration for text-in text-out
4. FP16 cache to halve memory footprint
5. Sliding window attention (Mistral-style) -- only cache the
   last W positions, which bounds memory at O(W) regardless
   of sequence length
