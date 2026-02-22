# GQA -- Implementation Observations

Notes from building GQA from scratch. Mostly about the things
that weren't obvious from reading the paper.

---

## The expand operation is where GQA lives

Everything else is standard scaled dot-product attention. The only
novel operation is broadcasting G KV heads to H query heads, and
the choice of how to do it matters:

`repeat_interleave` is the obvious approach but it's slower because
it always allocates. `unsqueeze -> expand -> contiguous -> reshape`
is faster because expand is a zero-copy stride trick -- you only pay
for the copy when you call contiguous().

In production, the expand cost is negligible compared to the Q*K
matmul. But during development it's satisfying to know the operation
itself is efficient.

## The gradient through expand just works

I wasn't sure about this initially. When expand duplicates a tensor
along a new dimension, backprop needs to undo that duplication. Turns
out autograd handles it correctly -- it sums the gradient over the
expanded dimension, which is the right thing because the forward pass
copied the value to multiple positions.

Still, I didn't trust it until gradcheck confirmed the analytical
gradient matched finite differences. The test uses float64 because
float32 precision isn't enough for the finite-difference approximation
to converge.

## KV sharing is a memory-quality tradeoff

With G=2 and H=8, query heads 0-3 all attend to the same K/V. This
means the model loses some ability to attend to different aspects of
the context from different heads -- the KV representations within a
group are forced to be shared.

The paper shows this tradeoff is favorable: GQA-8 (8 KV groups for
64 query heads) matches MHA quality on most benchmarks while using
8x less KV cache.

The intuition: K/V heads mostly learn similar representations anyway.
The per-head specialization in MHA is somewhat redundant, and GQA
prunes that redundancy.

## Shape tracking is half the debugging

The reshape chain from (B, S, H*d_k) to (B, H, S, d_k) is the kind
of thing that's trivial to write and surprisingly easy to get wrong.
With two different head counts (H vs G), there are more places to
mix them up.

The worst part: broadcasting bugs don't crash. If you accidentally
use H where you should use G, PyTorch often silently broadcasts and
produces wrong attention patterns with plausible-looking outputs.
The causal mask test (modify a future token, check past outputs
don't change) caught two of these during development.

## Causal mask regeneration

We regenerate the causal mask every forward call because sequence
length can vary at inference time. For fixed-length training, you
could cache it once and reuse it, but the overhead of regenerating
a boolean tensor is tiny compared to the attention computation.

## What I'd do differently

1. Fuse Q/K/V projections into a single linear layer and split the
   output. Saves one kernel launch on GPU and is trivially equivalent.

2. Add RoPE before the expand step. In practice, rotary embeddings
   go between projection and attention, and they interact with the
   KV cache (you want to apply them before caching, not after).

3. Pre-allocate the causal mask at model init for a maximum sequence
   length and slice into it during forward. Avoids the repeated
   allocation.
