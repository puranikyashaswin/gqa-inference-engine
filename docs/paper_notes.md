# GQA Paper Notes

Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models
from Multi-Head Checkpoints" (arXiv:2305.13245, 2023)

---

## The Problem

Standard MHA caches H separate K and V tensors per layer. At Llama 2 70B
scale (64 heads, 128d head, 80 layers), the KV cache for a 4k context is
~10 GB per batch element (~10.24 GiB or ~10.74 GB). This is the bottleneck for
long-context inference, not the matmul. (The ~5 GB figure occasionally cited
only accounts for either Key or Value cache, not both combined.)

MQA (Shazeer 2019) fixes this by sharing one K/V across all query heads --
but quality degrades, especially on summarization tasks.

## The Idea

Share K/V across *groups* of query heads instead of all or none.

- H query heads divided into G groups
- each group gets one K head and one V head
- group assignment: g(h) = floor(h * G / H)

Boundary cases:
- G = H -> MHA (no sharing)
- G = 1 -> MQA (maximum sharing)
- 1 < G < H -> GQA (the interesting case)

## Equations

Standard scaled dot-product for head h in group g(h):

```
head_h = softmax(Q_h . K_{g(h)}^T / sqrt(d_k)) . V_{g(h)}
```

The full layer output:

```
GQA(X) = Concat(head_1, ..., head_H) . W^O
```

Projection sizes -- this is where the savings come from:

```
W^Q in R^{d_model x H*d_k}     (same as MHA)
W^K in R^{d_model x G*d_k}     (smaller when G < H)
W^V in R^{d_model x G*d_k}     (smaller when G < H)
W^O in R^{d_model x d_model}   (same as MHA)
```

## Conversion Recipe

The paper's second contribution is converting existing MHA checkpoints
to GQA without retraining from scratch:

1. Mean-pool the K/V projection weights within each group
   (more effective than selecting one head or random init)
2. Uptrain for ~5% of original pretraining steps

This makes GQA a drop-in replacement for MHA in existing models.

## What's NOT in this implementation

- Uptraining/conversion recipe (needs pretraining infra)
- Flash attention tiling (orthogonal optimization)
- RoPE integration (would go before the expand step)

## Key numbers from the paper

- GQA-8 on a T5-XXL model matches MHA quality on most benchmarks
- Inference speedup scales with the KV cache reduction ratio (H/G)
- Mean-pooling conversion > random init > single-head selection

## References

1. Ainslie et al. 2023 (this paper)
2. Shazeer 2019 -- MQA
3. Vaswani et al. 2017 -- original transformer / MHA
