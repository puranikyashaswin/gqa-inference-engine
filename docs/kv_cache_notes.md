# KV Cache in Transformer Inference

Working notes on how KV caching changes the attention computation
during autoregressive token generation.

---

## The Problem

During training, attention processes the full sequence at once:

```
Q, K, V all come from X of shape (B, S, d_model)
attention is computed over the full (S, S) matrix
```

During inference (text generation), tokens are produced one at a time.
Naively, each new token requires reprocessing the entire sequence:

```
step 1:  process [t0]                -> 1 token
step 2:  process [t0, t1]            -> 2 tokens
step 3:  process [t0, t1, t2]        -> 3 tokens
...
step N:  process [t0, t1, ..., tN]   -> N tokens
```

Total work: 1 + 2 + 3 + ... + N = O(N^2). For 4096 tokens, that's
~8 million attention computations. Most of this is redundant -- we're
recomputing K and V for tokens we've already seen.

## The Fix: Cache K and V

The key observation: once a token is processed, its K and V vectors
never change (because of the causal mask, it can't attend to future
tokens that haven't been generated yet). So we cache them:

```
step 1:  compute K0, V0 from t0           cache: [K0], [V0]
step 2:  compute K1, V1 from t1 only      cache: [K0,K1], [V0,V1]
         Q1 attends to cached [K0,K1]
step 3:  compute K2, V2 from t2 only      cache: [K0,K1,K2], [V0,V1,V2]
         Q2 attends to cached [K0,K1,K2]
```

Now each step only processes 1 token (not the full sequence). The
attention scores are computed against the growing cache, but the
projection cost is constant per step.

## Two Phases of Inference

### Phase 1: Prefill

Process the full prompt at once (like training):

```
input:  prompt tokens (B, S_prompt, d_model)
output: K, V cached for all prompt positions
        + logits for the next token
```

This is efficient because batched matmul over the full prompt is
parallelizable on GPU. This phase fills the cache with the initial
context.

### Phase 2: Decode (Token-by-Token)

Generate one token at a time:

```
input:  single new token (B, 1, d_model)
        + cached K, V from all previous positions
output: logits for the next token
        + updated cache (appended with new K, V)
```

The Q is (B, H, 1, d_k) -- a single query position.
The K is (B, H, S_cached + 1, d_k) -- all past keys + new key.
The attention scores are (B, H, 1, S_cached + 1) -- a single row.

This is where the speedup comes from: we compute one row of the
attention matrix instead of the full (S, S) matrix.

## Exact Tensor Shapes

Using our GQA config: d_model=512, H=8, G=2, d_k=64.

### Without cache (step N):

```
x:          (B, N, 512)         <- reprocess everything
Q:          (B, 8, N, 64)
K:          (B, 8, N, 64)       <- recomputed from scratch
V:          (B, 8, N, 64)
scores:     (B, 8, N, N)        <- full NxN matrix
output:     (B, N, 512)
```

### With cache (step N, generating token N):

```
x_new:      (B, 1, 512)         <- only the new token
Q_new:      (B, 8, 1, 64)       <- single query position
K_new:      (B, 2, 1, 64)       <- single new KV (G=2 groups)
V_new:      (B, 2, 1, 64)

# append to cache
K_cached:   (B, 2, N, 64)       <- all previous + new (G heads, not H!)
V_cached:   (B, 2, N, 64)

# expand to match query heads
K_expanded: (B, 8, N, 64)
V_expanded: (B, 8, N, 64)

scores:     (B, 8, 1, N)        <- single row, not NxN!
output:     (B, 1, 512)         <- single output position
```

### Important: cache stores G heads, not H

With GQA, there's no point caching the expanded K/V (H heads).
We cache the original G KV heads (before expansion) because:
1. It uses G/H less memory
2. We expand after retrieving from cache anyway

This is the whole point of GQA for inference -- the cache is smaller.

## Cache Memory Formula

Per layer, per batch element:

```
cache_bytes = 2 * G * S * d_k * dtype_bytes
              ^   ^   ^   ^      ^
              K+V |   |   |      float16 = 2, float32 = 4
                  |   |   per-head dim
                  |   sequence length so far
                  num KV heads (NOT num query heads)
```

For Llama 2 70B (G=8, d_k=128, 80 layers, float16):
- Per token cached: 2 * 8 * 128 * 80 * 2 = 327,680 bytes = 320 KiB (or 327.68 KB)
- At 4096 context:
  - Binary calculation: 320 KiB * 4096 = 1,310,720 KiB = 1280 MiB = 1.25 GiB per batch element
  - Decimal calculation: 327.68 KB * 4096 = 1,342,177.28 KB = 1342.18 MB = 1.34 GB per batch element
  - (The mixed calculation `320 KiB * 4096 / 1024 / 1000` is sometimes written as 1.28 GB, but using standard binary units gives exactly 1.25 GiB, and standard decimal units gives 1.34 GB).

With full MHA (G=64 instead of 8):
  - Binary calculation: 1.25 GiB * 8 = 10.0 GiB per batch element
  - Decimal calculation: 1.342 GB * 8 = 10.74 GB per batch element
  - (Or 10.24 GB in mixed notation). That's why GQA matters.

## How the Forward Pass Changes

The GQA forward needs two modifications:

1. Accept optional `past_kv` tuple of (K_cached, V_cached)
2. Return updated KV for caching when `use_cache=True`

```
Without cache:
  forward(x)                          -> output

With cache (prefill):
  forward(x, past_kv=None, use_cache=True)  -> output, (K, V)

With cache (decode):
  forward(x_new, past_kv=(K, V), use_cache=True)  -> output, (K_new, V_new)
```

Inside the forward:
- Project new Q, K, V from the new input only
- If past_kv exists: concatenate cached K/V with new K/V along seq dim
- Run attention: Q_new against all K (cached + new)
- Return output + updated (K, V) if use_cache=True

## Causal Mask With Cache

When decoding with a cache, the causal mask changes:

Without cache: mask is (slen, slen) -- full square matrix
With cache:    mask is (1, S_total) -- single row, all positions visible

Since the new token is always the latest position and can attend to
all previous positions (which are in the cache), the mask for the
decode phase is trivially all-ones. We only need masking during
prefill (processing the full prompt).

## References

1. Vaswani et al. 2017 -- original transformer
2. Pope et al. 2022 -- "Efficiently Scaling Transformer Inference"
3. Kwon et al. 2023 -- "Efficient Memory Management for LLM Serving with PagedAttention" (vLLM)
