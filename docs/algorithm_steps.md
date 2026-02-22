# GQA Algorithm Steps

Concrete tensor shapes for d_model=512, H=8 query heads, G=2 KV groups,
d_k=64, N=4 queries per group, B=2, S=128.

---

## 1. Project Q, K, V

```
X:  (2, 128, 512)

Q = X @ W_q    ->  (2, 128, 512)   H*d_k = 8*64 = 512
K = X @ W_k    ->  (2, 128, 128)   G*d_k = 2*64 = 128  <- 4x smaller
V = X @ W_v    ->  (2, 128, 128)
```

K/V projection weights are (512, 128) instead of (512, 512).
This is where the parameter savings live.

## 2. Split into heads

Reshape then swap seq and head dims:

```
Q:  (2, 128, 512) -> view (2, 128, 8, 64) -> transpose -> (2, 8, 128, 64)
K:  (2, 128, 128) -> view (2, 128, 2, 64) -> transpose -> (2, 2, 128, 64)
V:  same as K
```

Q has 8 heads, K/V only have 2 -- shape mismatch for matmul.

## 3. Expand K/V to match Q's head count

The critical step. Each KV group gets copied N=4 times:

```
K:  (2, 2, 128, 64)
    -> unsqueeze(2)        (2, 2, 1, 128, 64)
    -> expand              (2, 2, 4, 128, 64)   <- free (stride trick)
    -> contiguous+reshape  (2, 8, 128, 64)      <- pays for the copy
V:  same
```

After this, Q/K/V are all (2, 8, 128, 64). Standard attention from here.

Group 0's K/V -> query heads 0,1,2,3.
Group 1's K/V -> query heads 4,5,6,7.

## 4. Attention scores

```
scores = Q @ K^T / sqrt(64)

Q:   (2, 8, 128, 64)
K^T: (2, 8, 64, 128)
scores: (2, 8, 128, 128)
```

Scaling by sqrt(d_k) = 8.0 keeps score variance around 1 so softmax
doesn't saturate.

## 5. Causal mask

Upper triangular -> -inf, so token i only attends to positions 0..i.
Required for decoder / autoregressive models.

```
scores: (2, 8, 128, 128)  <- future positions now -inf
```

## 6. Softmax -> weighted values -> output

```
weights = softmax(scores, dim=-1)     (2, 8, 128, 128)
context = weights @ V                 (2, 8, 128, 64)
context -> transpose -> reshape        (2, 128, 512)
output = context @ W_o                (2, 128, 512)
```

## Shape flow

```
X (B, S, d)
|-- W_q -> (B, S, H*dk) -> (B, H, S, dk)
|-- W_k -> (B, S, G*dk) -> (B, G, S, dk) -> expand -> (B, H, S, dk)
+-- W_v -> (B, S, G*dk) -> (B, G, S, dk) -> expand -> (B, H, S, dk)
                                                         |
                        Q @ K^T / sqrt(dk) -> scores (B, H, S, S)
                                                         |
                                mask + softmax -> weights (B, H, S, S)
                                                         |
                                    weights @ V -> (B, H, S, dk)
                                                         |
                             concat + W_o -> output (B, S, d)
```
