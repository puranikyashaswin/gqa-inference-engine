"""
GQA: Grouped Query Attention (Ainslie et al., 2023)

The whole trick is deceptively simple: instead of giving every query head
its own K/V, you share K/V across groups of queries. Llama 2 uses this
with 8 KV heads for 64 query heads -- 8x KV cache reduction, barely any
quality loss. The magic is in the expand step, not the math.

Extended with KV cache support for autoregressive decoding:
- prefill: process full prompt, fill cache
- decode: process one token at a time, reuse cached K/V
"""

import math
import torch
import torch.nn as nn

from typing import Optional, Tuple
from .kv_cache import KVCache


def expand_kv_to_query_heads(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Broadcast G KV heads -> H query-aligned heads via repeat-interleave.

    This is the operation that actually makes GQA different from MHA.
    Without it, there's a shape mismatch between Q (H heads) and K/V
    (G heads) that prevents the batched matmul from working.

    The expand() call is free (just a stride trick), but we pay for
    contiguous() + reshape to flatten back to 4D. In practice this
    overhead is negligible compared to the matmul that follows.
    """
    if n_rep == 1:
        # G == H -> standard MHA, nothing to do
        return kv

    bs, n_kv, slen, head_d = kv.shape

    # unsqueeze(2) -> expand -> reshape is cheaper than repeat_interleave
    # because expand doesn't allocate until we force contiguous layout
    return (
        kv.unsqueeze(2)
          .expand(bs, n_kv, n_rep, slen, head_d)
          .contiguous()
          .reshape(bs, n_kv * n_rep, slen, head_d)
    )


class GroupedQueryAttention(nn.Module):
    """GQA attention block. Set num_kv_heads = num_heads for vanilla MHA,
    num_kv_heads = 1 for MQA, anything in between for GQA.

    No bias on projections -- matches Llama/Mistral convention.

    Supports KV caching for autoregressive inference:
      forward(x)                                -> standard (training)
      forward(x, past_kv=cache, use_cache=True) -> cached (inference)
    """

    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.0, causal=True):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} not divisible by num_heads={num_heads}. "
                f"Can't split embedding into equal head slices."
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} not divisible by num_kv_heads={num_kv_heads}. "
                f"Each KV group must serve an equal number of query heads."
            )

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.causal = causal

        # how many query heads share each KV head
        self.queries_per_kv = num_heads // num_kv_heads

        # 1/sqrt(d_k) -- precomputed to avoid recomputing every forward
        self.inv_sqrt_dk = 1.0 / math.sqrt(self.head_dim)

        # Q gets the full H*d_k output, K/V only get G*d_k
        # this asymmetry is where the parameter savings come from
        self.W_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        bs, slen, _ = x.shape

        # project into Q/K/V subspaces -- only for the NEW tokens in x
        q = self.W_q(x)  # -> (bs, slen, H * d_k)
        k = self.W_k(x)  # -> (bs, slen, G * d_k)  <- smaller when G < H
        v = self.W_v(x)  # -> (bs, slen, G * d_k)

        # split into per-head views and move head dim before sequence
        q = q.view(bs, slen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, slen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, slen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # q: (bs, H, slen, d_k),  k/v: (bs, G, slen, d_k)

        # --- KV cache integration ---
        # if we have cached K/V from previous steps, concatenate them
        # with the newly computed K/V. the cache stores G-head layout
        # (before expand) to keep memory usage at G/H of full MHA.
        if past_kv is not None:
            k, v = past_kv.update(k, v)
            # k, v now include all previous + new positions
        elif use_cache:
            # first call with caching enabled -- initialize the cache
            past_kv = KVCache()
            k, v = past_kv.update(k, v)

        # total key length: could be longer than slen if cache exists
        kv_len = k.shape[2]

        # broadcast KV groups -> query heads (the core GQA operation)
        k_expanded = expand_kv_to_query_heads(k, self.queries_per_kv)
        v_expanded = expand_kv_to_query_heads(v, self.queries_per_kv)

        # Q attends to K -- during decode, this is (1 x kv_len) not (S x S)
        # which is where the speedup comes from
        attn_logits = torch.matmul(q, k_expanded.transpose(-2, -1)) * self.inv_sqrt_dk

        if self.causal:
            # during decode with cache, slen=1 and the single query position
            # is the latest token, which can attend to everything before it.
            # so we only need masking during prefill (slen > 1).
            if slen > 1:
                mask = torch.triu(
                    torch.ones(slen, kv_len, device=x.device, dtype=torch.bool),
                    diagonal=kv_len - slen + 1
                )
                attn_logits.masked_fill_(mask, float('-inf'))

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        context = torch.matmul(attn_weights, v_expanded)

        context = context.transpose(1, 2).contiguous().reshape(bs, slen, self.d_model)
        output = self.W_o(context)

        return output, past_kv if use_cache else None
