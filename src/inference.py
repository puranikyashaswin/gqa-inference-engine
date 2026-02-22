"""
Autoregressive token generation with KV caching.

Simulates the two-phase inference process used by every deployed LLM:
  1. Prefill: process the full prompt, fill KV cache
  2. Decode: generate tokens one at a time, reusing cached K/V

This is a single-layer demonstration -- real models stack N layers
and pass the output of each layer to the next, with a separate
KV cache per layer. The attention math is the same at each layer.
"""

import torch
import torch.nn as nn

from .attention import GroupedQueryAttention
from .kv_cache import KVCache


class SimpleTransformerLayer(nn.Module):
    """One transformer decoder layer: GQA attention + feed-forward.

    Intentionally minimal -- just enough structure to demonstrate
    caching behavior. No layer norm, no residual connections, because
    they don't affect the cache mechanics at all.
    """

    def __init__(self, d_model, num_heads, num_kv_heads, ff_dim=None):
        super().__init__()
        ff_dim = ff_dim or d_model * 4

        self.attn = GroupedQueryAttention(
            d_model, num_heads, num_kv_heads, causal=True, dropout=0.0
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim, bias=False),
            nn.SiLU(),
            nn.Linear(ff_dim, d_model, bias=False),
        )

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, updated_kv = self.attn(x, past_kv=past_kv, use_cache=use_cache)
        out = self.ff(attn_out)
        return out, updated_kv


def generate_no_cache(model, prompt_emb, n_tokens):
    """Naive generation: reprocess the ENTIRE sequence at each step."""
    sequence = prompt_emb
    outputs = []
    for _ in range(n_tokens):
        out, _ = model(sequence, use_cache=False)
        new_token = out[:, -1:, :]
        outputs.append(new_token)
        sequence = torch.cat([sequence, new_token], dim=1)
    return outputs


def generate_with_cache(model, prompt_emb, n_tokens):
    """Cached generation: process prompt once, then one token at a time."""
    outputs = []
    cache = KVCache()

    # prefill
    out, cache = model(prompt_emb, past_kv=cache, use_cache=True)
    new_token = out[:, -1:, :]
    outputs.append(new_token)

    # decode
    for _ in range(n_tokens - 1):
        out, cache = model(new_token, past_kv=cache, use_cache=True)
        new_token = out[:, -1:, :]
        outputs.append(new_token)

    return outputs
