"""
KV Cache for autoregressive decoding.

During generation, each transformer layer caches its K and V tensors
so they don't need to be recomputed for previously-seen tokens. This
turns O(N^2) per-step attention into O(N) -- the difference between
"unusably slow" and "real-time" for long sequences.

The cache stores KV in the *original* G-head layout (before GQA expand),
not the expanded H-head layout. This is important: with G=8 and H=64,
the cache is 8x smaller than if we naively cached the expanded tensors.
"""

import torch
from typing import Optional, Tuple


class KVCache:
    """Per-layer KV cache for autoregressive generation.

    Stores K and V tensors and grows along the sequence dimension
    as new tokens are generated. Tensors are stored in the G-head
    (num_kv_heads) layout: (batch, num_kv_heads, seq_len, head_dim).
    """

    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    @property
    def seq_len(self) -> int:
        if self.k is None:
            return 0
        return self.k.shape[2]

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V to the cache and return the full accumulated KV."""
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        return self.k, self.v

    def memory_bytes(self) -> int:
        if self.k is None:
            return 0
        return self.k.nelement() * self.k.element_size() * 2

    def reset(self):
        self.k = None
        self.v = None


class LayeredKVCache:
    """Manages KV caches across all transformer layers."""

    def __init__(self, num_layers: int):
        self.caches = [KVCache() for _ in range(num_layers)]

    def __getitem__(self, layer_idx: int) -> KVCache:
        return self.caches[layer_idx]

    @property
    def total_seq_len(self) -> int:
        return self.caches[0].seq_len

    def total_memory_bytes(self) -> int:
        return sum(c.memory_bytes() for c in self.caches)

    def reset(self):
        for c in self.caches:
            c.reset()
