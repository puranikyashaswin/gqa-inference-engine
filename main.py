"""Quick demo -- forward pass with and without KV cache."""

import torch
from src.attention import GroupedQueryAttention
from src.kv_cache import KVCache


def main():
    d_model, H, G = 512, 8, 2
    model = GroupedQueryAttention(d_model, H, G, causal=True)
    model.eval()

    x = torch.randn(2, 128, d_model)
    out, _ = model(x)
    print(f"uncached:  input {list(x.shape)} -> output {list(out.shape)}")

    prompt = torch.randn(1, 32, d_model)
    cache = KVCache()
    with torch.no_grad():
        out, cache = model(prompt, past_kv=cache, use_cache=True)
        print(f"prefill:   {list(prompt.shape)} -> {list(out.shape)}, cache={cache.seq_len} tokens")
        current = out[:, -1:, :]
        for step in range(5):
            out, cache = model(current, past_kv=cache, use_cache=True)
            current = out
            print(f"decode {step}: input (1,1,{d_model}) -> output {list(out.shape)}, cache={cache.seq_len} tokens")

    print(f"\ncache memory: {cache.memory_bytes() / 1024:.1f} KB")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
