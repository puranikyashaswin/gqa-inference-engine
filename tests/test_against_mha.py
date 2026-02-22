"""verify GQA(G=H) == hand-rolled MHA with shared weights"""
import torch
import torch.nn as nn
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.attention import GroupedQueryAttention


class ReferenceMHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        bs, slen, _ = x.shape
        H, dk = self.num_heads, self.head_dim
        q = self.W_q(x).view(bs, slen, H, dk).transpose(1, 2)
        k = self.W_k(x).view(bs, slen, H, dk).transpose(1, 2)
        v = self.W_v(x).view(bs, slen, H, dk).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(slen, slen, device=x.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return self.W_o(out.transpose(1, 2).contiguous().reshape(bs, slen, self.d_model))


def test_gqa_matches_mha():
    print("-- GQA vs reference MHA (G=H) --")
    d_model, H = 128, 4
    torch.manual_seed(7)
    gqa = GroupedQueryAttention(d_model, H, num_kv_heads=H, causal=True, dropout=0.0)
    ref = ReferenceMHA(d_model, H)
    ref.W_q.weight.data.copy_(gqa.W_q.weight.data)
    ref.W_k.weight.data.copy_(gqa.W_k.weight.data)
    ref.W_v.weight.data.copy_(gqa.W_v.weight.data)
    ref.W_o.weight.data.copy_(gqa.W_o.weight.data)
    gqa.eval(); ref.eval()
    x = torch.randn(2, 32, d_model)
    with torch.no_grad():
        out_gqa, _ = gqa(x)
        out_ref = ref(x)
    max_diff = (out_gqa - out_ref).abs().max().item()
    assert max_diff < 1e-5
    print(f"  max diff: {max_diff:.2e}")
    print(f"  ok")


def test_kv_diversity():
    print("\n-- KV diversity reduction --")
    d_model, H, G = 128, 8, 2
    model = GroupedQueryAttention(d_model, H, G, causal=False, dropout=0.0)
    model.eval()
    x = torch.randn(1, 16, d_model)
    with torch.no_grad():
        k = model.W_k(x).view(1, 16, G, d_model // H).transpose(1, 2)
    diff = (k[:, 0] - k[:, 1]).abs().mean().item()
    assert diff > 0.01
    print(f"  inter-group K difference: {diff:.4f}")
    print(f"  only {G} distinct KV heads for {H} query heads")
    print(f"  ok")


if __name__ == "__main__":
    print("GQA vs MHA comparison\n")
    test_gqa_matches_mha()
    test_kv_diversity()
    print("\nall passed")
