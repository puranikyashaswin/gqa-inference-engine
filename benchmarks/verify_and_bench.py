"""Verify every numeric README claim and save results with provenance.

  python benchmarks/verify_and_bench.py            # writes benchmarks/results_<date>.json
  RUNS=7 TORCH_THREADS=1 python benchmarks/verify_and_bench.py

1. runs every test file (exit codes recorded)
2. independent equivalence checks against torch's F.scaled_dot_product_attention
   (GQA with G=H vs MHA, G<H vs repeat-interleaved reference, cache vs no-cache,
   chunked prefill)
3. repeats benchmark.py's own timing functions RUNS times (interleaved), reports
   median and IQR of the speedup
4. measures real KVCache.memory_bytes() for MHA/GQA/MQA at 256 tokens
"""
import os, sys, json, time, subprocess, statistics, platform, datetime

T = os.environ.get("TORCH_THREADS")
if T:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = T
import torch
import torch.nn.functional as F
if T:
    torch.set_num_threads(int(T))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.attention import GroupedQueryAttention
from src.kv_cache import KVCache
from benchmark import time_generation_no_cache, time_generation_with_cache

RUNS = int(os.environ.get("RUNS", "7"))
res = {"date": datetime.datetime.now().isoformat(timespec="seconds")}

# 1. test files
tests = {}
for t in ["main.py", "tests/test_shapes.py", "tests/test_gradients.py", "tests/test_against_mha.py",
          "tests/test_cache.py", "tests/test_equivalence.py"]:
    p = subprocess.run([sys.executable, t], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    tests[t] = {"exit_code": p.returncode, "tail": p.stdout.strip().splitlines()[-1:] + p.stderr.strip().splitlines()[-2:]}
res["test_files"] = tests

# 2. independent equivalence checks
def ref_sdpa(m, x):
    bs, s, d = x.shape; H, G, hd = m.num_heads, m.num_kv_heads, m.head_dim
    q = m.W_q(x).view(bs, s, H, hd).transpose(1, 2)
    k = m.W_k(x).view(bs, s, G, hd).transpose(1, 2).repeat_interleave(H // G, dim=1)
    v = m.W_v(x).view(bs, s, G, hd).transpose(1, 2).repeat_interleave(H // G, dim=1)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return m.W_o(o.transpose(1, 2).reshape(bs, s, d))

eq = {}
with torch.no_grad():
    for name, (d, H, G) in {"mha_G_eq_H(512,8,8)": (512, 8, 8), "gqa(512,8,2)": (512, 8, 2),
                            "mqa(512,8,1)": (512, 8, 1), "mha_G_eq_H(128,4,4)": (128, 4, 4)}.items():
        diffs = []
        for seed in range(3):
            torch.manual_seed(seed)
            m = GroupedQueryAttention(d, H, G, causal=True).eval()
            x = torch.randn(2, 96, d)
            diffs.append((m(x)[0] - ref_sdpa(m, x)).abs().max().item())
        eq[f"vs_torch_sdpa/{name}"] = {"max_abs_diff": max(diffs), "seeds": 3, "shape": [2, 96, d]}

    # GQA(G=2) must equal MHA(G=8) whose K/V weights are the G=2 weights tiled per group
    torch.manual_seed(5)
    g2 = GroupedQueryAttention(256, 8, 2).eval(); g8 = GroupedQueryAttention(256, 8, 8).eval()
    hd = 32
    g8.W_q.weight.copy_(g2.W_q.weight); g8.W_o.weight.copy_(g2.W_o.weight)
    for src, dst in ((g2.W_k, g8.W_k), (g2.W_v, g8.W_v)):
        dst.weight.copy_(src.weight.view(2, hd, 256).repeat_interleave(4, dim=0).reshape(256, 256))
    x = torch.randn(2, 64, 256)
    eq["gqa_G2_equals_mha_with_tiled_kv_weights"] = {"max_abs_diff": (g2(x)[0] - g8(x)[0]).abs().max().item()}

    # cache vs no-cache at benchmark config, 128 decode steps, batch 2
    for (d, H, G) in [(512, 8, 2), (512, 8, 8), (512, 8, 1)]:
        torch.manual_seed(11)
        m = GroupedQueryAttention(d, H, G, causal=True).eval()
        prompt = torch.randn(2, 64, d)
        out, cache = m(prompt, past_kv=KVCache(), use_cache=True)
        cur = out[:, -1:, :]; seq = torch.cat([prompt, cur], 1); worst = 0.0; steps = 128
        for _ in range(steps):
            o_c, cache = m(cur, past_kv=cache, use_cache=True)
            o_f, _ = m(seq, use_cache=False)
            worst = max(worst, (o_c - o_f[:, -1:, :]).abs().max().item())
            cur = o_c; seq = torch.cat([seq, cur], 1)
        eq[f"cache_vs_nocache/({d},{H},{G})"] = {"prompt": 64, "decode_steps": steps, "batch": 2,
                                                 "max_abs_diff_over_all_steps": worst, "final_cache_len": cache.seq_len}

    # chunked prefill (40 + 24 tokens) vs single 64-token pass: exercises the offset causal mask
    torch.manual_seed(3)
    m = GroupedQueryAttention(512, 8, 2).eval(); x = torch.randn(1, 64, 512)
    full = m(x)[0]
    a, c = m(x[:, :40], past_kv=KVCache(), use_cache=True)
    b, c = m(x[:, 40:], past_kv=c, use_cache=True)
    eq["chunked_prefill_40+24_vs_full"] = {"max_abs_diff": (torch.cat([a, b], 1) - full).abs().max().item()}
res["equivalence"] = eq

# 3. benchmark: same functions/config as benchmark.py, repeated and interleaved
torch.manual_seed(0)
model = GroupedQueryAttention(512, 8, 2, causal=True, dropout=0.0)
prompt = torch.randn(1, 64, 512)
with torch.no_grad():
    model(prompt)
for n in (16, 32, 64, 128):  # warmup
    time_generation_no_cache(model, prompt, n); time_generation_with_cache(model, prompt, n)
raw = {n: {"no_cache_ms": [], "cached_ms": [], "loadavg": []} for n in (16, 32, 64, 128)}
for r in range(RUNS):
    for n in (16, 32, 64, 128):
        raw[n]["loadavg"].append(os.getloadavg())
        if r % 2 == 0:
            a = time_generation_no_cache(model, prompt, n); b, _ = time_generation_with_cache(model, prompt, n)
        else:
            b, _ = time_generation_with_cache(model, prompt, n); a = time_generation_no_cache(model, prompt, n)
        raw[n]["no_cache_ms"].append(a); raw[n]["cached_ms"].append(b)

def st(xs):
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return {"median": statistics.median(xs), "q1": q[0], "q3": q[2], "min": min(xs), "max": max(xs)}
bench = {}
for n, v in raw.items():
    sp = [a / b for a, b in zip(v["no_cache_ms"], v["cached_ms"])]
    bench[str(n)] = {"no_cache_ms": st(v["no_cache_ms"]), "cached_ms": st(v["cached_ms"]), "speedup": st(sp), "raw": v}
res["benchmark"] = {"config": {"d_model": 512, "H": 8, "G": 2, "prompt": 64, "batch": 1, "dtype": "float32",
                               "device": "cpu", "torch_threads": torch.get_num_threads(), "runs": RUNS,
                               "what": "single GQA attention layer (no FFN/LN), output fed back as next input; "
                                       "functions imported from benchmark.py"}, "results": bench}

# 4. measured cache memory, 256 tokens, batch 1
mem = {}
with torch.no_grad():
    for label, G in (("MHA (G=8)", 8), ("GQA (G=2)", 2), ("MQA (G=1)", 1)):
        _, c = GroupedQueryAttention(512, 8, G).eval()(torch.randn(1, 256, 512), use_cache=True)
        mem[label] = {"bytes": c.memory_bytes(), "KiB": c.memory_bytes() / 1024, "kB": c.memory_bytes() / 1000}
res["kv_cache_memory_256tok"] = mem
p = lambda G: sum(x.numel() for x in GroupedQueryAttention(512, 8, G).parameters())
res["params_512_8"] = {"MHA": p(8), "GQA_G2": p(2), "MQA": p(1)}

sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
res["provenance"] = {"chip": sh("sysctl -n machdep.cpu.brand_string"), "mem_bytes": sh("sysctl -n hw.memsize"),
                     "os": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__,
                     "git_commit": sh(f"git -C {ROOT} rev-parse HEAD"), "uptime_end": sh("uptime")}
out = os.path.join(ROOT, "benchmarks", f"results_{datetime.date.today()}.json")
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({"tests": {k: v["exit_code"] for k, v in tests.items()}, "equivalence": eq,
                  "speedup_median": {n: round(b["speedup"]["median"], 2) for n, b in bench.items()},
                  "kv_mem": mem}, indent=1))
print("wrote", out)
