# Part 3: interconnect roofline for H100 and A100, with the vLLM TP runs overlaid.
# x = FLOPs per byte sent over NVLink (TP all-reduce traffic), y = FLOPs/s per GPU.
import os
import numpy as np
import matplotlib.pyplot as plt
from common import (HARDWARE, MODELS, PLOTS, SERIES, TEXT, TEXT_2, SURFACE, setup_style, load_nccl,
                    peak_busbw, load_vllm_sweep, allreduce_bytes_per_token, flops_per_token_per_gpu)

setup_style()
fig, ax = plt.subplots(figsize=(8, 5.5))
ai = np.logspace(0, 5, 400)

for color, gpu in zip(SERIES, ["H100", "A100"]):
    hw = HARDWARE[gpu]
    measured_bw = peak_busbw(load_nccl(gpu))
    ridge = hw["peak_flops"] / measured_bw
    # measured ceiling solid, spec ceiling dashed (reference); ridge values live in the legend to keep the plot clear
    ax.loglog(ai, np.minimum(ai * measured_bw, hw["peak_flops"]), color=color,
              label=f"{hw['label']} measured ({measured_bw / 1e9:.0f} GB/s all-reduce busbw, ridge {ridge:,.0f} FLOP/B)")
    ax.loglog(ai, np.minimum(ai * hw["spec_bw"], hw["peak_flops"]), color=color, linewidth=1, linestyle="--",
              label=f"{hw['label']} spec ({hw['spec_bw'] / 1e9:.0f} GB/s NVLink)")

# vLLM runs on H100 (TP=1 has no all-reduce traffic, so it has no position on this x-axis)
for r in load_vllm_sweep():
    if r["tp"] == 1:
        continue
    x = flops_per_token_per_gpu(r["model"], r["tp"]) / allreduce_bytes_per_token(r["model"], r["tp"])
    y = flops_per_token_per_gpu(r["model"], r["tp"]) * r["total_tps"]
    marker = "o" if MODELS[r["model"]]["short"] == "32B" else "s"
    ax.plot(x, y, marker=marker, markersize=8, color=SERIES[2], markeredgecolor=SURFACE, markeredgewidth=2,
            linestyle="none")
    pct = y / HARDWARE["H100"]["peak_flops"]
    # TP=4 labels go left of their points, TP=2 labels right; 32B TP=4 drops below the A100 roof to stay clear of it
    left = r["tp"] == 4
    dy = -22 if (left and MODELS[r["model"]]["short"] == "32B") else 8
    ax.annotate(f"{MODELS[r['model']]['short']} TP={r['tp']} ({pct:.0%})", (x, y), xytext=(-8 if left else 8, dy),
                textcoords="offset points", ha="right" if left else "left", va="center", fontsize=8, color=TEXT)
for short, marker in [("32B", "o"), ("72B", "s")]:
    ax.plot([], [], marker=marker, markersize=8, color=SERIES[2], linestyle="none",
            label=f"vLLM Qwen2.5-{short} on H100 (% of peak)")

ax.set_xlim(1, 1e5)
ax.set_ylim(1e12, 2e15)
ax.set_xlabel("Arithmetic intensity (FLOPs per byte of all-reduce traffic)")
ax.set_ylabel("Performance per GPU (FLOPs/s, BF16)")
ax.set_title("Interconnect roofline: H100 vs A100, 4 GPUs over NVLink")
ax.legend(loc="lower right", fontsize=8)
fig.text(0.01, 0.01, "vLLM runs: 200 prompts, 1024 in / 256 out, all sent at once. FLOPs ≈ 2 × params per token.",
         fontsize=7, color=TEXT_2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = os.path.join(PLOTS, "roofline.png")
fig.savefig(out, dpi=200)
print(f"wrote {out}")
