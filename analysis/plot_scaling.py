# Part 5.2: measured vs ideal linear throughput scaling, one panel per model.
import os
import matplotlib.pyplot as plt
from common import MODELS, PLOTS, SERIES, TEXT, TEXT_2, TEXT_3, SURFACE, setup_style, load_vllm_sweep

setup_style()
runs = load_vllm_sweep()
models = [m for m in MODELS if any(r["model"] == m for r in runs)]
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5), sharey=True)

for ax, model in zip(axes, models):
    rs = [r for r in runs if r["model"] == model]
    tps, out = [r["tp"] for r in rs], [r["out_tps"] for r in rs]
    base_tp, base = tps[0], out[0]
    ideal = [base * tp / base_tp for tp in tps]

    ax.plot(tps, ideal, color=TEXT_3, linewidth=1, linestyle="--")
    ax.annotate("ideal linear", (tps[-1], ideal[-1]), xytext=(-6, 6), textcoords="offset points",
                ha="right", fontsize=8, color=TEXT_2)
    ax.plot(tps, out, color=SERIES[0], marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    # label each step with its speedup over the smallest TP size
    for tp, y in zip(tps[1:], out[1:]):
        ax.annotate(f"{y:,.0f} tok/s\n{y / base:.1f}× vs TP={base_tp}", (tp, y), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8, color=TEXT)
    ax.annotate(f"{base:,.0f} tok/s", (base_tp, base), xytext=(8, -2), textcoords="offset points",
                fontsize=8, color=TEXT)

    if base_tp > 1:
        ax.annotate("TP=1: doesn't fit\n(BF16 weights > 80 GB)", (0.8, 1500), ha="left", fontsize=8, color=TEXT_2)
    ax.set_title(f"Qwen2.5-{MODELS[model]['short']}-Instruct")
    ax.set_xticks([1, 2, 4])
    ax.set_xlim(0.7, 4.3)
    ax.set_xlabel("Tensor-parallel size (GPUs)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

axes[0].set_ylabel("Output throughput (tokens/s)")
axes[0].set_ylim(0, None)
fig.suptitle("vLLM throughput scaling on 4× H100 SXM (NVLink)", x=0.01, ha="left", fontweight="bold")
fig.text(0.01, 0.01, "Ideal line starts at each model's smallest TP size. "
         "Gains above ideal come from the larger KV cache, which lets more requests run at once.",
         fontsize=7, color=TEXT_2)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
out_path = os.path.join(PLOTS, "scaling_curve.png")
fig.savefig(out_path, dpi=200)
print(f"wrote {out_path}")
