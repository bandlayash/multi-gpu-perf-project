# Part 5.3: estimate how much of each decode step goes to TP all-reduces, using the Part 2 NCCL data.
# Decode batch is estimated with Little's law: sequences in flight ≈ output tokens/s × time per output token.
import csv, os
import matplotlib.pyplot as plt
from common import (MODELS, PLOTS, RESULTS, BYTES_PER_ELEM, SERIES, TEXT, TEXT_2, setup_style, load_nccl,
                    busbw_at, load_vllm_sweep)

nccl = load_nccl("H100")
rows = []
for r in load_vllm_sweep():
    if r["tp"] == 1:
        continue
    m = MODELS[r["model"]]
    batch = r["out_tps"] * r["tpot_s"]
    msg = batch * m["hidden"] * BYTES_PER_ELEM          # one all-reduce: activations for the whole batch
    bw = busbw_at(nccl, msg, r["tp"])
    t_ar = msg * 2 * (r["tp"] - 1) / r["tp"] / bw       # busbw = algbw × 2(n-1)/n
    t_step = 2 * m["layers"] * t_ar                     # 2 all-reduces per layer
    rows.append({"model": m["short"], "tp": r["tp"], "est_batch": round(batch), "msg_kb": msg / 1024,
                 "busbw_gbps": bw / 1e9, "allreduce_us": t_ar * 1e6, "comm_ms_per_step": t_step * 1e3,
                 "tpot_ms": r["tpot_s"] * 1e3, "comm_share": t_step / r["tpot_s"]})

hdr = f"{'model':>5} {'tp':>3} {'batch':>6} {'msg KB':>8} {'busbw GB/s':>11} {'AR µs':>7} {'comm ms':>8} {'TPOT ms':>8} {'share':>6}"
print(hdr)
for x in rows:
    print(f"{x['model']:>5} {x['tp']:>3} {x['est_batch']:>6} {x['msg_kb']:>8.0f} {x['busbw_gbps']:>11.1f} "
          f"{x['allreduce_us']:>7.1f} {x['comm_ms_per_step']:>8.2f} {x['tpot_ms']:>8.1f} {x['comm_share']:>6.1%}")

csv_path = os.path.join(RESULTS, "h100_4gpu", "vllm", "sweep", "comm_share.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {csv_path}")

setup_style()
fig, ax = plt.subplots(figsize=(7, 3.2))
labels = [f"{x['model']} TP={x['tp']}" for x in rows][::-1]
shares = [x["comm_share"] * 100 for x in rows][::-1]
bars = ax.barh(labels, shares, height=0.5, color=SERIES[0])
for bar, x in zip(bars, rows[::-1]):
    ax.annotate(f"{x['comm_share']:.0%}  ({x['comm_ms_per_step']:.1f} of {x['tpot_ms']:.0f} ms, batch ≈ {x['est_batch']})",
                (bar.get_width(), bar.get_y() + bar.get_height() / 2), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=TEXT)
ax.set_xlim(0, max(shares) * 1.9)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Estimated all-reduce time as % of time per output token")
ax.set_title("How much of each decode step is TP communication?")
fig.text(0.01, 0.01, "Estimate from NCCL all-reduce busbw at the decode message size. vLLM uses its own all-reduce "
         "for small messages, so treat as approximate.", fontsize=7, color=TEXT_2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
out = os.path.join(PLOTS, "comm_share.png")
fig.savefig(out, dpi=200)
print(f"wrote {out}")
