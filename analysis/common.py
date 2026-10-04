# Shared constants, data loaders and plot style for the Part 3/5 analysis scripts.
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")

# --- Hardware peaks (vendor spec sheets) ---
# NVLink bandwidth is per direction, which is what NCCL busbw is comparable to
HARDWARE = {
    "H100": {"label": "H100 SXM", "peak_flops": 989e12, "spec_bw": 450e9, "dir": "h100_4gpu"},  # BF16 dense, NVLink 4
    "A100": {"label": "A100 SXM", "peak_flops": 312e12, "spec_bw": 300e9, "dir": "a100_4gpu"},  # BF16 dense, NVLink 3
}

# --- Model shapes (from each model's config.json on Hugging Face) ---
MODELS = {
    "Qwen/Qwen2.5-32B-Instruct": {"short": "32B", "params": 32.76e9, "hidden": 5120, "layers": 64},
    "Qwen/Qwen2.5-72B-Instruct": {"short": "72B", "params": 72.71e9, "hidden": 8192, "layers": 80},
}
BYTES_PER_ELEM = 2  # BF16

# --- Palette (validated light-mode categorical slots 1-3) and text/surface tokens ---
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
TEXT_3 = "#8a8983"
GRID = "#e4e3df"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.edgecolor": GRID, "axes.linewidth": 1, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
        "axes.spines.top": False, "axes.axisbelow": True, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.labelcolor": TEXT_2,
    })
    os.makedirs(PLOTS, exist_ok=True)


def load_nccl(gpu="H100"):
    """Rows from a Part 2 sweep CSV, numeric fields converted; failed runs dropped."""
    path = os.path.join(RESULTS, HARDWARE[gpu]["dir"], "nccl_sweep_results.csv")
    rows = []
    for r in csv.DictReader(open(path)):
        if r["returncode"] != "0" or not r["size_bytes"]:
            continue
        rows.append({"op": r["op"], "gpus": int(r["gpus"]), "size": int(r["size_bytes"]),
                     "busbw_oop": float(r["busbw_oop_gbps"]) * 1e9, "busbw_ip": float(r["busbw_ip_gbps"]) * 1e9})
    return rows


def peak_busbw(rows, op="all_reduce_perf", gpus=4):
    return max(r["busbw_oop"] for r in rows if r["op"] == op and r["gpus"] == gpus)


def busbw_at(rows, size, gpus, op="all_reduce_perf"):
    """In-place busbw at an arbitrary message size, interpolated log-log between measured sizes."""
    pts = sorted((r["size"], r["busbw_ip"]) for r in rows if r["op"] == op and r["gpus"] == gpus and r["busbw_ip"] > 0)
    sizes, bws = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    return float(np.exp(np.interp(np.log(size), np.log(sizes), np.log(bws))))


def load_vllm_sweep():
    path = os.path.join(RESULTS, "h100_4gpu", "vllm", "sweep", "vllm_scaling_results.csv")
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({"model": r["model_id"], "tp": int(r["tp"]),
                     "out_tps": float(r["output_throughput"]), "total_tps": float(r["total_token_throughput"]),
                     "tpot_s": float(r["mean_tpot_ms"]) / 1000, "median_ttft_s": float(r["median_ttft_ms"]) / 1000})
    return sorted(rows, key=lambda r: (r["model"], r["tp"]))


def allreduce_bytes_per_token(model, tp):
    """Bytes each GPU sends per token for TP all-reduces: 2 per layer (attention + MLP), ring algorithm."""
    m = MODELS[model]
    return 2 * m["layers"] * (2 * (tp - 1) / tp) * m["hidden"] * BYTES_PER_ELEM


def flops_per_token_per_gpu(model, tp):
    # ~2 FLOPs per parameter per token; attention FLOPs ignored (small at 1-2k context)
    return 2 * MODELS[model]["params"] / tp
