# Collect vllm bench serve result JSONs (named <model>_tp<N>.json) into one CSV for Part 5 plotting.
# Usage: python3 inference_bench/collect_bench.py results/h100_4gpu/vllm/sweep
import csv, glob, json, os, re, sys

bench_dir = sys.argv[1] if len(sys.argv) > 1 else "."
fields = ["model_id", "tp", "request_rate", "num_prompts", "completed", "failed", "duration",
          "output_throughput", "total_token_throughput", "request_throughput",
          "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms"]
rows = []

for path in sorted(glob.glob(os.path.join(bench_dir, "*_tp*.json"))):
    m = re.search(r"_tp(\d+)\.json$", path)
    if not m:
        continue
    d = json.load(open(path))
    rows.append({**{k: d.get(k) for k in fields}, "tp": int(m.group(1))})

rows.sort(key=lambda r: (r["model_id"], r["tp"]))
out = os.path.join(bench_dir, "vllm_scaling_results.csv")
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {out}")
