import subprocess, csv, os, sys

# combinations of ops and counts
gpu_counts = [1,2,3,4]
ops = ["all_reduce_perf","all_gather_perf", "broadcast_perf", "alltoall_perf"]
# nccl-tests build dir; override with NCCL_TESTS_BUILD if not run from the nccl-tests dir
build_dir = os.environ.get("NCCL_TESTS_BUILD", "./build")
# NVLS (NVLink SHARP) fails on this RunPod 4-of-8 GPU slice at 3+ GPUs, so disable it by default;
# run with NCCL_NVLS_ENABLE=1 to try it anyway
env = {**os.environ, "NCCL_NVLS_ENABLE": os.environ.get("NCCL_NVLS_ENABLE", "0")}
rows = []

def parse_busbw(stdout):
    # data lines start with the message size; busbw is the 3rd-from-last column of each
    # half (out-of-place, in-place), so index from the end to handle both output layouts
    results = []
    for line in stdout.splitlines():
        fields = line.split()
        if not fields or not fields[0].isdigit():
            continue
        results.append({
            "size_bytes": int(fields[0]),
            "busbw_oop_gbps": float(fields[-6]),
            "busbw_ip_gbps": float(fields[-2]),
        })
    return results

# check all binaries up front so a missing one doesn't abort mid-sweep
missing = [op for op in ops if not os.path.isfile(os.path.join(build_dir, op))]
if missing:
    sys.exit(f"{missing} not found in {build_dir}; run from the nccl-tests dir or set NCCL_TESTS_BUILD")

# running tests with results capturing
for op in ops:
    binary = os.path.join(build_dir, op)
    for count in gpu_counts:
        results = subprocess.run([binary, "-b", "8", "-e", "8G", "-f", "2", "-g", str(count)],
                                 capture_output=True, text=True, env=env)
        parsed = parse_busbw(results.stdout) if results.returncode == 0 else []
        if not parsed:
            # keep a row for the failed run so it shows up in the CSV
            # nccl-tests prints its errors to stdout, so fall back to it when stderr is empty
            error = (results.stderr.strip() or results.stdout.strip())[-500:]
            print(f"FAILED {op} gpus={count} rc={results.returncode}: {error}", file=sys.stderr)
            rows.append({"op": op, "gpus": count, "size_bytes": "", "busbw_oop_gbps": "", "busbw_ip_gbps": "",
                         "returncode": results.returncode, "error": error})
            continue
        print(f"ok {op} gpus={count}: {len(parsed)} sizes")
        for r in parsed:
            rows.append({"op": op, "gpus": count, **r, "returncode": 0, "error": ""})

with open("nccl_sweep_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
