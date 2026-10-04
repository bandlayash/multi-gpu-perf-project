#!/usr/bin/env bash
# Part 5 scaling sweep: for each TP size, start vLLM, wait until healthy, run the benchmark, stop the server.
# Usage (on the GPU host): MODEL=Qwen/Qwen2.5-32B-Instruct TPS="1 2 4" bash run_vllm_sweep.sh
set -u

MODEL=${MODEL:-Qwen/Qwen2.5-32B-Instruct}
TPS=${TPS:-"1 2 4"}
# same memory setting for every TP size so runs are comparable
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.95}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
# request rate inf = send everything at once, so throughput measures capacity rather than the arrival rate
REQUEST_RATE=${REQUEST_RATE:-inf}
NUM_PROMPTS=${NUM_PROMPTS:-200}
OUT_DIR=${OUT_DIR:-/workspace/bench}
PORT=8000

export HF_HOME=${HF_HOME:-/workspace/hf}
export NCCL_NVLS_ENABLE=0  # NVLS fails on this RunPod 4-GPU slice
mkdir -p "$OUT_DIR"
name=$(basename "$MODEL" | tr 'A-Z.' 'a-z_')

for tp in $TPS; do
    tag="${name}_tp${tp}"
    echo "=== $tag: starting server"
    pkill -f "vllm serve" 2>/dev/null
    # wait for the previous server's workers to release GPU memory
    for _ in $(seq 1 60); do
        [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break
        sleep 2
    done
    vllm serve "$MODEL" --tensor-parallel-size "$tp" --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" --port "$PORT" > "$OUT_DIR/${tag}_serve.log" 2>&1 &
    pid=$!

    # startup includes compilation and can take ~10 min
    for _ in $(seq 1 180); do
        curl -sf "localhost:$PORT/health" > /dev/null && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 5
    done
    if ! curl -sf "localhost:$PORT/health" > /dev/null; then
        echo "=== $tag: FAILED to start, see $OUT_DIR/${tag}_serve.log"; tail -5 "$OUT_DIR/${tag}_serve.log"
        kill "$pid" 2>/dev/null; continue
    fi

    echo "=== $tag: benchmarking"
    vllm bench serve --backend vllm --model "$MODEL" --port "$PORT" \
        --dataset-name random --random-input-len 1024 --random-output-len 256 \
        --num-prompts "$NUM_PROMPTS" --request-rate "$REQUEST_RATE" --seed 0 \
        --save-result --result-dir "$OUT_DIR" --result-filename "${tag}.json" \
        > "$OUT_DIR/${tag}_bench.log" 2>&1 || echo "=== $tag: benchmark FAILED, see $OUT_DIR/${tag}_bench.log"

    kill "$pid"; wait "$pid" 2>/dev/null
    echo "=== $tag: done"
done
pkill -f "vllm serve" 2>/dev/null
echo "results in $OUT_DIR"
