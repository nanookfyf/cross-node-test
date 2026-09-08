#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../vllm" && pwd)"
VENV="${VENV:-$ROOT/.venv}"
MODEL="${MODEL:-/nvme1/fyf/model/DeepSeek-V2-Lite}"
GPUS="${GPUS:-1,2}"
HEAD_IP="${HEAD_IP:-<ip>}"
RAY_PORT="${RAY_PORT:-9305}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8267}"
API_HOST="${API_HOST:-<>}"
API_PORT="${API_PORT:-8005}"
NODE_IP="${NODE_IP:-$(ip -o -4 addr show | awk '$4 ~ /^10\.0\.0\./ {sub("/.*", "", $4); print $4; exit}')}"
DP_SIZE="${DP_SIZE:-4}"
DP_SIZE_LOCAL="${DP_SIZE_LOCAL:-2}"
GPU_UTIL="${GPU_UTIL:-0.35}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
ALL2ALL="${ALL2ALL:-nixl_ep}"
EPLB_USE_ASYNC="${EPLB_USE_ASYNC:-false}"
EPLB_COMM="${EPLB_COMM:-}"
EPLB_COMM_ARGS=()
[ -z "$EPLB_COMM" ] || EPLB_COMM_ARGS=(--eplb-config.communicator "$EPLB_COMM")
RAY_TMP="${RAY_TMP:-/tmp/elastic-ray-$(hostname -s)}"
LOG_FILE="${LOG_FILE:-/tmp/elastic-vllm.log}"
PID_FILE="${PID_FILE:-/tmp/elastic-vllm.pid}"

export PATH="$VENV/bin:$HOME/.local/bin:/shared/Applications/miniconda3/bin:$PATH"
NIXL_OPENSSL_DIR="${NIXL_OPENSSL_DIR:-/usr/local/cuda-12.8/nsight-systems-2024.6.2/host-linux-x64}"
export LD_LIBRARY_PATH="$NIXL_OPENSSL_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
export UCX_TLS="${UCX_TLS:-all}"
export RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL="${RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL:-minor}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"

is_head() {
    [ "$NODE_IP" = "$HEAD_IP" ]
}

ray_cmd() {
    "$VENV/bin/ray" "$@"
}

stop_managed_ray() {
    local pids
    pids=$(ps -eo pid=,args= | awk -v tmp="$RAY_TMP" \
        'index($0, "--temp-dir=" tmp) || index($0, "--temp_dir=" tmp) || index($0, "--session-dir=" tmp) || index($0, "--session_dir=" tmp) {print $1}')
    if [ -n "$pids" ]; then
        kill $pids 2>/dev/null || true
        sleep 2
        pids=$(ps -eo pid=,args= | awk -v tmp="$RAY_TMP" \
            'index($0, "--temp-dir=" tmp) || index($0, "--temp_dir=" tmp) || index($0, "--session-dir=" tmp) || index($0, "--session_dir=" tmp) {print $1}')
        [ -z "$pids" ] || kill -9 $pids 2>/dev/null || true
    fi
}

usage() {
    cat <<EOF
Usage: $0 {head|worker|start|serve|stop|status|scale} [target_dp]

head    Start the Ray head on ${HEAD_IP}.
worker  Join this host to the Ray cluster.
start   Start Ray according to NODE_ROLE, then start vLLM on the head.
serve   Start vLLM on the head using the existing Ray cluster.
stop    Stop the vLLM service on the head, or Ray on a worker.
status  Show Ray and HTTP service status.
scale   Change DP size, for example: $0 scale 2

Environment overrides: MODEL GPUS HEAD_IP RAY_PORT API_PORT DP_SIZE
DP_SIZE_LOCAL GPU_UTIL MAX_MODEL_LEN RAY_TMP LOG_FILE PID_FILE
EOF
}

start_head() {
    if timeout 5 "$VENV/bin/ray" status --address="$RAY_ADDRESS" >/dev/null 2>&1; then
        return
    fi
    CUDA_VISIBLE_DEVICES="$GPUS" ray_cmd start \
        --head \
        --node-ip-address="$HEAD_IP" \
        --port="$RAY_PORT" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="$RAY_DASHBOARD_PORT" \
        --temp-dir="$RAY_TMP" \
        --disable-usage-stats
}

start_worker() {
    stop_managed_ray
    CUDA_VISIBLE_DEVICES="$GPUS" ray_cmd start \
        --address="$RAY_ADDRESS" \
        --node-ip-address="$NODE_IP" \
        --temp-dir="$RAY_TMP" \
        --disable-usage-stats
}

serve() {
    if ! is_head; then
        echo "serve must run on the Ray head ${HEAD_IP}" >&2
        exit 2
    fi
    cd "$ROOT"
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "vLLM is already running with PID $(cat "$PID_FILE")"
        return
    fi
    mkdir -p "$(dirname "$LOG_FILE")"
    export CUDA_VISIBLE_DEVICES="$GPUS"
    export RAY_DEDUP_LOGS=0
    export VLLM_SERVER_DEV_MODE=1
    export PYTHONUNBUFFERED=1
    nohup "$VENV/bin/vllm" serve "$MODEL" \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port "$API_PORT" \
        --enforce-eager \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs 4 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization "$GPU_UTIL" \
        --enable-prefix-caching \
        --enable-expert-parallel \
        --enable-eplb \
        --enable-elastic-ep \
        --enable-sleep-mode \
        --eplb-config.num_redundant_experts 64 \
        --eplb-config.use_async "$EPLB_USE_ASYNC" \
        "${EPLB_COMM_ARGS[@]}" \
        --all2all-backend "$ALL2ALL" \
        --data-parallel-backend ray \
        --distributed-executor-backend ray \
        --data-parallel-address "$HEAD_IP" \
        --data-parallel-size "$DP_SIZE" \
        --data-parallel-size-local "$DP_SIZE_LOCAL" \
        >"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "vLLM started with PID $(cat "$PID_FILE"); log: $LOG_FILE"
}

stop() {
    if is_head && [ -f "$PID_FILE" ]; then
        kill "$(cat "$PID_FILE")" 2>/dev/null || true
        rm -f "$PID_FILE"
    fi
    stop_managed_ray
}

status() {
    ray_cmd status --address="$RAY_ADDRESS" || true
    curl -sS -o /dev/null -w "HTTP health: %{http_code}\n" "http://${API_HOST}:${API_PORT}/health" || true
}

scale() {
    TARGET="${2:-}"
    if [ -z "$TARGET" ]; then
        echo "scale requires a target DP size" >&2
        exit 2
    fi
    curl -fsS --max-time 300 \
        -X POST "http://${API_HOST}:${API_PORT}/flash_epscale" \
        -H 'Content-Type: application/json' \
        -d "{\"ep_size\":${TARGET},\"level\":${SLEEP_LEVEL:-1}}"
    echo
}

case "$ROLE" in
    head)
        start_head
        ;;
    worker)
        start_worker
        ;;
    start)
        if [ "${NODE_ROLE:-}" = "worker" ]; then
            start_worker
        else
            start_head
            serve
        fi
        ;;
    serve)
        serve
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    scale)
        scale "$@"
        ;;
    *)
        usage
        exit 2
        ;;
esac
