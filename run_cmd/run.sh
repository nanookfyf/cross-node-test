#!/usr/bin/env bash
set -euo pipefail

MODEL="models/Qwen3-30B-A3B-Instruct-2507"
LOG_FILE="/home/fengyunfei/eursys/logfscale.txt"
CUDA_VISIBLE_DEVICES=4,5,6,7
RAY_HOST="127.0.0.1"
RAY_PORT="9305"

export VLLM_NIXL_EP_DEBUG=1
# custom NCCL 2.30.7 with ncclCommSuspend/ncclCommResume, needed for
# flash_epscale to suspend the max-DP communicator after scale-down
export RAY_ADDRESS="${RAY_HOST}:${RAY_PORT}"
export VLLM_SERVER_DEV_MODE=1
export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1


export VLLM_MOE_SHAPE_DEBUG=1
mkdir -p "$(dirname "$LOG_FILE")"

# 如果本机没有固定 Ray 实例，就起一个。
# 如果你机器上可能有别的 Ray 任务，不建议自动 ray stop。
if ! ray status --address="$RAY_ADDRESS" >/dev/null 2>&1; then
    ray start --head --port="$RAY_PORT" --disable-usage-stats
fi
CUDA_LAUNCH_BLOCKING=1

vllm serve "$MODEL" --trust-remote-code \
    --host 0.0.0.0 \
    --port 8005 \
    --enforce-eager \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.5 \
    --enable-sleep-mode \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --enable-eplb \
    --enable-elastic-ep \
    --all2all-backend nixl_ep \
    --eplb-config.num_redundant_experts 128 \
    --data-parallel-backend ray \
    --distributed-executor-backend ray \
    --data-parallel-size 4  \
    --data-parallel-size-local  4\
    --data-parallel-rpc-port 9876 \
    --data-parallel-start-rank 0 \
    2>&1 | stdbuf -oL -eL sed -u \
        -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' \
        -e 's/(APIServer pid=[0-9]*) *//g' \
        -e 's/(DPMoEEngineCoreActor pid=[0-9]*) *//g' \
        -e 's/(RayWorkerWrapper pid=[0-9]*) *//g' \
        -e 's/^ *//' \
        | tee "$LOG_FILE"
