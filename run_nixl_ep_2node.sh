#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_nixl_ep_2node.sh --remote-host HOST --master-addr IP [options]

The local machine runs rank 0; HOST runs rank 1 over ssh.
Options after -- are passed to the Python benchmark.
EOF
}

REMOTE_HOST=""
MASTER_ADDR=""
REMOTE_ROOT="/path/to/cross-node"
ENV_FILE="nixl_ep_env.sh"
LOCAL_PYTHON=""
REMOTE_PYTHON=""
STORE_PORT="29591"
DEVICE0="0"
DEVICE1="0"
BENCH_ARGS=()

while (($#)); do
    case "$1" in
        --remote-host) REMOTE_HOST="$2"; shift 2 ;;
        --master-addr) MASTER_ADDR="$2"; shift 2 ;;
        --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
        --env-file) ENV_FILE="$2"; shift 2 ;;
        --local-python) LOCAL_PYTHON="$2"; shift 2 ;;
        --remote-python) REMOTE_PYTHON="$2"; shift 2 ;;
        --store-port) STORE_PORT="$2"; shift 2 ;;
        --device0) DEVICE0="$2"; shift 2 ;;
        --device1) DEVICE1="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; BENCH_ARGS+=("$@"); break ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$REMOTE_HOST" && -n "$MASTER_ADDR" ]] || { usage >&2; exit 2; }

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCRIPT="$SCRIPT_DIR/nixl_ep_cross_machine_bench.py"
ENV_PATH="$SCRIPT_DIR/../$ENV_FILE"
if [[ ! -f "$ENV_PATH" ]]; then
    ENV_PATH="$SCRIPT_DIR/$ENV_FILE"
fi
[[ -f "$ENV_PATH" ]] || { echo "environment file not found: $ENV_PATH" >&2; exit 2; }

if [[ -z "$LOCAL_PYTHON" ]]; then
    source "$ENV_PATH"
    LOCAL_PYTHON="${NIXL_VENV:?NIXL_VENV is not set}/bin/python"
fi
if [[ -z "$REMOTE_PYTHON" ]]; then
    REMOTE_PYTHON='${NIXL_VENV:?NIXL_VENV is not set}/bin/python'
fi

COMMON=(
    "$SCRIPT"
    --world-size 2
    --master-addr "$MASTER_ADDR"
    --store-port "$STORE_PORT"
    "${BENCH_ARGS[@]}"
)

cleanup() {
    if [[ -n "${REMOTE_PID:-}" ]]; then
        kill "$REMOTE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "$REMOTE_PYTHON" == *'${NIXL_VENV}'* ]]; then
    printf -v remote_command_text 'cd %q && source %q && exec %s %q --rank 1 --world-size 2 --master-addr %q --store-port %q --device %q' \
        "${REMOTE_ROOT}" "${REMOTE_ROOT}/${ENV_FILE}" "${REMOTE_PYTHON}" \
        "${REMOTE_ROOT}/test/nixl_ep_cross_machine_bench.py" "${MASTER_ADDR}" \
        "${STORE_PORT}" "${DEVICE1}"
else
    printf -v remote_command_text 'cd %q && source %q && exec %q %q --rank 1 --world-size 2 --master-addr %q --store-port %q --device %q' \
        "${REMOTE_ROOT}" "${REMOTE_ROOT}/${ENV_FILE}" "${REMOTE_PYTHON}" \
        "${REMOTE_ROOT}/test/nixl_ep_cross_machine_bench.py" "${MASTER_ADDR}" \
        "${STORE_PORT}" "${DEVICE1}"
fi
for argument in "${BENCH_ARGS[@]}"; do
    printf -v quoted_argument ' %q' "$argument"
    remote_command_text+="$quoted_argument"
done
echo "Starting remote rank 1 on $REMOTE_HOST"
ssh "$REMOTE_HOST" "$remote_command_text" &
REMOTE_PID=$!

sleep 1
printf 'Starting local rank 0\n'
"$LOCAL_PYTHON" "${COMMON[@]}" --rank 0 --device "$DEVICE0"
wait "$REMOTE_PID"
REMOTE_PID=""
