#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:?model path is required}
DEVICE_ID=${2:-7}
PORT=${3:-18080}
PYTHON_ENV=${PYTHON_ENV:?set PYTHON_ENV to the verified virtual environment}
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}
ATB_ENV=${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}

set +u
source "$CANN_ENV"
source "$ATB_ENV" --cxx_abi=1
set -u

RUNNER=("$PYTHON_ENV/bin/vllm-hust-ext" run -- "$PYTHON_ENV/bin/vllm")
if [[ "${DIRECT_VLLM_RUN:-0}" == "1" ]]; then
  RUNNER=("$PYTHON_ENV/bin/vllm")
fi
EXTRA_ARGS=(--additional-config '{"enable_cpu_binding":false}')

exec sudo -n env \
  HOME="$HOME" \
  PATH="$PATH" \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  PYTHONPATH="${PYTHONPATH:-}" \
  ASCEND_AICPU_PATH="${ASCEND_AICPU_PATH:-}" \
  ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-}" \
  ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-}" \
  ASCEND_RT_VISIBLE_DEVICES="$DEVICE_ID" \
  ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-}" \
  ASCEND_TOOLKIT_LATEST_HOME="${ASCEND_TOOLKIT_LATEST_HOME:-}" \
  ATB_HOME_PATH="${ATB_HOME_PATH:-}" \
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  VLLM_BATCH_INVARIANT=1 \
  VLLM_PLUGINS=ascend,operator_optimizations \
  VLLM_HUST_OPERATOR_OPTIMIZATIONS="${VLLM_HUST_OPERATOR_OPTIMIZATIONS:-persistent-matmul-empty,linear-swiglu-graph}" \
  "${RUNNER[@]}" serve "$MODEL_PATH" \
  --served-model-name qwen2.5-7b-plugin-smoke \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --block-size 128 \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.65 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  "${EXTRA_ARGS[@]}"
