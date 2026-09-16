#!/usr/bin/env bash
set -Eeuo pipefail

LABEL=${1:?arm label is required}
PLUGIN_ROOT=${2:?OPset checkout is required}
PORT=${3:-18080}

VLLM_ROOT=${VLLM_ROOT:?set VLLM_ROOT to the pinned vLLM checkout}
MODEL_ROOT=${MODEL_ROOT:?set MODEL_ROOT to the local model snapshot}
DATASET=${DATASET:?set DATASET to the prepared dataset JSON}
PROFILER_RUNNER=${PROFILER_RUNNER:?set PROFILER_RUNNER to the msprof wrapper}
RESULT_ROOT=${RESULT_ROOT:?set RESULT_ROOT to a writable result directory}
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_DIR="$RESULT_ROOT/profiling/$LABEL"

export PYTHONPATH="$PLUGIN_ROOT:$VLLM_ROOT:${PYTHONPATH:-}"
export VLLM_BATCH_INVARIANT=1
export VLLM_PLUGINS=ascend,msserviceprofiler
export VLLM_USE_V1=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_REPO="$VLLM_ROOT"
export VLLM_BIN="$PROFILER_RUNNER"
export PYTHON_BIN
export PROFILING_ROOT="$RESULT_ROOT/profiler-output"
export RUN_DIR
export MODEL_PATH="$MODEL_ROOT"
export DATASET_PATH="$DATASET"
export PROFILE_SCENARIO=sharegpt-online
export DEVICE=${DEVICE:-0}
export PORT
export ENFORCE_EAGER=false
export NUM_PROMPTS=32
export REQUEST_RATE=inf
export BENCH_OUTPUT_LEN=128
export MAX_MODEL_LEN=2048
export MAX_NUM_SEQS=4
export MAX_NUM_BATCHED_TOKENS=2048
export GPU_MEMORY_UTILIZATION=0.75
export ADDITIONAL_CONFIG='{"enable_cpu_binding":false}'
export PROFILER_TIMELIMIT=900

mkdir -p "$RUN_DIR"
{
  printf 'label=%s\n' "$LABEL"
  printf 'vllm_commit=%s\n' "$(git -C "$VLLM_ROOT" rev-parse HEAD)"
  printf 'plugin_commit=%s\n' "$(git -C "$PLUGIN_ROOT" rev-parse HEAD)"
  printf 'dataset_sha256=%s\n' "$(sha256sum "$DATASET" | awk '{print $1}')"
} > "$RUN_DIR/manifest.env"

bash "$PROFILER_RUNNER"
