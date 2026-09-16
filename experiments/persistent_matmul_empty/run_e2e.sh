#!/usr/bin/env bash
set -Eeuo pipefail

LABEL=${1:?arm label is required}
PLUGIN_ROOT=${2:?OPset checkout is required}
RESULT_ROOT=${3:?result directory is required}
PORT=${4:-18080}

VLLM_ROOT=${VLLM_ROOT:?set VLLM_ROOT to the pinned vLLM checkout}
MODEL_ROOT=${MODEL_ROOT:?set MODEL_ROOT to the local model snapshot}
DATASET=${DATASET:?set DATASET to the prepared dataset JSON}
PYTHON_BIN=${PYTHON_BIN:-python}
VLLM_BIN=${VLLM_BIN:-vllm}
DEVICE=${DEVICE:-0}

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/benchmark"
export PYTHONPATH="$PLUGIN_ROOT:$VLLM_ROOT:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"
export ASCEND_VISIBLE_DEVICES="$DEVICE"
export VLLM_PLUGINS=ascend
export VLLM_BATCH_INVARIANT=1
export VLLM_USE_V1=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export PROMETHEUS_MULTIPROC_DIR="$RESULT_ROOT/prometheus"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

{
  printf 'label=%s\n' "$LABEL"
  printf 'started_at=%s\n' "$(date -Is)"
  printf 'vllm_commit=%s\n' "$(git -C "$VLLM_ROOT" rev-parse HEAD)"
  printf 'plugin_commit=%s\n' "$(git -C "$PLUGIN_ROOT" rev-parse HEAD)"
  printf 'model_id=%s\n' "${MODEL_ID:-Qwen2.5-7B-Instruct}"
  printf 'dataset_id=%s\n' "${DATASET_ID:-custom}"
  printf 'dataset_sha256=%s\n' "$(sha256sum "$DATASET" | awk '{print $1}')"
  printf 'seed=12345\nnum_prompts=64\nmax_concurrency=4\noutput_len=128\n'
} > "$RESULT_ROOT/manifest.env"

setsid "$VLLM_BIN" serve "$MODEL_ROOT" \
  --served-model-name qwen2.5-7b \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --block-size 128 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.75 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --additional-config '{"enable_cpu_binding":false}' \
  > "$RESULT_ROOT/logs/vllm.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$RESULT_ROOT/server.pid"

for _ in $(seq 1 240); do
  if "$PYTHON_BIN" - "$PORT" <<'PY'
import sys, urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2) as r:
        raise SystemExit(0 if r.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
  then
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -100 "$RESULT_ROOT/logs/vllm.log"; exit 4; }
  sleep 2
done

"$PYTHON_BIN" - "$PORT" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
prompts = [
    "请简要介绍 Transformer 架构。",
    "请说明 Transformer 中自注意力机制的作用。",
    "请比较 Transformer 的编码器和解码器。",
    "请解释位置编码为什么重要。",
]
for prompt in prompts:
    body = json.dumps({
        "model": "qwen2.5-7b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 64,
    }, ensure_ascii=False).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        if response.status != 200:
            raise RuntimeError(response.status)
PY

"$VLLM_BIN" bench serve \
  --backend openai-chat \
  --base-url "http://127.0.0.1:$PORT" \
  --endpoint /v1/chat/completions \
  --model qwen2.5-7b \
  --tokenizer "$MODEL_ROOT" \
  --dataset-name sharegpt \
  --dataset-path "$DATASET" \
  --num-prompts 64 \
  --request-rate inf \
  --max-concurrency 4 \
  --seed 12345 \
  --temperature 0 \
  --output-len 128 \
  --save-result \
  --result-dir "$RESULT_ROOT/benchmark" \
  --result-filename result.json \
  2>&1 | tee "$RESULT_ROOT/logs/benchmark.log"

printf 'finished_at=%s\n' "$(date -Is)" >> "$RESULT_ROOT/manifest.env"
cleanup
SERVER_PID=""
