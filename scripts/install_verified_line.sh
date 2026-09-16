#!/usr/bin/env bash
set -euo pipefail

: "${VLLM_SRC:?set VLLM_SRC to the pinned vLLM checkout}"
: "${VLLM_ASCEND_SRC:?set VLLM_ASCEND_SRC to the pinned vLLM-Ascend-HUST checkout}"
: "${VLLM_ENV_PYTHON:?set VLLM_ENV_PYTHON to the target environment python}"

check_rev() {
  local path="$1" expected="$2" name="$3" actual
  actual="$(git -C "$path" rev-parse HEAD)"
  test "$actual" = "$expected" || {
    echo "$name revision mismatch: expected $expected, got $actual" >&2
    exit 2
  }
  test -z "$(git -C "$path" status --porcelain)" || {
    echo "$name checkout is dirty: $path" >&2
    exit 2
  }
}

check_rev "$VLLM_SRC" ba07e4a48fc951300d97eb506217dd530583dea3 vllm
check_rev "$VLLM_ASCEND_SRC" 588cdbdc577aa2c2d2e1e0ab362e39f6913d02e3 vllm-ascend-hust
if test -n "${TRITON_ASCEND_SRC:-}"; then
  check_rev "$TRITON_ASCEND_SRC" a4e5e4b53959de21563754641f2193eea067994e triton-ascend
fi

"$VLLM_ENV_PYTHON" - <<'PY'
from importlib.metadata import version

expected = {
    "torch": "2.10.0",
    "torch-npu": "2.10.0.post4",
    "triton-ascend": "3.2.2",
}
bad = []
for package, wanted in expected.items():
    got = version(package).split("+")[0]
    if got != wanted:
        bad.append(f"{package}: expected {wanted}, got {got}")
if bad:
    raise SystemExit("incompatible environment:\n" + "\n".join(bad))
print("verified package versions")
PY

WHEEL="${1:-dist/vllm_hust_operator_optimizations-0.1.1-py3-none-any.whl}"
"$VLLM_ENV_PYTHON" -m pip install --no-deps --force-reinstall "$WHEEL"
echo "installed verified operator-optimization line"
