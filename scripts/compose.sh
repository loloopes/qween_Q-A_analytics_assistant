#!/usr/bin/env bash
# Run docker compose for the LLM stack with CPU/GPU selected by LLM_USE_GPU in llm/.env
#
#   LLM_USE_GPU=false   → CPU (default for WSL/Mac without NVIDIA in Docker)
#   LLM_USE_GPU=true    → GPU (requires NVIDIA Container Toolkit)
#   LLM_USE_GPU=auto    → GPU only if `docker run --gpus all` works
#
# Examples:
#   ./scripts/compose.sh up -d --build
#   LLM_USE_GPU=true ./scripts/compose.sh up -d --build
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

_llm_use_gpu_mode() {
  local raw="${LLM_USE_GPU:-auto}"
  local value
  value="$(echo "$raw" | tr '[:upper:]' '[:lower:]')"
  case "$value" in
    1|true|yes|on|gpu) echo true ;;
    0|false|no|off|cpu) echo false ;;
    auto) echo auto ;;
    *) echo auto ;;
  esac
}

_gpu_available_in_docker() {
  docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1
}

compose_files=(-f docker-compose.yml)
mode="$(_llm_use_gpu_mode)"
use_gpu=false

case "$mode" in
  true) use_gpu=true ;;
  false) use_gpu=false ;;
  auto)
    if _gpu_available_in_docker; then
      use_gpu=true
    fi
    ;;
esac

if [[ "$use_gpu" == true ]]; then
  compose_files+=(-f docker-compose.gpu.yml)
  export TORCH_VARIANT=gpu
  export LLM_DEVICE=cuda
  echo "==> LLM runtime: GPU (LLM_USE_GPU=${LLM_USE_GPU:-auto})"
else
  export TORCH_VARIANT=cpu
  export LLM_DEVICE=cpu
  if [[ "$mode" == true ]]; then
    echo "WARNING: LLM_USE_GPU=true but NVIDIA is not available in Docker — falling back to CPU" >&2
  else
    echo "==> LLM runtime: CPU (LLM_USE_GPU=${LLM_USE_GPU:-auto})"
  fi
fi

exec docker compose "${compose_files[@]}" "$@"
