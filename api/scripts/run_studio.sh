#!/usr/bin/env bash
# LangGraph Studio — visual debugger at https://smith.langchain.com/studio/
# Requires LANGSMITH_API_KEY in llm/.env (see api/env.sample).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v langgraph >/dev/null 2>&1; then
  echo "Installing langgraph-cli..."
  pip install -U "langgraph-cli[inmem]" langsmith
fi

echo "Starting LangGraph dev server (use --tunnel if Studio cannot reach localhost)..."
exec langgraph dev --host 127.0.0.1 --port 2024 "$@"
