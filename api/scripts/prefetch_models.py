"""Optional build-time step: cache Hugging Face base model + LoRA adapter."""

import os
import sys

from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_REPO = os.getenv("ADAPTER_REPO", "Glccampos/llm_qween")
EMBEDDING_MODEL = os.getenv(
    "SEMANTIC_CACHE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
HF_TOKEN = os.getenv("HF_TOKEN") or None


def main() -> None:
    if not HF_TOKEN:
        print("HF_TOKEN not set; skipping model prefetch.", file=sys.stderr)
        return

    print(f"Prefetching tokenizer from {ADAPTER_REPO}...", file=sys.stderr)
    AutoTokenizer.from_pretrained(
        ADAPTER_REPO,
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    print(f"Prefetching base model {BASE_MODEL}...", file=sys.stderr)
    snapshot_download(BASE_MODEL, token=HF_TOKEN)

    print(f"Prefetching adapter {ADAPTER_REPO}...", file=sys.stderr)
    snapshot_download(ADAPTER_REPO, token=HF_TOKEN)

    print(f"Prefetching semantic cache embedder {EMBEDDING_MODEL}...", file=sys.stderr)
    snapshot_download(EMBEDDING_MODEL, token=HF_TOKEN)

    print("Verifying load (CPU)...", file=sys.stderr)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    PeftModel.from_pretrained(base, ADAPTER_REPO, token=HF_TOKEN)
    print("Prefetch complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
