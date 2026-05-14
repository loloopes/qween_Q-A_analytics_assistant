"""Download a Hugging Face model repo to a local folder (for offline / stable loading).

Reads from `.env` in this folder: `BASE_MODEL`, optional `HF_TOKEN`, optional `HF_ENDPOINT`.

If `huggingface.co` does not resolve (Windows error `getaddrinfo failed`), fix DNS/VPN or use a
mirror, for example in `.env`:

  HF_ENDPOINT=https://hf-mirror.com

Then open a **new** terminal so the variable is picked up, and run this script again.

Usage:
  python download_hf_model.py
  python download_hf_model.py --repo-id Qwen/Qwen2.5-0.5B-Instruct --local-dir ./hf_models/my-qwen
  python download_hf_model.py --endpoint https://hf-mirror.com
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install huggingface_hub: pip install -U huggingface_hub") from exc


def _sanitize_repo_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

    parser = argparse.ArgumentParser(description="Download a model snapshot from Hugging Face Hub.")
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("BASE_MODEL", "").strip(),
        help="Model repo id (default: BASE_MODEL from .env)",
    )
    parser.add_argument(
        "--local-dir",
        default="",
        help="Destination directory (default: ./hf_models/<repo>)",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "").strip(),
        help="Hub API base URL (or set HF_ENDPOINT in .env), e.g. https://hf-mirror.com",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel download workers",
    )
    args = parser.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint.rstrip("/")
        print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")

    repo_id = args.repo_id
    if not repo_id:
        raise SystemExit("Set BASE_MODEL in .env or pass --repo-id.")

    token = os.environ.get("HF_TOKEN") or None

    api_dir = Path(__file__).resolve().parent
    if args.local_dir:
        local_dir = Path(args.local_dir).expanduser().resolve()
    else:
        local_dir = (api_dir / "hf_models" / _sanitize_repo_name(repo_id)).resolve()

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Repo: {repo_id}")
    print(f"Destination: {local_dir}")
    print(
        "Starting snapshot_download. Large models (e.g. Qwen3-30B) are many GB and can take hours.\n"
    )

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=token,
            max_workers=args.max_workers,
        )
    except OSError as exc:
        msg = (
            "\nCould not reach the Hugging Face Hub. Check internet and DNS "
            "(try `ping huggingface.co`). If you are in a restricted region, set HF_ENDPOINT "
            "to a mirror (see docstring) and retry from a new shell.\n"
        )
        print(msg, file=sys.stderr)
        raise SystemExit(1) from exc

    print("\nDone. Use this folder as the model path (restart kernel after editing .env):")
    print(f"  BASE_MODEL={local_dir.as_posix()}")


if __name__ == "__main__":
    main()
