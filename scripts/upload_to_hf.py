"""Upload the prepared release checkpoint + model card to Hugging Face.

Creates the repo if it doesn't exist, then pushes pytorch_model.bin and
README.md. Idempotent — re-running with the same files just re-uploads.

Usage:
    python scripts/upload_to_hf.py
    python scripts/upload_to_hf.py --repo mirrorethic/t3-124m-v36 --private
    python scripts/upload_to_hf.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="mirrorethic/t3-124m-v36")
    ap.add_argument("--release-dir", default="release")
    ap.add_argument("--private", action="store_true",
                    help="Create as a private repo (default: public)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan without creating or uploading anything")
    args = ap.parse_args()

    rel = Path(args.release_dir)
    ckpt = rel / "pytorch_model.bin"
    card = rel / "README.md"
    for p in (ckpt, card):
        if not p.exists():
            raise SystemExit(f"missing release artifact: {p}")

    print(f"plan: upload to {args.repo}  (private={args.private})")
    print(f"  ckpt: {ckpt}  ({ckpt.stat().st_size / 1e6:.1f} MB)")
    print(f"  card: {card}  ({card.stat().st_size} bytes)")
    if args.dry_run:
        print("dry-run, exiting.")
        return

    api = HfApi()

    print(f"\ncreating repo {args.repo} (exist_ok=True) ...")
    create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    print(f"\nuploading README.md ...")
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message="Add model card (run-3 release)",
    )

    print(f"\nuploading pytorch_model.bin ({ckpt.stat().st_size / 1e6:.1f} MB) ...")
    api.upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo="pytorch_model.bin",
        repo_id=args.repo,
        repo_type="model",
        commit_message="Add canonical run-3 release checkpoint (val_ppl=27.76)",
    )

    print(f"\ndone. View at https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
