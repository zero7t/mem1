import argparse
import gzip
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def gunzip(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"exists: {dst}")
        return
    with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"wrote: {dst}")


def concat(parts, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"exists: {dst}")
        return
    with open(dst, "wb") as out:
        for part in parts:
            with open(part, "rb") as src:
                shutil.copyfileobj(src, out)
    print(f"wrote: {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_dir", default="/root/paddlejob/workspace/mem1/MEM1/assets/wiki-18")
    parser.add_argument("--model_dir", default="/root/paddlejob/workspace/mem1/MEM1/assets/models")
    parser.add_argument("--skip_models", action="store_true")
    args = parser.parse_args()

    asset_dir = Path(args.asset_dir)
    model_dir = Path(args.model_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    index_parts = []
    for filename in ["part_aa", "part_ab"]:
        path = hf_hub_download(
            repo_id="PeterJinGo/wiki-18-e5-index",
            filename=filename,
            repo_type="dataset",
            local_dir=str(asset_dir),
            resume_download=True,
        )
        index_parts.append(Path(path))
        print(f"downloaded: {path}")
    concat(index_parts, asset_dir / "e5_Flat.index")

    corpus_gz = hf_hub_download(
        repo_id="PeterJinGo/wiki-18-corpus",
        filename="wiki-18.jsonl.gz",
        repo_type="dataset",
        local_dir=str(asset_dir),
        resume_download=True,
    )
    print(f"downloaded: {corpus_gz}")
    gunzip(Path(corpus_gz), asset_dir / "wiki-18.jsonl")

    if not args.skip_models:
        for repo_id in ["intfloat/e5-base-v2", "Qwen/Qwen2.5-7B", "Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release"]:
            local = snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_dir / repo_id.replace("/", "__")),
                resume_download=True,
            )
            print(f"snapshot: {repo_id} -> {local}")


if __name__ == "__main__":
    main()
