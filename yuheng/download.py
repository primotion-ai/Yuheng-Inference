import argparse
import os
from pathlib import Path

from .releases import RELEASES


def main():
    parser = argparse.ArgumentParser(description="Download the published Yuheng weights")
    parser.add_argument("--model", choices=["4b", "27b", "both"], required=True)
    parser.add_argument("--source", choices=["official", "mirror"], default="official")
    parser.add_argument("--output-dir", type=Path, default=Path("weights"),
                        help="Parent directory for Yuheng-4B and Yuheng-27B")
    parser.add_argument("--dry-run", action="store_true", help="Resolve remote files and sizes without downloading")
    args = parser.parse_args()
    endpoint = {"official": "https://huggingface.co", "mirror": "https://hf-mirror.com"}[args.source]
    os.environ["HF_ENDPOINT"] = endpoint
    from huggingface_hub import snapshot_download

    for key in RELEASES if args.model == "both" else [args.model]:
        model = RELEASES[key]
        directory = args.output_dir / model.repo.split("/")[1]
        info = snapshot_download(model.repo, revision=model.revision, endpoint=endpoint, token=False,
                                 allow_patterns=list(model.filenames), local_dir=directory,
                                 dry_run=args.dry_run)
        if args.dry_run:
            if {f.filename for f in info} != set(model.filenames):
                raise RuntimeError(f"Incomplete weight listing from {endpoint}")
            print(f"{model.repo}: {len(info)} files, {sum(f.file_size for f in info):,} bytes -> {directory}")
        else:
            missing = [name for name in model.filenames if not (directory / name).is_file()]
            if missing:
                raise RuntimeError(f"Missing downloaded weights in {directory}: {missing}")
            print(directory.resolve())


if __name__ == "__main__":
    main()
