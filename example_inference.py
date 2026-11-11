"""Score both example trajectories against the successful demonstration."""

import argparse
import json
from pathlib import Path

import torch

from yuheng import Yuheng

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("4b", "27b"), default="4b")
    parser.add_argument("--weights-dir", type=Path, help="Directory containing the chosen model's weights")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    example = ROOT / "example"
    instruction = (example / "instruction.txt").read_text(encoding="utf-8").strip()
    reference_id, indices = (example / "keyframe.txt").read_text(encoding="utf-8").splitlines()
    reference = example / f"{reference_id.strip()}-success"
    keyframes = [int(value) for value in indices.split()]
    if not 2 <= len(keyframes) <= 11 or sorted(set(keyframes)) != keyframes:
        raise ValueError("Expected 2 to 11 strictly increasing keyframe indices")
    reference_images = [reference / f"{i}.png" for i in keyframes]
    for image in reference_images:
        if not image.is_file():
            raise FileNotFoundError(image)
    reference_progress = [i / (len(keyframes) - 1) for i in range(len(keyframes))]

    model = Yuheng(args.model, weights_dir=args.weights_dir, device=args.device)
    torch.cuda.reset_peak_memory_stats(model.device)
    trajectories = {}
    for trajectory in sorted(path for path in example.iterdir() if path.is_dir()):
        images = sorted(trajectory.glob("*.png"), key=lambda path: int(path.stem))
        if not images:
            raise ValueError(f"No PNG frames in {trajectory}")
        scores = model.predict_progress(instruction, reference_images, reference_progress, images,
                                        batch_size=args.batch_size)
        trajectories[trajectory.name] = [
            {"frame_index": int(image.stem), "image": image.relative_to(example).as_posix(), "progress": score}
            for image, score in zip(images, scores, strict=True)
        ]

    result = {
        "model": f"Yuheng-{args.model.upper()}",
        "instruction": instruction,
        "reference_trajectory": reference.name,
        "reference_frame_indices": keyframes,
        "reference_progress": reference_progress,
        "trajectories": trajectories,
    }
    output = ROOT / "outputs" / f"{result['model']}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    peak = torch.cuda.max_memory_allocated(model.device) / 2**30
    print(f"Saved {output} | peak allocated GPU memory: {peak:.2f} GiB")


if __name__ == "__main__":
    main()
