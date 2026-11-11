import math
from os import PathLike
from pathlib import Path
from typing import Sequence

from PIL import Image
import torch
from transformers import AutoProcessor

from .model import load_weights
from .prompt import prepare_batch
from .releases import RELEASES

ImageInput = str | PathLike | Image.Image


def _rgb(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as source:
        return source.convert("RGB")


class Yuheng:
    """Load a released model once and reuse it for independent query images."""

    def __init__(self, model: str = "4b", weights_dir=None, *, device="cuda:0", dtype="float32"):
        model = model.lower()
        if model not in RELEASES:
            raise ValueError("model must be '4b' or '27b'")
        if dtype not in ("float32", "bfloat16"):
            raise ValueError("dtype must be 'float32' or 'bfloat16'")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Yuheng requires a CUDA GPU with BF16 support")
        with torch.cuda.device(self.device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("The selected GPU must support BF16")
        release = RELEASES[model]
        directory = Path(weights_dir) if weights_dir is not None else Path("weights") / release.repo.split("/")[1]
        files = [directory / name for name in release.filenames]
        missing = [path.name for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {missing[0]} in {directory}. Run yuheng-download --model {model}.")
        assets = Path(__file__).parent / "assets" / model
        self.processor = AutoProcessor.from_pretrained(assets, local_files_only=True)
        self.processor.tokenizer.padding_side = "right"
        self.model = load_weights(assets, files, self.device, getattr(torch, dtype))

    @torch.inference_mode()
    def predict_progress(self, instruction: str, reference_images: Sequence[ImageInput],
                         reference_progress: Sequence[float], query_images: Sequence[ImageInput],
                         *, batch_size: int = 1) -> list[float]:
        """Return one progress value per query in input order, on the supplied [0, 1] scale."""
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a nonempty string")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(reference_images, (str, PathLike)) or isinstance(query_images, (str, PathLike)):
            raise TypeError("Pass image paths in a sequence, including for a single image")
        progress = [float(p) for p in reference_progress]
        if len(reference_images) != len(progress) or not 2 <= len(progress) <= 11:
            raise ValueError("Supply 2 to 11 reference images with one progress value per image")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in progress):
            raise ValueError("Progress values must be finite and between 0 and 1")
        if any(a > b for a, b in zip(progress, progress[1:])) or progress[-1] <= progress[0]:
            raise ValueError("Reference progress must be nondecreasing and span a nonzero interval")
        if len(query_images) == 0:
            return []
        start, end = progress[0], progress[-1]
        relative = [(p - start) / (end - start) for p in progress]
        refs = [_rgb(image) for image in reference_images]
        result = []
        for offset in range(0, len(query_images), batch_size):
            queries = [_rgb(image) for image in query_images[offset:offset + batch_size]]
            inputs = prepare_batch(self.processor, instruction, refs, relative, queries)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values = self.model(**inputs.to(self.device))
            if not torch.isfinite(values).all():
                raise FloatingPointError("Nonfinite predicted progress")
            result.extend(start + (end - start) * p for p in values.clamp(0, 1).tolist())
        return result
