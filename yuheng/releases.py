"""Immutable weight releases and their expected files."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Release:
    repo: str
    revision: str
    filenames: tuple[str, ...]


RELEASES = {
    "4b": Release("CarlDegio/Yuheng-4B", "096f6b489e77085a74d53ac8513bec4ba1b07463",
                  ("model.safetensors",)),
    "27b": Release("CarlDegio/Yuheng-27B", "9411f413bb7e7ed764dfb5836c565d7a6b2d1784",
                   tuple(f"model-{i:05d}-of-00022.safetensors" for i in range(1, 23))),
}
