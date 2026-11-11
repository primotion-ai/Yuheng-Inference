"""Public interface for reference-conditioned progress prediction."""
__all__ = ["Yuheng"]


def __getattr__(name):
    if name == "Yuheng":
        from .inference import Yuheng
        return Yuheng
    raise AttributeError(name)
