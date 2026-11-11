from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig, Qwen3VLModel, Qwen3_5Model


class ProgressModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.text_config.hidden_size
        dim = 256
        self.anchor_projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim), nn.GELU())
        self.query_projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim), nn.GELU())
        self.scorer = nn.Sequential(nn.Linear(3 * dim + 1, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, anchor_positions, query_positions, anchor_progress, **inputs):
        hidden = self.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        a = self.anchor_projection(hidden[batch[:, None], anchor_positions])
        q = self.query_projection(hidden[batch, query_positions])[:, None].expand_as(a)
        features = torch.cat((a, q, a * q, anchor_progress[..., None].to(a.dtype)), dim=-1)
        logits = self.scorer(features).squeeze(-1).float()
        return (logits.softmax(-1) * anchor_progress).sum(-1)


def load_weights(assets: Path, files, device, dtype):
    config = AutoConfig.from_pretrained(assets, local_files_only=True)
    config._attn_implementation = "sdpa"
    config.use_cache = config.text_config.use_cache = False
    backbone_cls = {"qwen3_vl": Qwen3VLModel, "qwen3_5": Qwen3_5Model}[config.model_type]
    with torch.cuda.device(device), init_empty_weights(include_buffers=False):
        model = ProgressModel(backbone_cls(config))
    expected = model.state_dict()
    weights = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                header = shard.get_slice(name)
                if name not in expected or name in weights:
                    raise ValueError(f"Unexpected or duplicate weight {name} in {path.name}")
                if tuple(header.get_shape()) != expected[name].shape or header.get_dtype() != "F32":
                    raise ValueError(f"Incorrect shape or dtype for {name} in {path.name}")
                weights[name] = shard.get_tensor(name).to(device=device, dtype=dtype)
    model.load_state_dict(weights, strict=True, assign=True)
    # Parameters already reside on the GPU. Move the real rotary/position buffers too.
    return model.to(device=device).eval().requires_grad_(False)
