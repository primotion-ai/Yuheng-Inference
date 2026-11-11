# Yuheng-Inference

Yuheng estimates task progress from images. Give it a task instruction, a sequence of reference images with known progress values, and one or more query images. It returns one progress estimate per query, in input order.

Both [Yuheng-4B](https://huggingface.co/CarlDegio/Yuheng-4B) and [Yuheng-27B](https://huggingface.co/CarlDegio/Yuheng-27B) use the same interface. The package includes the tokenizer, image processor, chat template, and architecture configuration for each model. Once you download the weights, inference runs offline.

## Install and download

Use Python 3.10 or later and an NVIDIA CUDA GPU with FP16 support. Install CUDA builds of PyTorch 2.8.0 and torchvision 0.23.0 compatible with your driver. Then run the following from the repository root.

```bash
python -m pip install .
yuheng-download --model 4b --source official
```

For either model, choose `official` for Hugging Face or `mirror` for hf-mirror.com. For example, to download 27B from the mirror, run the following command.

```bash
yuheng-download --model 27b --source mirror
```

Weights go into `weights/Yuheng-4B/` or `weights/Yuheng-27B/` under the current directory. Use `--output-dir DIR` to choose another parent directory, `--model both` to download both models, or `--dry-run` to check file counts and sizes before downloading. You can also invoke the downloader with `python -m yuheng.download`.

## Predict progress

```python
from yuheng import Yuheng

predictor = Yuheng("4b")  # Load once and reuse across tasks. Use "27b" for the larger model.
scores = predictor.predict_progress(
    instruction="Close the drawer",
    reference_images=["open.png", "partially_closed.png", "closed.png"],
    reference_progress=[0.0, 0.35, 1.0],
    query_images=["query_001.png", "query_002.png"],
    batch_size=2,
)
print(scores)  # list[float], one value per query image in input order
```

Pass local image paths or PIL images. Provide 2–11 reference images ordered by progress. Each query is scored independently against the same references, so you can pass every frame of a trajectory in one call. `batch_size` controls how many queries run in each forward pass.

Assign each reference a progress value in `[0, 1]`. Adjacent values can be equal, and the last value must be greater than the first. The references can also cover part of a task. For example, with `[0.2, 0.35, 0.8]`, the model scores each query relative to that segment and returns a value in `[0.2, 0.8]`.

The model applies softmax to the reference logits and uses the resulting probabilities to compute a weighted mean of the reference progress values. If you pass a reference image as a query, it receives a model prediction like any other query.

## GPU memory

Both models load their weights in FP16 and use FP16 autocast by default. The weights occupy approximately 8.27 GiB for 4B and 48.59 GiB for 27B. The forward pass needs additional memory for images and activations. Use `batch_size=1` to minimize query batch memory.

Use `Yuheng("27b", weights_dir="/path/to/Yuheng-27B", device="cuda:1")` to select a weight directory and GPU. Set `dtype="float32"` for the original evaluation precision (FP32 weights and BF16 autocast), or `dtype="bfloat16"` for BF16 weights and autocast. Both alternatives require BF16 support. Scores can vary slightly with precision.

The 27B model uses the Transformers implementation of Qwen3.5 linear attention. The original evaluation used fused kernels available with compatible versions of FLA and causal-conv1d.

The model configurations and processor assets come from Qwen3-VL-4B-Instruct and Qwen3.8-27B. Their license is included in [assets/LICENSE](yuheng/assets/LICENSE).
