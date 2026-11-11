<div align="center">

# Yuheng

**Yuheng: Toward Universal Progress Modeling through In-Context Learning and Semantic Alignment**

[![Project page](https://img.shields.io/badge/Project-page-76B900)](https://primotion-ai.github.io/yuheng.github.io/)

<a href="https://primotion-ai.github.io/yuheng.github.io/assets/project-video/project-video.mp4"><img src="assets/project-video-preview.webp" width="100%" alt="Yuheng project video showing reference keyframes and predicted progress across robot and human demonstrations"></a>

▶ [Watch the full project video with sound](https://primotion-ai.github.io/yuheng.github.io/assets/project-video/project-video.mp4) (1:48)

</div>

Yuheng estimates how much of a robot task has been completed in a given image. Choose keyframes from one successful demonstration and assign a progress value to each milestone. Given the task instruction, those labeled keyframes, and a new image, Yuheng predicts progress on your scale.

The same reference lets you compare attempts that move at different speeds, stop partway through, or use different embodiments, such as a human hand and a robot gripper. This repository provides the inference code for both model sizes and a two-trajectory example.

## Models

| Model | Hugging Face weights | Download size |
| --- | --- | ---: |
| Yuheng-4B | [CarlDegio/Yuheng-4B](https://huggingface.co/CarlDegio/Yuheng-4B) | 17.76 GB |
| Yuheng-27B | [CarlDegio/Yuheng-27B](https://huggingface.co/CarlDegio/Yuheng-27B) | 104.35 GB |

Both sizes use the same Python API. The package includes their tokenizers, image processors, chat templates, and architecture configurations. Inference runs offline once the weights are on disk.

## Quick start

Use Python 3.10 or later and an NVIDIA CUDA GPU with FP16 support. Install CUDA builds of PyTorch 2.8.0 and torchvision 0.23.0 compatible with your driver. From the repository root, install the package, download the 4B weights, and run the handoff example.

```bash
python -m pip install .
yuheng-download --model 4b --source official
python example_inference.py --model 4b
```

The task is “Grab the drink and perform a two-handed handoff.” The script uses seven keyframes from the successful demonstration, assigns them evenly spaced values from 0 to 1, and scores all 32 frames of each attempt on that scale, including the failed handoff.

The script writes `outputs/Yuheng-4B.json` with the reference indices and assigned values, plus a frame index, image path, and predicted progress for every query. The [task instruction](example/instruction.txt) and [reference keyframe selection](example/keyframe.txt) are included with the example. The first line of the selection file identifies the successful trajectory and the second lists its frame indices.

To run the 27B model, use `--model 27b` in both commands. Its results go to `outputs/Yuheng-27B.json`. The downloader stores weights under `weights/Yuheng-4B/` or `weights/Yuheng-27B/`.

### Download options

- `--source mirror` to download from hf-mirror.com.
- `--dry-run` to check file counts and sizes before downloading.
- `--model both` to download both models.
- `--output-dir DIR` to choose a different parent directory. Pass the resulting model directory to the example with `--weights-dir`.

The equivalent module entry point is `python -m yuheng.download`.

## Python API

The example below uses three milestones from the bundled handoff and assigns the middle one a progress value of 0.35. It scores the final image of each attempt against that reference.

```python
from pathlib import Path
from yuheng import Yuheng

example = Path("example")
model = Yuheng("4b")
scores = model.predict_progress(
    instruction=(example / "instruction.txt").read_text().strip(),
    reference_images=[example / "1-success" / f"{i}.png" for i in (0, 15, 31)],
    reference_progress=[0.0, 0.35, 1.0],
    query_images=[example / "1-success/31.png", example / "2-fail/31.png"],
    batch_size=2,
)
print(scores)  # One float per query, in input order.
```

Pass local image paths or PIL images. Supply 2–11 reference images with one value per image in nondecreasing order. Values must lie in `[0, 1]`, and the last must exceed the first. A reference can cover only part of a task. For example, values of `[0.2, 0.35, 0.8]` give predictions in `[0.2, 0.8]`.

The model scores each query independently, including images that also appear in the reference. Its progress head applies softmax to the reference logits and returns a weighted mean of the supplied progress values. Reuse the loaded model across tasks and set `batch_size` to control the number of queries per forward pass.

## GPU memory and precision

Both models load their weights in FP16 and use FP16 autocast by default. The table shows peak GPU memory allocated by PyTorch during prediction on one NVIDIA B200 with a query batch size of 2. Each peak includes the resident model weights.

| Model | FP16 weights | Handoff example peak (7 references, 568 × 320) | OOD sample peak (6 references, 640 × 480) |
| --- | ---: | ---: | ---: |
| Yuheng-4B | 8.27 GiB | 8.92 GiB | 9.19 GiB |
| Yuheng-27B | 48.59 GiB | 49.43 GiB | 49.94 GiB |

The example script reports its own peak using `torch.cuda.max_memory_allocated()`. Total device memory shown by `nvidia-smi` can be higher because it includes the CUDA context and reserved memory. Image resolution, reference count, and query batch size affect the peak. Use `--batch-size 1` in the example to reduce query batch memory.

Use `Yuheng("27b", weights_dir="/path/to/Yuheng-27B", device="cuda:1")` to select a weight directory and GPU. Set `dtype="float32"` for the original evaluation precision (FP32 weights and BF16 autocast), or `dtype="bfloat16"` for BF16 weights and autocast. Both alternatives require BF16 support. Scores can vary slightly with precision.

The 27B model uses the Transformers implementation of Qwen3.5 linear attention. The original evaluation used fused kernels available with compatible versions of FLA and causal-conv1d.

## Discussion (WeChat Group)

Scan either QR code in WeChat to join a Yuheng discussion group.

| WeChat Group 2 | WeChat Group 3 |
| :---: | :---: |
| [<img src="assets/wechat2.png" alt="QR code for Yuheng WeChat group 2" width="260">](assets/wechat2.png) | [<img src="assets/wechat3.png" alt="QR code for Yuheng WeChat group 3" width="260">](assets/wechat3.png) |

## Acknowledgments

The model configurations and processor assets come from Qwen3-VL-4B-Instruct and Qwen3.8-27B. Their license is included in [assets/LICENSE](yuheng/assets/LICENSE).
