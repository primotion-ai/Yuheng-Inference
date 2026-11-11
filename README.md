# Yuheng-Inference

根据任务指令、参考关键帧及其进度，为每张查询图预测一个进度值。支持 [Yuheng-4B](https://huggingface.co/CarlDegio/Yuheng-4B) 和 [Yuheng-27B](https://huggingface.co/CarlDegio/Yuheng-27B)。分词器、图像处理器、聊天模板及模型结构配置已随包提供；下载对应的 Yuheng 权重后即可离线推理。

## 安装与下载

在仓库根目录运行。需要 Python 3.10+ 和支持 BF16 的 NVIDIA GPU，请安装与驱动兼容的 CUDA 版 PyTorch 2.8.0 和 torchvision 0.23.0。

```bash
python -m pip install .
yuheng-download --model 4b --source official
yuheng-download --model 27b --source mirror
```

`--model both` 同时下载两个版本。`--output-dir DIR` 指定父目录，默认为当前目录的 `weights/`，其中分别生成 `Yuheng-4B/` 和 `Yuheng-27B/`。加 `--dry-run` 可先查看文件数量和大小。下载脚本也可通过 `python -m yuheng.download` 调用。

## 预测进度

```python
from yuheng import Yuheng

predictor = Yuheng("4b")  # 同一实例可用于多个任务；27B 使用 "27b"
scores = predictor.predict_progress(
    instruction="Close the drawer",
    reference_images=["open.png", "partially_closed.png", "closed.png"],
    reference_progress=[0.0, 0.35, 1.0],
    query_images=["query_001.png", "query_002.png"],
    batch_size=2,
)
print(scores)  # list[float]，按两张查询图的传入顺序返回
```

- 图片支持本地文件路径和 PIL 图像。参考帧按进度排列，支持 2–11 张。每张查询图独立与同一组参考帧比较；可以一次传入整条轨迹的帧，`batch_size` 控制每次前向计算的查询数。
- 参考进度用 `[0, 1]` 表示，允许相邻值相等，首末值须不同。若只给出一个任务片段（如 `[0.2, 0.35, 0.8]`），代码先按首末值归一化，模型在片段内定位，再将输出换回 `[0.2, 0.8]`。
- 模型对各参考关键帧输出 logits，经过 softmax 后对给定进度加权求和。参考图也可传作查询图，得到的是模型预测值。
- 可通过 `Yuheng("27b", weights_dir="/path/to/Yuheng-27B", device="cuda:1")` 指定本地模型目录和单张 GPU。默认保留 FP32 权重并使用 BF16 自动混合精度。仅权重约占 16.54 GiB（4B）或 97.19 GiB（27B），前向计算还需额外显存。显存有限时可用 `dtype="bfloat16"` 将权重占用减半，并将 `batch_size` 设为 1；该模式可能产生数值差异。

27B 使用 Transformers 的 Qwen3.5 线性注意力实现。已安装兼容的 FLA 和 causal-conv1d 时，Transformers 可使用融合内核；原评测使用了这条路径。

模型配置与处理器资产来自 Qwen3-VL-4B-Instruct 和 Qwen3.8-27B，其许可随附在 [assets/LICENSE](yuheng/assets/LICENSE)。
