# PACS 域泛化与自适应凸核对齐

本项目使用 ImageNet 预训练的 ResNet-50，在 PACS 图像分类数据集上训练域泛化模型。训练支持单源域和多源域，并可通过 `adaptive_convex_soft` 核对齐损失，结合多种核变换模拟域间关系变化。训练结束后会在指定目标域上评估准确率，并保存模型权重。

## 项目文件

| 文件 | 说明 |
| --- | --- |
| `train_dg.py` | 训练入口：数据加载、增强、模型训练、目标域评估和检查点保存。 |
| `kernel_strategies.py` | A–L 共 12 种核变换策略的 NumPy/PyTorch 实现。 |
| `augmentations.py` | `original` 增强方案所需的图像风格与像素级增强操作。 |
| `run_all_experiments.py` | 依次启动预设的 24 个 PACS 实验（8 种域配置 × 3 个随机种子）。 |
| `eval_dg.py` | 从检查点加载模型，在指定目标域上独立评估。 |
| `utils.py` | 特征提取辅助函数。 |

## 环境依赖

需要 Python 3、PyTorch、torchvision、NumPy、SciPy、Pillow、OpenCV（`opencv-python`）和 tqdm。训练脚本会根据环境自动选择 CUDA 或 CPU；建议使用与本机 CUDA 驱动匹配的 PyTorch 版本。项目没有附带锁定版本的依赖清单。

安装 Python 依赖的示例：

```bash
pip install torch torchvision numpy scipy pillow opencv-python tqdm
```

首次构建 ResNet-50 时，`torchvision` 会尝试获取 ImageNet 预训练权重；请确保权重已缓存或运行环境可以访问相应下载源。

## 数据集准备

脚本使用 `torchvision.datasets.ImageFolder` 读取图片。将 PACS 数据整理为以下目录结构（类别目录名须在各域中一致）：

```text
data/PACS/
├── art_painting/
│   ├── dog/
│   ├── elephant/
│   └── ...
├── cartoon/
├── photo/
└── sketch/
```

每个域目录下应直接包含类别子目录，类别子目录中放置对应图片。训练会检查各源域的类别到索引映射是否一致。仓库中的 `data` 目录当前未展示数据文件；请按实际存放位置设置 `--data_root`。

## 训练

在项目目录运行单源域示例：

```bash
python train_dg.py \
  --data_root ./data/PACS \
  --source photo \
  --targets art_painting,sketch,cartoon
```

多源域示例（以 sketch 为目标域）：

```bash
python train_dg.py \
  --data_root ./data/PACS \
  --sources photo,art_painting,cartoon \
  --targets sketch
```

常用参数如下；完整参数可通过 `python train_dg.py --help` 查看。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--source` / `--sources` | 无 | 单个源域或逗号分隔的多个源域，二者至少提供一个。 |
| `--targets` | 必填 | 逗号分隔的目标域名称。 |
| `--iterations` | `5000` | 训练迭代次数。 |
| `--batch_size` | `64` | 批次大小。 |
| `--lr` | `1e-3` | SGD 初始学习率；约在总迭代数的 80% 处衰减为原来的 0.1。 |
| `--lambda_ka` | `0.1` | 核对齐损失权重；前 `--ka_warmup` 次迭代线性增加。设为 0 可关闭核对齐项。 |
| `--strategies` | `A,C,D,F,H,I` | 逗号分隔的核变换代码，可从 A–L 中选择。 |
| `--aug_type` | `original` | 数据增强方案：`original`、`minimal` 或 `TA`。 |
| `--seed` | `1` | 随机种子。 |
| `--eval_every` | `500` | 打印训练指标的间隔；目标域评估在训练结束时执行。 |

其他核对齐参数包括 `--alpha_couple`、`--convex_temperature`、`--convex_uniform_mix`、`--ka_warmup`、`--coupling` 和 `--no_detach_target`。当前仅实现 `adaptive_convex_soft` coupling。

### 核变换策略

| 代码 | 操作 |
| --- | --- |
| A | 特征值扰动 |
| B | 流形/特征向量随机旋转 |
| C | 与随机噪声核混合 |
| D | 随机元素掩蔽并投影到半正定矩阵 |
| E | 频谱滤波 |
| F | 加性噪声 |
| G | 块掩蔽 |
| H | 特征向量扰动 |
| I | 特征值对比度缩放 |
| J | PCA 子空间替换 |
| K | 类条件扰动（需要标签） |
| L | 核元素 dropout |

## 批量实验

`run_all_experiments.py` 按脚本顶部的常量配置运行预设的 24 个实验。默认数据路径是 `./data/PACS`，每项训练 5000 次迭代、批次大小 64，并使用 3 个种子。运行前请确认该路径相对于当前工作目录有效；需要调整参数时，编辑脚本顶部的常量后再执行：

```bash
python run_all_experiments.py
```

脚本逐项调用 `train_dg.py`，某个子任务失败时会打印返回码并继续后续任务。

## 评估与输出

训练脚本结束时会直接在 `--targets` 指定的目标域上报告各域准确率和平均准确率。模型状态字典保存在当前工作目录下的 `checkpoints/`，文件名包含源域、核策略、损失权重、增强方案和随机种子等信息。

也可使用独立评估脚本：

```bash
python eval_dg.py \
  --data_root ./data/PACS \
  --source photo \
  --targets art_painting,sketch,cartoon \
  --ckpt ./checkpoints/<训练生成的模型文件>.pth
```

`eval_dg.py` 的 `--source` 默认值为 `photo`，用于确定分类类别数；若模型使用了不同的源域类别目录，请传入对应源域。该脚本当前将模型和输入直接放到 CUDA 上，因此需要可用的 NVIDIA GPU。训练脚本则支持 CPU 回退。
