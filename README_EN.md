# Domain Generalization with Adaptive Convex Kernel Alignment

This repository contains experimental code for domain generalization in image classification. The implementation uses an ImageNet-pretrained ResNet-50 and supports single-source and multi-source training. An optional `adaptive_convex_soft` kernel alignment loss combines kernel transformations to model changes in relationships across domains.

The experiments use three benchmark datasets: **PACS**, **VLCS**, and **Office-Home**. The training and evaluation scripts read datasets in `torchvision.datasets.ImageFolder` format. Dataset images are not included in this repository; obtain them from their official or authorized distribution channels and follow their respective terms of use.

## Repository contents

| File | Description |
| --- | --- |
| `train_dg.py` | Training entry point for data loading, augmentation, model training, target-domain evaluation, and checkpoint saving. |
| `kernel_strategies.py` | NumPy and PyTorch implementations of 12 kernel transformation strategies (A–L). |
| `augmentations.py` | Image style and pixel-level augmentations used by the `original` augmentation pipeline. |
| `run_all_experiments.py` | Runs 24 predefined PACS experiments (8 domain configurations × 3 random seeds). |
| `eval_dg.py` | Loads a trained checkpoint and evaluates it on specified target domains. |
| `utils.py` | Helper functions for feature extraction. |
| `README_CN.md` | Chinese documentation. |

## Requirements

The code requires Python 3, PyTorch, torchvision, NumPy, SciPy, Pillow, OpenCV (`opencv-python`), and tqdm. Training uses CUDA when available and otherwise falls back to CPU. Install a PyTorch build compatible with your CUDA setup.

```bash
pip install torch torchvision numpy scipy pillow opencv-python tqdm
```

The ResNet-50 initialization uses ImageNet pretrained weights provided by `torchvision`. Ensure that the weights are cached or that your environment can download them.

## Dataset setup

### Supported benchmarks and domains

| Dataset | Domains |
| --- | --- |
| PACS | `art_painting`, `cartoon`, `photo`, `sketch` |
| VLCS | `CALTECH`, `LABELME`, `PASCAL`, `SUN` |
| Office-Home | `Art`, `Clipart`, `Product`, `Real_World` |

Domain directory names are case-sensitive. The names above are conventional examples; if your copy uses different names, use the actual directory names in the command-line arguments.

### Required directory format

Both training and evaluation use `ImageFolder`: each domain directory must contain one subdirectory per class, and each class directory must contain its images. Keep the class directory names consistent across domains in a dataset so that class indices match.

For example, a PACS copy can be arranged as follows:

```text
data/
└── PACS/
    ├── art_painting/
    │   ├── dog/
    │   ├── elephant/
    │   └── ...
    ├── cartoon/
    ├── photo/
    └── sketch/
```

The same layout applies to VLCS and Office-Home, with their corresponding domain directories. For example:

```text
data/
├── VLCS/
│   ├── CALTECH/
│   ├── LABELME/
│   ├── PASCAL/
│   └── SUN/
└── OfficeHome/
    ├── Art/
    ├── Clipart/
    ├── Product/
    └── Real_World/
```

Each domain directory shown above must itself contain the class subdirectories and images. If the downloaded archive has an extra wrapper directory (for example, `PACS/kfold/`), set `--data_root` to the directory whose immediate children are the domain directories, or reorganize the files accordingly.

The scripts do not download or preprocess these datasets. They accept a dataset root path and domain names; they do not require a fixed dataset name. **The provided `run_all_experiments.py` configuration is PACS-specific** and does not define VLCS or Office-Home experiment lists. Use `train_dg.py` directly for those datasets, or edit the batch script to specify the desired domain splits.

## Training

Run commands from the repository directory. The `--data_root` argument should point to the directory containing the domain directories, not to an individual domain.

### PACS example (single source)

```bash
python train_dg.py \
  --data_root ./data/PACS \
  --source photo \
  --targets art_painting,sketch,cartoon
```

### VLCS example (single source)

```bash
python train_dg.py \
  --data_root ./data/VLCS \
  --source CALTECH \
  --targets LABELME,PASCAL,SUN
```

### Office-Home example (single source)

```bash
python train_dg.py \
  --data_root ./data/OfficeHome \
  --source Product \
  --targets Art,Clipart,Real_World
```

### Multi-source example

Specify multiple source domains with `--sources` and one or more held-out target domains with `--targets`. For example, for PACS with `sketch` held out:

```bash
python train_dg.py \
  --data_root ./data/PACS \
  --sources photo,art_painting,cartoon \
  --targets sketch
```

Common options are listed below; run `python train_dg.py --help` for the full list.

| Option | Default | Description |
| --- | --- | --- |
| `--source` / `--sources` | None | A single source domain or a comma-separated list of source domains. Provide one of these options. |
| `--targets` | Required | Comma-separated target domain names used for final evaluation. |
| `--iterations` | `5000` | Number of training iterations. |
| `--batch_size` | `64` | Batch size. |
| `--lr` | `1e-3` | Initial SGD learning rate; decayed to 10% at about 80% of training. |
| `--lambda_ka` | `0.1` | Kernel alignment loss weight, linearly warmed up over `--ka_warmup` iterations. Use `0` to disable the term. |
| `--strategies` | `A,C,D,F,H,I` | Comma-separated kernel transformation codes selected from A–L. |
| `--aug_type` | `original` | Augmentation pipeline: `original`, `minimal`, or `TA`. |
| `--seed` | `1` | Random seed. |
| `--eval_every` | `500` | Interval for printing training metrics. Target evaluation is performed after training. |

Additional kernel alignment options include `--alpha_couple`, `--convex_temperature`, `--convex_uniform_mix`, `--ka_warmup`, `--coupling`, and `--no_detach_target`. Only `adaptive_convex_soft` coupling is implemented.

### Kernel transformation strategies

| Code | Operation |
| --- | --- |
| A | Eigenvalue perturbation |
| B | Random manifold/eigenvector rotation |
| C | Mix with a random noise kernel |
| D | Random element masking followed by positive semidefinite projection |
| E | Spectral filtering |
| F | Additive noise |
| G | Block masking |
| H | Eigenvector perturbation |
| I | Eigenvalue contrast scaling |
| J | PCA subspace replacement |
| K | Class-conditioned perturbation (requires labels) |
| L | Kernel element dropout |

## Batch experiments

`run_all_experiments.py` runs 24 predefined PACS experiments: 8 domain configurations × 3 random seeds. It uses `./data/PACS` by default, with 5000 iterations and a batch size of 64. Confirm that the data path is correct before running:

```bash
python run_all_experiments.py
```

This batch script is currently PACS-specific. For VLCS and Office-Home, configure the desired source/target splits and domain names in the script or launch experiments with `train_dg.py` directly. The batch script invokes training sequentially and continues after a failed experiment while reporting its return code.

## Evaluation and outputs

Training evaluates the model on every domain listed in `--targets` at the end of the run and reports individual and average accuracy. Model state dictionaries are saved under `checkpoints/` in the current working directory. Checkpoint filenames include source domain(s), kernel strategy, loss weight, augmentation type, and random seed.

To evaluate a checkpoint separately, provide a source domain whose class directory defines the class count, target domain names, and the same dataset root used for the corresponding benchmark:

```bash
python eval_dg.py \
  --data_root ./data/PACS \
  --source photo \
  --targets art_painting,sketch,cartoon \
  --ckpt ./checkpoints/<checkpoint-file>.pth
```

For VLCS or Office-Home, change `--data_root`, `--source`, and `--targets` to the corresponding paths and domains. The standalone `eval_dg.py` currently places the model and input batches directly on CUDA and therefore requires an available NVIDIA GPU. The training script supports CPU fallback.

## Data, checkpoints, and GitHub

Dataset images and trained checkpoints are not included in this repository. Check the terms of each dataset before sharing or redistributing any data. In most cases, provide dataset citations and official access instructions, and let users download the files themselves. Keep local data and generated checkpoints out of Git commits (for example, ignore `data/`, `checkpoints/`, and Python cache files).

When reporting results, document the dataset version/source, domain split, class mapping, training arguments, and random seeds. The supplied batch script records a PACS experiment configuration; it does not by itself define the VLCS or Office-Home splits used in your study.
