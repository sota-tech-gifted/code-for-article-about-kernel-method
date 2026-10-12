"""
PACS experiments across all domains (Python version)
Method: adaptive_convex_soft (soft worst-case convex combination)
Runs each configuration with three seeds, for 24 training tasks total.
"""
import subprocess
import sys

# Base parameters (edit as needed).
DATA_ROOT = "./data/PACS"
ITERATIONS = 5000
BATCH_SIZE = 64
LR = 1e-3
LAMBDA_KA = 0.1
ALPHA_COUPLE = 0.2
STRATEGIES = "A,C,D,F,H,I"
COUPLING = "adaptive_convex_soft"
TEMP = 0.30
MIX = 0.40
WARMUP = 500

# List of experiment configurations.
experiments = []

# ---------- Single-source domains ----------
# photo
for seed in [1, 2, 3]:
    experiments.append({
        'source': 'photo',
        'targets': 'art_painting,sketch,cartoon',
        'seed': seed
    })

# art_painting
for seed in [1, 2, 3]:
    experiments.append({
        'source': 'art_painting',
        'targets': 'photo,sketch,cartoon',
        'seed': seed
    })

# cartoon
for seed in [1, 2, 3]:
    experiments.append({
        'source': 'cartoon',
        'targets': 'photo,art_painting,sketch',
        'seed': seed
    })

# sketch
for seed in [1, 2, 3]:
    experiments.append({
        'source': 'sketch',
        'targets': 'photo,art_painting,cartoon',
        'seed': seed
    })

# ---------- Multi-source domains ----------
# Target: sketch
for seed in [1, 2, 3]:
    experiments.append({
        'sources': 'photo,art_painting,cartoon',
        'targets': 'sketch',
        'seed': seed
    })

# Target: photo
for seed in [1, 2, 3]:
    experiments.append({
        'sources': 'art_painting,sketch,cartoon',
        'targets': 'photo',
        'seed': seed
    })

# Target: art_painting
for seed in [1, 2, 3]:
    experiments.append({
        'sources': 'photo,sketch,cartoon',
        'targets': 'art_painting',
        'seed': seed
    })

# Target: cartoon
for seed in [1, 2, 3]:
    experiments.append({
        'sources': 'photo,art_painting,sketch',
        'targets': 'cartoon',
        'seed': seed
    })

# ---------- Run experiments ----------
total = len(experiments)
print(f"共有 {total} 个实验任务，开始执行...")

for i, exp in enumerate(experiments, 1):
    # Build the command line.
    cmd = [
        "python", "train_dg.py",
        "--data_root", DATA_ROOT,
        "--targets", exp['targets'],
        "--iterations", str(ITERATIONS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--lambda_ka", str(LAMBDA_KA),
        "--alpha_couple", str(ALPHA_COUPLE),
        "--strategies", STRATEGIES,
        "--coupling", COUPLING,
        "--convex_temperature", str(TEMP),
        "--convex_uniform_mix", str(MIX),
        "--ka_warmup", str(WARMUP),
        "--seed", str(exp['seed'])
    ]

    if 'sources' in exp:
        cmd += ["--sources", exp['sources']]
    else:
        cmd += ["--source", exp['source']]

    # Print progress.
    source_str = exp.get('source', exp.get('sources'))
    print(f"\n{'='*60}")
    print(f"[{i}/{total}] 开始训练: source={source_str}, targets={exp['targets']}, seed={exp['seed']}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")

    # Run the command.
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"❌ 实验失败！返回码: {ret.returncode}")
        # Optionally exit here instead of continuing.
        # sys.exit(1)

    print(f"✅ [{i}/{total}] 完成")

print("\n========== 全部实验结束 ==========")
