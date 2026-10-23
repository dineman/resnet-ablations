#!/usr/bin/env bash
# Reproduce everything. The ablations need a GPU: on a laptop, run only the first block.
# Usage: conda activate resnet && ./run_all.sh [EPOCHS]     (EPOCHS: 182 = paper budget, default 60)
set -euo pipefail
cd "$(dirname "$0")"
EPOCHS="${1:-60}"

# --- runs anywhere (laptop CPU is fine) ---
python -m scripts.download_data           # CIFAR-10 binary, MD5-checked (~170 MB)
python -m pytest -q                       # 41 tests, ~1 min on CPU
python -m scripts.grad_flow               # gradient flow at init (~1 min)

# --- GPU: 27 runs (9 variants x 3 seeds). See notebooks/run_on_gpu.ipynb for Kaggle/Colab ---
python -m scripts.benchmark               # measure s/epoch here and print the cost of each budget
python -m scripts.run_ablation all --epochs "$EPOCHS"
python -m scripts.analyze --runs "runs/e$EPOCHS"
