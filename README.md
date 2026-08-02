# I2MD4Fair: Intra- and Inter-Modality Debiasing for Item-Side Exposure Fairness in Multimodal Recommendation

## Installation

```bash
pip install -r requirements.txt
pip install sentence-transformers scikit-learn
```

## Data

### Download

All preprocessed datasets (Baby, Clothing, MIND) are available at:

> https://drive.google.com/file/d/152sgC9vkuyxk1vs8QhXygoH5YDxIRan8/view?usp=drive_link

Download and extract to `data/damrs/` so that the directory structure looks like:

```
data/damrs/
├── baby/        {baby.inter, image_feat.npy, text_feat.npy, image_adj_80.pt, text_adj_80.pt, ...}
├── clothing/    {clothing.inter, ...}
└── mind/        {mind.inter, ...}
```

A small **demo** dataset (50 users, 100 items, 32-dim features) is already included in `data/damrs/demo/` for quick testing without downloading anything.

### Dataset Statistics

| Dataset | Users | Items | Interactions | Sparsity |
|---------|-------|-------|--------------|----------|
| Baby    | 19,445| 7,050 | 160,792      | 99.88%   |
| Clothing| 39,387| 23,033| 278,677      | 99.97%   |
| MIND    | 750,434| 96,700| 17,491,799  | 99.98%   |

## Quick Demo

```bash
# Train I2MD4Fair on the demo dataset (CPU, ~30 seconds)
python scripts/run_main.py --dataset demo --model I2MD4Fair --device cpu --max_epochs 10 --n_runs 1 --batch_size 256

# Run ablation study on demo
python scripts/run_ablation.py --dataset demo --device cpu --max_epochs 10 --n_runs 1 --batch_size 256

# Run hyperparameter sensitivity on demo
python scripts/run_hyperparam.py --dataset demo --device cpu --max_epochs 10 --n_runs 1 --batch_size 256

# Run backbone transferability on demo
python scripts/run_transfer.py --dataset demo --device cpu --max_epochs 10 --n_runs 1 --batch_size 256
```

## Running Experiments

### Main model

```bash
# I2MD4Fair
python scripts/run_main.py --dataset baby --model I2MD4Fair --device cuda
python scripts/run_main.py --dataset clothing --model I2MD4Fair --device cuda
python scripts/run_main.py --dataset mind --model I2MD4Fair --device cuda

# Baselines
python scripts/run_main.py --dataset baby --model MMSSL --device cuda
python scripts/run_main.py --dataset baby --model MMSSL+DPR --device cuda
python scripts/run_main.py --dataset baby --model MMSSL+FairDual --device cuda
# ... (see all available models below)
```

Available models: `I2MD4Fair`, `LightGCN`, `VBPR`, `MMGCN`, `GRCN`, `LATTICE`, `FREEDOM`, `LGMRec`, `BM3`, `SLMRec`, `MMSSL`, `DiffMM`, `MENTOR`, `DMRL`, `CLUSSL`, `MMSSL+MD`, `DiffMM+MD`, `LGMRec+MD`, `MENTOR+MD`, `MMSSL+DPR`, `DiffMM+DPR`, `LGMRec+DPR`, `MENTOR+DPR`, `MMSSL+FairDual`, `DiffMM+FairDual`, `LGMRec+FairDual`, `MENTOR+FairDual`

### Ablation and hyperparameter studies

```bash
# Component analysis
python scripts/run_ablation.py --dataset baby --device cuda

# Hyperparameter sensitivity
python scripts/run_hyperparam.py --dataset baby --device cuda

# Backbone transferability (Intra-MDM / Inter-MDM on other backbones)
python scripts/run_transfer.py --dataset baby --device cuda
```
