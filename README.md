# I2MD4Fair: Intra- and Inter-Modality Debiasing for Item-Side Exposure Fairness in Multimodal Recommendation

Official implementation of **I2MD4Fair** (TOIS 2026).

## Overview

I2MD4Fair is a representation-level framework for mitigating modality-associated concentration in item-side exposure. It consists of two complementary modules:

- **Intra-Modality Debiasing Module (Intra-MDM)**: Uses entropy-regularized optimal transport (Sinkhorn) to learn balanced soft assignments between items and trainable semantic prototypes, followed by prototype-induced hypergraph propagation to strengthen representations in weakly supported semantic regions.
- **Inter-Modality Debiasing Module (Inter-MDM)**: Combines CLUB-based information compression with adaptive p-norm aggregation of modality-specific ranking objectives to regulate cross-modality optimization.

A cross-modal InfoNCE objective further aligns complementary representations before they are fused with collaborative ID embeddings.

## Project Structure

```
I2MD4Fair/
├── models/
│   └── i2md4fair.py          # Main model (SinkhornOT, IntraMDM, InterMDM, CLUBEstimator, InfoNCE)
├── baseline/
│   ├── lightgcn.py           # LightGCN (ID-only baseline)
│   ├── vbpr.py               # VBPR (Visual BPR)
│   ├── mmgcn.py              # MMGCN
│   ├── grcn.py               # GRCN
│   ├── lattice.py            # LATTICE
│   ├── freedom.py            # FREEDOM
│   ├── lgmrec.py             # LGMRec
│   ├── bm3.py                # BM3
│   ├── slmrec.py             # SLMRec
│   ├── mmssl.py              # MMSSL
│   ├── diffmm.py             # DiffMM
│   ├── mentor.py             # MENTOR
│   └── md_wrapper.py         # Modality Debiasing wrapper (fairness-aware baselines)
├── data/
│   ├── dataset.py            # DAMRSDataset loading & graph construction
│   ├── damrs/                # DA-MRS format preprocessed data
│   │   ├── baby/             # Baby dataset
│   │   ├── clothing/         # Clothing dataset
│   │   └── mind/             # MIND dataset
│   └── demo/                 # Demo dataset (small, for quick testing)
├── utils/
│   ├── metrics.py            # Accuracy, fairness, and relevance-exposure metrics
│   └── data_utils.py         # BPR data loader
├── scripts/
│   ├── run_main.py           # Main experiment script (Algorithm 1)
│   ├── run_ablation.py       # Ablation study (Table 5)
│   ├── run_hyperparam.py     # Hyperparameter sensitivity (Figs. 5-6)
│   ├── preprocess_mind.py    # MIND preprocessing (raw -> DA-MRS format)
│   └── extract_features.py   # Amazon visual feature extraction (legacy)
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
pip install sentence-transformers scikit-learn
```

## Quick Demo

A small demo dataset is included under `data/damrs/demo/` (50 users, 100 items, ~500 interactions, 32-dim features) so you can verify the full pipeline without downloading any data:

```bash
# Train I2MD4Fair on the demo dataset (CPU, ~30 seconds)
python scripts/run_main.py --dataset demo --model I2MD4Fair --device cpu --max_epochs 10 --n_runs 1 --batch_size 256

# Run ablation study on demo
python scripts/run_ablation.py --dataset demo --device cpu --max_epochs 10 --n_runs 1 --batch_size 256

# Run hyperparameter sensitivity on demo
python scripts/run_hyperparam.py --dataset demo --device cpu --max_epochs 10 --n_runs 1 --batch_size 256
```

## Data Format (DA-MRS)

All datasets use the **DA-MRS format**, stored under `data/damrs/<dataset_name>/`:

| File | Format | Description |
|------|--------|-------------|
| `<name>.inter` | Tab-separated text | Interactions: `userID itemID rating timestamp x_label` (x_label: 0=train, 1=val, 2=test, 8:1:1 random split) |
| `image_feat.npy` | numpy (n_items, d_v) float32 | Visual features |
| `text_feat.npy` | numpy (n_items, d_t) float32 | Textual features |
| `image_adj_80.pt` | torch tensor (n_items, n_items) float32 | Top-80 visual similarity adjacency (row-normalized) |
| `text_adj_80.pt` | torch tensor (n_items, n_items) float32 | Top-80 textual similarity adjacency (row-normalized) |
| `u_id_mapping.csv` | CSV | Original user ID -> mapped integer ID |
| `i_id_mapping.csv` | CSV | Original item ID -> mapped integer ID |

## Dataset Preparation

### Baby & Clothing

Baby and Clothing datasets use pre-computed features from the DA-MRS project. Place them in `data/damrs/baby/` and `data/damrs/clothing/` with the files listed above.

| Dataset | Users | Items | Interactions | Sparsity |
|---------|-------|-------|--------------|----------|
| Baby    | 19,445| 7,050 | 160,792      | 99.88%   |
| Clothing| 39,387| 23,033| 278,677      | 99.97%   |
| MIND    | 750,434| 96,700| 17,491,799  | 99.98%   |

### MIND (From raw data)

MIND requires preprocessing from raw data to DA-MRS format. Steps:

1. **Download raw data** — Place the following in `data/MIND/`:
   - `MINDlarge_train.zip` (from [MIND dataset](https://msnews.github.io/)) — contains `news.tsv` + `behaviors.tsv`
   - `MINDlarge_dev.zip` — contains `news.tsv` + `behaviors.tsv`
   - News images (`N*.jpg`) — from [IM-MIND](https://drive.google.com/file/d/1gx0OzN7qSuyRlvN0cfVUjB4tmoKvvQk1/view), extract to `data/MIND/`

2. **Run preprocessing** (each step can run independently, supports GPU):

```bash
python scripts/preprocess_mind.py 1   # Build interactions + 8:1:1 split
python scripts/preprocess_mind.py 2   # Extract visual features (ResNet-50 avgpool -> 2048-dim)
python scripts/preprocess_mind.py 3   # Extract textual features (all-MiniLM-L6-v2 -> 384-dim)
python scripts/preprocess_mind.py 4   # Compute top-80 similarity adjacency matrices
python scripts/preprocess_mind.py 0   # Run all steps
```

## Running Experiments

### Main experiments

```bash
# I2MD4Fair on Baby
python scripts/run_main.py --dataset baby --model I2MD4Fair --device cuda

# I2MD4Fair on Clothing
python scripts/run_main.py --dataset clothing --model I2MD4Fair --device cuda

# I2MD4Fair on MIND
python scripts/run_main.py --dataset mind --model I2MD4Fair --device cuda
```

Available models: `I2MD4Fair`, `LightGCN`, `VBPR`, `MMGCN`, `GRCN`, `LATTICE`, `FREEDOM`, `LGMRec`, `BM3`, `SLMRec`, `MMSSL`, `DiffMM`, `MENTOR`, `MMSSL+MD`, `DiffMM+MD`, `LGMRec+MD`, `MENTOR+MD`

### Ablation and hyperparameter studies

```bash
# Component analysis (Table 5)
python scripts/run_ablation.py --dataset baby --device cuda

# Hyperparameter sensitivity (Figs. 5-6)
python scripts/run_hyperparam.py --dataset baby --device cuda
```

### Output

Each run prints averaged metrics over `n_runs`:

```text
N@10 N@20 R@10 R@20 HR@10 HR@20 G@10 G@20 E@10 E@20 C@10 C@20 REG@10 REG@20
```

`N` = NDCG, `R` = Recall, `HR` = Hit Rate, `G` = Gini (lower is better), `E` = Entropy (higher is better), `C` = Coverage (higher is better), `REG` = Relevance-aware Exposure Gap (lower is better).

## Model Architecture (Paper Alignment)

The implementation aligns with all equations in the paper:

| Paper Equation | Implementation | Notes |
|----------|---------------|-------|
| Eq 2 | `modality_item_encoders[k]` — MLP projection | Maps raw features to d-dimensional latent space |
| Eqs 3-5 | `_modality_init_propagation()` with `D_U^{-1}R` and `D_V^{-1}R^T` (asymmetric normalization) | L_m=1 propagation layers; layer-wise average |
| Eqs 6-7 | `_id_message_passing()` with symmetric normalized Laplacian + layer-wise average | LightGCN-style, L=2 layers |
| Eqs 8-11 | `SinkhornOT` + `SoftPrototypeClustering` | Cosine dissimilarity cost; 3 Sinkhorn iterations; eps=0.1 |
| Eqs 12-13 | `SinkhornOT.forward()` — alternating row/column normalization | Denominators lower-bounded by 1e-12 |
| Eq 14 | `IntraMDM.forward()` — `incidence = B_I * gamma` | Hypergraph incidence matrix from OT transport plan |
| Eq 15 | `HypergraphConv` — `D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2} Z W` | Normalized hypergraph convolution |
| Eq 16 | `HypergraphConv.forward()` with **ReLU** activation | 1 propagation layer |
| Eqs 17-19 | `_reconstruct_modality_user()` + per-modality BPR | One-hop aggregation from debiased items to users |
| Eqs 20-22 | `CLUBEstimator` — 2-layer MLP (hidden=64) + diagonal Gaussian | Log-variance clamped to [-10, 10]; MI estimated via permutation |
| Eqs 24-28 | `InterMDM.adaptive_loss()` — p-norm aggregation | p=2; greater emphasis on modalities with larger objectives |
| Eq 30 | `InfoNCELoss` — all ordered modality pairs, divided by M(M-1) | Applied to debiased item representations; tau=0.01 |
| Eqs 31-33 | Concatenation fusion + BPR loss | `hat_X = [X_ID \|\| Z^1 \|\| ... \|\| Z^M]` |
| Eq 34 | Total loss = BPR + lambda_amb * L_AdaMB + lambda_cl * L_InfoNCE + lambda_reg * L2 | CLUB parameters excluded from Theta |

**Training (Algorithm 1):**
- CLUB estimators and recommender optimized alternately per mini-batch
- 5-epoch warm-up: CLUB trained but MI term excluded from recommender objective
- Gradient clipping with max_norm=5.0 on both optimization stages
- Prototypes are trainable parameters (nn.Parameter), optimized via backprop
- Early stopping on validation Recall@10 (patience=50, max 1000 epochs)
- Xavier initialization on all trainable parameters

## Hyperparameters (Paper Configuration)

| Parameter | Value | Description |
|-----------|-------|-------------|
| d | 64 | Embedding dimension |
| batch_size | 4096 | Training batch size |
| T | 64 | Number of semantic prototypes |
| epsilon | 0.1 | Sinkhorn entropy coefficient |
| Sinkhorn iters | 3 | Number of Sinkhorn iterations |
| L | 2 | LightGCN propagation layers |
| L_m | 1 | Modality-aware propagation layers |
| lambda_ib | 0.01 | Information compression weight |
| lambda_amb | 0.1 | Adaptive modality-balanced loss weight |
| lambda_cl | 0.1 | Cross-modal alignment loss weight |
| lambda_reg | 1e-4 | L2 regularization weight |
| p | 2 | Adaptive p-norm order |
| tau | 0.01 | InfoNCE temperature |
| lr | 1e-3 | Learning rate (Adam, beta1=0.9, beta2=0.999) |
| warmup_epochs | 5 | CLUB warm-up epochs |
| grad_clip | 5.0 | Gradient clipping max norm |
| early_stop | 50 | Early stopping patience on Recall@10 |
| max_epochs | 1000 | Maximum training epochs |
| n_runs | 5 | Number of repeated runs (averaged) |

## Evaluation Metrics

**Ranking utility (higher is better):** Recall@10, NDCG@10, HR@10

**Item-side exposure:**
- **Gini@10** (lower is better): Concentration of aggregate inclusion exposure
- **Entropy@10** (higher is better): Dispersion of aggregate exposure
- **Coverage@10** (higher is better): Fraction of catalog items appearing in at least one recommendation list
- **REG@10** (lower is better): Relevance-aware exposure gap — total variation distance between discounted exposure and held-out relevance demand

All metrics use full-catalog evaluation at cutoff K=10. Training and validation interactions are masked from test-time scoring.

## Citation

```bibtex
@article{I2MD4Fair2026,
  title={Intra- and Inter-Modality Debiasing for Item-Side Exposure Fairness in Multimodal Recommendation},
  author={Geng, Renwu and Wang, Fan and Liu, Weiming and Chen, Chaochao},
  journal={ACM Transactions on Information Systems (TOIS)},
  year={2026}
}
```
