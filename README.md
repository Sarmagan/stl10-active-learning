# stl10-active-learning

> Comparing uncertainty-based active learning query strategies against random sampling on STL-10 using MobileNetV3-Small.

---

## Overview

This project benchmarks four active learning query strategies on the [STL-10](https://cs.stanford.edu/~acoates/stl10/) image classification dataset. Starting from a small labeled pool of 500 samples, each strategy iteratively queries the most informative unlabeled samples, measuring how efficiently test accuracy improves as the labeled pool grows. Results are averaged over 3 random seeds and reported as mean ± std.

---

## Dataset

**STL-10** is a 10-class image classification dataset derived from ImageNet, designed for developing unsupervised and semi-supervised learning algorithms.

| Property | Value |
|---|---|
| Image size | 96 × 96 RGB |
| Training set | 5,000 labeled images (500 per class) |
| Test set | 8,000 images |
| Classes | airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck |

### Data Split (per seed)

| Split | Size | Purpose |
|---|---|---|
| Initial labeled pool | 500 | Training at iteration 0 |
| Validation set | 500 | Model selection / early stopping |
| Unlabeled pool | 4,000 | Query candidates for AL |

---

## Model

**MobileNetV3-Small** pretrained on ImageNet, fine-tuned on STL-10.

| Property | Value |
|---|---|
| Architecture | MobileNetV3-Small |
| Parameters | ~2.5M |
| Pretrained | ImageNet (torchvision) |
| Classifier head | Dropout(0.3) → Linear(576, 10) |

MobileNetV3-Small was chosen over larger models (e.g. ResNet18 at 11M params) to reduce overfitting on the small labeled pools used in active learning iterations.

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| LR schedule | Linear warmup (3 epochs) → CosineAnnealing |
| Epochs per AL iteration | 10 |
| Batch size | 64 |
| Label smoothing | 0.1 |
| Dropout | 0.3 |

### Data Augmentation

Training uses augmented transforms; validation and query inference always use clean transforms to avoid noise in uncertainty scores.

| Split | Transforms |
|---|---|
| Train | RandomHorizontalFlip, RandomCrop(96, padding=12), ColorJitter, RandomGrayscale, Normalize |
| Val / Query | ToTensor, Normalize |

### Model Reset

The model is **fully reset to pretrained ImageNet weights** at the start of each AL iteration. This prevents carrying over local minima and ensures a fair comparison across pool sizes.

---

## Active Learning Setup

| Property | Value |
|---|---|
| Initial labeled pool | 500 |
| Query size per iteration | 200 |
| AL iterations | 5 |
| Final labeled pool | 1,500 |
| Seeds | 42, 7, 123 |

---

## Query Strategies

### Random Sampling (baseline)
Selects samples uniformly at random from the unlabeled pool. Serves as the lower-bound baseline — no model information is used.

### Least Confidence
Queries samples where the model's maximum class probability is lowest:

$$x^* = \arg\min_{x} \max_c \, p(y=c \mid x)$$

### Margin Sampling
Queries samples where the gap between the top-2 class probabilities is smallest:

$$x^* = \arg\min_{x} \left( p_1(x) - p_2(x) \right)$$

### Entropy Sampling
Queries samples with the highest predictive entropy:

$$x^* = \arg\max_{x} -\sum_c p(y=c \mid x) \log p(y=c \mid x)$$

---

## Results

### Full Dataset Baseline

Training on all 4,500 available samples (labeled + unlabeled, excluding validation) for 60 epochs:

| Training Set Size | Test Accuracy |
|---|---|
| 4,500 (full) | **86.89%** |

---

### Active Learning Comparison

Test accuracy (mean ± std over 3 seeds) as the labeled pool grows from 500 to 1,500 samples.

| Pool Size | Random | Least Confidence | Margin Sampling | Entropy Sampling |
|---|---|---|---|---|
| 500 | 66.50 ± 0.93 | 65.75 ± 2.15 | 64.62 ± 0.87 | 64.27 ± 1.86 |
| 700 | 69.77 ± 2.23 | 70.92 ± 1.00 | 71.48 ± 0.39 | 70.77 ± 0.86 |
| 900 | 70.72 ± 0.02 | 73.11 ± 0.62 | 73.13 ± 0.77 | 73.27 ± 0.48 |
| 1100 | 75.43 ± 0.68 | 77.00 ± 1.05 | 76.34 ± 1.02 | 77.09 ± 1.44 |
| 1300 | 77.89 ± 0.46 | 78.85 ± 0.72 | 78.43 ± 0.13 | 78.48 ± 0.51 |
| **1500** | 78.92 ± 0.60 | 80.08 ± 0.66 | 80.05 ± 0.49 | **80.42 ± 0.22** |

**Best strategy at 1,500 labels:** Entropy Sampling (80.42%) — narrowly ahead of Least Confidence (80.08%) and Margin Sampling (80.05%), all notably above Random (78.92%).

**Gap vs. full dataset:** The best AL strategy at 1,500 labels (80.42%) achieves ~92.6% of the full-dataset accuracy (86.89%) using only 33% of the available labeled data.

---

## Key Observations

- **All uncertainty strategies outperform random sampling** consistently from pool size 700 onward, confirming that model-guided querying is beneficial even on small datasets.
- **Entropy and margin sampling converge tightly** across all pool sizes, suggesting that the difference in uncertainty signal between these two criteria is marginal at this scale.
- **Standard deviation decreases** as the pool grows for all methods, indicating that results become more stable as more labeled data is available.

---

## Usage

```bash
# Install dependencies
pip install torch torchvision wandb scikit-learn matplotlib

# Full dataset baseline (single run)
python active_learning_mobilenet.py --method full_dataset

# Active learning strategies (3 seeds by default)
python active_learning_mobilenet.py --method random_sampling
python active_learning_mobilenet.py --method least_confidence
python active_learning_mobilenet.py --method margin_sampling
python active_learning_mobilenet.py --method entropy_sampling

# Custom seeds
python active_learning_mobilenet.py --method entropy_sampling --seeds 0 1 2 3 4
```

All runs are logged to [Weights & Biases](https://wandb.ai) under the project `stl10-active-learning`. Each seed produces an individual run; a `summary_{method}` run aggregates mean ± std for cross-method comparison charts.

---

## Repository Structure

```
stl10-active-learning/
├── active_learning_mobilenet.py   # Main training & evaluation script
├── README.md
└── data/                          # STL-10 downloaded automatically by torchvision
```

---

## Dependencies

| Package | Purpose |
|---|---|
| torch / torchvision | Model, training, datasets |
| wandb | Experiment tracking |
| scikit-learn | Confusion matrix |
| matplotlib | Confusion matrix visualization |
| numpy | Array operations |