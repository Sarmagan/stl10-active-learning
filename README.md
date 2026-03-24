# stl10-active-learning

> Comparing uncertainty-based active learning query strategies and semi-supervised learning (FixMatch) against random sampling on STL-10 using MobileNetV3-Small.

---

## Overview

This project benchmarks four active learning query strategies alongside a semi-supervised learning (SSL) baseline on the [STL-10](https://cs.stanford.edu/~acoates/stl10/) image classification dataset. Starting from a small labeled pool of 500 samples, the active learning strategies iteratively query the most informative unlabeled samples. The SSL baseline leverages the remaining unlabeled data at each pool size via consistency regularization. The goal is to measure how efficiently test accuracy improves as the labeled pool grows, and at what point targeted active learning overtakes semi-supervised learning. Results are averaged over 3 random seeds and reported as mean ± std.

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
| Unlabeled pool | 4,000 | Query candidates for AL / Unlabeled pool for SSL |

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
| Epochs per Evaluation | 10 |
| Batch size | 64 |
| Label smoothing | 0.1 |
| Dropout | 0.3 |

### Data Augmentation

Training uses augmented transforms; validation and query inference always use clean transforms. The SSL approach (FixMatch) utilizes a dual-augmentation strategy.

| Split / Method | Transforms |
|---|---|
| Train (AL / Supervised) | RandomHorizontalFlip, RandomCrop(96, padding=12), ColorJitter, RandomGrayscale, Normalize |
| Val / Query | ToTensor, Normalize |
| FixMatch (Weak) | RandomHorizontalFlip, RandomCrop(96, padding=12), ToTensor, Normalize |
| FixMatch (Strong) | RandomHorizontalFlip, RandomCrop(96, padding=12), RandAugment(2, 9), ToTensor, Normalize |

### Model Reset

The model is **fully reset to pretrained ImageNet weights** at the start of each evaluation step (whether querying in AL or expanding the pool in SSL). This prevents carrying over local minima and ensures a fair comparison across pool sizes.

---

## Evaluation Setup

| Property | Value |
|---|---|
| Initial labeled pool | 500 |
| Step size per iteration | 200 |
| Evaluation points | 5 |
| Final labeled pool | 1,500 |
| Seeds | 42, 7, 123 |

---

## Strategies Evaluated

### Active Learning Strategies
* **Random Sampling (baseline):** Selects samples uniformly at random from the unlabeled pool. Serves as the lower-bound baseline.
* **Least Confidence:** Queries samples where the model's maximum class probability is lowest:
  $$x^* = \arg\min_{x} \max_c \, p(y=c \mid x)$$
* **Margin Sampling:** Queries samples where the gap between the top-2 class probabilities is smallest:
  $$x^* = \arg\min_{x} \left( p_1(x) - p_2(x) \right)$$
* **Entropy Sampling:** Queries samples with the highest predictive entropy:
  $$x^* = \arg\max_{x} -\sum_c p(y=c \mid x) \log p(y=c \mid x)$$

### Semi-Supervised Baseline
* **FixMatch:** At each labeled pool size, the model trains on the labeled subset while applying consistency regularization to the remaining unlabeled data. Weakly augmented images generate pseudo-labels (if confidence $\ge 0.95$), which are then used as targets for strongly augmented versions of the same images.

---

## Results

### Full Dataset Baseline

Training on all 4,500 available samples (labeled + unlabeled, excluding validation) for 60 epochs:

| Training Set Size | Test Accuracy |
|---|---|
| 4,500 (full) | **86.89%** |

---

### Active Learning vs. Semi-Supervised Comparison

Test accuracy (mean ± std over 3 seeds) as the labeled pool grows from 500 to 1,500 samples.

| Pool Size | Random | Least Confidence | Margin Sampling | Entropy Sampling | SSL (FixMatch) |
|---|---|---|---|---|---|
| **500** | 66.50 ± 0.93 | 65.75 ± 2.15 | 64.62 ± 0.87 | 64.27 ± 1.86 | **69.21 ± 0.16** |
| **700** | 69.77 ± 2.23 | 70.92 ± 1.00 | 71.48 ± 0.39 | 70.77 ± 0.86 | **72.75 ± 0.71** |
| **900** | 70.72 ± 0.02 | 73.11 ± 0.62 | 73.13 ± 0.77 | 73.27 ± 0.48 | **75.08 ± 0.59** |
| **1100** | 75.43 ± 0.68 | 77.00 ± 1.05 | 76.34 ± 1.02 | 77.09 ± 1.44 | **77.45 ± 0.63** |
| **1300** | 77.89 ± 0.46 | **78.85 ± 0.72** | 78.43 ± 0.13 | 78.48 ± 0.51 | 78.82 ± 0.72 |
| **1500** | 78.92 ± 0.60 | 80.08 ± 0.66 | 80.05 ± 0.49 | **80.42 ± 0.22** | 79.64 ± 0.34 |

---

## Key Observations

- **The Low-Data SSL Advantage:** In the extremely low-data regime (500–1,100 labels), FixMatch heavily dominates. Because the AL feature extractors are weak at this stage, uncertainty metrics are largely inaccurate. FixMatch bypasses this by leveraging the 4,000+ unlabeled images immediately to learn robust features.
- **The Active Learning Crossover:** As the labeled pool reaches 1,300–1,500 samples, Active Learning (specifically Entropy and Least Confidence) overtakes SSL. Once the model is competent enough, querying highly informative targeted labels yields better performance than training on random labels combined with pseudo-labels.
- **Uncertainty Strategies vs. Random:** All uncertainty strategies consistently outperform random sampling from pool size 700 onward, confirming that model-guided querying remains highly beneficial.
- **Stability:** FixMatch shows remarkable stability (low standard deviation) even at the lowest data scales, whereas AL methods initially suffer from higher variance until the feature representations mature.

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

# Semi-supervised baseline (FixMatch)
python active_learning_mobilenet.py --method ssl_fixmatch

# Custom seeds
python active_learning_mobilenet.py --method entropy_sampling --seeds 0 1 2 3 4