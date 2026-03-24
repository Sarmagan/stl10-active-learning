import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import random
import copy
import wandb

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Identical to AL script for a fair comparison
INITIAL_LABELED_SIZE = 500
VALIDATION_SIZE      = 500
QUERY_SIZE           = 200          # SSL has no queries, but we evaluate at the
AL_ITERATIONS        = 5            # same labeled pool sizes as AL (500,700,...,1500)
                                    # by subsampling the labeled pool
EPOCHS_PER_ITER      = 10
BATCH_SIZE           = 64
NUM_CLASSES          = 10
INITIAL_LR           = 0.001
WEIGHT_DECAY         = 1e-4
WARMUP_EPOCHS        = 3
DROPOUT_P            = 0.3
SEEDS                = [42, 7, 123]

# FixMatch hyperparameters
CONFIDENCE_THRESHOLD = 0.95         # pseudo-label confidence threshold
LAMBDA_U             = 1.0          # weight of unsupervised loss
SUPERVISED_WARMUP    = 3            # epochs of pure supervised training before SSL kicks in
MU                   = 3            # unlabeled batch size multiplier (unlabeled = MU * BATCH_SIZE)

STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
                 'dog', 'horse', 'monkey', 'ship', 'truck']

# ==========================================
# 2. Transforms
#    FixMatch uses weak aug for pseudo-label generation,
#    strong aug for the consistency target.
# ==========================================
weak_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(96, padding=12),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

strong_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(96, padding=12),
    transforms.RandAugment(num_ops=2, magnitude=9),   # strong aug
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# ==========================================
# 3. Dataset that returns (weak_aug, strong_aug) pairs
#    Used for the unlabeled pool in FixMatch training.
# ==========================================
class UnlabeledPairDataset(Dataset):
    """
    Wraps an STL-10 Subset and returns (weak_aug, strong_aug) pairs.
    Labels are ignored — we only use the images.
    """
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices      = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[self.indices[idx]]
        # base_dataset already applied ToTensor+Normalize — we need the raw PIL image
        # So we use the raw_dataset (no transform) stored separately
        return idx   # placeholder; see note in train_fixmatch


class RawSTL10(Dataset):
    """STL-10 with no transform — returns PIL images for dual augmentation."""
    def __init__(self, root, split, download=False):
        self.base = torchvision.datasets.STL10(root=root, split=split,
                                               download=download, transform=None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, label   # PIL image, int label


class LabeledDataset(Dataset):
    """Labeled subset — applies weak transform to labeled images."""
    def __init__(self, raw_dataset, indices, transform):
        self.raw       = raw_dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.raw[self.indices[idx]]
        return self.transform(img), label


class UnlabeledDataset(Dataset):
    """
    Unlabeled subset — returns (weak_aug, strong_aug) pairs.
    Labels are discarded; ground-truth never used during training.
    """
    def __init__(self, raw_dataset, indices):
        self.raw     = raw_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, _ = self.raw[self.indices[idx]]   # label intentionally dropped
        return weak_transform(img), strong_transform(img)


# ==========================================
# 4. Seeding
# ==========================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 5. Model
# ==========================================
def get_new_model(pretrained_state: dict = None):
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=DROPOUT_P),
        nn.Linear(in_features, NUM_CLASSES)
    )
    model = model.to(DEVICE)
    if pretrained_state is not None:
        model.load_state_dict(copy.deepcopy(pretrained_state))
    return model

# ==========================================
# 6. Scheduler
# ==========================================
def get_scheduler(optimizer, epochs):
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - WARMUP_EPOCHS, 1), eta_min=1e-6
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
    )

# ==========================================
# 7. FixMatch Training
#
#    Each epoch:
#      - Supervised: standard cross-entropy on labeled batch
#      - Unsupervised (after warmup): 
#          1. Get pseudo-label from weak-aug pass (no grad)
#          2. Only keep if max confidence > threshold
#          3. Apply cross-entropy loss on strong-aug pass
#          Total loss = L_supervised + lambda_u * L_unsupervised
# ==========================================
def train_fixmatch(model, labeled_indices, unlabeled_indices,
                   raw_dataset, val_loader, epochs, iteration):

    labeled_dataset   = LabeledDataset(raw_dataset, labeled_indices, weak_transform)
    unlabeled_dataset = UnlabeledDataset(raw_dataset, unlabeled_indices)

    labeled_loader   = DataLoader(labeled_dataset,   batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE * MU,
                                  shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)

    criterion_sup = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_u   = nn.CrossEntropyLoss(reduction='none')   # per-sample, for masking
    optimizer     = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler     = get_scheduler(optimizer, epochs)

    best_val_loss, best_model_state = float('inf'), None

    for epoch in range(epochs):
        current_lr   = scheduler.get_last_lr()[0]
        use_ssl      = epoch >= SUPERVISED_WARMUP   # pure supervised warmup first

        model.train()
        total_loss_sum   = 0.0
        sup_loss_sum     = 0.0
        unsup_loss_sum   = 0.0
        mask_ratio_sum   = 0.0
        num_batches      = 0

        unlabeled_iter = iter(unlabeled_loader)

        for labeled_imgs, labels in labeled_loader:
            labeled_imgs = labeled_imgs.to(DEVICE)
            labels       = labels.to(DEVICE)

            # --- Supervised loss ---
            logits_sup = model(labeled_imgs)
            loss_sup   = criterion_sup(logits_sup, labels)

            # --- Unsupervised loss (FixMatch) ---
            loss_u    = torch.tensor(0.0, device=DEVICE)
            mask_ratio = 0.0

            if use_ssl:
                try:
                    weak_imgs, strong_imgs = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    weak_imgs, strong_imgs = next(unlabeled_iter)

                weak_imgs   = weak_imgs.to(DEVICE)
                strong_imgs = strong_imgs.to(DEVICE)

                # Pseudo-labels from weak augmentation (no gradient)
                with torch.no_grad():
                    probs_weak    = torch.softmax(model(weak_imgs), dim=1)
                    conf, pseudo  = probs_weak.max(dim=1)   # confidence, pseudo-label

                # Confidence mask — only train on high-confidence pseudo-labels
                mask       = (conf >= CONFIDENCE_THRESHOLD).float()
                mask_ratio = mask.mean().item()

                # Consistency loss on strong augmentation
                logits_strong = model(strong_imgs)
                loss_u_all    = criterion_u(logits_strong, pseudo)
                loss_u        = (loss_u_all * mask).mean()

            total_loss = loss_sup + LAMBDA_U * loss_u

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_sum  += total_loss.item()
            sup_loss_sum    += loss_sup.item()
            unsup_loss_sum  += loss_u.item()
            mask_ratio_sum  += mask_ratio
            num_batches     += 1

        avg_total  = total_loss_sum  / num_batches
        avg_sup    = sup_loss_sum    / num_batches
        avg_unsup  = unsup_loss_sum  / num_batches
        avg_mask   = mask_ratio_sum  / num_batches

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, lbls in val_loader:
                inputs, lbls = inputs.to(DEVICE), lbls.to(DEVICE)
                val_loss += criterion_sup(model(inputs), lbls).item() * inputs.size(0)
        epoch_val_loss = val_loss / len(val_loader.dataset)

        ssl_str = f"MaskRatio: {avg_mask:.2f} | Unsup: {avg_unsup:.4f}" if use_ssl else "SSL: warmup"
        print(f"   Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | "
              f"Sup: {avg_sup:.4f} | Val: {epoch_val_loss:.4f} | {ssl_str}")

        wandb.log({
            "ssl_iteration":      iteration,
            "epoch":              epoch + 1,
            "train_loss_total":   avg_total,
            "train_loss_sup":     avg_sup,
            "train_loss_unsup":   avg_unsup,
            "pseudo_mask_ratio":  avg_mask,
            "epoch_val_loss":     epoch_val_loss,
            "learning_rate":      current_lr,
            "labeled_pool_size":  len(labeled_indices),
        })

        if epoch_val_loss < best_val_loss:
            best_val_loss    = epoch_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        scheduler.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"   --> Best weights loaded (Val Loss: {best_val_loss:.4f})")

    return model, best_val_loss

# ==========================================
# 8. Evaluation
# ==========================================
def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            preds = torch.max(model(inputs.to(DEVICE)), dim=1)[1].cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    overall_acc = 100 * (all_preds == all_labels).mean()

    per_class = {}
    for cls_idx, cls_name in enumerate(STL10_CLASSES):
        mask = all_labels == cls_idx
        per_class[f"acc_{cls_name}"] = (
            100 * (all_preds[mask] == all_labels[mask]).mean() if mask.sum() > 0 else 0.0
        )
    return overall_acc, per_class

# ==========================================
# 9. One SSL run (single seed)
#
#    Key difference from AL:
#    - The full unlabeled pool (4,000 samples) is ALWAYS used for SSL training
#    - We evaluate at the same labeled pool sizes as AL (500, 700, ..., 1500)
#      by subsampling the labeled pool — NOT by querying
#    - This isolates the effect of SSL vs AL fairly
# ==========================================
def run_ssl_experiment(seed, raw_dataset, val_raw_dataset, test_loader,
                       pretrained_state, run_name):
    set_seed(seed)

    all_indices = np.arange(len(raw_dataset))
    np.random.shuffle(all_indices)

    # Use the same split structure as the AL script
    initial_labeled = all_indices[:INITIAL_LABELED_SIZE].tolist()
    val_indices     = all_indices[INITIAL_LABELED_SIZE : INITIAL_LABELED_SIZE + VALIDATION_SIZE].tolist()
    unlabeled_pool  = all_indices[INITIAL_LABELED_SIZE + VALIDATION_SIZE:].tolist()

    # Validation loader — uses val_transform via a LabeledDataset with val_transform
    val_dataset = LabeledDataset(val_raw_dataset, val_indices, val_transform)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Pool sizes to evaluate — mirrors AL experiment exactly
    pool_sizes = [INITIAL_LABELED_SIZE + i * QUERY_SIZE for i in range(AL_ITERATIONS + 1)]

    # We have 4,500 - 500 (val) = 4,000 remaining.
    # initial_labeled (500) + unlabeled_pool (4,000) = 4,500 total non-val samples.
    # For each pool size, we use the first N as labeled and the rest as unlabeled.
    # This means SSL always has access to 4000 - (pool_size - 500) unlabeled samples.
    all_non_val = initial_labeled + unlabeled_pool   # fixed order, seed-controlled

    wandb.init(
        project="stl10-active-learning",
        name=run_name,
        config={
            "method":             "ssl_fixmatch",
            "seed":               seed,
            "model":              "MobileNetV3-Small",
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "lambda_u":           LAMBDA_U,
            "mu":                 MU,
            "supervised_warmup":  SUPERVISED_WARMUP,
            "validation_size":    VALIDATION_SIZE,
            "epochs_per_eval":    EPOCHS_PER_ITER,
            "batch_size":         BATCH_SIZE,
            "lr":                 INITIAL_LR,
            "weight_decay":       WEIGHT_DECAY,
            "dropout":            DROPOUT_P,
            "warmup_epochs":      WARMUP_EPOCHS,
            "label_smoothing":    0.1,
            "scheduler":          "LinearWarmup+CosineAnnealing",
        },
        reinit=True,
    )

    iteration_accs = []

    for i, pool_size in enumerate(pool_sizes):
        labeled_indices   = all_non_val[:pool_size]
        unlabeled_indices = all_non_val[pool_size:]   # remaining samples used as unlabeled

        print(f"\n[{run_name}] Pool {pool_size} "
              f"| Labeled: {len(labeled_indices)} | Unlabeled: {len(unlabeled_indices)}")

        # Fresh pretrained weights each evaluation point — same as AL script
        model = get_new_model(pretrained_state)
        model, best_val_loss = train_fixmatch(
            model, labeled_indices, unlabeled_indices,
            raw_dataset, val_loader, EPOCHS_PER_ITER, iteration=i
        )

        acc, per_class = evaluate_model(model, test_loader)
        print(f"  Test Acc: {acc:.2f}%")
        wandb.log({
            "iteration":         i,
            "labeled_pool_size": pool_size,
            "unlabeled_pool_size": len(unlabeled_indices),
            "val_loss":          best_val_loss,
            "test_accuracy":     acc,
            **per_class
        })
        iteration_accs.append(acc)

    wandb.finish()
    return iteration_accs

# ==========================================
# 10. Main
# ==========================================
if __name__ == '__main__':
    print(f"Device : {DEVICE}")
    print(f"Method : SSL_FIXMATCH")
    print(f"Seeds  : {SEEDS}")
    print(f"Confidence threshold : {CONFIDENCE_THRESHOLD}")
    print(f"Lambda_u             : {LAMBDA_U}")
    print(f"Unlabeled batch mult : {MU}x")

    # Raw (PIL) datasets — needed for dual augmentation in FixMatch
    raw_train   = RawSTL10(root='./data', split='train', download=True)
    raw_val     = RawSTL10(root='./data', split='train', download=False)

    # Standard test loader with val_transform
    test_dataset = torchvision.datasets.STL10(root='./data', split='test',
                                              download=True, transform=val_transform)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Cache pretrained weights once
    print("\nCaching pretrained MobileNetV3-Small weights...")
    pretrained_state = copy.deepcopy(get_new_model().state_dict())

    all_accs = []

    for seed in SEEDS:
        run_name = f"run_ssl_fixmatch_seed{seed}"
        print(f"\n{'='*55}\n  {run_name}\n{'='*55}")
        accs = run_ssl_experiment(
            seed=seed,
            raw_dataset=raw_train,
            val_raw_dataset=raw_val,
            test_loader=test_loader,
            pretrained_state=pretrained_state,
            run_name=run_name,
        )
        all_accs.append(accs)

    all_accs  = np.array(all_accs)
    mean_accs = all_accs.mean(axis=0)
    std_accs  = all_accs.std(axis=0)

    pool_sizes = [INITIAL_LABELED_SIZE + i * QUERY_SIZE for i in range(AL_ITERATIONS + 1)]

    print(f"\n{'='*55}")
    print(f" SUMMARY — SSL_FIXMATCH ({len(SEEDS)} seeds)")
    print(f"{'='*55}")
    for pool_size, m, s in zip(pool_sizes, mean_accs, std_accs):
        print(f"  Pool {pool_size:>4} | {m:.2f}% ± {s:.2f}%")

    # Summary W&B run for cross-method comparison
    wandb.init(project="stl10-active-learning",
               name="summary_ssl_fixmatch",
               config={"method": "ssl_fixmatch", "num_seeds": len(SEEDS)},
               reinit=True)
    for pool_size, m, s in zip(pool_sizes, mean_accs, std_accs):
        wandb.log({
            "labeled_pool_size":  pool_size,
            "mean_test_accuracy": m,
            "std_test_accuracy":  s,
        })
    wandb.finish()