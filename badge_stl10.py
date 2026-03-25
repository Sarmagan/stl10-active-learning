import argparse
import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INITIAL_LABELED_SIZE = 500
VALIDATION_SIZE = 500
QUERY_SIZE = 200
AL_ITERATIONS = 5
EPOCHS_PER_ITER = 10
BATCH_SIZE = 64
NUM_CLASSES = 10
INITIAL_LR = 0.001
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
DROPOUT_P = 0.3
SEEDS = [42, 7, 123]

STL10_CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


# ==========================================
# 2. Transforms
# ==========================================
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(96, padding=12),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


# ==========================================
# 3. Seeding
# ==========================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 4. Model
# ==========================================
def get_new_model(pretrained_state: dict = None):
    """
    MobileNetV3-Small (~2.5M params).
    Pass pretrained_state to restore cached ImageNet weights without re-downloading.
    """
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=DROPOUT_P),
        nn.Linear(in_features, NUM_CLASSES),
    )
    model = model.to(DEVICE)
    if pretrained_state is not None:
        model.load_state_dict(copy.deepcopy(pretrained_state))
    return model


def get_feature_probabilities(model, indices, query_dataset):
    """Collect penultimate features and probabilities on the clean query transform."""
    model.eval()
    subset = Subset(query_dataset, indices)
    loader = DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    all_features = []
    all_probs = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(DEVICE)
            features = model.features(inputs)
            features = model.avgpool(features)
            features = torch.flatten(features, 1)
            logits = model.classifier(features)
            probs = torch.softmax(logits, dim=1)

            all_features.append(features.cpu())
            all_probs.append(probs.cpu())

    return torch.cat(all_features, dim=0), torch.cat(all_probs, dim=0)


# ==========================================
# 5. BADGE Query Strategy
# ==========================================
def get_badge_embeddings(model, unlabeled_indices, query_dataset):
    """
    Approximate last-layer gradient embeddings:
    g_x = f(x) outer (one_hot(argmax p) - p).
    """
    features, probs = get_feature_probabilities(model, unlabeled_indices, query_dataset)
    preds = probs.argmax(dim=1)
    one_hot = F.one_hot(preds, num_classes=NUM_CLASSES).float()
    grad_embeddings = features.unsqueeze(1) * (one_hot - probs).unsqueeze(2)
    return grad_embeddings.reshape(grad_embeddings.size(0), -1)


def badge_kmeanspp(embeddings, query_size):
    """k-means++ seeding over BADGE gradient embeddings."""
    num_points = embeddings.size(0)
    if num_points == 0:
        return []

    num_to_select = min(query_size, num_points)
    selected = [torch.norm(embeddings, dim=1).argmax().item()]

    min_dist_sq = torch.cdist(
        embeddings, embeddings[selected[0] : selected[0] + 1]
    ).pow(2).squeeze(1)

    while len(selected) < num_to_select:
        total_dist = min_dist_sq.sum().item()
        if total_dist <= 0:
            remaining = [idx for idx in range(num_points) if idx not in set(selected)]
            selected.extend(remaining[: num_to_select - len(selected)])
            break

        probs = (min_dist_sq / min_dist_sq.sum()).numpy()
        next_idx = int(np.random.choice(num_points, p=probs))
        while next_idx in selected:
            next_idx = int(np.random.choice(num_points, p=probs))

        selected.append(next_idx)
        new_dist_sq = torch.cdist(
            embeddings, embeddings[next_idx : next_idx + 1]
        ).pow(2).squeeze(1)
        min_dist_sq = torch.minimum(min_dist_sq, new_dist_sq)

    return selected


def query_badge(model, unlabeled_indices, query_dataset, query_size):
    embeddings = get_badge_embeddings(model, unlabeled_indices, query_dataset)
    selected_local = badge_kmeanspp(embeddings, query_size)
    return [unlabeled_indices[i] for i in selected_local]


# ==========================================
# 6. Scheduler
# ==========================================
def get_scheduler(optimizer, epochs):
    """Linear warmup -> cosine annealing."""
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
# 7. Training
# ==========================================
def train_model(model, train_indices, train_dataset, val_loader, epochs, current_iteration):
    subset = Subset(train_dataset, train_indices)
    loader = DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer, epochs)

    best_val_loss, best_model_state = float("inf"), None

    for epoch in range(epochs):
        current_lr = scheduler.get_last_lr()[0]

        model.train()
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / len(subset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                val_loss += criterion(model(inputs), labels).item() * inputs.size(0)
        epoch_val_loss = val_loss / len(val_loader.dataset)

        print(
            f"   Epoch {epoch + 1}/{epochs} | LR: {current_lr:.6f} | "
            f"Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}"
        )

        wandb.log(
            {
                "al_iteration": current_iteration,
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "epoch_val_loss": epoch_val_loss,
                "learning_rate": current_lr,
            }
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        scheduler.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"   --> Best weights loaded (Val Loss: {best_val_loss:.4f})")

    return model, best_val_loss


# ==========================================
# 8. Evaluation
# ==========================================
def evaluate_model(model, loader, log_confusion: bool = False, iteration: int = 0):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            preds = torch.max(model(inputs.to(DEVICE)), dim=1)[1].cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    overall_acc = 100 * (all_preds == all_labels).mean()

    per_class = {}
    for cls_idx, cls_name in enumerate(STL10_CLASSES):
        mask = all_labels == cls_idx
        per_class[f"acc_{cls_name}"] = (
            100 * (all_preds[mask] == all_labels[mask]).mean() if mask.sum() > 0 else 0.0
        )

    if log_confusion:
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(NUM_CLASSES),
            yticks=np.arange(NUM_CLASSES),
            xticklabels=STL10_CLASSES,
            yticklabels=STL10_CLASSES,
            xlabel="Predicted",
            ylabel="True",
            title=f"Confusion Matrix - Iteration {iteration}",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        thresh = cm.max() / 2.0
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8,
                )
        fig.tight_layout()
        wandb.log({"confusion_matrix": wandb.Image(fig), "al_iteration": iteration})
        plt.close(fig)

    return overall_acc, per_class


# ==========================================
# 9. One full AL run (single seed)
# ==========================================
def run_experiment(seed, train_dataset, query_dataset, test_loader, pretrained_state, run_name):
    set_seed(seed)

    all_indices = np.arange(len(train_dataset))
    np.random.shuffle(all_indices)

    labeled_indices = all_indices[:INITIAL_LABELED_SIZE].tolist()
    val_indices = all_indices[
        INITIAL_LABELED_SIZE : INITIAL_LABELED_SIZE + VALIDATION_SIZE
    ].tolist()
    unlabeled_set = set(all_indices[INITIAL_LABELED_SIZE + VALIDATION_SIZE :].tolist())

    val_loader = DataLoader(
        Subset(query_dataset, val_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    wandb.init(
        project="stl10-active-learning",
        name=run_name,
        config={
            "method": "badge",
            "seed": seed,
            "model": "MobileNetV3-Small",
            "initial_pool": INITIAL_LABELED_SIZE,
            "validation_size": VALIDATION_SIZE,
            "query_size": QUERY_SIZE,
            "al_iterations": AL_ITERATIONS,
            "epochs_per_iter": EPOCHS_PER_ITER,
            "batch_size": BATCH_SIZE,
            "lr": INITIAL_LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT_P,
            "warmup_epochs": WARMUP_EPOCHS,
            "label_smoothing": 0.1,
            "scheduler": "LinearWarmup+CosineAnnealing",
            "query_strategy": "badge",
            "feature_space": "last_layer_gradient_embeddings",
        },
        reinit=True,
    )

    model = get_new_model(pretrained_state)
    print(f"\n[{run_name}] Iter 0 - pool: {len(labeled_indices)}")
    model, best_val_loss = train_model(
        model, labeled_indices, train_dataset, val_loader, EPOCHS_PER_ITER, 0
    )
    acc, per_class = evaluate_model(model, test_loader, log_confusion=True, iteration=0)
    print(f"  Test Acc: {acc:.2f}%")
    wandb.log(
        {
            "iteration": 0,
            "labeled_pool_size": len(labeled_indices),
            "val_loss": best_val_loss,
            "test_accuracy": acc,
            **per_class,
        }
    )

    iteration_accs = [acc]

    for iteration in range(1, AL_ITERATIONS + 1):
        print(f"\n[{run_name}] Iter {iteration}")

        unlabeled_list = list(unlabeled_set)
        queried = query_badge(model, unlabeled_list, query_dataset, QUERY_SIZE)

        labeled_indices.extend(queried)
        unlabeled_set -= set(queried)

        print(f"  Labeled: {len(labeled_indices)} | Unlabeled: {len(unlabeled_set)}")

        model = get_new_model(pretrained_state)
        model, best_val_loss = train_model(
            model,
            labeled_indices,
            train_dataset,
            val_loader,
            EPOCHS_PER_ITER,
            iteration,
        )

        log_cm = iteration == AL_ITERATIONS
        acc, per_class = evaluate_model(
            model, test_loader, log_confusion=log_cm, iteration=iteration
        )
        print(f"  Test Acc: {acc:.2f}%")
        wandb.log(
            {
                "iteration": iteration,
                "labeled_pool_size": len(labeled_indices),
                "val_loss": best_val_loss,
                "test_accuracy": acc,
                **per_class,
            }
        )
        iteration_accs.append(acc)

    wandb.finish()
    return iteration_accs


# ==========================================
# 10. Main
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BADGE Active Learning on STL-10")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help=f"Seeds for multi-run averaging (default: {SEEDS})",
    )
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print("Method : BADGE")
    print(f"Seeds  : {args.seeds}")

    train_dataset = torchvision.datasets.STL10(
        root="./data", split="train", download=True, transform=train_transform
    )
    query_dataset = torchvision.datasets.STL10(
        root="./data", split="train", download=True, transform=val_transform
    )
    test_dataset = torchvision.datasets.STL10(
        root="./data", split="test", download=True, transform=val_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    print("\nCaching pretrained MobileNetV3-Small weights...")
    pretrained_state = copy.deepcopy(get_new_model().state_dict())

    all_accs = []
    for seed in args.seeds:
        run_name = f"run_badge_seed{seed}"
        print(f"\n{'=' * 55}\n  {run_name}\n{'=' * 55}")
        accs = run_experiment(
            seed=seed,
            train_dataset=train_dataset,
            query_dataset=query_dataset,
            test_loader=test_loader,
            pretrained_state=pretrained_state,
            run_name=run_name,
        )
        all_accs.append(accs)

    all_accs = np.array(all_accs)
    mean_accs = all_accs.mean(axis=0)
    std_accs = all_accs.std(axis=0)

    print(f"\n{'=' * 55}")
    print(f" SUMMARY - BADGE ({len(args.seeds)} seeds)")
    print(f"{'=' * 55}")
    for i, (m, s) in enumerate(zip(mean_accs, std_accs)):
        pool = INITIAL_LABELED_SIZE + i * QUERY_SIZE
        print(f"  Iter {i} | Pool {pool:>4} | {m:.2f}% +/- {s:.2f}%")

    wandb.init(
        project="stl10-active-learning",
        name="summary_badge",
        config={"method": "badge", "num_seeds": len(args.seeds)},
        reinit=True,
    )
    for i, (m, s) in enumerate(zip(mean_accs, std_accs)):
        wandb.log(
            {
                "iteration": i,
                "labeled_pool_size": INITIAL_LABELED_SIZE + i * QUERY_SIZE,
                "mean_test_accuracy": m,
                "std_test_accuracy": s,
            }
        )
    wandb.finish()
