from __future__ import annotations

import copy
import json
import time
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as tv_models
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import ImageFolder

warnings.filterwarnings("ignore")

# ============================================================
# GLOBALS
# ============================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("/content/drive/MyDrive/dataset/cattle")
SAVE_DIR = Path("cnn_cv_cattle_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Using device: {DEVICE}")


# ============================================================
# DATA HELPERS
# ============================================================
class RGBImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if isinstance(sample, Image.Image):
            sample = sample.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


class TransformSubset(Dataset):
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.targets = [subset.dataset.targets[i] for i in subset.indices]
        self.classes = subset.dataset.classes

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def get_cattle_loaders(
    data_dir: Path = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
    num_workers: int = NUM_WORKERS,
):
    train_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    eval_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    base_dataset = RGBImageFolder(data_dir, transform=None)
    class_names = base_dataset.classes
    num_classes = len(class_names)

    total_size = len(base_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size

    train_subset, val_subset, test_subset = random_split(
        base_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_ds = TransformSubset(train_subset, transform=train_tf)
    val_ds = TransformSubset(val_subset, transform=eval_tf)
    test_ds = TransformSubset(test_subset, transform=eval_tf)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)

    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Split sizes -> train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    return train_loader, val_loader, test_loader, class_names, num_classes


# ============================================================
# MODELS
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, residual: bool = False):
        super().__init__()
        self.residual = residual and (in_ch == out_ch) and (stride == 1)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.residual:
            out = out + x
        return self.relu(out)


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ConvBlock(32, 64, residual=False)
        self.pool1 = nn.MaxPool2d(2)
        self.layer2 = ConvBlock(64, 128, residual=False)
        self.pool2 = nn.MaxPool2d(2)
        self.layer3 = ConvBlock(128, 256, residual=False)
        self.pool3 = nn.MaxPool2d(2)
        self.layer4 = ConvBlock(256, 256, residual=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = self.layer4(x)
        x = self.gap(x)
        return self.classifier(x)


def build_transfer_model(arch: str, num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    if arch == "resnet18":
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        model = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    return model


# ============================================================
# TRAIN / EVAL
# ============================================================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 4,
    label: str = "model",
):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item() * xb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total += xb.size(0)

        model.eval()
        va_loss = va_correct = va_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss += loss.item() * xb.size(0)
                va_correct += (logits.argmax(1) == yb).sum().item()
                va_total += xb.size(0)

        scheduler.step()

        tr_acc = tr_correct / tr_total
        va_acc = va_correct / va_total
        history["train_loss"].append(tr_loss / tr_total)
        history["val_loss"].append(va_loss / va_total)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(
            f"[{label}] Epoch {epoch:02d}/{epochs} | "
            f"train_loss={tr_loss/tr_total:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss/va_total:.4f} val_acc={va_acc:.4f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping for {label} at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, class_names: list[str]) -> dict:
    model.eval().to(DEVICE)
    all_preds, all_labels, all_probs = [], [], []

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.numpy())
        all_probs.extend(probs.cpu().numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    metrics = {
        "acc": float((y_pred == y_true).mean()),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "y_pred": y_pred,
        "y_true": y_true,
    }

    try:
        metrics["auc_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        metrics["auc_macro"] = None

    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    return metrics


# ============================================================
# PLOTS
# ============================================================
def plot_training_history(history: dict, label: str, save_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve - {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{label.replace(' ', '_').lower()}_accuracy_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{label.replace(' ', '_').lower()}_loss_curve.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names: list[str], label: str, save_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {label}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_dir / f"{label.replace(' ', '_').lower()}_confusion_matrix.png", dpi=150)
    plt.close()


def plot_model_comparison(results: list[dict], save_dir: Path):
    df = pd.DataFrame(results)
    df.to_csv(save_dir / "final_results.csv", index=False)
    df.to_json(save_dir / "final_results.json", orient="records", indent=2)

    metrics = ["acc", "precision_macro", "recall_macro", "f1_macro"]
    for metric in metrics:
        plt.figure(figsize=(9, 5))
        plt.bar(df["model"], df[metric])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(f"{metric} Comparison")
        plt.tight_layout()
        plt.savefig(save_dir / f"comparison_{metric}.png", dpi=150)
        plt.close()

    df_plot = df.set_index("model")[metrics]
    ax = df_plot.plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel("Score")
    ax.set_title("Final Model Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_dir / "final_comparison_all_metrics.png", dpi=150)
    plt.close()

    best_idx = df["acc"].idxmax()
    best_row = df.loc[best_idx].to_dict()
    with open(save_dir / "best_model_summary.txt", "w", encoding="utf-8") as f:
        f.write("Best model based on test accuracy\n")
        f.write(json.dumps(best_row, indent=2))


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 72)
    print("CNN Computer Vision - Modified for Cattle Breed Dataset")
    print("=" * 72)

    train_loader, val_loader, test_loader, class_names, num_classes = get_cattle_loaders()

    all_results = []

    configs = [
        {
            "name": "Custom CNN",
            "model": CustomCNN(num_classes=num_classes),
            "epochs": 15,
            "lr": 1e-3,
            "patience": 4,
        },
        {
            "name": "ResNet18 TL",
            "model": build_transfer_model("resnet18", num_classes=num_classes),
            "epochs": 12,
            "lr": 5e-4,
            "patience": 3,
        },
        {
            "name": "MobileNetV3 TL",
            "model": build_transfer_model("mobilenet_v3_small", num_classes=num_classes),
            "epochs": 12,
            "lr": 5e-4,
            "patience": 3,
        },
    ]

    for cfg in configs:
        print("\n" + "-" * 72)
        print(f"Running model: {cfg['name']}")
        print("-" * 72)
        start = time.time()

        history = train_model(
            cfg["model"], train_loader, val_loader,
            epochs=cfg["epochs"], lr=cfg["lr"], patience=cfg["patience"], label=cfg["name"]
        )
        metrics = evaluate_model(cfg["model"], test_loader, class_names)
        elapsed = time.time() - start

        plot_training_history(history, cfg["name"], SAVE_DIR)
        plot_confusion_matrix(metrics["y_true"], metrics["y_pred"], class_names, cfg["name"], SAVE_DIR)

        result_row = {
            "model": cfg["name"],
            "acc": metrics["acc"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "auc_macro": metrics["auc_macro"],
            "time_sec": round(elapsed, 2),
        }
        all_results.append(result_row)
        print(result_row)

        report_path = SAVE_DIR / f"{cfg['name'].replace(' ', '_').lower()}_classification_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics["classification_report"], f, indent=2)

    plot_model_comparison(all_results, SAVE_DIR)
    print(f"\nAll outputs saved in: {SAVE_DIR.resolve()}")


if __name__ == "__main__":
    main()
