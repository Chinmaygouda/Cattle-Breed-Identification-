"""
╔══════════════════════════════════════════════════════════════════════╗
║          CNN-BASED COMPUTER VISION  ·  Full Pipeline                ║
║──────────────────────────────────────────────────────────────────────║
║  Covers:                                                             ║
║    1. Custom CNN Architecture (from scratch)                         ║
║    2. Transfer Learning  (ResNet-18, MobileNetV3)                    ║
║    3. Data Augmentation  (albumentations-style with torchvision)     ║
║    4. Object Detection   (sliding-window + Grad-CAM localisation)    ║
║    5. Semantic Segmentation  (lightweight U-Net)                     ║
║    6. Grad-CAM Visualisation                                         ║
║    7. Full Training Loop w/ LR Scheduling + Early Stopping           ║
║    8. Evaluation  (accuracy, F1, confusion matrix, ROC-AUC)          ║
║                                                                      ║
║  Dataset : CIFAR-10 (auto-downloaded via torchvision)                ║
║  Framework: PyTorch + torchvision                                    ║
║                                                                      ║
║  Install:                                                            ║
║    pip install torch torchvision matplotlib seaborn scikit-learn     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import copy
import time
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.models as tv_models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  GLOBALS
# ─────────────────────────────────────────────
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR   = Path("cnn_cv_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"🖥  Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════
# 1.  DATA LOADING & AUGMENTATION
# ══════════════════════════════════════════════════════════════

def get_cifar10_loaders(
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """CIFAR-10 with strong train augmentation."""

    # ── Train augmentation ──────────────────────────────────
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomRotation(15),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])

    # ── Val / test: only normalise ──────────────────────────
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_tf
    )
    test_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=eval_tf
    )

    n_val = int(len(full_train) * val_split)
    n_tr  = len(full_train) - n_val
    tr_ds, val_ds = random_split(
        full_train, [n_tr, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    # Override val transform
    val_ds.dataset.transform = eval_tf

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    tr_loader  = DataLoader(tr_ds,  shuffle=True,  **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    te_loader  = DataLoader(test_ds, shuffle=False, **kw)

    print(f"✅  CIFAR-10  train={n_tr}  val={n_val}  test={len(test_ds)}")
    return tr_loader, val_loader, te_loader


# ══════════════════════════════════════════════════════════════
# 2.  CNN ARCHITECTURES
# ══════════════════════════════════════════════════════════════

# ── 2a. Custom CNN from scratch ────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BN → ReLU (+ optional residual)."""
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
    """
    Small residual CNN for CIFAR-10 (32×32 input).
    ~1.5M parameters.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ConvBlock(32,  64,  stride=1, residual=False)   # 32×32
        self.pool1  = nn.MaxPool2d(2)                                  # 16×16

        self.layer2 = ConvBlock(64,  128, stride=1, residual=False)
        self.pool2  = nn.MaxPool2d(2)                                  #  8×8

        self.layer3 = ConvBlock(128, 256, stride=1, residual=False)
        self.pool3  = nn.MaxPool2d(2)                                  #  4×4

        self.layer4 = ConvBlock(256, 256, stride=1, residual=True)    #  4×4

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
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = self.layer4(x)
        x = self.gap(x)
        return self.classifier(x)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return final conv feature map (before GAP) for Grad-CAM."""
        x = self.stem(x)
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        return self.layer4(x)                # (B, 256, 4, 4)


# ── 2b. Transfer Learning wrapper ─────────────────────────────

def build_transfer_model(
    arch: str = "resnet18",
    num_classes: int = 10,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Fine-tune a pretrained backbone on CIFAR-10.

    arch: "resnet18" | "mobilenet_v3_small"
    """
    if arch == "resnet18":
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        # CIFAR images are 32×32; replace first conv + remove maxpool
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif arch == "mobilenet_v3_small":
        model = tv_models.mobilenet_v3_small(
            weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT
        )
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown arch: {arch}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    return model


# ══════════════════════════════════════════════════════════════
# 3.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:       int   = 20,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    patience:     int   = 5,
    label:        str   = "model",
) -> dict:
    """
    Training loop with:
      • AdamW optimiser
      • Cosine annealing LR
      • Early stopping
      • Best-model checkpointing
    Returns history dict.
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    history = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])
    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss    += loss.item() * xb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += xb.size(0)

        # ── Validate ─────────────────────────────────────────
        model.eval()
        va_loss = va_correct = va_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss   = criterion(logits, yb)
                va_loss    += loss.item() * xb.size(0)
                va_correct += (logits.argmax(1) == yb).sum().item()
                va_total   += xb.size(0)

        scheduler.step()

        tr_acc = tr_correct / tr_total
        va_acc = va_correct / va_total
        history["train_loss"].append(tr_loss / tr_total)
        history["val_loss"].append(va_loss / va_total)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"  [{label}] Ep {epoch:>3}/{epochs}  "
              f"tr_loss={tr_loss/tr_total:.4f}  tr_acc={tr_acc:.4f}  "
              f"va_loss={va_loss/va_total:.4f}  va_acc={va_acc:.4f}")

        # ── Early stopping ────────────────────────────────────
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  ⏹  Early stop at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    print(f"  ✅  Best val acc: {best_val_acc:.4f}")
    return history


# ══════════════════════════════════════════════════════════════
# 4.  EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader) -> dict:
    model.eval().to(DEVICE)
    all_preds, all_labels, all_probs = [], [], []

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        probs  = F.softmax(logits, dim=1)
        preds  = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.numpy())
        all_probs.extend(probs.cpu().numpy())

    y_pred  = np.array(all_preds)
    y_true  = np.array(all_labels)
    y_prob  = np.array(all_probs)

    acc    = (y_pred == y_true).mean()
    f1_mac = f1_score(y_true, y_pred, average="macro")
    auc    = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

    return dict(acc=acc, f1_macro=f1_mac, auc=auc,
                y_pred=y_pred, y_true=y_true, y_prob=y_prob)


# ══════════════════════════════════════════════════════════════
# 5.  GRAD-CAM
# ══════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Maps for the CustomCNN.
    Hooks into the last conv layer (layer4).
    """
    def __init__(self, model: CustomCNN):
        self.model      = model
        self.gradients  = None
        self.activations = None

        # Register hooks on the final conv block
        target = model.layer4.block[-1]   # last BatchNorm in residual block

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target.register_forward_hook(fwd_hook)
        target.register_full_backward_hook(bwd_hook)

    def __call__(
        self, x: torch.Tensor, class_idx: Optional[int] = None
    ) -> np.ndarray:
        """Return CAM heatmap for input x (1×C×H×W) as (H, W) array in [0,1]."""
        self.model.eval()
        x = x.to(DEVICE).requires_grad_(True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Global average-pool the gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)

        # Normalise & resize to input spatial dims
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


# ══════════════════════════════════════════════════════════════
# 6.  LIGHTWEIGHT U-NET  (semantic segmentation demo)
# ══════════════════════════════════════════════════════════════

class UNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LightUNet(nn.Module):
    """
    Lightweight U-Net for semantic segmentation.
    Input: (B, 3, H, W)  →  Output: (B, num_classes, H, W)
    """
    def __init__(self, num_classes: int = 2, base_ch: int = 32):
        super().__init__()
        b = base_ch
        # Encoder
        self.enc1 = UNetBlock(3,    b)
        self.enc2 = UNetBlock(b,   b*2)
        self.enc3 = UNetBlock(b*2, b*4)
        # Bottleneck
        self.bottleneck = UNetBlock(b*4, b*8)
        # Decoder
        self.up3 = nn.ConvTranspose2d(b*8, b*4, 2, stride=2)
        self.dec3 = UNetBlock(b*8, b*4)
        self.up2 = nn.ConvTranspose2d(b*4, b*2, 2, stride=2)
        self.dec2 = UNetBlock(b*4, b*2)
        self.up1 = nn.ConvTranspose2d(b*2, b,   2, stride=2)
        self.dec1 = UNetBlock(b*2, b)
        # Head
        self.head = nn.Conv2d(b, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bn = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def demo_unet_segmentation(save_dir: Path):
    """
    Forward-pass a random image through the U-Net
    and visualise predicted segmentation mask.
    """
    model = LightUNet(num_classes=3, base_ch=32).eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        logits = model(x)
    seg_mask = logits.argmax(dim=1).squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.axis("off")

    img_np = ((x.squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1))
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image (random)", color="#e8e0d4", fontsize=11)

    axes[1].imshow(seg_mask, cmap="tab20", vmin=0, vmax=2)
    axes[1].set_title("U-Net Segmentation Mask", color="#e8e0d4", fontsize=11)

    patches = [mpatches.Patch(color=plt.cm.tab20(i / 20), label=f"Class {i}")
               for i in range(3)]
    axes[1].legend(handles=patches, loc="lower right",
                   fontsize=8, facecolor="#1a1f2e",
                   labelcolor="#e8e0d4", edgecolor="#333")

    fig.suptitle("Lightweight U-Net — Semantic Segmentation Demo",
                 color="#c8a96e", fontsize=13)
    fig.tight_layout()
    path = save_dir / "unet_segmentation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  📊 Saved: {path}")


# ══════════════════════════════════════════════════════════════
# 7.  VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════

BG  = "#0d1117"
FG  = "#e8e0d4"
ACC = "#c8a96e"

def plot_training_history(history: dict, label: str, save_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG)
        ax.spines[:].set_color("#2a3040")
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], color=ACC,    lw=2, label="Train")
    ax1.plot(epochs, history["val_loss"],   color="#6e9cc8", lw=2, label="Val", linestyle="--")
    ax1.set_title("Loss Curve"); ax1.set_xlabel("Epoch"); ax1.legend(facecolor="#1a1f2e", labelcolor=FG)

    ax2.plot(epochs, history["train_acc"], color=ACC,    lw=2, label="Train")
    ax2.plot(epochs, history["val_acc"],   color="#6e9cc8", lw=2, label="Val", linestyle="--")
    ax2.set_title("Accuracy Curve"); ax2.set_xlabel("Epoch"); ax2.legend(facecolor="#1a1f2e", labelcolor=FG)

    fig.suptitle(f"Training History — {label}", color=FG, fontsize=13, y=1.02)
    fig.tight_layout()
    path = save_dir / f"history_{label.replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  📊 Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, classes: list[str], label: str, save_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(11, 9), facecolor=BG)
    ax.set_facecolor(BG)
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.4, linecolor="#1a1f2e",
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(f"Confusion Matrix — {label}", color=FG, fontsize=13, pad=12)
    fig.tight_layout()
    path = save_dir / f"cm_{label.replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  📊 Saved: {path}")


def plot_gradcam(
    model: CustomCNN,
    test_loader: DataLoader,
    save_dir: Path,
    n: int = 8,
):
    """Show Grad-CAM overlaid on a grid of CIFAR-10 test images."""
    cam_gen  = GradCAM(model)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2470, 0.2435, 0.2616])

    images, labels = next(iter(test_loader))
    images = images[:n]; labels = labels[:n]

    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 6), facecolor=BG)
    fig.suptitle("Grad-CAM Visualisation", color=FG, fontsize=14, y=1.01)

    for i in range(n):
        x    = images[i:i+1]
        cam, pred = cam_gen(x)

        # Denormalise
        img_np = (images[i].permute(1, 2, 0).numpy() * std + mean).clip(0, 1)

        # Row 0: original
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(
            f"True: {CIFAR10_CLASSES[labels[i]]}", color=FG, fontsize=7, pad=3
        )
        axes[0, i].axis("off")

        # Row 1: Grad-CAM overlay
        axes[1, i].imshow(img_np)
        axes[1, i].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, i].set_title(
            f"Pred: {CIFAR10_CLASSES[pred]}", color=ACC, fontsize=7, pad=3
        )
        axes[1, i].axis("off")

    fig.tight_layout()
    path = save_dir / "gradcam.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  📊 Saved: {path}")


def plot_model_comparison(results: dict[str, dict], save_dir: Path):
    metrics = ["acc", "f1_macro", "auc"]
    labels  = ["Accuracy", "F1 Macro", "ROC-AUC"]
    names   = list(results.keys())
    colors  = [ACC, "#6e9cc8", "#a8c890"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        vals = [results[n][metric] for n in names]
        bars = ax.barh(names, vals, color=color, edgecolor="#2a3040", height=0.5)
        ax.set_xlim(0, 1.1)
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG)
        ax.spines[:].set_color("#2a3040")
        ax.set_title(label, color=FG, fontsize=12, pad=8)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", color=FG, fontsize=9)

    fig.suptitle("Model Comparison — Test Set", color=FG, fontsize=14, y=1.02)
    fig.tight_layout()
    path = save_dir / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  📊 Saved: {path}")


def plot_augmentation_grid(dataset, save_dir: Path, n: int = 8):
    """Visualise augmented training samples."""
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2470, 0.2435, 0.2616])

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.5), facecolor=BG)
    fig.suptitle("Data Augmentation Samples", color=FG, fontsize=13)
    indices = np.random.choice(len(dataset), n, replace=False)
    for ax, idx in zip(axes, indices):
        img, label = dataset[int(idx)]
        img_np = (img.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        ax.imshow(img_np)
        ax.set_title(CIFAR10_CLASSES[label], color=ACC, fontsize=8, pad=3)
        ax.axis("off")
    fig.tight_layout()
    path = save_dir / "augmentation_samples.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  📊 Saved: {path}")


# ══════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 65)
    print("  CNN-BASED COMPUTER VISION  ·  Full Pipeline")
    print("═" * 65 + "\n")

    # ── Data ────────────────────────────────────────────────
    tr_loader, va_loader, te_loader = get_cifar10_loaders(batch_size=128)

    # Visualise augmented samples
    plot_augmentation_grid(tr_loader.dataset, SAVE_DIR)

    # ── U-Net demo ──────────────────────────────────────────
    print("\n── Semantic Segmentation Demo (U-Net) ─────────────────")
    demo_unet_segmentation(SAVE_DIR)

    all_results: dict[str, dict] = {}

    # ── Model 1: Custom CNN ──────────────────────────────────
    print("\n── Model 1: Custom CNN ────────────────────────────────")
    custom_model = CustomCNN(num_classes=10)
    n_params = sum(p.numel() for p in custom_model.parameters())
    print(f"  Parameters: {n_params:,}")

    h1 = train_model(
        custom_model, tr_loader, va_loader,
        epochs=25, lr=1e-3, patience=6, label="CustomCNN",
    )
    plot_training_history(h1, "Custom CNN", SAVE_DIR)
    res1 = evaluate_model(custom_model, te_loader)
    print(f"  Test → acc={res1['acc']:.4f}  f1={res1['f1_macro']:.4f}  auc={res1['auc']:.4f}")
    plot_confusion_matrix(res1["y_true"], res1["y_pred"], CIFAR10_CLASSES, "Custom CNN", SAVE_DIR)
    all_results["Custom CNN"] = res1

    # ── Grad-CAM ────────────────────────────────────────────
    print("\n── Grad-CAM Visualisation ─────────────────────────────")
    plot_gradcam(custom_model, te_loader, SAVE_DIR)

    # ── Model 2: ResNet-18 Transfer Learning ────────────────
    print("\n── Model 2: ResNet-18 (Transfer Learning) ─────────────")
    resnet = build_transfer_model("resnet18", num_classes=10, freeze_backbone=False)
    n_params_rn = sum(p.numel() for p in resnet.parameters())
    print(f"  Parameters: {n_params_rn:,}")

    h2 = train_model(
        resnet, tr_loader, va_loader,
        epochs=20, lr=5e-4, patience=5, label="ResNet18",
    )
    plot_training_history(h2, "ResNet-18 TL", SAVE_DIR)
    res2 = evaluate_model(resnet, te_loader)
    print(f"  Test → acc={res2['acc']:.4f}  f1={res2['f1_macro']:.4f}  auc={res2['auc']:.4f}")
    plot_confusion_matrix(res2["y_true"], res2["y_pred"], CIFAR10_CLASSES, "ResNet-18", SAVE_DIR)
    all_results["ResNet-18 TL"] = res2

    # ── Model 3: MobileNetV3-Small ──────────────────────────
    print("\n── Model 3: MobileNetV3-Small (Transfer Learning) ─────")
    mobilenet = build_transfer_model("mobilenet_v3_small", num_classes=10)
    n_params_mn = sum(p.numel() for p in mobilenet.parameters())
    print(f"  Parameters: {n_params_mn:,}")

    h3 = train_model(
        mobilenet, tr_loader, va_loader,
        epochs=20, lr=5e-4, patience=5, label="MobileNetV3",
    )
    plot_training_history(h3, "MobileNetV3-Small TL", SAVE_DIR)
    res3 = evaluate_model(mobilenet, te_loader)
    print(f"  Test → acc={res3['acc']:.4f}  f1={res3['f1_macro']:.4f}  auc={res3['auc']:.4f}")
    plot_confusion_matrix(res3["y_true"], res3["y_pred"], CIFAR10_CLASSES, "MobileNetV3-S", SAVE_DIR)
    all_results["MobileNetV3-S TL"] = res3

    # ── Final comparison ────────────────────────────────────
    print("\n── Model Comparison ───────────────────────────────────")
    plot_model_comparison(all_results, SAVE_DIR)

    # ── Print summary ───────────────────────────────────────
    print("\n" + "═" * 65)
    print(f"  {'Model':<25} {'Acc':>8} {'F1 Mac':>8} {'AUC':>8}")
    print("  " + "─" * 55)
    for name, r in all_results.items():
        print(f"  {name:<25} {r['acc']:>8.4f} {r['f1_macro']:>8.4f} {r['auc']:>8.4f}")
    print("═" * 65)

    best = max(all_results, key=lambda k: all_results[k]["acc"])
    print(f"\n🏆  Best model: {best}  acc={all_results[best]['acc']:.4f}")

    print(f"\n📁  All outputs saved to: {SAVE_DIR.resolve()}/")

    # Detailed classification report
    print("\n── Detailed Report (Custom CNN) ───────────────────────")
    print(classification_report(
        res1["y_true"], res1["y_pred"],
        target_names=CIFAR10_CLASSES, digits=4,
    ))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n⏱  Total time: {(time.time()-t0)/60:.1f} min")
