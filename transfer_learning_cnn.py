"""
╔══════════════════════════════════════════════════════════════════════╗
║          TRANSFER LEARNING ON CNNs  ·  Complete Guide               ║
║──────────────────────────────────────────────────────────────────────║
║  Strategies covered:                                                 ║
║    1. Feature Extraction   – frozen backbone, train head only        ║
║    2. Fine-Tuning          – unfreeze top layers progressively       ║
║    3. Full Fine-Tuning     – unfreeze entire network (low LR)        ║
║    4. Domain Adaptation    – custom augmentation per domain          ║
║    5. Multi-backbone bench – ResNet-18/50, EfficientNet-B0,          ║
║                              MobileNetV3-S, DenseNet-121             ║
║    6. Layer-wise LR decay  – deeper layers get smaller LR           ║
║    7. Visualisations       – feature maps, weight histograms,        ║
║                              t-SNE embedding, training curves,       ║
║                              Grad-CAM comparison                     ║
║                                                                      ║
║  Dataset : CIFAR-10 (auto-downloaded)                                ║
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
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.models as tv_models
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
SEED     = 42
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("tl_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

torch.manual_seed(SEED); np.random.seed(SEED)
print(f"🖥  Device: {DEVICE}\n")


# ══════════════════════════════════════════════════════════════
# 1.  DATA
# ══════════════════════════════════════════════════════════════

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def get_loaders(batch_size: int = 128, val_frac: float = 0.1):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.2, 0.05),
        T.RandomRotation(12),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

    full   = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test   = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=eval_tf)

    n_val = int(len(full) * val_frac)
    tr_ds, va_ds = random_split(full, [len(full)-n_val, n_val],
                                generator=torch.Generator().manual_seed(SEED))
    va_ds.dataset.transform = eval_tf          # no augment on val

    kw = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    return (DataLoader(tr_ds, shuffle=True, **kw),
            DataLoader(va_ds, shuffle=False, **kw),
            DataLoader(test,  shuffle=False, **kw))


# ══════════════════════════════════════════════════════════════
# 2.  BACKBONE REGISTRY
# ══════════════════════════════════════════════════════════════

def _patch_for_cifar(model: nn.Module, arch: str) -> nn.Module:
    """
    CIFAR images are 32×32. Backbones pretrained on 224×224 ImageNet
    need their first layer patched so spatial resolution isn't crushed.
    """
    if arch.startswith("resnet"):
        model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif arch.startswith("densenet"):
        model.features.conv0   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.features.pool0   = nn.Identity()
    # EfficientNet / MobileNet handle small images reasonably well as-is
    return model


BACKBONE_REGISTRY: dict[str, tuple] = {
    "resnet18":          (tv_models.resnet18,          tv_models.ResNet18_Weights.DEFAULT,          "fc"),
    "resnet50":          (tv_models.resnet50,           tv_models.ResNet50_Weights.DEFAULT,           "fc"),
    "efficientnet_b0":   (tv_models.efficientnet_b0,    tv_models.EfficientNet_B0_Weights.DEFAULT,    "classifier"),
    "mobilenet_v3_small":(tv_models.mobilenet_v3_small, tv_models.MobileNet_V3_Small_Weights.DEFAULT, "classifier"),
    "densenet121":       (tv_models.densenet121,        tv_models.DenseNet121_Weights.DEFAULT,        "classifier"),
}

def build_model(
    arch:       str = "resnet18",
    strategy:   str = "feature_extract",   # feature_extract | finetune_top | full_finetune
    num_classes: int = 10,
) -> nn.Module:
    """
    strategy
    --------
    feature_extract  – pretrained backbone frozen; only new head trains
    finetune_top     – last 2 blocks + head unfrozen
    full_finetune    – entire network unfrozen (use low LR)
    """
    fn, weights, head_attr = BACKBONE_REGISTRY[arch]
    model = fn(weights=weights)
    model = _patch_for_cifar(model, arch)

    # ── Freeze all ────────────────────────────────────────────
    for p in model.parameters():
        p.requires_grad = False

    # ── Replace head ──────────────────────────────────────────
    head = getattr(model, head_attr)
    if isinstance(head, nn.Linear):
        in_f = head.in_features
        setattr(model, head_attr, nn.Linear(in_f, num_classes))
    elif isinstance(head, nn.Sequential):
        # EfficientNet / MobileNet use a Sequential classifier
        in_f = head[-1].in_features
        head[-1] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Unknown head type: {type(head)}")

    # ── Unfreeze per strategy ─────────────────────────────────
    if strategy == "feature_extract":
        pass   # head already has requires_grad=True by default (new layer)

    elif strategy == "finetune_top":
        children = list(model.named_children())
        # Unfreeze last 2 child modules + head
        unfreeze_names = {n for n, _ in children[-3:]}
        for name, module in model.named_children():
            if name in unfreeze_names:
                for p in module.parameters():
                    p.requires_grad = True

    elif strategy == "full_finetune":
        for p in model.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  [{arch} / {strategy}]  "
          f"trainable={trainable:,}  total={total:,}  "
          f"({100*trainable/total:.1f}%)")
    return model


# ══════════════════════════════════════════════════════════════
# 3.  LAYER-WISE LR DECAY
# ══════════════════════════════════════════════════════════════

def layerwise_lr_params(
    model: nn.Module,
    base_lr: float,
    decay: float = 0.3,
) -> list[dict]:
    """
    Assign exponentially decaying LRs: deeper (earlier) layers get smaller LR.
    This implements Discriminative Fine-Tuning (Howard & Ruder, 2018).
    """
    layers = list(model.named_children())
    n = len(layers)
    param_groups = []
    for i, (name, module) in enumerate(layers):
        # layer 0 (stem) gets base_lr * decay^(n-1), last layer gets base_lr
        lr_i = base_lr * (decay ** (n - 1 - i))
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": lr_i, "name": name})
    return param_groups


# ══════════════════════════════════════════════════════════════
# 4.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def train(
    model:        nn.Module,
    tr_loader:    DataLoader,
    va_loader:    DataLoader,
    epochs:       int   = 20,
    lr:           float = 1e-3,
    layerwise:    bool  = False,
    sched:        str   = "cosine",   # cosine | onecycle
    patience:     int   = 6,
    label:        str   = "model",
) -> tuple[nn.Module, dict]:
    model = model.to(DEVICE)

    if layerwise:
        param_groups = layerwise_lr_params(model, base_lr=lr, decay=0.4)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4,
        )

    if sched == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    else:   # onecycle
        scheduler = OneCycleLR(optimizer, max_lr=lr,
                               steps_per_epoch=len(tr_loader), epochs=epochs)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    history   = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[], lr=[])
    best_acc  = 0.0
    best_wts  = None
    no_imp    = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tl = tc = tt = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if sched == "onecycle":
                scheduler.step()
            tl += loss.item() * xb.size(0)
            preds = model(xb).argmax(1)
            tc += (preds == yb).sum().item(); tt += xb.size(0)

        # Val
        model.eval()
        vl = vc = vt = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                vl += criterion(logits, yb).item() * xb.size(0)
                vc += (logits.argmax(1) == yb).sum().item(); vt += xb.size(0)

        if sched == "cosine":
            scheduler.step()

        tr_acc = tc/tt; va_acc = vc/vt
        history["train_loss"].append(tl/tt); history["val_loss"].append(vl/vt)
        history["train_acc"].append(tr_acc); history["val_acc"].append(va_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"  [{label}] {epoch:>3}/{epochs}  "
              f"tr={tr_acc:.4f}  va={va_acc:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if va_acc > best_acc:
            best_acc = va_acc; best_wts = copy.deepcopy(model.state_dict()); no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  ⏹  Early stop @ epoch {epoch}")
                break

    model.load_state_dict(best_wts)
    print(f"  ✅  Best val acc: {best_acc:.4f}\n")
    return model, history


# ══════════════════════════════════════════════════════════════
# 5.  EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval().to(DEVICE)
    preds, labels, probs = [], [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        probs.extend(F.softmax(logits, 1).cpu().numpy())
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(yb.numpy())
    y_pred = np.array(preds); y_true = np.array(labels); y_prob = np.array(probs)
    return dict(
        acc     = accuracy_score(y_true, y_pred),
        f1      = f1_score(y_true, y_pred, average="macro"),
        auc     = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"),
        y_pred  = y_pred, y_true = y_true, y_prob = y_prob,
    )


# ══════════════════════════════════════════════════════════════
# 6.  FEATURE EMBEDDINGS  (for t-SNE)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model: nn.Module, loader: DataLoader, max_batches: int = 10):
    """
    Extract penultimate-layer embeddings by hooking just before the classifier.
    Works for ResNet (model.avgpool output) via a generic approach:
    register a hook on the second-to-last module.
    """
    model.eval().to(DEVICE)
    feats_list, labels_list = [], []
    hook_out = {}

    # Find the last children that is NOT the classifier/fc head
    children = list(model.named_children())
    # Second-to-last child is typically avgpool / features
    target_name, target_module = children[-2]

    def hook_fn(module, inp, out):
        hook_out["feat"] = out

    handle = target_module.register_forward_hook(hook_fn)

    for i, (xb, yb) in enumerate(loader):
        if i >= max_batches:
            break
        model(xb.to(DEVICE))
        f = hook_out["feat"]
        # Flatten any spatial dims
        if f.dim() > 2:
            f = f.flatten(1)
        feats_list.append(f.cpu().numpy())
        labels_list.extend(yb.numpy())

    handle.remove()
    return np.concatenate(feats_list, axis=0), np.array(labels_list)


# ══════════════════════════════════════════════════════════════
# 7.  GRAD-CAM
# ══════════════════════════════════════════════════════════════

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.grads = self.acts = None
        target_layer.register_forward_hook( lambda m,i,o: setattr(self, "acts", o.detach()))
        target_layer.register_full_backward_hook(lambda m,gi,go: setattr(self, "grads", go[0].detach()))

    def __call__(self, x: torch.Tensor, cls: Optional[int] = None):
        self.model.eval()
        x = x.to(DEVICE).requires_grad_(True)
        logits = self.model(x)
        cls = cls or logits.argmax(1).item()
        self.model.zero_grad(); logits[0, cls].backward()
        w   = self.grads.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), cls


def get_last_conv(model: nn.Module) -> nn.Module:
    """Return last Conv2d in model."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


# ══════════════════════════════════════════════════════════════
# 8.  VISUALISATIONS
# ══════════════════════════════════════════════════════════════

BG="#0d1117"; FG="#e8e0d4"; AC="#c8a96e"; BL="#6e9cc8"; GR="#7ec894"

def _style(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=FG)
    ax.spines[:].set_color("#2a3040")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(FG)


def plot_training_curves(histories: dict[str, dict], save_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    palette = [AC, BL, GR, "#e87c7c", "#c890e8"]
    for (label, h), color in zip(histories.items(), palette):
        ep = range(1, len(h["train_acc"]) + 1)
        axes[0].plot(ep, h["train_loss"], color=color, lw=1.5, linestyle="--", alpha=0.6)
        axes[0].plot(ep, h["val_loss"],   color=color, lw=2,   label=label)
        axes[1].plot(ep, h["train_acc"],  color=color, lw=1.5, linestyle="--", alpha=0.6)
        axes[1].plot(ep, h["val_acc"],    color=color, lw=2,   label=label)
    for ax, title in zip(axes, ["Loss (val=solid)", "Accuracy (val=solid)"]):
        _style(ax); ax.set_title(title, color=FG, fontsize=12, pad=8)
        ax.set_xlabel("Epoch", color=FG)
        ax.legend(facecolor="#1a1f2e", labelcolor=FG, fontsize=8)
    fig.suptitle("Transfer Learning Strategy Comparison — Training Curves",
                 color=FG, fontsize=13, y=1.01)
    fig.tight_layout()
    p = save_dir / "training_curves.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"  📊 {p}")


def plot_lr_schedule(histories: dict[str, dict], save_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG); _style(ax)
    palette = [AC, BL, GR, "#e87c7c"]
    for (label, h), color in zip(histories.items(), palette):
        ax.plot(h["lr"], color=color, lw=2, label=label)
    ax.set_title("Learning Rate Schedules", color=FG, fontsize=12, pad=8)
    ax.set_xlabel("Step", color=FG); ax.set_ylabel("LR", color=FG)
    ax.legend(facecolor="#1a1f2e", labelcolor=FG, fontsize=9)
    fig.tight_layout()
    p = save_dir / "lr_schedules.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"  📊 {p}")


def plot_tsne(model: nn.Module, loader: DataLoader,
              label: str, save_dir: Path):
    print(f"  t-SNE for {label} …")
    feats, y = extract_embeddings(model, loader, max_batches=8)
    z = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(feats)
    cmap = plt.cm.get_cmap("tab10", 10)
    fig, ax = plt.subplots(figsize=(9, 8), facecolor=BG); _style(ax)
    for cls in range(10):
        mask = y == cls
        ax.scatter(z[mask, 0], z[mask, 1], c=[cmap(cls)],
                   s=12, alpha=0.7, label=CLASSES[cls])
    ax.set_title(f"t-SNE Feature Embeddings — {label}", color=FG, fontsize=12, pad=8)
    ax.legend(facecolor="#1a1f2e", labelcolor=FG, fontsize=7,
              ncol=2, markerscale=1.5)
    fig.tight_layout()
    p = save_dir / f"tsne_{label.replace(' ','_')}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"  📊 {p}")


def plot_confusion(y_true, y_pred, label: str, save_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
    ax.set_facecolor(BG)
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr",
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.3, linecolor="#1a1f2e", ax=ax,
                cbar_kws={"shrink": 0.75})
    ax.tick_params(colors=FG, labelsize=8)
    ax.set_xlabel("Predicted", color=FG); ax.set_ylabel("True", color=FG)
    ax.set_title(f"Confusion Matrix — {label}", color=FG, fontsize=12, pad=10)
    fig.tight_layout()
    p = save_dir / f"cm_{label.replace(' ','_')}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"  📊 {p}")


def plot_gradcam_grid(models_dict: dict[str, nn.Module],
                      loader: DataLoader, save_dir: Path, n: int = 6):
    mean = np.array(CIFAR_MEAN); std = np.array(CIFAR_STD)
    imgs, labels = next(iter(loader))
    imgs = imgs[:n]; labels = labels[:n]

    model_names = list(models_dict.keys())
    n_models = len(model_names)
    fig, axes = plt.subplots(n_models + 1, n,
                             figsize=(2.3*n, 2.5*(n_models+1)), facecolor=BG)
    fig.suptitle("Grad-CAM Comparison Across Backbones", color=FG, fontsize=13, y=1.01)

    # Row 0: original images
    for j in range(n):
        img = (imgs[j].permute(1,2,0).numpy() * std + mean).clip(0,1)
        axes[0, j].imshow(img); axes[0, j].axis("off")
        axes[0, j].set_title(CLASSES[labels[j]], color=AC, fontsize=7, pad=2)
    axes[0, 0].set_ylabel("Original", color=FG, fontsize=8)

    for i, name in enumerate(model_names, start=1):
        model = models_dict[name]
        target_layer = get_last_conv(model)
        gcam = GradCAM(model, target_layer)
        for j in range(n):
            img_np = (imgs[j].permute(1,2,0).numpy() * std + mean).clip(0,1)
            cam, pred = gcam(imgs[j:j+1])
            axes[i, j].imshow(img_np)
            axes[i, j].imshow(cam, cmap="jet", alpha=0.45)
            axes[i, j].axis("off")
            axes[i, j].set_title(CLASSES[pred], color=BL, fontsize=7, pad=2)
        axes[i, 0].set_ylabel(name, color=FG, fontsize=7)

    fig.tight_layout()
    p = save_dir / "gradcam_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"  📊 {p}")


def plot_weight_histogram(models_dict: dict[str, nn.Module], save_dir: Path):
    fig, axes = plt.subplots(1, len(models_dict),
                             figsize=(5*len(models_dict), 4), facecolor=BG)
    if len(models_dict) == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, models_dict.items()):
        all_w = []
        for p in model.parameters():
            all_w.append(p.data.cpu().flatten().numpy())
        all_w = np.concatenate(all_w)
        _style(ax)
        ax.hist(all_w, bins=120, color=AC, edgecolor="none", alpha=0.8)
        ax.set_title(name, color=FG, fontsize=10, pad=6)
        ax.set_xlabel("Weight value", color=FG)
        ax.set_ylabel("Count", color=FG)
    fig.suptitle("Weight Distribution per Backbone", color=FG, fontsize=12, y=1.02)
    fig.tight_layout()
    p = save_dir / "weight_histograms.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"  📊 {p}")


def plot_benchmark(results: dict[str, dict], save_dir: Path):
    names   = list(results.keys())
    metrics = ["acc", "f1", "auc"]
    labels  = ["Accuracy", "F1 Macro", "ROC-AUC"]
    colors  = [AC, BL, GR]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        vals = [results[n][metric] for n in names]
        bars = ax.barh(names, vals, color=color, edgecolor="#2a3040", height=0.5)
        ax.set_xlim(0, 1.08); _style(ax)
        ax.set_title(label, color=FG, fontsize=11, pad=8)
        for bar, v in zip(bars, vals):
            ax.text(v+0.005, bar.get_y()+bar.get_height()/2,
                    f"{v:.3f}", va="center", color=FG, fontsize=8)
    fig.suptitle("Transfer Learning Benchmark — Test Set", color=FG, fontsize=14, y=1.02)
    fig.tight_layout()
    p = save_dir / "benchmark.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"  📊 {p}")


# ══════════════════════════════════════════════════════════════
# 9.  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("═"*65)
    print("  TRANSFER LEARNING ON CNNs  ·  Full Pipeline")
    print("═"*65 + "\n")

    tr_loader, va_loader, te_loader = get_loaders(batch_size=128)

    histories: dict[str, dict]  = {}
    results:   dict[str, dict]  = {}
    trained:   dict[str, nn.Module] = {}

    # ── Experiment 1: Strategy comparison on ResNet-18 ────────
    print("━"*55)
    print("  Experiment 1 — Strategy Comparison (ResNet-18)")
    print("━"*55)

    strategy_cfg = [
        ("ResNet-18 Feature Extract", "resnet18", "feature_extract", 1e-3, False, "cosine"),
        ("ResNet-18 Finetune Top",    "resnet18", "finetune_top",    5e-4, False, "cosine"),
        ("ResNet-18 Full Finetune",   "resnet18", "full_finetune",   1e-4, True,  "cosine"),
        ("ResNet-18 LayerLR Decay",   "resnet18", "full_finetune",   3e-4, True,  "onecycle"),
    ]

    for label, arch, strategy, lr, layerwise, sched in strategy_cfg:
        print(f"\n▸ {label}")
        model = build_model(arch, strategy, num_classes=10)
        model, h = train(model, tr_loader, va_loader,
                         epochs=15, lr=lr, layerwise=layerwise,
                         sched=sched, patience=5, label=label)
        r = evaluate(model, te_loader)
        print(f"  acc={r['acc']:.4f}  f1={r['f1']:.4f}  auc={r['auc']:.4f}")
        histories[label] = h; results[label] = r; trained[label] = model

    # ── Experiment 2: Multi-backbone benchmark ─────────────────
    print("\n" + "━"*55)
    print("  Experiment 2 — Multi-Backbone Benchmark (Full Finetune)")
    print("━"*55)

    backbone_cfg = [
        ("ResNet-50",       "resnet50",          1e-4),
        ("EfficientNet-B0", "efficientnet_b0",   5e-4),
        ("MobileNetV3-S",   "mobilenet_v3_small",5e-4),
        ("DenseNet-121",    "densenet121",        1e-4),
    ]

    for label, arch, lr in backbone_cfg:
        print(f"\n▸ {label}")
        model = build_model(arch, "full_finetune", num_classes=10)
        model, h = train(model, tr_loader, va_loader,
                         epochs=15, lr=lr, layerwise=False,
                         sched="cosine", patience=5, label=label)
        r = evaluate(model, te_loader)
        print(f"  acc={r['acc']:.4f}  f1={r['f1']:.4f}  auc={r['auc']:.4f}")
        histories[label] = h; results[label] = r; trained[label] = model

    # ── Visualisations ─────────────────────────────────────────
    print("\n" + "━"*55)
    print("  Generating Visualisations")
    print("━"*55)

    plot_training_curves(histories, SAVE_DIR)
    plot_lr_schedule(histories, SAVE_DIR)
    plot_benchmark(results, SAVE_DIR)
    plot_weight_histogram(
        {k: trained[k] for k in ["ResNet-18 Feature Extract",
                                  "ResNet-18 Full Finetune",
                                  "EfficientNet-B0"]},
        SAVE_DIR,
    )

    # Grad-CAM across backbones
    cam_models = {k: trained[k] for k in
                  ["ResNet-18 Full Finetune", "EfficientNet-B0",
                   "MobileNetV3-S", "DenseNet-121"]}
    plot_gradcam_grid(cam_models, te_loader, SAVE_DIR)

    # Confusion matrix for best model
    best_key = max(results, key=lambda k: results[k]["acc"])
    r_best   = results[best_key]
    plot_confusion(r_best["y_true"], r_best["y_pred"], best_key, SAVE_DIR)

    # t-SNE for feature extractor vs full finetune
    for key in ["ResNet-18 Feature Extract", "ResNet-18 Full Finetune"]:
        plot_tsne(trained[key], te_loader, key, SAVE_DIR)

    # ── Summary table ──────────────────────────────────────────
    print("\n" + "═"*65)
    print(f"  {'Model':<32} {'Acc':>7} {'F1':>7} {'AUC':>7}")
    print("  " + "─"*55)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["acc"]):
        print(f"  {name:<32} {r['acc']:>7.4f} {r['f1']:>7.4f} {r['auc']:>7.4f}")
    print("═"*65)
    print(f"\n🏆  Best: {best_key}  acc={r_best['acc']:.4f}")

    # Detailed classification report for best
    print(f"\n── Classification Report: {best_key} ─────────────────────")
    print(classification_report(r_best["y_true"], r_best["y_pred"],
                                 target_names=CLASSES, digits=4))

    print(f"\n📁  All outputs → {SAVE_DIR.resolve()}/")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n⏱  Total time: {(time.time()-t0)/60:.1f} min")
