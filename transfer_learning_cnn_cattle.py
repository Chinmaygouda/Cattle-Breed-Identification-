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
import torchvision
import torchvision.transforms as T
import torchvision.models as tv_models
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, random_split, Subset

warnings.filterwarnings("ignore")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("tl_cattle_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("/content/drive/MyDrive/dataset/cattle")
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2

BG = "#0d1117"
FG = "#e8e0d4"
AC = "#c8a96e"
BL = "#6e9cc8"
GR = "#7ec894"

np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"Using device: {DEVICE}")


class RGBImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


def get_loaders(batch_size: int = BATCH_SIZE):
    train_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(0.2, 0.2, 0.15, 0.03),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    base_ds = RGBImageFolder(DATA_DIR, transform=train_tf)
    class_names = base_ds.classes
    num_classes = len(class_names)

    n_total = len(base_ds)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        base_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )

    val_base = RGBImageFolder(DATA_DIR, transform=eval_tf)
    test_base = RGBImageFolder(DATA_DIR, transform=eval_tf)
    val_ds = Subset(val_base, val_ds.indices)
    test_ds = Subset(test_base, test_ds.indices)

    kw = dict(batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    print("Classes:", class_names)
    print("Number of classes:", num_classes)
    return train_loader, val_loader, test_loader, class_names, num_classes


BACKBONE_REGISTRY = {
    "resnet18": (tv_models.resnet18, tv_models.ResNet18_Weights.DEFAULT, "fc"),
    "resnet50": (tv_models.resnet50, tv_models.ResNet50_Weights.DEFAULT, "fc"),
    "efficientnet_b0": (tv_models.efficientnet_b0, tv_models.EfficientNet_B0_Weights.DEFAULT, "classifier"),
    "mobilenet_v3_small": (tv_models.mobilenet_v3_small, tv_models.MobileNet_V3_Small_Weights.DEFAULT, "classifier"),
    "densenet121": (tv_models.densenet121, tv_models.DenseNet121_Weights.DEFAULT, "classifier"),
}


def build_model(arch: str = "resnet18", strategy: str = "feature_extract", num_classes: int = 2) -> nn.Module:
    fn, weights, head_attr = BACKBONE_REGISTRY[arch]
    model = fn(weights=weights)

    for p in model.parameters():
        p.requires_grad = False

    head = getattr(model, head_attr)
    if isinstance(head, nn.Linear):
        in_f = head.in_features
        setattr(model, head_attr, nn.Linear(in_f, num_classes))
    elif isinstance(head, nn.Sequential):
        in_f = head[-1].in_features
        head[-1] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Unknown head type: {type(head)}")

    if strategy == "feature_extract":
        pass
    elif strategy == "finetune_top":
        children = list(model.named_children())
        unfreeze_names = {n for n, _ in children[-3:]}
        for name, module in model.named_children():
            if name in unfreeze_names:
                for p in module.parameters():
                    p.requires_grad = True
    elif strategy == "full_finetune":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError("Invalid strategy")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  [{arch} / {strategy}] trainable={trainable:,} total={total:,} ({100*trainable/total:.1f}%)")
    return model


def layerwise_lr_params(model: nn.Module, base_lr: float, decay: float = 0.3):
    layers = list(model.named_children())
    n = len(layers)
    param_groups = []
    for i, (name, module) in enumerate(layers):
        lr_i = base_lr * (decay ** (n - 1 - i))
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": lr_i, "name": name})
    return param_groups


def train(model, tr_loader, va_loader, epochs=10, lr=1e-3, layerwise=False, sched="cosine", patience=4, label="model"):
    model = model.to(DEVICE)
    if layerwise:
        param_groups = layerwise_lr_params(model, base_lr=lr, decay=0.4)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    if sched == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    else:
        scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(tr_loader), epochs=epochs)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    history = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[], lr=[])
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    no_imp = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tl = tc = tt = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if sched == "onecycle":
                scheduler.step()
            tl += loss.item() * xb.size(0)
            preds = logits.argmax(1)
            tc += (preds == yb).sum().item()
            tt += xb.size(0)

        model.eval()
        vl = vc = vt = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                vl += loss.item() * xb.size(0)
                vc += (logits.argmax(1) == yb).sum().item()
                vt += xb.size(0)

        if sched == "cosine":
            scheduler.step()

        tr_acc = tc / max(tt, 1)
        va_acc = vc / max(vt, 1)
        history["train_loss"].append(tl / max(tt, 1))
        history["val_loss"].append(vl / max(vt, 1))
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"  [{label}] {epoch:02d}/{epochs} | train_loss={tl/max(tt,1):.4f} train_acc={tr_acc:.4f} | val_loss={vl/max(vt,1):.4f} val_acc={va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            best_wts = copy.deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_wts)
    print(f"  Best val acc: {best_acc:.4f}\n")
    return model, history


@torch.no_grad()
def evaluate(model, loader):
    model.eval().to(DEVICE)
    preds, labels, probs = [], [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        prob = F.softmax(logits, 1)
        probs.extend(prob.cpu().numpy())
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(yb.numpy())
    y_pred = np.array(preds)
    y_true = np.array(labels)
    y_prob = np.array(probs)
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "y_pred": y_pred,
        "y_true": y_true,
    }
    try:
        out["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        out["auc"] = float("nan")
    return out


@torch.no_grad()
def extract_embeddings(model, loader, max_batches=6):
    model.eval().to(DEVICE)
    feats_list, labels_list = [], []
    hook_out = {}
    children = list(model.named_children())
    target_name, target_module = children[-2]

    def hook_fn(module, inp, out):
        hook_out["feat"] = out

    handle = target_module.register_forward_hook(hook_fn)
    for i, (xb, yb) in enumerate(loader):
        if i >= max_batches:
            break
        model(xb.to(DEVICE))
        f = hook_out["feat"]
        if isinstance(f, tuple):
            f = f[0]
        if f.dim() > 2:
            f = f.flatten(1)
        feats_list.append(f.cpu().numpy())
        labels_list.extend(yb.numpy())
    handle.remove()
    return np.concatenate(feats_list, axis=0), np.array(labels_list)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.grads = None
        self.acts = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o.detach()))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0].detach()))

    def __call__(self, x: torch.Tensor, cls: Optional[int] = None):
        self.model.eval()
        x = x.to(DEVICE).requires_grad_(True)
        logits = self.model(x)
        cls = int(cls if cls is not None else logits.argmax(1).item())
        self.model.zero_grad()
        logits[0, cls].backward()
        w = self.grads.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), cls


def get_last_conv(model: nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for side in ax.spines.values():
        side.set_color("#2a3040")


def plot_training_curves(histories, save_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    palette = [AC, BL, GR, "#e87c7c", "#c890e8", "#8fd3ff", "#ffb86c", "#50fa7b"]
    for (label, h), color in zip(histories.items(), palette):
        ep = range(1, len(h["train_acc"]) + 1)
        axes[0].plot(ep, h["train_loss"], color=color, lw=1.5, linestyle="--", alpha=0.6)
        axes[0].plot(ep, h["val_loss"], color=color, lw=2, label=label)
        axes[1].plot(ep, h["train_acc"], color=color, lw=1.5, linestyle="--", alpha=0.6)
        axes[1].plot(ep, h["val_acc"], color=color, lw=2, label=label)
    for ax, title in zip(axes, ["Loss (val=solid)", "Accuracy (val=solid)"]):
        _style(ax)
        ax.set_title(title, color=FG)
        ax.set_xlabel("Epoch", color=FG)
        ax.legend(facecolor="#1a1f2e", labelcolor=FG, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_lr_schedule(histories, save_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    _style(ax)
    palette = [AC, BL, GR, "#e87c7c", "#c890e8", "#8fd3ff", "#ffb86c", "#50fa7b"]
    for (label, h), color in zip(histories.items(), palette):
        ax.plot(h["lr"], color=color, lw=2, label=label)
    ax.set_title("Learning Rate Schedules", color=FG)
    ax.legend(facecolor="#1a1f2e", labelcolor=FG, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "lr_schedules.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_tsne(model, loader, label, class_names, save_dir: Path):
    feats, y = extract_embeddings(model, loader, max_batches=6)
    if len(np.unique(y)) < 2 or len(feats) < 10:
        return
    perplexity = min(30, max(5, len(feats) // 10))
    z = TSNE(n_components=2, random_state=SEED, perplexity=perplexity).fit_transform(feats)
    fig, ax = plt.subplots(figsize=(9, 8), facecolor=BG)
    _style(ax)
    unique_classes = np.unique(y)
    cmap = plt.cm.get_cmap("tab20", len(unique_classes))
    for idx, cls in enumerate(unique_classes):
        mask = y == cls
        ax.scatter(z[mask, 0], z[mask, 1], c=[cmap(idx)], s=10, alpha=0.65, label=class_names[int(cls)])
    ax.set_title(f"t-SNE — {label}", color=FG)
    ax.legend(facecolor="#1a1f2e", labelcolor=FG, fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(save_dir / f"tsne_{label.replace(' ', '_')}.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_confusion(y_true, y_pred, label, class_names, save_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12), facecolor=BG)
    ax.set_facecolor(BG)
    sns.heatmap(cm, cmap="YlOrBr", xticklabels=class_names, yticklabels=class_names, ax=ax, cbar_kws={"shrink": 0.6})
    ax.tick_params(colors=FG, labelsize=7)
    ax.set_xlabel("Predicted", color=FG)
    ax.set_ylabel("True", color=FG)
    ax.set_title(f"Confusion Matrix — {label}", color=FG)
    fig.tight_layout()
    fig.savefig(save_dir / f"cm_{label.replace(' ', '_')}.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_gradcam_grid(models_dict, loader, class_names, save_dir: Path, n=5):
    imgs, labels = next(iter(loader))
    imgs = imgs[:n]
    labels = labels[:n]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    model_names = list(models_dict.keys())
    fig, axes = plt.subplots(len(model_names) + 1, n, figsize=(2.4 * n, 2.6 * (len(model_names) + 1)), facecolor=BG)
    if len(model_names) == 0:
        return
    for j in range(n):
        img = (imgs[j].permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        axes[0, j].imshow(img)
        axes[0, j].axis("off")
        axes[0, j].set_title(class_names[int(labels[j])], color=AC, fontsize=7)
    axes[0, 0].set_ylabel("Original", color=FG, fontsize=8)

    for i, name in enumerate(model_names, start=1):
        model = models_dict[name]
        target_layer = get_last_conv(model)
        gcam = GradCAM(model, target_layer)
        for j in range(n):
            img_np = (imgs[j].permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
            cam, pred = gcam(imgs[j:j+1])
            axes[i, j].imshow(img_np)
            axes[i, j].imshow(cam, cmap="jet", alpha=0.45)
            axes[i, j].axis("off")
            axes[i, j].set_title(class_names[int(pred)], color=BL, fontsize=7)
        axes[i, 0].set_ylabel(name, color=FG, fontsize=7)
    fig.tight_layout()
    fig.savefig(save_dir / "gradcam_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_weight_histogram(models_dict, save_dir: Path):
    fig, axes = plt.subplots(1, len(models_dict), figsize=(5 * len(models_dict), 4), facecolor=BG)
    if len(models_dict) == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, models_dict.items()):
        all_w = np.concatenate([p.data.detach().cpu().flatten().numpy() for p in model.parameters()])
        _style(ax)
        ax.hist(all_w, bins=120, color=AC, edgecolor="none", alpha=0.8)
        ax.set_title(name, color=FG, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_dir / "weight_histograms.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_benchmark(results, save_dir: Path):
    names = list(results.keys())
    metrics = ["acc", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1 Macro"]
    colors = [AC, BL, GR, "#e87c7c"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor=BG)
    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        vals = [results[n][metric] for n in names]
        bars = ax.barh(names, vals, color=color, edgecolor="#2a3040", height=0.5)
        ax.set_xlim(0, 1.05)
        _style(ax)
        ax.set_title(label, color=FG)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", color=FG, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "benchmark.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def save_final_outputs(results, best_key, save_dir: Path):
    rows = []
    for name, r in sorted(results.items(), key=lambda x: -x[1]["acc"]):
        rows.append({
            "Model": name,
            "Accuracy": float(r["acc"]),
            "Precision": float(r["precision"]),
            "Recall": float(r["recall"]),
            "F1": float(r["f1"]),
            "AUC": None if np.isnan(r["auc"]) else float(r["auc"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "final_results.csv", index=False)
    with open(save_dir / "final_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    best_row = next(r for r in rows if r["Model"] == best_key)
    with open(save_dir / "best_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_key}\n")
        for k, v in best_row.items():
            if k != "Model":
                f.write(f"{k}: {v}\n")


def main():
    tr_loader, va_loader, te_loader, class_names, num_classes = get_loaders(batch_size=BATCH_SIZE)
    histories = {}
    results = {}
    trained = {}

    strategy_cfg = [
        ("ResNet-18 Feature Extract", "resnet18", "feature_extract", 1e-3, False, "cosine"),
        ("ResNet-18 Finetune Top", "resnet18", "finetune_top", 5e-4, False, "cosine"),
        ("ResNet-18 Full Finetune", "resnet18", "full_finetune", 1e-4, True, "cosine"),
        ("ResNet-18 LayerLR Decay", "resnet18", "full_finetune", 3e-4, True, "onecycle"),
    ]

    backbone_cfg = [
        ("ResNet-50", "resnet50", 1e-4),
        ("EfficientNet-B0", "efficientnet_b0", 3e-4),
        ("MobileNetV3-S", "mobilenet_v3_small", 3e-4),
        ("DenseNet-121", "densenet121", 1e-4),
    ]

    for label, arch, strategy, lr, layerwise, sched in strategy_cfg:
        print(f"\nRunning: {label}")
        model = build_model(arch, strategy, num_classes=num_classes)
        model, h = train(model, tr_loader, va_loader, epochs=8, lr=lr, layerwise=layerwise, sched=sched, patience=3, label=label)
        r = evaluate(model, te_loader)
        print(f"  Test -> acc={r['acc']:.4f} precision={r['precision']:.4f} recall={r['recall']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f}" if not np.isnan(r['auc']) else f"  Test -> acc={r['acc']:.4f} precision={r['precision']:.4f} recall={r['recall']:.4f} f1={r['f1']:.4f}")
        histories[label] = h
        results[label] = r
        trained[label] = model

    for label, arch, lr in backbone_cfg:
        print(f"\nRunning: {label}")
        model = build_model(arch, "full_finetune", num_classes=num_classes)
        model, h = train(model, tr_loader, va_loader, epochs=8, lr=lr, layerwise=False, sched="cosine", patience=3, label=label)
        r = evaluate(model, te_loader)
        print(f"  Test -> acc={r['acc']:.4f} precision={r['precision']:.4f} recall={r['recall']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f}" if not np.isnan(r['auc']) else f"  Test -> acc={r['acc']:.4f} precision={r['precision']:.4f} recall={r['recall']:.4f} f1={r['f1']:.4f}")
        histories[label] = h
        results[label] = r
        trained[label] = model

    plot_training_curves(histories, SAVE_DIR)
    plot_lr_schedule(histories, SAVE_DIR)
    plot_benchmark(results, SAVE_DIR)
    subset_hist_models = {k: trained[k] for k in list(trained.keys())[:3]}
    plot_weight_histogram(subset_hist_models, SAVE_DIR)
    cam_keys = [k for k in ["ResNet-18 Full Finetune", "EfficientNet-B0", "MobileNetV3-S", "DenseNet-121"] if k in trained]
    plot_gradcam_grid({k: trained[k] for k in cam_keys}, te_loader, class_names, SAVE_DIR)

    best_key = max(results, key=lambda k: results[k]["acc"])
    plot_confusion(results[best_key]["y_true"], results[best_key]["y_pred"], best_key, class_names, SAVE_DIR)
    for key in [k for k in ["ResNet-18 Feature Extract", "ResNet-18 Full Finetune"] if k in trained]:
        plot_tsne(trained[key], te_loader, key, class_names, SAVE_DIR)

    save_final_outputs(results, best_key, SAVE_DIR)

    print("\n" + "=" * 75)
    print(f"{'Model':<32} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 75)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["acc"]):
        auc_str = f"{r['auc']:.4f}" if not np.isnan(r['auc']) else "N/A"
        print(f"{name:<32} {r['acc']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} {auc_str:>7}")
    print("=" * 75)
    print(f"\nBest model: {best_key}  acc={results[best_key]['acc']:.4f}")
    print(f"Outputs saved to: {SAVE_DIR.resolve()}")
    print("\nClassification Report:\n")
    print(classification_report(results[best_key]["y_true"], results[best_key]["y_pred"], target_names=class_names, digits=4, zero_division=0))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {(time.time() - t0)/60:.1f} min")
