from __future__ import annotations

import copy
import json
import math
import time
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
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import ImageFolder

SEED = 42
DATA_DIR = "/content/drive/MyDrive/dataset/cattle"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("tl_attention_cattle_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 12
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 4
NUM_WORKERS = 2

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def rgb_loader(path: str):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class TransformSubset(Dataset):
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = self.subset.dataset.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_loaders(batch_size: int = BATCH_SIZE):
    train_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    base_ds = ImageFolder(DATA_DIR, loader=rgb_loader)
    classes = base_ds.classes
    num_classes = len(classes)

    n = len(base_ds)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_subset, val_subset, test_subset = random_split(
        base_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_ds = TransformSubset(train_subset, train_tf)
    val_ds = TransformSubset(val_subset, eval_tf)
    test_ds = TransformSubset(test_subset, eval_tf)

    kw = dict(batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    print(f"Using device: {DEVICE}")
    print(f"Classes: {classes}")
    print(f"Number of classes: {num_classes}")
    print(f"Split sizes -> train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_loader, val_loader, test_loader, classes, num_classes


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        hidden = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out) * out
        out = self.sa(out) * out
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class ResNetCBAM(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18_cbam(num_classes: int):
    return ResNetCBAM(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def train_one_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, label="cbam"):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train_correct = train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += xb.size(0)

        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        scheduler.step()

        tr_loss = train_loss / train_total
        va_loss = val_loss / val_total
        tr_acc = train_correct / train_total
        va_acc = val_correct / val_total
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"[{label}] Epoch {epoch:02d}/{epochs} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate_model(model, loader, classes, label="cbam"):
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

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0))
    return {
        "model": label,
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_training(history, save_dir: Path, label: str):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{label} Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f'{label}_loss_curve.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{label} Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f'{label}_accuracy_curve.png', dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred, classes, save_dir: Path, label: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{label} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_dir / f'{label}_confusion_matrix.png', dpi=150)
    plt.close()


def save_results(result: dict, save_dir: Path):
    df = pd.DataFrame([{k: v for k, v in result.items() if k not in ['y_true', 'y_pred']}])
    df.to_csv(save_dir / 'final_results.csv', index=False)

    with open(save_dir / 'final_results.json', 'w') as f:
        json.dump({k: v for k, v in result.items() if k not in ['y_true', 'y_pred']}, f, indent=2)

    with open(save_dir / 'best_model_summary.txt', 'w') as f:
        f.write(f"Model: {result['model']}\n")
        f.write(f"Accuracy: {result['accuracy']:.4f}\n")
        f.write(f"Precision (macro): {result['precision_macro']:.4f}\n")
        f.write(f"Recall (macro): {result['recall_macro']:.4f}\n")
        f.write(f"F1 (macro): {result['f1_macro']:.4f}\n")

    plt.figure(figsize=(7, 5))
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    values = [result[m] for m in metrics]
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.title('Final Metrics')
    plt.tight_layout()
    plt.savefig(save_dir / 'final_metrics_graph.png', dpi=150)
    plt.close()


def main():
    start = time.time()
    train_loader, val_loader, test_loader, classes, num_classes = get_loaders()
    model = resnet18_cbam(num_classes=num_classes)
    model, history = train_one_model(model, train_loader, val_loader, label='resnet18_cbam')
    result = evaluate_model(model, test_loader, classes, label='resnet18_cbam')
    plot_training(history, SAVE_DIR, 'resnet18_cbam')
    plot_confusion(result['y_true'], result['y_pred'], classes, SAVE_DIR, 'resnet18_cbam')
    save_results(result, SAVE_DIR)
    print(f"\nOutputs saved to: {SAVE_DIR.resolve()}")
    print(f"Total time: {(time.time() - start)/60:.1f} min")


if __name__ == '__main__':
    main()
