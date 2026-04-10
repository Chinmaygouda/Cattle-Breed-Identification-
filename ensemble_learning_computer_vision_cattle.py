from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.svm import SVC
from torchvision.datasets import ImageFolder
from tqdm import tqdm

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/content/drive/MyDrive/dataset/cattle"
SAVE_DIR = Path("ensemble_cv_cattle_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
IMG_SIZE = 128
MAX_SAMPLES = None  # set integer like 5000 if you want a faster demo


# -------------------------------------------------------------
# 1. DATASET
# -------------------------------------------------------------

def pil_loader_rgb(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def load_cattle_dataset(data_dir: str = DATA_DIR, img_size: int = IMG_SIZE, max_samples: int | None = MAX_SAMPLES):
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])

    ds = ImageFolder(root=data_dir, loader=pil_loader_rgb, transform=tf)
    class_names = ds.classes

    images = []
    labels = []
    for i in tqdm(range(len(ds)), desc="Loading cattle images"):
        x, y = ds[i]
        images.append(x.permute(1, 2, 0).numpy())
        labels.append(y)

    X = np.stack(images).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    if max_samples is not None and max_samples < len(X):
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    print(f"Using device: {DEVICE}")
    print(f"Loaded {len(X)} images")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    return X, y, class_names


# -------------------------------------------------------------
# 2. FEATURE EXTRACTION
# -------------------------------------------------------------

def extract_flat(images: np.ndarray) -> np.ndarray:
    return images.reshape(len(images), -1)


def extract_hog(images: np.ndarray) -> np.ndarray:
    """HOG over grayscale resized images."""
    hog = cv2.HOGDescriptor(
        (64, 64),   # winSize
        (16, 16),   # blockSize
        (8, 8),     # blockStride
        (8, 8),     # cellSize
        9,          # nbins
    )
    feats = []
    for img in tqdm(images, desc="HOG features", leave=False):
        img_u8 = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (64, 64))
        feats.append(hog.compute(gray).flatten())
    return np.array(feats, dtype=np.float32)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        z = self.embedding(self.features(x))
        return self.classifier(z)

    def extract_features(self, x):
        return self.embedding(self.features(x))


def train_cnn_extractor(X_train: np.ndarray, y_train: np.ndarray, num_classes: int, epochs: int = 5) -> SimpleCNN:
    model = SimpleCNN(num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X_train).permute(0, 3, 1, 2).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
        print(f"   CNN epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}  acc={correct/total:.4f}")

    model.eval()
    return model


def extract_cnn(model: SimpleCNN, images: np.ndarray) -> np.ndarray:
    X_t = torch.from_numpy(images).permute(0, 3, 1, 2).to(DEVICE)
    feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_t), 64):
            feats.append(model.extract_features(X_t[i:i+64]).cpu().numpy())
    return np.concatenate(feats, axis=0)


def build_resnet_extractor() -> nn.Module:
    resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet.eval().to(DEVICE)
    return resnet


def extract_resnet(model: nn.Module, images: np.ndarray) -> np.ndarray:
    tf = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    feats = []
    model.eval()
    with torch.no_grad():
        for img in tqdm(images, desc="ResNet features", leave=False):
            img_u8 = (img * 255).astype(np.uint8)
            tensor = tf(img_u8).unsqueeze(0).to(DEVICE)
            feats.append(model(tensor).squeeze().cpu().numpy())
    return np.array(feats, dtype=np.float32)


# -------------------------------------------------------------
# 3. BASE LEARNERS
# -------------------------------------------------------------

def make_svm() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=50, random_state=SEED)),
        ("clf", SVC(kernel="rbf", C=10, probability=True, random_state=SEED)),
    ])


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)


def make_gb() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=SEED)


def make_lr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=SEED)),
    ])


# -------------------------------------------------------------
# 4. ENSEMBLE METHODS
# -------------------------------------------------------------

def build_ensembles(X_train, y_train) -> dict[str, object]:
    ensembles: dict[str, object] = {}

    print("\n[1/4] Training Voting Ensemble …")
    voting_hard = VotingClassifier(
        estimators=[("svm", make_svm()), ("rf", make_rf()), ("lr", make_lr())],
        voting="hard", n_jobs=-1,
    )
    voting_soft = VotingClassifier(
        estimators=[("svm", make_svm()), ("rf", make_rf()), ("lr", make_lr())],
        voting="soft", n_jobs=-1,
    )
    voting_hard.fit(X_train, y_train)
    voting_soft.fit(X_train, y_train)
    ensembles["Voting (hard)"] = voting_hard
    ensembles["Voting (soft)"] = voting_soft

    print("[2/4] Training Bagging Ensemble …")
    bagging = BaggingClassifier(
        estimator=make_lr(),
        n_estimators=20, max_samples=0.8, max_features=0.8,
        n_jobs=-1, random_state=SEED,
    )
    bagging.fit(X_train, y_train)
    ensembles["Bagging (LR)"] = bagging

    print("[3/4] Training AdaBoost …")
    ada = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=SEED, algorithm="SAMME")
    ada.fit(X_train, y_train)
    ensembles["AdaBoost"] = ada

    print("[4/4] Training Stacking Ensemble …")
    stacking = StackingClassifier(
        estimators=[("svm", make_svm()), ("rf", make_rf()), ("lr", make_lr())],
        final_estimator=LogisticRegression(max_iter=500),
        cv=3, n_jobs=-1,
    )
    stacking.fit(X_train, y_train)
    ensembles["Stacking"] = stacking

    return ensembles


# -------------------------------------------------------------
# 5. EVALUATION
# -------------------------------------------------------------

def evaluate(name: str, model, X_test, y_test, num_classes: int) -> dict:
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inf_ms = (time.perf_counter() - t0) * 1000

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    auc = float("nan")
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
            y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))
            auc = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
        except Exception:
            pass

    print(f"  {name:<22} acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}  auc={auc:.4f}  inf={inf_ms:.1f}ms")
    return {
        "name": name,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1_macro": f1,
        "auc": auc,
        "inf_ms": inf_ms,
        "y_pred": y_pred,
    }


# -------------------------------------------------------------
# 6. VISUALS + SAVES
# -------------------------------------------------------------

PALETTE = "#0a0a0f"
FG = "#e8e0d4"
ACCENT = "#c8a96e"


def plot_results(results: list[dict], y_test, class_names, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    names = [r["name"] for r in results]
    accs = [r["acc"] for r in results]
    precs = [r["precision"] for r in results]
    recs = [r["recall"] for r in results]
    f1s = [r["f1_macro"] for r in results]

    metrics = [accs, precs, recs, f1s]
    titles = ["Accuracy", "Precision", "Recall", "F1 Macro"]
    files = ["accuracy_comparison.png", "precision_comparison.png", "recall_comparison.png", "f1_comparison.png"]

    for vals, title, fname in zip(metrics, titles, files):
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE)
        ax.set_facecolor(PALETTE)
        bars = ax.barh(names, vals, color=ACCENT, edgecolor="#333344")
        ax.set_xlim(0, 1.05)
        ax.set_title(title, color=FG)
        ax.tick_params(colors=FG)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", color=FG)
        fig.tight_layout()
        fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight", facecolor=PALETTE)
        plt.close(fig)

    # combined graph
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=PALETTE)
    ax.set_facecolor(PALETTE)
    ax.bar(x - 1.5 * width, accs, width, label="Accuracy")
    ax.bar(x - 0.5 * width, precs, width, label="Precision")
    ax.bar(x + 0.5 * width, recs, width, label="Recall")
    ax.bar(x + 1.5 * width, f1s, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", color=FG)
    ax.tick_params(axis='y', colors=FG)
    ax.set_title("Ensemble Model Comparison", color=FG)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "final_comparison_all_metrics.png", dpi=150, bbox_inches="tight", facecolor=PALETTE)
    plt.close(fig)

    best = max(results, key=lambda r: r["acc"])
    fig, ax = plt.subplots(figsize=(18, 16), facecolor=PALETTE)
    ax.set_facecolor(PALETTE)
    ConfusionMatrixDisplay.from_predictions(y_test, best["y_pred"], display_labels=class_names, xticks_rotation=90, ax=ax, colorbar=False, cmap="YlOrBr")
    ax.set_title(f"Confusion Matrix - {best['name']}", color=FG)
    ax.tick_params(colors=FG)
    fig.tight_layout()
    fig.savefig(save_dir / "best_model_confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor=PALETTE)
    plt.close(fig)


# -------------------------------------------------------------
# 7. MAIN
# -------------------------------------------------------------

def main():
    X_images, y, class_names = load_cattle_dataset()
    num_classes = len(class_names)

    X_train_img, X_test_img, y_train, y_test = train_test_split(
        X_images, y, test_size=0.2, stratify=y, random_state=SEED
    )

    print("\n── Feature Extraction ─────────────────────────────────")
    X_hog_tr = extract_hog(X_train_img)
    X_hog_te = extract_hog(X_test_img)
    print("  [HOG] train:", X_hog_tr.shape, " test:", X_hog_te.shape)

    print("  [CNN] training CNN extractor …")
    cnn = train_cnn_extractor(X_train_img, y_train, num_classes=num_classes, epochs=5)
    X_cnn_tr = extract_cnn(cnn, X_train_img)
    X_cnn_te = extract_cnn(cnn, X_test_img)
    print("  [CNN] train:", X_cnn_tr.shape, " test:", X_cnn_te.shape)

    if DEVICE.type == "cuda":
        print("  [ResNet] extracting with GPU …")
        resnet = build_resnet_extractor()
        X_rn_tr = extract_resnet(resnet, X_train_img)
        X_rn_te = extract_resnet(resnet, X_test_img)
        print("  [ResNet] train:", X_rn_tr.shape, " test:", X_rn_te.shape)
        X_combo_tr = np.concatenate([X_hog_tr, X_cnn_tr, X_rn_tr], axis=1)
        X_combo_te = np.concatenate([X_hog_te, X_cnn_te, X_rn_te], axis=1)
    else:
        print("  [ResNet] CPU detected – skipping ResNet features.")
        X_combo_tr = np.concatenate([X_hog_tr, X_cnn_tr], axis=1)
        X_combo_te = np.concatenate([X_hog_te, X_cnn_te], axis=1)

    print(f"\nCombined feature dim: {X_combo_tr.shape[1]}")

    print("\n── Ensemble Training ──────────────────────────────────")
    ensembles = build_ensembles(X_combo_tr, y_train)

    print("\nTraining baselines …")
    baselines = {
        "RF (baseline)": make_rf(),
        "SVM (baseline)": make_svm(),
        "GB (baseline)": make_gb(),
    }
    for name, clf in baselines.items():
        clf.fit(X_combo_tr, y_train)

    all_models = {**baselines, **ensembles}

    print("\n── Evaluation ─────────────────────────────────────────")
    results = [evaluate(name, clf, X_combo_te, y_test, num_classes) for name, clf in all_models.items()]

    best = max(results, key=lambda r: r["acc"])
    print(f"\n── Best model: {best['name']}  acc={best['acc']:.4f} ──")
    print(classification_report(y_test, best["y_pred"], target_names=class_names, digits=4, zero_division=0))

    plot_results(results, y_test, class_names, SAVE_DIR)

    serializable_results = [
        {k: v for k, v in r.items() if k != "y_pred"}
        for r in results
    ]
    with open(SAVE_DIR / "final_results.json", "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)

    import pandas as pd
    pd.DataFrame(serializable_results).to_csv(SAVE_DIR / "final_results.csv", index=False)

    with open(SAVE_DIR / "best_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model: {best['name']}\n")
        f.write(f"Accuracy: {best['acc']:.4f}\n")
        f.write(f"Precision: {best['precision']:.4f}\n")
        f.write(f"Recall: {best['recall']:.4f}\n")
        f.write(f"F1 Macro: {best['f1_macro']:.4f}\n")
        f.write(f"AUC: {best['auc']:.4f}\n")

    print(f"\n✅ All done. Outputs saved to: {SAVE_DIR.resolve()}")


if __name__ == "__main__":
    main()
