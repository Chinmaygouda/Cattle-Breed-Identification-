"""
================================================================
  Ensemble Learning + Computer Vision
  ---------------------------------------------------------------
  Combines multiple CV models (CNN, HOG+SVM, ResNet feature
  extractor) via Bagging, Boosting, and Stacking ensembles.

  Techniques covered
  ------------------
  1. Feature extraction  – HOG, CNN (simple), ResNet transfer
  2. Base learners       – SVM, Random Forest, Gradient Boost
  3. Ensemble methods    – Voting, Bagging, AdaBoost, Stacking
  4. Evaluation          – accuracy, confusion matrix, ROC-AUC

  Dependencies
  ------------
  pip install numpy scikit-learn torch torchvision opencv-python
              matplotlib seaborn tqdm
================================================================
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from sklearn.datasets import fetch_openml
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
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# 1.  DATASET  (MNIST-784 via sklearn – works offline)
# ──────────────────────────────────────────────────────────────

def load_mnist(n_samples: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Load a subset of MNIST; returns (X_images, y) where
    X_images.shape = (N, 28, 28) and y.shape = (N,)."""
    print("⏳  Fetching MNIST …")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0          # (70000, 784)
    y = mnist.target.astype(int)

    idx = np.random.choice(len(X), n_samples, replace=False)
    X, y = X[idx], y[idx]

    X_images = X.reshape(-1, 28, 28)                   # (N, 28, 28)
    print(f"✅  Loaded {n_samples} samples, classes: {np.unique(y)}")
    return X_images, y


# ──────────────────────────────────────────────────────────────
# 2.  FEATURE EXTRACTION STRATEGIES
# ──────────────────────────────────────────────────────────────

# 2-a. Flatten (baseline)
def extract_flat(images: np.ndarray) -> np.ndarray:
    return images.reshape(len(images), -1)


# 2-b. HOG features via OpenCV
def extract_hog(images: np.ndarray) -> np.ndarray:
    """Return HOG feature vectors for a batch of 28×28 images."""
    win = (28, 28)
    hog = cv2.HOGDescriptor(
        win,          # winSize
        (14, 14),     # blockSize
        (7, 7),       # blockStride
        (14, 14),     # cellSize
        9,            # nbins
    )
    feats = []
    for img in images:
        img_u8 = (img * 255).astype(np.uint8)
        feats.append(hog.compute(img_u8).flatten())
    return np.array(feats, dtype=np.float32)


# 2-c. Simple custom CNN as feature extractor (PyTorch)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14×14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  #  7×7
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
        )

    def forward(self, x):
        return self.head(self.body(x))


def train_cnn_extractor(
    X_train: np.ndarray, y_train: np.ndarray, epochs: int = 5
) -> SimpleCNN:
    """Quick supervised pre-training so CNN features are meaningful."""
    model = SimpleCNN().to(DEVICE)
    classifier_head = nn.Linear(128, 10).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier_head.parameters()), lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X_train[:, None, :, :]).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)

    model.train(); classifier_head.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            feats = model(xb)
            loss = criterion(classifier_head(feats), yb)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"   CNN epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")

    model.eval()
    return model


def extract_cnn(model: SimpleCNN, images: np.ndarray) -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(images[:, None, :, :]).to(DEVICE)
    feats = []
    with torch.no_grad():
        for i in range(0, len(X_t), 256):
            feats.append(model(X_t[i : i + 256]).cpu().numpy())
    return np.concatenate(feats, axis=0)


# 2-d. Transfer learning – ResNet-18 penultimate layer
def build_resnet_extractor() -> nn.Module:
    resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()          # strip classifier; output dim = 512
    resnet.eval().to(DEVICE)
    return resnet


def extract_resnet(model: nn.Module, images: np.ndarray) -> np.ndarray:
    """Resize 28×28 grey → 224×224 RGB required by ResNet."""
    tf = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
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


# ──────────────────────────────────────────────────────────────
# 3.  BASE LEARNERS
# ──────────────────────────────────────────────────────────────

def make_svm() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=50, random_state=SEED)),
        ("clf", SVC(kernel="rbf", C=10, probability=True, random_state=SEED)),
    ])


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=SEED
    )


def make_gb() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4, random_state=SEED
    )


def make_lr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=SEED)),
    ])


# ──────────────────────────────────────────────────────────────
# 4.  ENSEMBLE METHODS
# ──────────────────────────────────────────────────────────────

def build_ensembles(X_train, y_train) -> dict[str, object]:
    """Construct and fit each ensemble type."""
    ensembles: dict[str, object] = {}

    # 4-a. Hard + Soft Voting
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

    # 4-b. Bagging
    print("[2/4] Training Bagging Ensemble …")
    bagging = BaggingClassifier(
        estimator=make_lr(),
        n_estimators=20, max_samples=0.8, max_features=0.8,
        n_jobs=-1, random_state=SEED,
    )
    bagging.fit(X_train, y_train)
    ensembles["Bagging (LR)"] = bagging

    # 4-c. AdaBoost
    print("[3/4] Training AdaBoost …")
    ada = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=SEED, algorithm="SAMME")
    ada.fit(X_train, y_train)
    ensembles["AdaBoost"] = ada

    # 4-d. Stacking
    print("[4/4] Training Stacking Ensemble …")
    stacking = StackingClassifier(
        estimators=[("svm", make_svm()), ("rf", make_rf()), ("lr", make_lr())],
        final_estimator=LogisticRegression(max_iter=500),
        cv=3, n_jobs=-1,
    )
    stacking.fit(X_train, y_train)
    ensembles["Stacking"] = stacking

    return ensembles


# ──────────────────────────────────────────────────────────────
# 5.  EVALUATION HELPERS
# ──────────────────────────────────────────────────────────────

def evaluate(name: str, model, X_test, y_test) -> dict:
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inf_ms = (time.perf_counter() - t0) * 1000

    acc = accuracy_score(y_test, y_pred)

    # ROC-AUC (one-vs-rest)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    else:
        auc = float("nan")

    print(f"  {name:<22} acc={acc:.4f}  auc={auc:.4f}  inf={inf_ms:.1f}ms")
    return {"name": name, "acc": acc, "auc": auc, "inf_ms": inf_ms, "y_pred": y_pred}


# ──────────────────────────────────────────────────────────────
# 6.  VISUALISATIONS
# ──────────────────────────────────────────────────────────────

PALETTE = "#0a0a0f"   # bg
FG      = "#e8e0d4"   # text
ACCENT  = "#c8a96e"   # gold accent

def plot_results(results: list[dict], X_test, y_test, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Bar chart: accuracy comparison ---
    names = [r["name"] for r in results]
    accs  = [r["acc"] for r in results]
    aucs  = [r["auc"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             facecolor=PALETTE)
    for ax in axes:
        ax.set_facecolor(PALETTE)
        ax.tick_params(colors=FG)
        ax.spines[:].set_color("#333344")

    # Accuracy
    bars = axes[0].barh(names, accs, color=ACCENT, edgecolor="#333344", height=0.55)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Accuracy", color=FG, fontsize=11)
    axes[0].set_title("Ensemble Accuracy Comparison", color=FG, fontsize=13, pad=12)
    for bar, v in zip(bars, accs):
        axes[0].text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{v:.3f}", va="center", color=FG, fontsize=9)

    # ROC-AUC
    bars2 = axes[1].barh(names, aucs, color="#6e9cc8", edgecolor="#333344", height=0.55)
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("ROC-AUC (macro OvR)", color=FG, fontsize=11)
    axes[1].set_title("Ensemble ROC-AUC Comparison", color=FG, fontsize=13, pad=12)
    for bar, v in zip(bars2, aucs):
        lbl = f"{v:.3f}" if not np.isnan(v) else "N/A"
        axes[1].text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                     lbl, va="center", color=FG, fontsize=9)

    fig.tight_layout(pad=2)
    path = save_dir / "comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE)
    print(f"  📊 Saved: {path}")
    plt.close(fig)

    # --- Confusion matrix for best model ---
    best = max(results, key=lambda r: r["acc"])
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=PALETTE)
    ax.set_facecolor(PALETTE)
    cm_disp = ConfusionMatrixDisplay.from_predictions(
        y_test, best["y_pred"],
        ax=ax,
        colorbar=False,
        cmap="YlOrBr",
    )
    ax.set_title(f"Confusion Matrix – {best['name']}", color=FG, fontsize=13, pad=12)
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    path_cm = save_dir / "confusion_matrix.png"
    fig.savefig(path_cm, dpi=150, bbox_inches="tight", facecolor=PALETTE)
    print(f"  📊 Saved: {path_cm}")
    plt.close(fig)

    # --- Feature importance from RF ---
    rf_result = next((r for r in results if "Stacking" in r["name"]), None)
    # Instead, extract RF importance directly
    return


def plot_feature_spaces(X_flat, X_hog, y, save_dir: Path):
    """PCA 2-D scatter: flat vs HOG."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=PALETTE)
    cmap = plt.cm.get_cmap("tab10", 10)

    for ax, X, title in zip(axes,
                             [X_flat, X_hog],
                             ["Flat pixels (PCA 2D)", "HOG features (PCA 2D)"]):
        pca = PCA(n_components=2, random_state=SEED)
        Z = pca.fit_transform(X)
        ax.set_facecolor(PALETTE)
        ax.tick_params(colors=FG)
        ax.spines[:].set_color("#333344")
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10",
                        s=8, alpha=0.6, linewidths=0)
        ax.set_title(title, color=FG, fontsize=12, pad=8)
        plt.colorbar(sc, ax=ax).ax.tick_params(colors=FG)

    fig.suptitle("Feature Space Visualisation", color=FG, fontsize=14, y=1.01)
    fig.tight_layout()
    path = save_dir / "feature_spaces.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE)
    print(f"  📊 Saved: {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 7.  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────

def main():
    SAVE_DIR = Path("ensemble_cv_outputs")

    # ── Load data ──────────────────────────────────────────────
    X_images, y = load_mnist(n_samples=3000)        # keep small for demo
    X_tr_img, X_te_img, y_train, y_test = train_test_split(
        X_images, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # ── Feature extraction ─────────────────────────────────────
    print("\n── Feature Extraction ─────────────────────────────────")
    X_flat_tr  = extract_flat(X_tr_img)
    X_flat_te  = extract_flat(X_te_img)
    print("  [flat]   train:", X_flat_tr.shape, " test:", X_flat_te.shape)

    X_hog_tr   = extract_hog(X_tr_img)
    X_hog_te   = extract_hog(X_te_img)
    print("  [HOG]    train:", X_hog_tr.shape,  " test:", X_hog_te.shape)

    print("  [CNN]    training CNN extractor …")
    cnn = train_cnn_extractor(X_tr_img, y_train, epochs=5)
    X_cnn_tr   = extract_cnn(cnn, X_tr_img)
    X_cnn_te   = extract_cnn(cnn, X_te_img)
    print("  [CNN]    train:", X_cnn_tr.shape,  " test:", X_cnn_te.shape)

    # ResNet features are slow on CPU; skip if no GPU detected
    if DEVICE.type == "cuda":
        print("  [ResNet] extracting with GPU …")
        resnet = build_resnet_extractor()
        X_rn_tr = extract_resnet(resnet, X_tr_img)
        X_rn_te = extract_resnet(resnet, X_te_img)
        print("  [ResNet] train:", X_rn_tr.shape, " test:", X_rn_te.shape)
        USE_RESNET = True
    else:
        print("  [ResNet] CPU detected – skipping (would be very slow).")
        USE_RESNET = False

    # Visualise feature spaces
    plot_feature_spaces(X_flat_tr, X_hog_tr, y_train, SAVE_DIR)

    # ── Choose feature set for ensembles (concatenate HOG + CNN) ─
    X_combo_tr = np.concatenate([X_hog_tr, X_cnn_tr], axis=1)
    X_combo_te = np.concatenate([X_hog_te, X_cnn_te], axis=1)
    print(f"\n  Combined feature dim: {X_combo_tr.shape[1]}")

    # ── Build & train ensembles ────────────────────────────────
    print("\n── Ensemble Training ──────────────────────────────────")
    ensembles = build_ensembles(X_combo_tr, y_train)

    # ── Also include single RF / SVM baselines for reference ───
    print("\n  Training baselines …")
    baselines: dict[str, object] = {
        "RF (baseline)":  make_rf(),
        "SVM (baseline)": make_svm(),
    }
    for name, clf in baselines.items():
        clf.fit(X_combo_tr, y_train)
    all_models = {**baselines, **ensembles}

    # ── Evaluate ───────────────────────────────────────────────
    print("\n── Evaluation ─────────────────────────────────────────")
    results = [evaluate(name, clf, X_combo_te, y_test) for name, clf in all_models.items()]

    # ── Detailed report for best model ─────────────────────────
    best = max(results, key=lambda r: r["acc"])
    print(f"\n── Best model: {best['name']}  acc={best['acc']:.4f} ──")
    print(classification_report(y_test, best["y_pred"], digits=4))

    # ── Cross-validation on stacking ───────────────────────────
    print("\n── 3-Fold CV on Stacking ensemble ─────────────────────")
    cv_scores = cross_val_score(
        ensembles["Stacking"], X_combo_tr, y_train, cv=3, scoring="accuracy", n_jobs=-1
    )
    print(f"  CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Plots ──────────────────────────────────────────────────
    print("\n── Saving plots ───────────────────────────────────────")
    plot_results(results, X_combo_te, y_test, SAVE_DIR)

    print(f"\n✅  All done. Outputs saved to: {SAVE_DIR.resolve()}")


if __name__ == "__main__":
    main()
