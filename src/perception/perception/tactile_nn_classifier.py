"""
Texture classifier using deep CNN-style features + PCA.

Two feature backends are supported and selected automatically:
  1. ResNet-18 (torch + torchvision required) — preferred
  2. HOG + LBP + Gabor (OpenCV / numpy only) — fallback, runs without torch

For the PCA visualization / training-data exploration run:
    python3 tactile_nn_classifier.py <base_dir> [out.png]

To install torch (CPU-only):
    sudo apt install python3-pip
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Texture class registry — keep in sync with tactile_classification_utils.py
# ---------------------------------------------------------------------------

TEXTURE_CLASSES = [
    (1, 'texture1_dense'),
    (3, 'texture3_square'),
]
IMG_SIZE = (224, 224)  # resize target (width, height) for both backends

# ---------------------------------------------------------------------------
# Feature extraction backends
# ---------------------------------------------------------------------------

try:
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as tvt
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _build_resnet_extractor():
    """Return (model_fn, transform_fn) using ResNet-18 avgpool features (512-d)."""
    model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    # Drop the final FC layer — avgpool gives (1, 512, 1, 1) output
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    transform = tvt.Compose([
        tvt.ToPILImage(),
        tvt.Resize(IMG_SIZE),
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract(bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0)
        with torch.no_grad():
            feat = model(tensor).squeeze().numpy()   # (512,)
        return feat.astype(np.float32)

    return extract, 'ResNet-18 avgpool (512-d)'


def _gabor_responses(gray: np.ndarray) -> np.ndarray:
    """Mean and variance of responses to a bank of Gabor filters."""
    responses = []
    for theta in np.linspace(0, np.pi, 6, endpoint=False):
        for freq in (0.1, 0.25, 0.4):
            kern = cv2.getGaborKernel((21, 21), 4.0, theta, 1.0 / freq, 0.5, 0)
            resp = cv2.filter2D(gray, cv2.CV_32F, kern)
            responses.extend([resp.mean(), resp.var()])
    return np.array(responses, dtype=np.float32)


def _lbp_histogram(gray: np.ndarray, radius: int = 2, n_points: int = 16) -> np.ndarray:
    """Uniform LBP histogram over the whole image."""
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.uint8)
    angles = [2 * np.pi * i / n_points for i in range(n_points)]
    offsets = [(int(round(radius * np.sin(a))), int(round(radius * np.cos(a)))) for a in angles]
    center = gray.astype(np.int16)
    code = np.zeros_like(gray, dtype=np.uint32)
    for bit, (dy, dx) in enumerate(offsets):
        shifted = np.roll(np.roll(gray.astype(np.int16), dy, axis=0), dx, axis=1)
        code |= ((shifted >= center).astype(np.uint32) << bit)
    hist, _ = np.histogram(code.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-6
    return hist


def _build_opencv_extractor():
    """Return (extract_fn, description) using HOG + LBP + Gabor (no torch needed)."""
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )

    def extract(bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(
            cv2.resize(bgr, IMG_SIZE), cv2.COLOR_BGR2GRAY
        )
        hog_small = cv2.resize(gray, (128, 128))
        hog_feat = hog.compute(hog_small).flatten()          # ~3780-d
        lbp_feat = _lbp_histogram(gray)                      # 256-d
        gabor_feat = _gabor_responses(gray)                  # 36-d
        feat = np.concatenate([hog_feat, lbp_feat, gabor_feat])
        return feat.astype(np.float32)

    dim = len(extract(np.zeros((240, 320, 3), dtype=np.uint8)))
    return extract, f'HOG+LBP+Gabor ({dim}-d, no-torch fallback)'


def _get_extractor():
    if _TORCH_AVAILABLE:
        return _build_resnet_extractor()
    return _build_opencv_extractor()


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------

class ClassFeatures(NamedTuple):
    class_id: int
    features: np.ndarray   # (n_samples, feature_dim)
    names: list[str]


def load_reference_features(base_dir: Path, extract_fn) -> list[ClassFeatures]:
    result = []
    for class_id, folder_name in TEXTURE_CLASSES:
        folder = base_dir / folder_name
        feats, names = [], []
        for path in sorted(folder.glob('*.png')):
            bgr = cv2.imread(str(path))
            if bgr is None:
                print(f'  [warn] could not read {path}')
                continue
            feats.append(extract_fn(bgr))
            names.append(path.stem)
        if feats:
            result.append(ClassFeatures(class_id, np.stack(feats), names))
            print(f'  class {class_id} ({folder_name}): {len(feats)} images, feat_dim={feats[0].shape[0]}')
        else:
            print(f'  [warn] no images found in {folder}')
    return result


# ---------------------------------------------------------------------------
# PCA helpers (numpy SVD — no sklearn needed)
# ---------------------------------------------------------------------------

class PCAModel(NamedTuple):
    mean: np.ndarray        # (feature_dim,)
    components: np.ndarray  # (n_components, feature_dim)
    explained_ratio: np.ndarray  # (n_components,)


def fit_pca(X: np.ndarray, n_components: int) -> tuple[PCAModel, np.ndarray]:
    """Fit PCA on X (n_samples, feature_dim). Returns (model, X_projected)."""
    mean = X.mean(axis=0)
    Xc = X - mean
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    total_var = (S ** 2).sum() + 1e-12
    explained = (S[:n_components] ** 2) / total_var
    model = PCAModel(mean, Vt[:n_components], explained)
    X_proj = Xc @ Vt[:n_components].T
    return model, X_proj


def project(bgr: np.ndarray, extract_fn, pca: PCAModel) -> np.ndarray:
    feat = extract_fn(bgr)
    return ((feat - pca.mean) @ pca.components.T)


# ---------------------------------------------------------------------------
# Classifier (nearest centroid in PCA space)
# ---------------------------------------------------------------------------

class NNClassifier(NamedTuple):
    extract_fn: object
    pca: PCAModel
    centroids: dict[int, np.ndarray]  # class_id → mean PCA coords
    n_components: int


def build_classifier(base_dir: Path, n_components: int = 16) -> NNClassifier:
    extract_fn, desc = _get_extractor()
    print(f'Feature backend: {desc}')
    class_features = load_reference_features(base_dir, extract_fn)
    X = np.vstack([cf.features for cf in class_features])
    pca, X_proj = fit_pca(X, n_components)
    idx = 0
    centroids = {}
    for cf in class_features:
        n = len(cf.features)
        centroids[cf.class_id] = X_proj[idx:idx + n].mean(axis=0)
        idx += n
    return NNClassifier(extract_fn, pca, centroids, n_components)


def classify_texture_nn(query_bgr: np.ndarray, clf: NNClassifier) -> int:
    """Return the class_id whose PCA centroid is closest to the query image."""
    coords = project(query_bgr, clf.extract_fn, clf.pca)
    best_class = min(clf.centroids, key=lambda cid: np.linalg.norm(coords - clf.centroids[cid]))
    return best_class


# ---------------------------------------------------------------------------
# PCA visualisation
# ---------------------------------------------------------------------------

def visualize_pca(class_features: list[ClassFeatures], out_path: Path | None = None) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    X = np.vstack([cf.features for cf in class_features])
    pca, X_proj = fit_pca(X, n_components=min(2, X.shape[1]))

    palette = plt.cm.tab10.colors
    color_map = {cf.class_id: palette[i] for i, cf in enumerate(class_features)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- 2-D scatter (PC1 vs PC2) ---
    ax = axes[0]
    idx = 0
    for cf in class_features:
        n = len(cf.features)
        xs = X_proj[idx:idx + n, 0]
        ys = X_proj[idx:idx + n, 1] if X_proj.shape[1] > 1 else np.zeros(n)
        ax.scatter(xs, ys, c=[color_map[cf.class_id]] * n, s=90,
                   edgecolors='k', linewidths=0.6, zorder=3)
        for j, name in enumerate(cf.names):
            ax.annotate(name, (xs[j], ys[j]), fontsize=7,
                        xytext=(4, 4), textcoords='offset points')
        # draw centroid
        ax.scatter(xs.mean(), ys.mean(), marker='*', s=250,
                   c=[color_map[cf.class_id]], edgecolors='k', linewidths=1.2, zorder=4)
        idx += n
    ax.set_xlabel(f'PC1 ({pca.explained_ratio[0]*100:.1f}% var)')
    if X_proj.shape[1] > 1:
        ax.set_xlabel(f'PC1 ({pca.explained_ratio[0]*100:.1f}% var)')
        ax.set_ylabel(f'PC2 ({pca.explained_ratio[1]*100:.1f}% var)')
    ax.set_title('PCA of reference images (2D projection)')
    patches = [mpatches.Patch(color=color_map[cf.class_id], label=f'class {cf.class_id}')
               for cf in class_features]
    ax.legend(handles=patches)
    ax.grid(True, alpha=0.3)

    # --- Explained variance curve ---
    ax2 = axes[1]
    _, S, _ = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    cum_var = np.cumsum(S ** 2) / (S ** 2).sum()
    ax2.plot(np.arange(1, len(cum_var) + 1), cum_var * 100, 'o-', ms=4)
    ax2.axhline(90, color='red', linestyle='--', alpha=0.6, label='90% threshold')
    ax2.axhline(95, color='orange', linestyle='--', alpha=0.6, label='95% threshold')
    ax2.set_xlabel('Number of components')
    ax2.set_ylabel('Cumulative explained variance (%)')
    ax2.set_title('Scree plot')
    ax2.set_xlim(1, min(30, len(cum_var)))
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Print inter-class separation metric
    if len(class_features) == 2:
        cf_a, cf_b = class_features
        n_a, n_b = len(cf_a.features), len(cf_b.features)
        proj_a = X_proj[:n_a]
        proj_b = X_proj[n_a:n_a + n_b]
        centroid_dist = np.linalg.norm(proj_a.mean(axis=0) - proj_b.mean(axis=0))
        spread = (np.std(proj_a, axis=0).mean() + np.std(proj_b, axis=0).mean()) / 2
        print(f'\nSeparability (2D PCA):')
        print(f'  centroid distance : {centroid_dist:.3f}')
        print(f'  mean within-class spread : {spread:.3f}')
        print(f'  separation ratio (dist/spread) : {centroid_dist / (spread + 1e-6):.2f}')
        print(f'  (ratio > 2 → clusters likely separable)')

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(str(out_path), dpi=150)
        print(f'\nSaved PCA plot → {out_path}')
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    out  = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f'Loading references from {base}')
    extract_fn, desc = _get_extractor()
    print(f'Feature backend: {desc}')

    class_features = load_reference_features(base, extract_fn)
    if not class_features:
        print('No reference images found — check the base_dir path.')
        sys.exit(1)

    visualize_pca(class_features, out_path=out)
