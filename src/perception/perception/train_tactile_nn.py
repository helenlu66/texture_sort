"""
Texture classifier using deep CNN-style features + PCA.

Two feature backends are supported and selected automatically:
  1. ResNet-18 (torch + torchvision required) — preferred
  2. HOG + LBP + Gabor (OpenCV / numpy only) — fallback, runs without torch

For the PCA visualization / training-data exploration run:
    python3 train_tactile_nn.py <base_dir> [out.png]

For training a small neural net on frozen ResNet features:
    python3 train_tactile_nn.py <base_dir> --train --model-out tactile_resnet_nn.pt

To install torch (CPU-only):
    sudo apt install python3-pip
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import NamedTuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Texture class registry — keep in sync with tactile_classification_utils.py
# ---------------------------------------------------------------------------

TEXTURE_CLASSES = [
    (1, 'texture1_dense'),
    (2, 'texture2_smooth'),
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


def _rotate_image(bgr: np.ndarray, degrees: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)
    return cv2.warpAffine(
        bgr,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _vignette_edge_blur(bgr: np.ndarray, blur_kernel: int = 31) -> np.ndarray:
    h, w = bgr.shape[:2]
    blurred = cv2.GaussianBlur(bgr, (blur_kernel, blur_kernel), 0)
    yy, xx = np.mgrid[0:h, 0:w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    radius = np.sqrt(((xx - cx) / max(cx, 1.0)) ** 2 + ((yy - cy) / max(cy, 1.0)) ** 2)
    alpha = np.clip((radius - 0.45) / 0.45, 0.0, 1.0).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 5.0)[..., None]
    mixed = bgr.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _one_sided_contact_blur(bgr: np.ndarray, side: str, blur_kernel: int = 41) -> np.ndarray:
    h, w = bgr.shape[:2]
    blurred = cv2.GaussianBlur(bgr, (blur_kernel, blur_kernel), 0)
    if side in {'left', 'right'}:
        ramp = np.linspace(1.0, 0.0, w, dtype=np.float32)
        if side == 'right':
            ramp = ramp[::-1]
        alpha = np.tile(ramp, (h, 1))
    else:
        ramp = np.linspace(1.0, 0.0, h, dtype=np.float32)
        if side == 'bottom':
            ramp = ramp[::-1]
        alpha = np.tile(ramp[:, None], (1, w))
    alpha = np.clip((alpha - 0.25) / 0.75, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 7.0)[..., None]
    mixed = bgr.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha
    return np.clip(mixed, 0, 255).astype(np.uint8)


def augment_image(bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Deterministic tactile augmentations for pose/contact variation."""
    augmented = [('orig', bgr)]
    for degrees in (-20.0, -10.0, 10.0, 20.0, 90.0, 180.0, 270.0):
        augmented.append((f'rot_{degrees:g}', _rotate_image(bgr, degrees)))
    augmented.append(('vignette_edge_blur', _vignette_edge_blur(bgr)))
    for side in ('left', 'right', 'top', 'bottom'):
        augmented.append((f'contact_blur_{side}', _one_sided_contact_blur(bgr, side)))
    return augmented


def load_reference_features(
    base_dir: Path,
    extract_fn,
    augment: bool = False,
) -> list[ClassFeatures]:
    result = []
    for class_id, folder_name in TEXTURE_CLASSES:
        folder = base_dir / folder_name
        feats, names = [], []
        for path in sorted(folder.glob('*.png')):
            bgr = cv2.imread(str(path))
            if bgr is None:
                print(f'  [warn] could not read {path}')
                continue
            images = augment_image(bgr) if augment else [('orig', bgr)]
            for suffix, image in images:
                feats.append(extract_fn(image))
                names.append(path.stem if suffix == 'orig' else f'{path.stem}__{suffix}')
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
# Neural net trained on frozen ResNet features
# ---------------------------------------------------------------------------

def _stack_features(class_features: list[ClassFeatures]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_parts = []
    y_parts = []
    names = []
    for cf in class_features:
        X_parts.append(cf.features)
        y_parts.append(np.full(len(cf.features), cf.class_id, dtype=np.int64))
        names.extend(cf.names)
    return np.vstack(X_parts), np.concatenate(y_parts), names


def _base_sample_name(name: str) -> str:
    return name.split('__', 1)[0]


def _stratified_split(
    y_class_ids: np.ndarray,
    names: list[str],
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    train_indices = []
    val_indices = []
    for class_id in sorted(set(int(v) for v in y_class_ids)):
        base_names = sorted({
            _base_sample_name(names[i])
            for i, v in enumerate(y_class_ids)
            if int(v) == class_id
        })
        rng.shuffle(base_names)
        n_val = max(1, int(round(len(base_names) * val_fraction)))
        n_val = min(n_val, max(1, len(base_names) - 1))
        val_base_names = set(base_names[:n_val])
        for i, v in enumerate(y_class_ids):
            if int(v) != class_id:
                continue
            is_original = '__' not in names[i]
            is_val_base = _base_sample_name(names[i]) in val_base_names
            if is_val_base and is_original:
                val_indices.append(i)
            elif not is_val_base:
                train_indices.append(i)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.array(train_indices, dtype=np.int64), np.array(val_indices, dtype=np.int64)


class ResNetFeatureMLP(torch.nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_resnet_feature_nn(
    base_dir: Path,
    model_out: Path,
    epochs: int = 500,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    dropout: float = 0.15,
    val_fraction: float = 0.25,
    seed: int = 7,
    augment: bool = True,
) -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError('torch and torchvision are required for neural-net training.')

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    extract_fn, desc = _build_resnet_extractor()
    print(f'Feature backend: {desc}')
    class_features = load_reference_features(base_dir, extract_fn, augment=augment)
    if len(class_features) < 2:
        raise RuntimeError('Need at least two texture classes with reference images.')

    X, y_class_ids, names = _stack_features(class_features)
    class_ids = sorted(int(class_id) for class_id in set(y_class_ids))
    class_to_index = {class_id: i for i, class_id in enumerate(class_ids)}
    index_to_class = {i: class_id for class_id, i in class_to_index.items()}
    y = np.array([class_to_index[int(class_id)] for class_id in y_class_ids], dtype=np.int64)

    train_idx, val_idx = _stratified_split(
        y_class_ids=y_class_ids,
        names=names,
        val_fraction=val_fraction,
        seed=seed,
    )
    print(f'Training samples: {len(train_idx)}; validation originals: {len(val_idx)}')
    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.long)
    val_x = torch.tensor(X_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.long)

    model = ResNetFeatureMLP(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        num_classes=len(class_ids),
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = -1.0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_acc = float((model(train_x).argmax(dim=1) == train_y).float().mean())
            val_logits = model(val_x)
            val_loss = float(loss_fn(val_logits, val_y))
            val_pred = val_logits.argmax(dim=1)
            val_acc = float((val_pred == val_y).float().mean())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(
                f'epoch {epoch:04d}: '
                f'loss={float(loss.detach()):.4f}, train_acc={train_acc:.3f}, '
                f'val_loss={val_loss:.4f}, val_acc={val_acc:.3f}'
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_pred = model(val_x).argmax(dim=1).cpu().numpy()

    print('\nValidation predictions:')
    for idx, pred_index in zip(val_idx, val_pred):
        truth_class = int(y_class_ids[idx])
        pred_class = index_to_class[int(pred_index)]
        print(f'  {names[idx]}: true={truth_class}, pred={pred_class}')

    artifact = {
        'model_state_dict': model.state_dict(),
        'input_dim': int(X.shape[1]),
        'hidden_dim': int(hidden_dim),
        'num_classes': len(class_ids),
        'dropout': float(dropout),
        'class_ids': class_ids,
        'class_to_index': class_to_index,
        'index_to_class': index_to_class,
        'feature_mean': mean.astype(np.float32),
        'feature_std': std.astype(np.float32),
        'texture_classes': TEXTURE_CLASSES,
        'img_size': IMG_SIZE,
        'augment': bool(augment),
        'backend': desc,
        'best_val_acc': float(best_val_acc),
        'best_epoch': int(best_epoch),
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, model_out)

    metrics_path = model_out.with_suffix('.json')
    metrics = {
        'model_out': str(model_out),
        'backend': desc,
        'samples': int(len(y)),
        'train_samples': int(len(train_idx)),
        'val_samples': int(len(val_idx)),
        'augment': bool(augment),
        'class_ids': class_ids,
        'best_val_acc': float(best_val_acc),
        'best_epoch': int(best_epoch),
        'val_predictions': [
            {
                'name': names[int(idx)],
                'true_class': int(y_class_ids[int(idx)]),
                'predicted_class': int(index_to_class[int(pred)]),
            }
            for idx, pred in zip(val_idx, val_pred)
        ],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f'\nSaved neural net model -> {model_out}')
    print(f'Saved metrics -> {metrics_path}')
    print(f'Best validation accuracy: {best_val_acc:.3f} at epoch {best_epoch}')


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
    parser = argparse.ArgumentParser(description='Train/visualize tactile texture classifiers.')
    parser.add_argument('base_dir', nargs='?', type=Path, default=Path(__file__).parent)
    parser.add_argument('out', nargs='?', type=Path, default=None, help='PCA plot path when not training.')
    parser.add_argument('--train', action='store_true', help='Train an MLP on frozen ResNet-18 features.')
    parser.add_argument('--model-out', type=Path, default=Path(__file__).with_name('tactile_resnet_nn.pt'))
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--val-fraction', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--no-augment', action='store_true', help='Train only on original images.')
    args = parser.parse_args()

    base = args.base_dir
    if args.train:
        train_resnet_feature_nn(
            base,
            args.model_out,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            val_fraction=args.val_fraction,
            seed=args.seed,
            augment=not args.no_augment,
        )
        raise SystemExit(0)

    print(f'Loading references from {base}')
    extract_fn, desc = _get_extractor()
    print(f'Feature backend: {desc}')

    class_features = load_reference_features(base, extract_fn)
    if not class_features:
        print('No reference images found — check the base_dir path.')
        sys.exit(1)

    visualize_pca(class_features, out_path=args.out)
