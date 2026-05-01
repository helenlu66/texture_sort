from pathlib import Path

import cv2
import numpy as np


# (class_id, folder_name) pairs; extend to add more texture classes.
TEXTURE_CLASSES = [
    (1, 'texture1_dense'),
    (2, 'texture2_smooth'),
]

REF_SIZE = (320, 240)  # OpenCV resize order: (width, height)

# Primary decision thresholds — sit in the clean gap between class distributions:
#   class 1 (dense)  bv: [185.8, 227.1]   tc: [0.897, 0.943]
#   class 2 (smooth) bv: [5.5,   27.5]    tc: [0.000, 0.033]
#   bv gap: [27.5, 185.8] → threshold 100
#   tc gap: [0.033, 0.897] → threshold 0.5
# OR logic: class 1 if EITHER feature exceeds its threshold.
# This tolerates partial contact degrading one feature while the other stays strong.
BV_THRESH = 100.0
TC_THRESH = 0.5

# Retained for the combined-score fallback (both features in the ambiguous zone).
NCC_WEIGHT = 0.1
BLOCK_VARIANCE_WEIGHT = 0.5
TEXTURE_COVERAGE_WEIGHT = 0.4
DENSE_CLASS_ID = 1
SMOOTH_CLASS_ID = 2


def gradient_feature(gray: np.ndarray) -> np.ndarray:
    """CLAHE-enhanced gradient magnitude for local texture structure."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def block_variance(gray: np.ndarray, block: int = 8) -> float:
    """Mean variance of non-overlapping blocks to discriminate texture density."""
    h, w = gray.shape
    return float(np.mean([
        gray[y:y + block, x:x + block].var()
        for y in range(0, h - block, block)
        for x in range(0, w - block, block)
    ]))


def texture_coverage(gray: np.ndarray, block: int = 16, variance_threshold: float = 80.0) -> float:
    """Fraction of image blocks containing visible texture.

    Dense samples should have texture in most blocks, while square samples have
    large smooth holes where block variance stays low.
    """
    h, w = gray.shape
    block_variances = [
        gray[y:y + block, x:x + block].var()
        for y in range(0, h - block + 1, block)
        for x in range(0, w - block + 1, block)
    ]
    if not block_variances:
        return 0.0
    return float(np.mean(np.asarray(block_variances) > variance_threshold))


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation between two arrays of the same shape."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 1e-6 else 0.0


class ClassRefs:
    """Pre-computed features for one texture class."""

    def __init__(self, class_id: int, folder: Path):
        self.class_id = class_id
        self.grad_features: list[np.ndarray] = []
        self.block_variances: list[float] = []
        self.texture_coverages: list[float] = []
        for path in sorted(folder.glob('*.png')):
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            gray = cv2.resize(gray, REF_SIZE)
            self.grad_features.append(gradient_feature(gray))
            self.block_variances.append(block_variance(gray))
            self.texture_coverages.append(texture_coverage(gray))
        self.mean_block_variance = float(np.mean(self.block_variances)) if self.block_variances else 0.0
        self.mean_texture_coverage = float(np.mean(self.texture_coverages)) if self.texture_coverages else 0.0
        # Medians are outlier-robust and used by the combined-score fallback.
        self.median_block_variance = float(np.median(self.block_variances)) if self.block_variances else 0.0
        self.median_texture_coverage = float(np.median(self.texture_coverages)) if self.texture_coverages else 0.0

    def __len__(self):
        return len(self.grad_features)


def load_refs(base_dir: Path) -> list[ClassRefs]:
    return [ClassRefs(class_id, base_dir / folder) for class_id, folder in TEXTURE_CLASSES]


# ---------------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------------

def _build_feature_matrix(refs: list[ClassRefs]) -> tuple[np.ndarray, list[int], list[str]]:
    """Stack gradient features into (n_samples × n_features), with matching label and name lists."""
    rows, labels, names = [], [], []
    for cr in refs:
        for i, feat in enumerate(cr.grad_features):
            rows.append(feat.flatten().astype(np.float32))
            labels.append(cr.class_id)
            names.append(f'c{cr.class_id}[{i}]')
    X = np.stack(rows, axis=0)
    return X, labels, names


def _fit_pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Center X and project onto its top principal components via SVD.

    Returns:
        X_proj: (n_samples, n_components) coordinates in PC space.
        explained_ratio: variance fraction explained by each component.
    """
    X = X - X.mean(axis=0)
    # economy SVD: X = U S Vt, principal components are rows of Vt
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    X_proj = X @ Vt[:n_components].T
    total_variance = (S ** 2).sum()
    explained_ratio = (S[:n_components] ** 2) / (total_variance + 1e-12)
    return X_proj, explained_ratio


def visualize_pca(refs: list[ClassRefs], out_path: Path | None = None) -> None:
    """Create a 2-D PCA scatter plot of all reference images, coloured by class."""
    import matplotlib.pyplot as plt

    X, labels, names = _build_feature_matrix(refs)
    X_proj, explained = _fit_pca(X, n_components=2)

    class_ids = sorted({cr.class_id for cr in refs})
    palette = plt.cm.tab10.colors
    color_map = {cid: palette[i] for i, cid in enumerate(class_ids)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for cid in class_ids:
        mask = [l == cid for l in labels]
        xs = X_proj[mask, 0]
        ys = X_proj[mask, 1]
        ax.scatter(xs, ys, c=[color_map[cid]], label=f'class {cid}', s=80, edgecolors='k', linewidths=0.5)

    for i, name in enumerate(names):
        ax.annotate(name, (X_proj[i, 0], X_proj[i, 1]),
                    fontsize=7, xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}% var)')
    ax.set_title('PCA of gradient features — reference images')
    ax.legend()
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(str(out_path), dpi=150)
        print(f'Saved PCA plot to {out_path}')
    else:
        plt.show()


if __name__ == '__main__':
    import sys
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    print(f'Loading references from {base}')
    refs = load_refs(base)
    for cr in refs:
        print(
            f'  class {cr.class_id}: {len(cr)} images, '
            f'mean_block_var={cr.mean_block_variance:.1f}, '
            f'mean_texture_coverage={cr.mean_texture_coverage:.3f}'
        )
    visualize_pca(refs, out_path=out)


def texture_scores(query_bgr: np.ndarray, refs: list[ClassRefs]) -> dict[int, dict[str, float]]:
    """Return per-class classifier score components for query_bgr.

    Score combines mean NCC of CLAHE gradient features and inverse distance to
    class mean block variance. Both signals are normalized before averaging.
    """
    query_gray = cv2.cvtColor(cv2.resize(query_bgr, REF_SIZE), cv2.COLOR_BGR2GRAY)
    query_feature = gradient_feature(query_gray)
    query_block_variance = block_variance(query_gray)
    query_texture_coverage = texture_coverage(query_gray)

    ncc_scores = []
    bv_scores = []
    coverage_scores = []
    for class_refs in refs:
        if not class_refs.grad_features:
            ncc_scores.append(0.0)
            bv_scores.append(0.0)
            coverage_scores.append(0.0)
            continue
        ncc_scores.append(float(np.mean([ncc(query_feature, ref) for ref in class_refs.grad_features])))
        bv_scores.append(1.0 / (abs(query_block_variance - class_refs.mean_block_variance) + 1.0))
        coverage_scores.append(1.0 / (abs(query_texture_coverage - class_refs.mean_texture_coverage) + 1.0))

    ncc_sum = sum(abs(score) for score in ncc_scores) + 1e-9
    bv_sum = sum(bv_scores) + 1e-9
    coverage_sum = sum(coverage_scores) + 1e-9
    combined = [
        NCC_WEIGHT * ncc_scores[i] / ncc_sum
        + BLOCK_VARIANCE_WEIGHT * bv_scores[i] / bv_sum
        + TEXTURE_COVERAGE_WEIGHT * coverage_scores[i] / coverage_sum
        for i in range(len(refs))
    ]
    return {
        refs[i].class_id: {
            'combined': float(combined[i]),
            'ncc': float(ncc_scores[i]),
            'block_variance_score': float(bv_scores[i]),
            'texture_coverage': float(coverage_scores[i]),
            'query_block_variance': float(query_block_variance),
            'query_texture_coverage': float(query_texture_coverage),
        }
        for i in range(len(refs))
    }


def classify_texture(query_bgr: np.ndarray, refs: list[ClassRefs]) -> int:
    """Return the class_id that best matches query_bgr.

    OR logic: class 1 (dense) if bv >= BV_THRESH OR tc >= TC_THRESH, else class 2 (smooth).
    The feature gap between classes is large enough that a single threshold on either
    feature is sufficient; the OR rule tolerates partial contact degrading one feature.
    """
    query_gray = cv2.cvtColor(cv2.resize(query_bgr, REF_SIZE), cv2.COLOR_BGR2GRAY)
    bv = block_variance(query_gray)
    tc = texture_coverage(query_gray)

    if bv >= BV_THRESH or tc >= TC_THRESH:
        return DENSE_CLASS_ID
    return SMOOTH_CLASS_ID
