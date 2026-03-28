"""SLoD axis computation and projection."""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def compute_centroid_axis(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute SLoD axis as normalized difference of macro/micro centroids.

    Direction: macro (0) -> micro (2), positive = toward micro.

    Args:
        embeddings: (N, D) embeddings
        labels: (N,) labels (0=macro, 2=micro; 1=meso ignored)

    Returns:
        axis: (D,) unit vector
    """
    macro_mask = labels == 0
    micro_mask = labels == 2
    macro_centroid = embeddings[macro_mask].mean(axis=0)
    micro_centroid = embeddings[micro_mask].mean(axis=0)
    axis = micro_centroid - macro_centroid
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        raise ValueError("Macro and micro centroids are identical!")
    axis = axis / norm
    print(f"Centroid axis computed: macro n={macro_mask.sum()}, micro n={micro_mask.sum()}")
    print(f"  Axis norm before normalization: {norm:.4f}")
    return axis


def compute_lda_axis(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute SLoD axis via LDA (macro vs micro only).

    Args:
        embeddings: (N, D) embeddings
        labels: (N,) labels (0=macro, 2=micro; 1=meso ignored)

    Returns:
        axis: (D,) unit vector (LDA direction)
    """
    # Filter to macro and micro only
    mask = (labels == 0) | (labels == 2)
    X = embeddings[mask]
    y = labels[mask]

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X, y)
    axis = lda.scalings_.flatten()
    axis = axis / np.linalg.norm(axis)

    # Ensure direction is macro -> micro (positive = micro)
    macro_proj = X[y == 0] @ axis
    micro_proj = X[y == 2] @ axis
    if macro_proj.mean() > micro_proj.mean():
        axis = -axis

    print(f"LDA axis computed: n={len(X)}")
    return axis


def project_onto_axis(embeddings: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Project embeddings onto a 1D axis.

    Args:
        embeddings: (N, D) array
        axis: (D,) unit vector

    Returns:
        projections: (N,) array of scalar projections
    """
    return embeddings @ axis


def validate_axis(embeddings: np.ndarray, labels: np.ndarray, axis: np.ndarray) -> dict:
    """Validate SLoD axis on test data.

    Returns dict with per-class mean projection and Cohen's d between macro/micro.
    """
    projections = project_onto_axis(embeddings, axis)

    stats = {}
    for label, name in [(0, 'macro'), (1, 'meso'), (2, 'micro')]:
        mask = labels == label
        proj = projections[mask]
        stats[f'{name}_mean'] = float(proj.mean())
        stats[f'{name}_std'] = float(proj.std())
        stats[f'{name}_n'] = int(mask.sum())

    # Cohen's d between macro and micro
    macro_proj = projections[labels == 0]
    micro_proj = projections[labels == 2]
    pooled_std = np.sqrt((macro_proj.std()**2 + micro_proj.std()**2) / 2)
    cohens_d = (micro_proj.mean() - macro_proj.mean()) / pooled_std
    stats['cohens_d'] = float(cohens_d)
    stats['separation_correct'] = bool(stats['macro_mean'] < stats['meso_mean'] < stats['micro_mean'])

    print(f"Axis validation:")
    print(f"  Macro: mean={stats['macro_mean']:.4f} +/- {stats['macro_std']:.4f}")
    print(f"  Meso:  mean={stats['meso_mean']:.4f} +/- {stats['meso_std']:.4f}")
    print(f"  Micro: mean={stats['micro_mean']:.4f} +/- {stats['micro_std']:.4f}")
    print(f"  Cohen's d (macro vs micro): {cohens_d:.4f}")
    print(f"  Ordering correct: {stats['separation_correct']}")
    return stats
