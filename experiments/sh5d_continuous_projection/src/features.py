"""Feature engineering: per-trace embedding-space features."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine as cosine_distance


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance = 1 - cosine_similarity. Returns 0 for identical vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    sim = np.dot(a, b) / (norm_a * norm_b)
    # Clip to handle floating point issues
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def compute_trace_features(step_embeddings: np.ndarray,
                           slod_axis: np.ndarray) -> dict:
    """Compute all 15 features for a single trace.

    Args:
        step_embeddings: (n_steps, 768) embeddings for this trace's steps (ordered by step_index)
        slod_axis: (768,) unit vector for SLoD axis

    Returns:
        dict of feature name -> value (NaN for undefined features)
    """
    n_steps = step_embeddings.shape[0]
    features = {}

    # --- SLoD axis projections ---
    projections = step_embeddings @ slod_axis  # (n_steps,)
    features['slod_axis_mean'] = float(projections.mean())
    features['slod_axis_variance'] = float(projections.var()) if n_steps > 1 else 0.0
    features['slod_axis_range'] = float(projections.max() - projections.min())

    if n_steps < 2:
        # No pairwise features possible
        features['mean_cosine_dist'] = np.nan
        features['max_cosine_dist'] = np.nan
        features['embedding_path_length'] = np.nan
        features['embedding_displacement'] = np.nan
        features['path_efficiency'] = np.nan
        features['slod_axis_drift_mean'] = np.nan
        features['slod_axis_drift_max'] = np.nan
        features['slod_axis_direction'] = np.nan
        features['slod_axis_monotonicity'] = np.nan
        features['orthogonal_drift_mean'] = np.nan
        features['orthogonal_variance'] = np.nan
        features['slod_ratio'] = np.nan
        return features

    # --- Full embedding features ---
    cos_dists = []
    for i in range(n_steps - 1):
        d = cosine_dist(step_embeddings[i], step_embeddings[i + 1])
        cos_dists.append(d)
    cos_dists = np.array(cos_dists)

    features['mean_cosine_dist'] = float(cos_dists.mean())
    features['max_cosine_dist'] = float(cos_dists.max())
    features['embedding_path_length'] = float(cos_dists.sum())
    features['embedding_displacement'] = cosine_dist(step_embeddings[0], step_embeddings[-1])

    if features['embedding_path_length'] > 1e-10:
        features['path_efficiency'] = features['embedding_displacement'] / features['embedding_path_length']
    else:
        features['path_efficiency'] = np.nan

    # --- SLoD axis drift features ---
    proj_deltas = np.abs(np.diff(projections))
    features['slod_axis_drift_mean'] = float(proj_deltas.mean())
    features['slod_axis_drift_max'] = float(proj_deltas.max())
    features['slod_axis_direction'] = float(np.sign(projections[-1] - projections[0]))

    # Monotonicity: Spearman correlation of projection vs step index
    if n_steps > 2:
        rho, _ = spearmanr(np.arange(n_steps), projections)
        features['slod_axis_monotonicity'] = float(rho) if not np.isnan(rho) else 0.0
    else:
        # 2 steps: monotonicity is trivially +1 or -1 depending on direction
        features['slod_axis_monotonicity'] = float(np.sign(projections[-1] - projections[0]))

    # --- Orthogonal features ---
    # For each consecutive pair, decompose the embedding difference into
    # parallel (SLoD axis) and orthogonal components.
    # total_diff = parallel + orthogonal
    # parallel = (diff . axis) * axis
    # |orthogonal|^2 = |total_diff|^2 - |parallel|^2
    orthogonal_drifts = []
    slod_drifts_euclidean = []
    total_drifts = []

    for i in range(n_steps - 1):
        diff = step_embeddings[i + 1] - step_embeddings[i]
        total_dist_sq = float(np.dot(diff, diff))
        parallel_component = float(np.dot(diff, slod_axis))
        parallel_dist_sq = parallel_component ** 2
        orthogonal_dist_sq = max(0.0, total_dist_sq - parallel_dist_sq)
        orthogonal_drifts.append(np.sqrt(orthogonal_dist_sq))
        slod_drifts_euclidean.append(abs(parallel_component))
        total_drifts.append(np.sqrt(total_dist_sq))

    orthogonal_drifts = np.array(orthogonal_drifts)
    slod_drifts_euclidean = np.array(slod_drifts_euclidean)
    total_drifts = np.array(total_drifts)

    features['orthogonal_drift_mean'] = float(orthogonal_drifts.mean())

    # Orthogonal variance: variance of the orthogonal component of each step's embedding
    # Project out the SLoD axis component to get orthogonal embeddings
    parallel_projections = (step_embeddings @ slod_axis).reshape(-1, 1) * slod_axis.reshape(1, -1)
    orthogonal_embeddings = step_embeddings - parallel_projections
    # Variance = mean of per-dimension variances summed
    features['orthogonal_variance'] = float(orthogonal_embeddings.var(axis=0).sum())

    # SLoD ratio: fraction of drift along SLoD axis
    mean_total = total_drifts.mean()
    if mean_total > 1e-10:
        features['slod_ratio'] = float(slod_drifts_euclidean.mean() / mean_total)
    else:
        features['slod_ratio'] = np.nan

    return features


def compute_all_trace_features(steps_df: pd.DataFrame,
                                step_embeddings: np.ndarray,
                                slod_axis: np.ndarray,
                                answer_scores: pd.DataFrame) -> pd.DataFrame:
    """Compute features for all traces.

    Args:
        steps_df: DataFrame with question_id, condition, step_index columns
        step_embeddings: (N, 768) embeddings aligned with steps_df rows
        slod_axis: (768,) unit vector
        answer_scores: DataFrame with question_id, condition, answer_token_f1, attribution_f1

    Returns:
        DataFrame with one row per trace, all features + quality metrics
    """
    # Group by trace
    traces = steps_df.groupby(['question_id', 'condition'])
    all_features = []

    for (qid, cond), group in traces:
        # Get embeddings for this trace, sorted by step_index
        group = group.sort_values('step_index')
        indices = group.index.values
        trace_embs = step_embeddings[indices]

        feats = compute_trace_features(trace_embs, slod_axis)
        feats['question_id'] = qid
        feats['condition'] = cond
        feats['n_steps'] = len(group)
        all_features.append(feats)

    features_df = pd.DataFrame(all_features)

    # Merge with answer scores
    merged = features_df.merge(answer_scores[['question_id', 'condition',
                                               'answer_token_f1', 'attribution_f1']],
                                on=['question_id', 'condition'], how='left')
    print(f"Computed features for {len(merged)} traces")
    print(f"  With quality scores: {merged['answer_token_f1'].notna().sum()}")
    return merged
