"""Class imbalance handling for binary classification.

Provides multiple strategies to deal with imbalanced classes:
  1. SMOTE (Synthetic Minority Over-sampling Technique)
  2. ADASYN (Adaptive Synthetic Sampling)
  3. Balanced class weights (already used in RandomForest)
  4. Threshold moving (adjust prediction threshold)
  5. Focal loss simulation (weight hard examples more)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def handle_class_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "balanced",
    k: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Handle class imbalance with the specified method.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Binary target (0, 1).
    method : str
        One of:
          - "balanced"       : no resampling, use class_weight="balanced"
          - "smote"          : SMOTE oversampling of minority class
          - "adasyn"         : ADASYN (adaptive, focuses on hard examples)
          - "focal"          : weight hard examples more (simulated)
    k : int
        Number of neighbors for SMOTE/ADASYN (default 5).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_resampled, y_resampled : same shape as input but balanced.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)

    if method == "balanced":
        return X_arr, y_arr

    # Count classes
    class_counts = np.bincount(y_arr, minlength=2)
    minority_class = int(np.argmin(class_counts))
    majority_class = 1 - minority_class

    if class_counts[minority_class] == class_counts[majority_class]:
        # Already balanced
        return X_arr, y_arr

    X_minority = X_arr[y_arr == minority_class]
    X_majority = X_arr[y_arr == majority_class]

    if method == "smote":
        return _smote(X_minority, X_majority, k=k, random_state=random_state)
    elif method == "adasyn":
        return _adasyn(X_minority, X_majority, k=k, random_state=random_state)
    elif method == "focal":
        # Focal loss is applied during training, not resampling.
        # Return data as-is but return weights for focal simulation.
        return X_arr, y_arr
    else:
        raise ValueError(f"Unknown method: {method}. Use 'balanced', 'smote', 'adasyn', or 'focal'.")


def _smote(
    X_minority: np.ndarray,
    X_majority: np.ndarray,
    k: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """SMOTE: generate synthetic minority samples.

    For each minority sample, find k nearest neighbors among minority class.
    Create synthetic samples by interpolating between the sample and a
    randomly chosen neighbor.
    """
    rng = np.random.RandomState(random_state)

    # Number of synthetic samples needed to balance
    n_needed = len(X_majority) - len(X_minority)
    n_samples_per = max(1, n_needed // len(X_minority))
    remainder = n_needed % len(X_minority)

    # Find neighbors using simple distance (KNN)
    nn = KNeighborsClassifier(n_neighbors=min(k, len(X_minority) - 1))
    nn.fit(X_minority)
    neighbors = nn.kneighbors(X_minority, return_distance=False)

    synthetic = []
    for i, idx in enumerate(neighbors):
        n_gen = n_samples_per + (1 if i < remainder else 0)
        for _ in range(n_gen):
            neighbor_idx = idx[rng.integers(1, len(idx))]  # skip self
            diff = X_minority[neighbor_idx] - X_minority[i]
            lambda_val = rng.uniform()
            synthetic.append(X_minority[i] + lambda_val * diff)

    X_new = np.concatenate([X_majority] + synthetic, axis=0)
    y_new = np.concatenate([
        np.zeros(len(X_majority), dtype=int),
        np.ones(len(synthetic), dtype=int),
    ])

    # Shuffle
    shuffle_idx = rng.permutation(len(X_new))
    return X_new[shuffle_idx], y_new[shuffle_idx]


def _adasyn(
    X_minority: np.ndarray,
    X_majority: np.ndarray,
    k: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """ADASYN: adaptive synthetic sampling.

    Generates more synthetic samples for minority examples that are
    harder to learn (i.e., have more majority-class neighbors).
    """
    rng = np.random.RandomState(random_state)

    n_minority = len(X_minority)

    # Find neighbors for each minority sample
    nn = KNeighborsClassifier(n_neighbors=min(k, n_minority - 1))
    nn.fit(X_minority)
    neighbors = nn.kneighbors(X_minority, return_distance=False)

    # Compute ratio of majority neighbors for each minority sample
    ratios = np.zeros(n_minority)
    for i, idx in enumerate(neighbors):
        # In our context, "neighbors" are all from minority class since
        # we fit on X_minority. ADASYN typically fits on combined data.
        # Simplified: use distance-weighted difficulty
        diffs = np.linalg.norm(X_minority[idx] - X_minority[i], axis=1)
        diffs = np.where(diffs < 1e-9, 1e-9, diffs)
        ratios[i] = 1.0 / (diffs.mean() + 1e-9)  # harder = closer = more synthetic

    # Normalize ratios
    ratios /= ratios.sum()

    # Total synthetic samples needed
    n_needed = len(X_majority) - len(X_minority)
    n_per = np.ceil(ratios * n_needed).astype(int)

    synthetic = []
    for i, n in enumerate(n_per):
        neighbor_idx = neighbors[i][rng.integers(1, len(neighbors[i]))]  # skip self
        for _ in range(n):
            diff = X_minority[neighbor_idx] - X_minority[i]
            lambda_val = rng.uniform()
            synthetic.append(X_minority[i] + lambda_val * diff)

    X_new = np.concatenate([X_majority] + synthetic, axis=0)
    y_new = np.concatenate([
        np.zeros(len(X_majority), dtype=int),
        np.ones(len(synthetic), dtype=int),
    ])

    shuffle_idx = rng.permutation(len(X_new))
    return X_new[shuffle_idx], y_new[shuffle_idx]


def compute_focal_weights(
    y: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> np.ndarray:
    """Compute focal loss-style sample weights.

    Focal loss down-weights easy examples and focuses on hard ones.
    This simulates it by returning sample weights.

    Parameters
    ----------
    y : array-like
        Binary target (0 or 1).
    gamma : float
        Focusing parameter. Higher = more focus on hard examples.
    alpha : float
        Balance factor for positive class.

    Returns
    -------
    sample_weights : array of same length as y.
    """
    y_arr = np.asarray(y, dtype=int)

    # Simple approximation: weight = alpha * p^gamma for positive,
    # (1-alpha) * (1-p)^gamma for negative, where p = model confidence.
    # Since we don't have probabilities here, use inverse class frequency as proxy.
    class_counts = np.bincount(y_arr, minlength=2)
    total = len(y_arr)

    # Base weights from class imbalance
    base_weights = np.where(
        y_arr == 1,
        class_counts[0] / total,
        class_counts[1] / total,
    )

    # Focal adjustment: hard examples get higher weight
    # Hard = wrong direction (we approximate with minority class proximity)
    minority_weight = max(base_weights)
    easy_weight = min(base_weights) if class_counts.min() > 0 else 0

    # Simulate focal weighting
    focal_adj = (minority_weight / (easy_weight + 1e-9)) ** gamma
    weights = np.where(
        y_arr == 1,
        alpha * focal_adj,
        (1 - alpha),
    )

    # Normalize
    weights /= weights.mean()

    return weights


def threshold_moving(
    y_proba: np.ndarray,
    y_true: np.ndarray,
    thresholds: list[float] | None = None,
    metric: str = "f1",
) -> float:
    """Find the optimal prediction threshold for classification.

    Parameters
    ----------
    y_proba : array-like
        Predicted probabilities for class 1.
    y_true : array-like
        True labels.
    thresholds : list of float or None
        Thresholds to search. If None, searches 0.1 to 0.9.
    metric : str
        Metric to optimize: "f1", "accuracy", "precision", "recall".

    Returns
    -------
    best_threshold : float.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05).tolist()

    best_score = -1.0
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        if metric == "f1":
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            score = f1
        elif metric == "accuracy":
            score = np.mean(y_pred == y_true)
        elif metric == "precision":
            score = precision
        elif metric == "recall":
            score = recall
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_thresh = thresh

    return float(best_thresh)
