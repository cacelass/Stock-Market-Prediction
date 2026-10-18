"""Purged K-Fold Cross-Validation with embargo for time series.

Prevents information leakage between train/test sets by:
  - Purging: removing test samples that overlap with train labels
  - Embargo: removing samples near the boundary to prevent leakage

Much better than simple temporal split for hyperparameter tuning.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class PurgedKFold:
    """Purged K-Fold splitter for time series.

    Parameters
    ----------
    n_splits : int
        Number of folds (default 5).
    embargo_days : int
        Number of samples after each train fold to discard from the next
        test fold. Prevents leakage from overlapping windows.
    purge_pct : float
        Fraction of the test fold to purge (remove). Samples near the
        train/test boundary in the test set are discarded too.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 5,
        purge_pct: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.purge_pct = purge_pct
        self.random_state = random_state
        self.kf = KFold(n_splits=n_splits, shuffle=False)

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        groups: pd.Series | None = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) with purging and embargo.

        For each fold:
          1. Standard K-Fold split into train/test.
          2. Purge: remove samples from the test set that are within
             `embargo_days` of any train sample (by index proximity).
          3. Further purge a percentage from the start of the test set.

        This prevents label leakage in time series with overlapping
        windows (e.g., 20-day rolling volatility features).
        """
        n_samples = len(X)

        for train_idx, test_idx in self.kf.split(X):
            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)

            # Sort test indices (K-Fold returns sorted by design for shuffle=False)
            test_idx.sort()

            # Embargo: remove test samples near train boundary
            if len(train_idx) > 0:
                max_train = train_idx.max()
                embargo_mask = test_idx > (max_train + self.embargo_days)
                test_idx = test_idx[embargo_mask]

            # Purge: remove a percentage from the start of the test set
            if len(test_idx) > 0 and self.purge_pct > 0:
                purge_count = max(1, int(len(test_idx) * self.purge_pct))
                test_idx = test_idx[purge_count:]

            if len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    def get_n_splits(
        self,
        X: pd.DataFrame | np.ndarray | None = None,
        y: pd.Series | np.ndarray | None = None,
        groups: pd.Series | None = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits


def purged_kfold_cv(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_splits: int = 5,
    embargo_days: int = 5,
    purge_pct: float = 0.05,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convenience function: return list of (train, test) index arrays.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    n_splits : int
        Number of folds.
    embargo_days : int
        Samples after each train fold to discard.
    purge_pct : float
        Fraction of test set to purge.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    list of (train_idx, test_idx) tuples.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    pkf = PurgedKFold(
        n_splits=n_splits,
        embargo_days=embargo_days,
        purge_pct=purge_pct,
        random_state=random_state,
    )

    folds = []
    for train_idx, test_idx in pkf.split(X_arr, y_arr):
        folds.append((train_idx, test_idx))

    return folds


def walk_forward_cv(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_splits: int = 5,
    initial_train_size: int | None = None,
    expand: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding window walk-forward validation.

    Unlike PurgedKFold, this is a pure temporal split without purging.
    Useful as a baseline for comparison.

    Parameters
    ----------
    X : array-like
    y : array-like
    n_splits : int
        Number of test folds.
    initial_train_size : int or None
        Minimum training set size. Defaults to max(10, n_samples // n_splits).
    expand : bool
        If True, training set expands with each fold. If False, rolling window.

    Returns
    -------
    list of (train_idx, test_idx) tuples.
    """
    X_arr = np.asarray(X)
    n_samples = len(X_arr)

    if initial_train_size is None:
        initial_train_size = max(10, n_samples // max(n_splits, 2))

    folds = []
    test_size = (n_samples - initial_train_size) // n_splits

    for i in range(n_splits):
        if expand:
            train_end = initial_train_size + i * test_size
        else:
            train_start = max(0, initial_train_size + i * test_size - test_size)
            train_end = initial_train_size + i * test_size

        test_start = train_end
        test_end = min(train_end + test_size, n_samples)

        if test_end <= test_start:
            break

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        folds.append((train_idx, test_idx))

    return folds
