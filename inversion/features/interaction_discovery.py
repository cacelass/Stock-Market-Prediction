"""Feature interaction auto-discovery.

Uses mutual information and tree-based feature importance to find:
  - Pairwise interactions that add predictive power
  - Non-linear combinations of features
  - Interaction terms that complement base features

This is a lightweight, deterministic approach that can be used during
feature engineering to identify which feature pairs are most valuable.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score


def discover_interactions(
    X: pd.DataFrame,
    y: np.ndarray,
    max_interactions: int = 20,
    n_bins: int = 10,
    random_state: int = 42,
) -> List[Dict[str, object]]:
    """Auto-discover feature interactions that improve prediction.

    For each pair of features:
      1. Compute MI of the interaction (product) with the target.
      2. Compute MI of each base feature with the target.
      3. Interaction strength = MI(product, y) - max(MI(f1, y), MI(f2, y)).
         Positive = the pair adds information beyond individual features.

    Parameters
    ----------
    X : DataFrame
        Feature matrix (only numeric columns used).
    y : array-like
        Binary target.
    max_interactions : int
        Maximum number of interaction pairs to return.
    n_bins : int
        Number of bins for discretization before MI computation.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    List of dicts, each with:
      - "features": (feature1, feature2)
      - "mi_interaction": MI of the product with y
      - "mi_improvement": MI(product) - max(MI(f1), MI(f2))
      - "rf_importance": feature importance of the product in a RF
    """
    X_num = X.select_dtypes(include=[np.number])
    feature_names = list(X_num.columns)
    y_arr = np.asarray(y)

    if len(feature_names) < 2:
        return []

    # Compute MI for each base feature
    base_mi: Dict[str, float] = {}
    for col in feature_names:
        discrete = pd.cut(X_num[col], bins=n_bins, labels=False, include_lowest=True)
        base_mi[col] = mutual_info_score(
            pd.factorize(discrete)[0],
            y_arr,
            random_state=random_state,
        )

    # Compute pairwise interactions
    interactions: List[Dict[str, object]] = []

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            f1, f2 = feature_names[i], feature_names[j]

            # Product interaction
            product = (X_num[f1] * X_num[f2]).fillna(0.0)
            discrete_prod = pd.cut(product, bins=n_bins, labels=False, include_lowest=True)
            mi_interaction = mutual_info_score(
                pd.factorize(discrete_prod)[0],
                y_arr,
                random_state=random_state,
            )

            # Improvement over best base
            improvement = mi_interaction - max(base_mi[f1], base_mi[f2])

            # Quick RF importance test
            rf = RandomForestClassifier(
                n_estimators=20, max_depth=3, random_state=random_state,
            )
            X_test = np.column_stack([product.values.reshape(-1, 1), X_num[f1].fillna(0), X_num[f2].fillna(0)])
            try:
                rf.fit(X_test, y_arr)
                rf_imp = float(rf.feature_importances_[0])
            except Exception:
                rf_imp = 0.0

            interactions.append({
                "features": (f1, f2),
                "mi_interaction": round(mi_interaction, 6),
                "mi_improvement": round(improvement, 6),
                "rf_importance": round(rf_imp, 6),
            })

    # Sort by improvement (descending) and return top N
    interactions.sort(key=lambda d: d["mi_improvement"], reverse=True)
    return interactions[:max_interactions]


def create_interaction_features(
    df: pd.DataFrame,
    selected_interactions: List[Tuple[str, str]] | None = None,
    max_interactions: int = 20,
) -> pd.DataFrame:
    """Create interaction features from selected feature pairs.

    If no interactions are selected, auto-discovers the best ones.

    Parameters
    ----------
    df : DataFrame
        Input features.
    selected_interactions : list of (f1, f2) tuples or None.
        If None, auto-discovers.
    max_interactions : int
        Limit for auto-discovery.

    Returns
    -------
    DataFrame with new interaction columns.
    """
    out = df.copy()

    if selected_interactions is None:
        result = discover_interactions(out, out["target"], max_interactions=max_interactions)
        selected_interactions = [d["features"] for d in result if d["mi_improvement"] > 0]

    for f1, f2 in selected_interactions:
        if f1 in out.columns and f2 in out.columns:
            col = f"{f1}_x_{f2}"
            out[col] = (out[f1] * out[f2]).fillna(0.0)

    return out
