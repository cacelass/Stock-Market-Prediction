"""SHAP-based feature selection.

Computes SHAP importance values for a trained model and selects the top-N
features.  Useful for pruning noise features (especially sentiment ones)
that add little predictive value.

Functions
---------
compute_shap_importance : calculate mean |SHAP| per feature
select_top_features : pick the N most important features
build_optimized_feature_set : return a DataFrame with only top features
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ── SHAP helpers ─────────────────────────────────────────────────────────────

def _import_shap() -> Any | None:
    """Lazily import shap to avoid hard dependency in test envs."""
    try:
        import shap  # noqa: F811
        return shap
    except ImportError:  # pragma: no cover
        return None


def compute_shap_importance(
    model: Any,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    sample_size: int = 500,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute mean absolute SHAP value per feature.

    Parameters
    ----------
    model : fitted scikit-learn classifier (RF, HGB, …)
    X : feature DataFrame
    feature_names : column names (defaults to X.columns)
    sample_size : max samples to use for SHAP computation
    random_state : seed for reproducibility

    Returns
    -------
    dict {feature_name: mean_abs_shap}
    """
    shap = _import_shap()
    if shap is None:  # pragma: no cover — SHAP not installed
        # Fallback: use tree-based feature_importances_ if available
        if hasattr(model, "feature_importances_"):
            names = feature_names or list(X.columns)
            return dict(zip(names, model.feature_importances_))
        return {}

    X_sample = X.sample(n=min(sample_size, len(X)), random_state=random_state)
    feature_names = feature_names or list(X.columns)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Mean |SHAP| across samples and classes
    if isinstance(shap_values, list):  # multi-class
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values)

    importance: Dict[str, float] = {}
    for i, name in enumerate(feature_names):
        if i < len(mean_abs):
            importance[name] = float(np.mean(mean_abs[:, i]))

    return importance


def select_top_features(
    model: Any,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    n_top: int = 15,
) -> List[str]:
    """Select top-N features by SHAP importance.

    Parameters
    ----------
    model : fitted classifier
    X : feature DataFrame
    feature_names : explicit names (optional)
    n_top : number of features to keep

    Returns
    -------
    list of feature names, ordered by importance (descending)
    """
    importance = compute_shap_importance(model, X, feature_names)
    if not importance:
        return feature_names or list(X.columns)[:n_top]

    sorted_features = sorted(importance, key=lambda f: importance[f], reverse=True)
    return sorted_features[:n_top]


def build_optimized_feature_set(
    df: pd.DataFrame,
    model: Any,
    n_top: int = 15,
    candidate_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build a feature set optimised for a specific model.

    Parameters
    ----------
    df : full DataFrame (features + target)
    model : fitted classifier
    n_top : keep top-N by SHAP importance
    candidate_features : subset of columns to evaluate (defaults to all
                         numeric columns except 'target')

    Returns
    -------
    DataFrame with only the selected features + target
    """
    if candidate_features is None:
        candidate_features = [c for c in df.select_dtypes(include=[np.number]).columns if c != "target"]

    X = df[candidate_features]
    selected = select_top_features(model, X, candidate_features, n_top=n_top)
    out = df[selected].copy() if "target" in df.columns else X[selected]

    # Ensure target column is included if present
    if "target" in df.columns and "target" not in out.columns:
        out["target"] = df["target"]

    return out
