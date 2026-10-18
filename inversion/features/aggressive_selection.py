"""Aggressive feature selection using SHAP importance.

Reduces the feature set to the top-N most important features as measured by
mean absolute SHAP value (or tree-based feature_importances_ as fallback).
Helps when noise features drown the signal — especially sentiment or
macro columns that add dimensionality without predictive value.

Functions
---------
select_top_features_shap : rank features by mean |SHAP| and return top-N
train_with_selected_features : end-to-end: train RF with only top-N features
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from inversion.features.feature_selection import compute_shap_importance
from inversion.features.build_features import fit_and_save_scaler
from inversion.models.train_model import (
    FEATURE_COLS,
    available_feature_cols,
    train_rf_model,
    evaluate_model,
)
from inversion.utils import paths


def select_top_features_shap(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str] | None = None,
    n_top: int = 5,
) -> list[str]:
    """Select top-N features using SHAP importance.

    Falls back to model.feature_importances_ when SHAP is not installed.
    Returns feature names ordered by importance (descending), truncated to
    ``n_top``.

    Parameters
    ----------
    model : fitted classifier (RF, HGB, …)
    X : feature DataFrame used for SHAP computation
    feature_names : explicit names (defaults to X.columns)
    n_top : number of features to keep

    Returns
    -------
    list of feature names, length <= n_top
    """
    importance = compute_shap_importance(model, X, feature_names)
    if not importance:
        names = feature_names or list(X.columns)
        return names[:n_top]

    sorted_features = sorted(importance, key=lambda f: importance[f], reverse=True)
    return sorted_features[:n_top]


def train_with_selected_features(
    ticker: str,
    csv_path: Path,
    n_top: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train an RF model using only the top-N SHAP-selected features.

    1. Trains a full RF on all features.
    2. Computes SHAP importance on the training set.
    3. Selects top-N features.
    4. Retrains a second RF on only those features.
    5. Evaluates both models for comparison.

    Parameters
    ----------
    ticker : ticker symbol (for model file naming)
    csv_path : path to features CSV
    n_top : number of features to keep
    random_state : seed

    Returns
    -------
    dict with full and reduced model metrics, plus the selected feature list
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    feature_cols = available_feature_cols(df)
    df = df.dropna(subset=[*feature_cols, "target"]).sort_values("timestamp")

    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    X_train_full = train_df[feature_cols]
    y_train = train_df["target"]
    X_test_full = test_df[feature_cols]
    y_test = test_df["target"]

    # Fit scaler on full feature set
    scaler = fit_and_save_scaler(X_train_full, filename=f"scaler_{ticker}_selected.pkl")
    X_train_s = scaler.transform(X_train_full)
    X_test_s = scaler.transform(X_test_full)

    # Step 1: Train full model for SHAP
    full_model = train_rf_model(
        X_train_s, y_train, filename=f"rf_{ticker}_selected_full.pkl",
        random_state=random_state,
    )

    # Step 2: Compute SHAP importance
    selected_features = select_top_features_shap(full_model, X_train_full, feature_cols, n_top=n_top)
    print(f"  Top-{n_top} SHAP features for {ticker}: {selected_features}")

    # Step 3: Retrain on selected features only
    X_train_sel = X_train_s[:, [list(X_train_full.columns).index(f) for f in selected_features if f in X_train_full.columns]]
    X_test_sel = X_test_s[:, [list(X_test_full.columns).index(f) for f in selected_features if f in X_test_full.columns]]

    reduced_model = train_rf_model(
        X_train_sel, y_train, filename=f"rf_{ticker}_selected.pkl",
        random_state=random_state,
    )

    # Evaluate both
    full_metrics = evaluate_model(full_model, X_test_full, y_test)
    reduced_metrics = evaluate_model(reduced_model, X_test_sel, y_test)

    return {
        "ticker": ticker,
        "n_top": n_top,
        "selected_features": selected_features,
        "n_selected_features": len(selected_features),
        "full_model": {"accuracy": full_metrics["accuracy"], "auc": full_metrics["auc"]},
        "reduced_model": {"accuracy": reduced_metrics["accuracy"], "auc": reduced_metrics["auc"]},
        "delta_accuracy": round(reduced_metrics["accuracy"] - full_metrics["accuracy"], 4),
        "delta_auc": round(reduced_metrics["auc"] - full_metrics["auc"], 4),
        "n_train": int(len(X_train_full)),
        "n_test": int(len(X_test_full)),
    }
