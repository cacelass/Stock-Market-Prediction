"""Continuous target regression instead of binary classification.

Classifying "up or down" throws away magnitude information. A regression
model that predicts the actual next-day return can capture richer signal
and be converted to a classifier at any threshold later.

This module trains an RF regressor to predict next-day return (continuous)
and evaluates with R², MAE, and Spearman correlation — plus converts
the regression predictions into a classifier (return > 0 → up) to compare
fairly with the classification baseline.

Functions
---------
train_regression : train RF regressor for continuous returns
train_regression_ticker : end-to-end per-ticker training
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from inversion.models.train_model import available_feature_cols
from inversion.utils import paths


def _make_regression_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Create continuous target: return over the next `horizon` trading days.

    Purely future: shift(-horizon). No leakage.
    """
    if "return" not in df.columns:
        raise ValueError("Dataset must have a 'return' column.")
    return df["return"].shift(-horizon)


def train_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train an RF regressor on continuous returns.

    Evaluates with R², MAE, and Spearman correlation on the test set.
    Also converts regression predictions to a binary classifier (pred > 0)
    and reports accuracy and AUC for comparison with the classification baseline.

    Parameters
    ----------
    X_train, y_train : training data (continuous returns as target)
    X_test, y_test : test data
    random_state : seed

    Returns
    -------
    dict with regression metrics and derived classification metrics
    """
    model = RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=40,
        max_features="sqrt", random_state=random_state, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    spearman, _ = spearmanr(y_test, y_pred)
    spearman = float(spearman)

    # Convert to classifier: pred > 0 → up
    y_pred_cls = (y_pred > 0).astype(int)
    y_test_cls = (y_test > 0).astype(int)

    cls_accuracy = float(np.mean(y_pred_cls == y_test_cls))

    # AUC for classifier comparison
    try:
        from sklearn.metrics import roc_auc_score
        cls_auc = float(roc_auc_score(y_test_cls, y_pred))
    except Exception:
        cls_auc = 0.0

    return {
        "model": model,
        "r2": r2,
        "mae": mae,
        "spearman": spearman,
        "cls_accuracy": cls_accuracy,
        "cls_auc": cls_auc,
    }


def train_regression_ticker(
    ticker: str,
    csv_path: Path,
    horizon: int = 1,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a regression model for a single ticker.

    1. Loads features + returns
    2. Creates continuous target: next-day return
    3. Splits 80/20 temporally
    4. Trains RF regressor
    5. Saves model to models/regression_<TICKER>.pkl

    Parameters
    ----------
    ticker : ticker symbol
    csv_path : path to features CSV (must have 'return')
    horizon : number of days ahead to predict (default 1)
    random_state : seed

    Returns
    -------
    dict with metrics and model path
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    feature_cols = available_feature_cols(df)

    # Create regression target
    df = df.copy()
    df["reg_target"] = _make_regression_target(df, horizon)

    df = df.dropna(subset=[*feature_cols, "reg_target"]).sort_values("timestamp")

    if len(df) < 100:
        raise ValueError(f"Dataset too short for {ticker}: {len(df)} rows after dropna")

    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    X_train = train_df[feature_cols].values
    y_train = train_df["reg_target"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["reg_target"].values

    result = train_regression(X_train, y_train, X_test, y_test, random_state)
    result["ticker"] = ticker
    result["n_train"] = int(len(X_train))
    result["n_test"] = int(len(X_test))
    result["horizon"] = horizon
    result["feature_cols"] = feature_cols

    # Save model
    save_path = paths.MODELS_DIR / f"regression_{ticker}.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": result["model"],
        "features": feature_cols,
        "horizon": horizon,
    }, save_path)

    print(
        f"  Regression {ticker}: R²={result['r2']:.3f} MAE={result['mae']:.4f} "
        f"Spearman={result['spearman']:.3f} cls_acc={result['cls_accuracy']:.3f} "
        f"cls_auc={result['cls_auc']:.3f}"
    )
    return result
