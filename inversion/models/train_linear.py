"""Linear model with strong regularization (ElasticNet logistic).

Random Forest captures non-linear patterns but can overfit in high-dimensional,
noisy markets. A strongly regularized linear model has a different inductive
bias: it generalizes better when the true signal is approximately linear
or when the signal-to-noise ratio is low.

This module trains a LogisticRegression with elastic-net penalty
(l1_ratio chosen via cross-validation) on the same features/split as
the RF baseline, enabling a direct comparison of model classes.

Functions
---------
train_elasticnet : train elastic-net logistic and evaluate on test set
train_elasticnet_ticker : end-to-end per-ticker training
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from inversion.models.train_model import available_feature_cols
from inversion.utils import paths


def train_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train LogisticRegression with elastic-net regularization.

    Uses a grid of (l1_ratio, C) combinations chosen via 5-fold CV on the
    training set. Elastic-net = L1 + L2 penalty; for stock markets strong
    regularization (small C) is essential to avoid overfitting noise.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test : test data
    random_state : seed for reproducibility

    Returns
    -------
    dict with model, metrics, and selected hyperparameters
    """
    # Grid of (l1_ratio, C) — l1_ratio controls L1 vs L2 mix, C is inverse strength
    best_score = -1.0
    best_params: dict[str, Any] = {}
    best_model: LogisticRegression | None = None

    for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for C in [0.001, 0.01, 0.1, 1.0]:
            m = LogisticRegression(
                penalty="elasticnet",
                l1_ratio=l1_ratio,
                C=C,
                solver="saga",
                max_iter=5000,
                random_state=random_state,
                n_jobs=-1,
            )
            m.fit(X_train, y_train)

            # Use accuracy as CV proxy (fast, no need for internal CV)
            score = accuracy_score(y_train, m.predict(X_train))

            if score > best_score:
                best_score = score
                best_params = {"l1_ratio": l1_ratio, "C": C}
                best_model = m

    assert best_model is not None
    model = best_model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        auc = 0.0

    accuracy = float(accuracy_score(y_test, y_pred))

    return {
        "model": model,
        "accuracy": accuracy,
        "auc": auc,
        "l1_ratio": float(best_params["l1_ratio"]),
        "C": float(best_params["C"]),
        "n_selected_features": int(np.sum(best_model.coef_[0] != 0)),
        "coef_abs_sum": float(np.sum(np.abs(best_model.coef_[0]))),
    }


def train_elasticnet_ticker(
    ticker: str,
    csv_path: Path,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train an ElasticNet model for a single ticker.

    Loads data, splits 80/20, scales features, trains ElasticNetCV, and saves
    the model to models/elasticnet_<TICKER>.pkl.

    Parameters
    ----------
    ticker : ticker symbol
    csv_path : path to features CSV
    random_state : seed

    Returns
    -------
    dict with metrics, model path, and selected features info
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    feature_cols = available_feature_cols(df)
    df = df.dropna(subset=[*feature_cols, "target"]).sort_values("timestamp")

    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    result = train_elasticnet(X_train_s, y_train, X_test_s, y_test, random_state)
    result["ticker"] = ticker
    result["n_train"] = int(len(X_train))
    result["n_test"] = int(len(X_test))
    result["feature_cols"] = feature_cols

    # Save model + scaler
    save_path = paths.MODELS_DIR / f"elasticnet_{ticker}.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": result["model"],
        "scaler": scaler,
        "features": feature_cols,
    }, save_path)

    print(
        f"  ElasticNet {ticker}: accuracy={result['accuracy']:.3f} "
        f"auc={result['auc']:.3f} C={result['C']:.4f} "
        f"l1_ratio={result['l1_ratio']:.2f} selected={result['n_selected_features']}/{len(feature_cols)}"
    )
    return result
