"""Regime-specific models: train separate classifiers per market regime.

Financial markets behave differently in bull, bear, and neutral regimes. A
single model averages over these regimes and may underperform compared to
regime-aware specialists. This module detects regimes from price/vol data
and trains separate models per regime.

Functions
---------
detect_regimes : classify each row as bull/bear/neutral
train_regime_models : train a model per regime, return combined metrics
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from inversion.features.regime_detection import (
    Regime,
    detect_regime,
    detect_regimes,
)
from inversion.models.train_model import available_feature_cols
from inversion.utils import paths


def _prepare_regime_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int = 60,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Split data into bull/bear/neutral subsets with train/test per regime.

    Returns dict mapping regime -> (train_df, test_df), only regimes with
    >= 30 samples each.
    """
    df = df.copy()
    df["regime"] = detect_regimes(df, lookback=lookback)

    result: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for regime in [Regime.BULL, Regime.BEAR, Regime.NEUTRAL]:
        regime_df = df[df["regime"] == regime].sort_values("timestamp")
        regime_df = regime_df.dropna(subset=[*feature_cols, "target"])

        if len(regime_df) < 30:
            print(f"  ⚠ Regime {regime}: only {len(regime_df)} samples, skipped")
            continue

        split = int(len(regime_df) * 0.8)
        train_df, test_df = regime_df.iloc[:split], regime_df.iloc[split:]
        result[regime] = (train_df, test_df)

    return result


def _train_single_regime(
    regime: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    random_state: int = 42,
) -> dict[str, Any]:
    """Train an RF model on a single regime's data."""
    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    model = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=50,
        max_features="sqrt", class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        auc = 0.0

    return {
        "model": model,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": auc,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def train_regime_models(
    ticker: str,
    csv_path: Path,
    lookback: int = 60,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train separate models for bull/bear/neutral regimes.

    For each regime with enough data:
    1. Detect regime labels
    2. Split into train/test (temporal, 80/20)
    3. Train RF with class-weighted balancing

    Also saves a combined model (all data) for comparison.

    Parameters
    ----------
    ticker : ticker symbol
    csv_path : path to features CSV
    lookback : window for regime detection
    random_state : seed

    Returns
    -------
    dict with per-regime metrics and a combined baseline
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    feature_cols = available_feature_cols(df)
    df = df.sort_values("timestamp").dropna(subset=[*feature_cols, "target"])

    # Detect regimes on the main df (not a copy) so we can report distribution later
    df["regime"] = detect_regimes(df, lookback=lookback)

    regime_splits = _prepare_regime_data(df, feature_cols, lookback=lookback)

    if not regime_splits:
        print(f"  ⚠ {ticker}: no regime with enough samples")
        return {"ticker": ticker, "regimes": {}, "error": "no data"}

    regime_metrics: dict[str, dict[str, Any]] = {}
    regime_models: dict[str, Any] = {}

    for regime, (train_df, test_df) in regime_splits.items():
        result = _train_single_regime(regime, train_df, test_df, feature_cols, random_state)
        regime_metrics[regime] = result
        regime_models[regime] = result["model"]
        print(
            f"  {ticker}/{regime}: accuracy={result['accuracy']:.3f} "
            f"auc={result['auc']:.3f} (n_train={result['n_train']}, n_test={result['n_test']})"
        )

    # Combined baseline: single model on all data
    all_train = pd.concat([t for t, _ in regime_splits.values()])
    all_test = pd.concat([t for _, t in regime_splits.values()])
    X_all = all_train[feature_cols].values
    y_all = all_train["target"].values
    X_all_test = all_test[feature_cols].values
    y_all_test = all_test["target"].values

    combined_model = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=50,
        max_features="sqrt", class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )
    combined_model.fit(X_all, y_all)

    combined_pred = combined_model.predict(X_all_test)
    combined_proba = combined_model.predict_proba(X_all_test)[:, 1]

    try:
        combined_auc = float(roc_auc_score(y_all_test, combined_proba))
    except Exception:
        combined_auc = 0.0

    combined_accuracy = float(accuracy_score(y_all_test, combined_pred))

    # Save regime models
    save_path = paths.MODELS_DIR / f"regime_{ticker}.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "models": regime_models,
        "combined_model": combined_model,
        "features": feature_cols,
    }, save_path)

    return {
        "ticker": ticker,
        "regimes": {r: {k: v for k, v in m.items() if k != "model"} for r, m in regime_metrics.items()},
        "combined_model": {
            "accuracy": combined_accuracy,
            "auc": combined_auc,
            "n_train": int(len(X_all)),
            "n_test": int(len(X_all_test)),
        },
        "regime_distribution": {
            regime: int(len(df[df["regime"] == regime]))
            for regime in [Regime.BULL, Regime.BEAR, Regime.NEUTRAL]
        },
    }
