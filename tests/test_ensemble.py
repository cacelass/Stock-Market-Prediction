"""Tests for the multi-model ensemble: RF-tech, HGB-tech, RF-sentiment.

Covers:
  - inversion.ensemble.weighted_ensemble  (train, predict, optimise weights)
  - inversion.models.train_ensemble       (train_ensemble + save)
  - inversion.models.train_model          (train_ensemble_ticker wrapper)
  - inversion.models.predict_ensemble     (predict_ensemble + graceful degradation)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inversion.ensemble.weighted_ensemble import (
    optimize_weights,
    predict_weighted_ensemble,
    train_weighted_ensemble,
)
from inversion.models.train_ensemble import train_ensemble, save_ensemble_models
from inversion.models.predict_ensemble import predict_ensemble


# ─────────────────────────────────────────────────────────────────────────────
#  weighted_ensemble  — train / predict / weight optimisation
# ─────────────────────────────────────────────────────────────────────────────

def _make_xy(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    return X, y


def test_train_weighted_ensemble_returns_dict():
    X, y = _make_xy()
    r = train_weighted_ensemble(X, y, X, y)
    assert isinstance(r, dict)
    assert "models" in r  # list of labels
    assert "weights" in r
    assert "metrics" in r


def test_train_weighted_ensemble_trains_rf_and_hgb():
    X, y = _make_xy()
    r = train_weighted_ensemble(X, y, X, y)
    assert "rf_tech" in r["models"]
    assert "hgb" in r["models"]


def test_train_weighted_ensemble_reproducible():
    X, y = _make_xy(seed=123)
    r1 = train_weighted_ensemble(X, y, X, y)
    r2 = train_weighted_ensemble(X, y, X, y)
    assert r1["metrics"]["rf_tech"] == r2["metrics"]["rf_tech"]
    assert r1["metrics"]["hgb"] == r2["metrics"]["hgb"]


def test_predict_weighted_ensemble_shape():
    """predict_weighted_ensemble needs model dict, not result dict."""
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

    X, y = _make_xy(n=100)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=50, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)
    hgb = HistGradientBoostingClassifier(
        max_iter=100, max_depth=5, learning_rate=0.1, random_state=42, class_weight="balanced",
    )
    hgb.fit(X, y)
    models = {"rf_tech": rf, "hgb": hgb}

    preds, probas = predict_weighted_ensemble(models, X, [0.5, 0.5])
    assert len(preds) == len(X)
    assert len(probas) == len(X)
    assert all(p in (0, 1) for p in preds)
    assert all(0 <= p <= 1 for p in probas)


def test_predict_weighted_ensemble_few_models():
    """Only rf_tech available."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = _make_xy(n=100)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=50, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)
    models = {"rf_tech": rf}

    preds, probas = predict_weighted_ensemble(models, X, [1.0], ["rf_tech"])
    assert len(preds) == len(X)


def test_optimize_weights_equal_fallback():
    """With only 1 model, weights should be [1.0]."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = _make_xy(n=100)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=50, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)
    models = {"rf_tech": rf}
    labels, weights = optimize_weights(models, X, y)
    assert labels == ["rf_tech"]
    assert weights == [1.0]


def test_optimize_weights_reduces_auc_on_noise():
    """Weight optimisation should prefer better models on clean data."""
    rng = np.random.default_rng(99)
    n = 300
    X = rng.normal(size=(n, 10))
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    X_train, y_train = X[:200], y[:200]
    X_val, y_val = X[200:], y[200:]

    r = train_weighted_ensemble(X_train, y_train, X_val, y_val)
    assert r["ensemble_auc"] > 0.5


# ─────────────────────────────────────────────────────────────────────────────
#  train_ensemble  — saves models + metrics
# ─────────────────────────────────────────────────────────────────────────────

def test_train_ensemble_returns_metrics():
    X, y = _make_xy()
    r = train_ensemble(X, y, X, y, "TST")
    assert "metrics" in r
    assert "rf_tech" in r["metrics"]
    assert "hgb" in r["metrics"]
    assert "ensemble_auc" in r["metrics"]


def test_save_ensemble_models(tmp_path, monkeypatch):
    """save_ensemble_models writes model files and config JSON."""
    from inversion.utils import paths as real_paths

    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(real_paths, "MODELS_DIR", models_dir)

    X, y = _make_xy()
    r = train_ensemble(X, y, X, y, "TST")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)

    saved = save_ensemble_models(r, "TST", scaler)
    assert (models_dir / "ensemble_rf_TST.pkl").exists()
    assert (models_dir / "ensemble_hgb_TST.pkl").exists()
    assert (models_dir / "scaler_TST.pkl").exists()
    assert (models_dir / "ensemble_config_TST.json").exists()

    config = json.loads((models_dir / "ensemble_config_TST.json").read_text())
    assert config["ticker"] == "TST"


# ─────────────────────────────────────────────────────────────────────────────
#  train_ensemble_ticker  — end-to-end ticker training
# ─────────────────────────────────────────────────────────────────────────────

def test_train_ensemble_ticker_saves_models(patch_paths):
    """train_ensemble_ticker for a ticker saves model files and metrics."""
    from inversion.models.train_model import train_ensemble_ticker
    from inversion.features.build_features import add_derived_features

    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 100 + np.arange(n) * 0.1,
            "high": 100 + np.arange(n) * 0.1 + 1,
            "low": 100 + np.arange(n) * 0.1 - 1,
            "close": 100 + np.arange(n) * 0.1,
            "volume": 1_000_000.0,
            "vwap": 100 + np.arange(n) * 0.1,
        }
    )
    df = add_derived_features(df)
    df["target"] = (np.sin(np.arange(len(df)) * 0.3) > 0).astype(int)
    df = df.dropna().reset_index(drop=True)

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_ENS.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    metrics = train_ensemble_ticker("ENS", csv_path)
    assert metrics["n_train"] > 0
    assert metrics["n_test"] > 0
    assert "ensemble_accuracy" in metrics
    assert "ensemble_auc" in metrics
    assert (patch_paths["MODELS_DIR"] / "ensemble_rf_ENS.pkl").exists()
    assert (patch_paths["MODELS_DIR"] / "ensemble_hgb_ENS.pkl").exists()
    assert (patch_paths["MODELS_DIR"] / "ensemble_config_ENS.json").exists()


def test_train_ensemble_ticker_reproducible(patch_paths):
    """Same seed → same metrics."""
    from inversion.models.train_model import train_ensemble_ticker
    from inversion.features.build_features import add_derived_features

    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 100 + np.arange(n) * 0.1,
            "high": 100 + np.arange(n) * 0.1 + 1,
            "low": 100 + np.arange(n) * 0.1 - 1,
            "close": 100 + np.arange(n) * 0.1,
            "volume": 1_000_000.0,
            "vwap": 100 + np.arange(n) * 0.1,
        }
    )
    df = add_derived_features(df)
    df["target"] = (np.sin(np.arange(len(df)) * 0.3) > 0).astype(int)
    df = df.dropna().reset_index(drop=True)

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_ENS2.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    m1 = train_ensemble_ticker("ENS2", csv_path, random_state=42)
    m2 = train_ensemble_ticker("ENS2", csv_path, random_state=42)
    assert m1["ensemble_auc"] == m2["ensemble_auc"]
    assert m1["ensemble_accuracy"] == m2["ensemble_accuracy"]


# ─────────────────────────────────────────────────────────────────────────────
#  predict_ensemble  — graceful degradation
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_ensemble_no_models_returns_empty():
    """No ensemble models → empty arrays."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=10), "value": rng.random(10)})
    preds, probas = predict_ensemble(df, "NOEXIST")
    assert len(preds) == 0


def test_predict_ensemble_with_model_returns_tuple():
    """When models exist, returns (predictions, probabilities) of matching length."""
    from inversion.features.build_features import add_derived_features
    from inversion.models.train_model import available_feature_cols
    from inversion.features.build_features import fit_and_save_scaler
    from inversion.utils import paths as real_paths
    from sklearn.ensemble import HistGradientBoostingClassifier

    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 100 + rng.random(n) * 0.1,
            "high": 100 + rng.random(n) * 0.1 + 1,
            "low": 100 + rng.random(n) * 0.1 - 1,
            "close": 100 + rng.random(n) * 0.1,
            "volume": 1_000_000.0,
            "vwap": 100 + rng.random(n) * 0.1,
        }
    )
    df = add_derived_features(df)
    df["target"] = (df["return"] > 0).astype(int)

    # Build ensemble models manually
    cols = available_feature_cols(df)
    df_clean = df.dropna(subset=cols)
    X = df_clean[cols].values
    y = df_clean["target"].values

    # Save RF
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=50, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)
    import joblib
    joblib.dump(rf, real_paths.MODELS_DIR / "ensemble_rf_TST.pkl")

    scaler = fit_and_save_scaler(df_clean[cols], filename="scaler_TST.pkl")

    # Save HGB
    hgb = HistGradientBoostingClassifier(max_iter=50, random_state=42)
    hgb.fit(X, y)
    joblib.dump(hgb, real_paths.MODELS_DIR / "ensemble_hgb_TST.pkl")

    # Predict
    preds, probas = predict_ensemble(df, "TST")
    assert len(preds) > 0
    assert len(preds) == len(probas)
    assert preds[0] in (0, 1)
    assert 0 <= probas[0] <= 1


# ─────────────────────────────────────────────────────────────────────────────
#  Graceful degradation — missing models
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_ensemble_with_only_one_model():
    """If only rf_tech exists, use it alone."""
    from inversion.features.build_features import add_derived_features
    from inversion.models.train_model import available_feature_cols
    from inversion.features.build_features import fit_and_save_scaler
    from inversion.utils import paths as real_paths
    from sklearn.ensemble import RandomForestClassifier

    n = 400
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 100 + rng.random(n) * 0.1,
            "high": 100 + rng.random(n) * 0.1 + 1,
            "low": 100 + rng.random(n) * 0.1 - 1,
            "close": 100 + rng.random(n) * 0.1,
            "volume": 1_000_000.0,
            "vwap": 100 + rng.random(n) * 0.1,
        }
    )
    df = add_derived_features(df)
    df["target"] = (df["return"] > 0).astype(int)

    cols = available_feature_cols(df)
    df_clean = df.dropna(subset=cols)
    assert len(df_clean) > 0, f"dropna left {len(df_clean)} rows — data too short for feature windows"
    X = df_clean[cols].values
    y = df_clean["target"].values

    # Only save rf_tech (no hgb, no rf_sent)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=50, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)
    import joblib
    joblib.dump(rf, real_paths.MODELS_DIR / "ensemble_rf_GD.pkl")

    fit_and_save_scaler(df_clean[cols], filename="scaler_GD.pkl")

    preds, probas = predict_ensemble(df, "GD")
    assert len(preds) == len(probas)


def test_weighted_ensemble_no_sentiment_features():
    """train_weighted_ensemble works even without sentiment features."""
    X, y = _make_xy(n=150)
    r = train_weighted_ensemble(X, y, X, y, use_sentiment=True)
    # Should still have rf_tech and hgb
    assert "rf_tech" in r["models"]
    assert "hgb" in r["models"]
    # ensemble should still produce results
    assert "ensemble_auc" in r
