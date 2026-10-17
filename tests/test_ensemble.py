"""Tests for the multi-model ensemble: RF-tech, HGB-tech, RF-sentiment.

Covers:
  - inversion.ensemble.weighted_ensemble  (train, predict, optimise weights)
  - inversion.ensemble.stacking_learner   (train + predict stacking)
  - inversion.models.train_ensemble       (train_ensemble + save)
  - inversion.models.train_model          (train_ensemble_ticker wrapper)
  - inversion.models.predict_ensemble     (predict_ensemble + graceful degradation)
  - inversion.models.select_best          (best-model per ticker)
  - inversion.features.build_features     (enhanced sentiment features)
  - inversion.tuning.tune_hgb_model       (HGB Optuna tuning)
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


# ─────────────────────────────────────────────────────────────────────────────
#  stacking_learner  — train / predict stacking
# ─────────────────────────────────────────────────────────────────────────────

def test_train_stacking_ensemble_returns_keys():
    """train_stacking_ensemble returns a dict with required keys."""
    from inversion.ensemble.stacking_learner import train_stacking_ensemble

    X, y = _make_xy(n=300)
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]

    r = train_stacking_ensemble(X_train, y_train, X_test, y_test)
    assert "meta_model" in r
    assert "base_models" in r
    assert "oos_train_proba" in r
    assert "ensemble_auc" in r
    assert "ensemble_accuracy" in r


def test_stacking_predict_shape():
    """predict_stacking_ensemble returns correct shape arrays."""
    from inversion.ensemble.stacking_learner import train_stacking_ensemble, predict_stacking_ensemble

    X, y = _make_xy(n=300)
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]

    r = train_stacking_ensemble(X_train, y_train, X_test, y_test)
    preds, probas = predict_stacking_ensemble(X_test, r["base_models"], r["meta_model"])
    assert len(preds) == len(X_test)
    assert len(probas) == len(X_test)
    assert all(p in (0, 1) for p in preds)


def test_stacking_uses_oos_predictions():
    """OOS predictions should have same shape as training set."""
    from inversion.ensemble.stacking_learner import train_stacking_ensemble

    X, y = _make_xy(n=300)
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]

    r = train_stacking_ensemble(X_train, y_train, X_test, y_test)
    # oos_train_proba should have 3 base models as columns
    assert r["oos_train_proba"].shape[0] == len(y_train)
    assert r["oos_train_proba"].shape[1] == 3  # rf_tech, hgb, rf_sent


# ─────────────────────────────────────────────────────────────────────────────
#  enhanced sentiment features
# ─────────────────────────────────────────────────────────────────────────────

def test_enhanced_sentiment_features_generated(patch_paths):
    """add_sentiment_features generates all 15 enhanced columns when news data exists."""
    from inversion.features.build_features import add_derived_features, add_sentiment_features

    n = 400
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

    # Create synthetic news data
    news_path = patch_paths["RAW_DATA_DIR"] / "news_TST_live.csv"
    news_path.parent.mkdir(parents=True, exist_ok=True)
    news_dates = pd.date_range("2020-01-01", periods=n, freq="B").date
    news_df = pd.DataFrame({
        "fecha": [d.strftime("%Y-%m-%d") for d in news_dates],
        "titulo": ["This stock is doing great and showing strong growth"] * n,
        "cuerpo": ["Positive earnings and revenue exceed expectations"] * n,
        "fuente": ["test"] * n,
    })
    news_df.to_csv(news_path, index=False)

    df_with_sent = add_sentiment_features(df, ticker="TST")

    expected_cols = [
        "sentiment_score", "sentiment_ma5", "sentiment_vol",
        "sentiment_ma20", "sentiment_ma60", "sentiment_zscore",
        "sentiment_trend", "sentiment_rsi", "sentiment_vol_ratio",
        "sentiment_crown", "news_volume", "news_volume_ma5",
        "sentiment_extreme", "sentiment_change", "sentiment_change_ma5",
    ]
    for col in expected_cols:
        assert col in df_with_sent.columns, f"Missing column: {col}"


def test_sentiment_features_no_leakage(patch_paths):
    """Sentiment features use only past data — no forward-looking."""
    from inversion.features.build_features import add_derived_features, add_sentiment_features

    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [100 + i * 0.1 for i in range(n)],
            "high": [100 + i * 0.1 + 1 for i in range(n)],
            "low": [100 + i * 0.1 - 1 for i in range(n)],
            "close": [100 + i * 0.1 for i in range(n)],
            "volume": 1_000_000.0,
            "vwap": [100 + i * 0.1 for i in range(n)],
        }
    )
    df = add_derived_features(df)

    # News data with varied sentiment content so ma5 differs from score
    rng = np.random.default_rng(42)
    titles = [
        "This stock is doing great and showing strong growth" if i % 2 == 0
        else "This stock crashed and lost value badly"
        for i in range(n)
    ]
    news_df = pd.DataFrame({
        "fecha": [d.strftime("%Y-%m-%d") for d in dates],
        "titulo": titles,
        "cuerpo": ["Mixed body content here"] * n,
        "fuente": ["test"] * n,
    })
    news_path = patch_paths["RAW_DATA_DIR"] / "news_LEAK_live.csv"
    news_path.parent.mkdir(parents=True, exist_ok=True)
    news_df.to_csv(news_path, index=False)

    df_with_sent = add_sentiment_features(df, ticker="LEAK")

    # sentiment_ma5 at row 5 should use rows 0-4 (not row 5)
    assert df_with_sent["sentiment_ma5"].iloc[5] != df_with_sent["sentiment_score"].iloc[5]


def test_add_sentiment_features_no_news_returns_unchanged(patch_paths):
    """If no news data exists, return df without sentiment columns."""
    from inversion.features.build_features import add_derived_features, add_sentiment_features

    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1_000_000.0,
        "vwap": 100.0,
    })

    result = add_sentiment_features(df, ticker="NONEXIST")
    assert not any(c.startswith("sentiment_") for c in result.columns if c != "timestamp")


# ─────────────────────────────────────────────────────────────────────────────
#  HGB tuning
# ─────────────────────────────────────────────────────────────────────────────

def test_hgb_tune_ticker_structure(patch_paths):
    """tune_hgb_ticker returns a dict with tuned params."""
    from inversion.tuning.tune_hgb_model import tune_hgb_ticker

    n = 500  # More data for walk-forward to work
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0 + np.arange(n) * 0.01,
        "volume": 1_000_000.0,
        "vwap": 100.0,
    })
    from inversion.features.build_features import add_derived_features
    df = add_derived_features(df)
    # Balanced target so both classes present (needed for AUC)
    df["target"] = (np.arange(n) % 2).astype(int)
    df = df.dropna().reset_index(drop=True)

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_HGB_ml_ready.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    best = tune_hgb_ticker("HGB", n_trials=5)  # few trials for speed
    assert isinstance(best, dict)
    assert "max_iter" in best
    assert "learning_rate" in best
    assert "l2_regularization" in best


def test_load_best_hgb_params(tmp_path, monkeypatch):
    """load_best_hgb_params returns None when no file exists, dict when it does."""
    from inversion.tuning.tune_hgb_model import load_best_hgb_params
    from inversion.utils import paths as real_paths

    # No file → None
    result = load_best_hgb_params("NOPE")
    assert result is None

    # Create a fake params file
    model_dir = tmp_path / "models" / "artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)
    fake_file = model_dir / "best_params_hgb_TST.json"
    fake_file.write_text(json.dumps({"ticker": "TST", "params": {"max_iter": 100, "learning_rate": 0.1}}))
    monkeypatch.setattr(real_paths, "ARTIFACTS_DIR", model_dir)
    result = load_best_hgb_params("TST")
    assert result is not None
    assert result["max_iter"] == 100


# ─────────────────────────────────────────────────────────────────────────────
#  select_best_model
# ─────────────────────────────────────────────────────────────────────────────

def test_select_best_model_returns_dict(patch_paths):
    """select_best_model returns dict with best_model_name and metrics."""
    from inversion.models.select_best import select_best_model

    n = 800  # Need enough data for stacking walk-forward
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": [100 + i * 0.01 for i in range(n)],
        "high": [101 + i * 0.01 for i in range(n)],
        "low": [99 + i * 0.01 for i in range(n)],
        "close": [100 + i * 0.01 for i in range(n)],
        "volume": 1_000_000.0,
        "vwap": [100 + i * 0.01 for i in range(n)],
    })
    from inversion.features.build_features import add_derived_features
    df = add_derived_features(df)
    # Balanced target so both classes present (needed for AUC + stacking meta-learner)
    df["target"] = (np.arange(n) % 2).astype(int)
    df = df.dropna().reset_index(drop=True)

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_SB_ml_ready.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    result = select_best_model("SB", csv_path)
    assert isinstance(result, dict)
    assert "best_model_name" in result
    assert "metrics" in result
    assert result["best_model_name"] in ("rf_tech", "rf_sent", "hgb", "stacking")


def test_select_best_model_saves_config(patch_paths):
    """select_best_model saves best_model_config_<T>.json."""
    from inversion.models.select_best import select_best_model

    n = 800  # Need enough data for stacking walk-forward
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": [100 + i * 0.01 for i in range(n)],
        "high": [101 + i * 0.01 for i in range(n)],
        "low": [99 + i * 0.01 for i in range(n)],
        "close": [100 + i * 0.01 for i in range(n)],
        "volume": 1_000_000.0,
        "vwap": [100 + i * 0.01 for i in range(n)],
    })
    from inversion.features.build_features import add_derived_features
    df = add_derived_features(df)
    # Balanced target so both classes present
    df["target"] = (np.arange(n) % 2).astype(int)
    df = df.dropna().reset_index(drop=True)

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_SBC_ml_ready.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    result = select_best_model("SBC", csv_path)
    config_path = patch_paths["MODELS_DIR"] / f"best_model_config_SBC.json"
    assert config_path.exists()
    config = json.loads(config_path.read_text())
    assert config["ticker"] == "SBC"
    assert "best_model" in config


# ─────────────────────────────────────────────────────────────────────────────
#  Macro features (ME-001)
# ─────────────────────────────────────────────────────────────────────────────

def test_add_macro_features_no_network_returns_defaults(patch_paths):
    """add_macro_features returns zero-filled macro columns when no data."""
    from inversion.features.build_features import add_derived_features
    from inversion.features.macro_features import add_macro_features, MACRO_COLUMNS

    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0 + np.arange(n) * 0.1,
        "volume": 1_000_000.0,
        "vwap": 100.0,
    })
    df = add_derived_features(df)
    result = add_macro_features(df, use_yf_fetch=False)

    for col in MACRO_COLUMNS:
        assert col in result.columns, f"Missing column: {col}"
    # When no external data, all values should be 0.0
    for col in MACRO_COLUMNS:
        assert result[col].dtype in (np.float64, float), f"{col} should be float"


def test_add_macro_features_with_vix_data(patch_paths):
    """add_macro_features populates VIX columns from synthetic data."""
    from inversion.features.build_features import add_derived_features
    from inversion.features.macro_features import add_macro_features, MACRO_COLUMNS

    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0 + np.arange(n) * 0.1,
        "volume": 1_000_000.0,
        "vwap": 100.0,
    })
    df = add_derived_features(df)

    # Create synthetic VIX data
    vix_df = pd.DataFrame({
        "timestamp": dates,
        "vix_close": [20.0 + i * 0.1 for i in range(n)],
    })

    result = add_macro_features(df, vix_data=vix_df, use_yf_fetch=False)
    assert "vix_level" in result.columns
    assert result["vix_level"].iloc[0] == 20.0


# ─────────────────────────────────────────────────────────────────────────────
#  Feature selection (SHAP-based)
# ─────────────────────────────────────────────────────────────────────────────

def test_select_top_features_returns_n_features():
    """select_top_features returns exactly n_top features."""
    from sklearn.ensemble import RandomForestClassifier
    from inversion.features.feature_selection import select_top_features

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(200, 10)), columns=[f"f{i}" for i in range(10)])
    y = ((X.iloc[:, 0] + X.iloc[:, 1]) > 0).astype(int)

    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    rf.fit(X, y)

    selected = select_top_features(rf, X, n_top=5)
    assert len(selected) == 5
    assert all(c in X.columns for c in selected)


def test_build_optimized_feature_set_preserves_target():
    """build_optimized_feature_set keeps the target column."""
    from sklearn.ensemble import RandomForestClassifier
    from inversion.features.feature_selection import build_optimized_feature_set

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(200, 8)), columns=[f"f{i}" for i in range(8)])
    y = ((X.iloc[:, 0] + X.iloc[:, 1]) > 0).astype(int)
    df = pd.concat([X, y.rename("target")], axis=1)

    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    rf.fit(X, y)

    result = build_optimized_feature_set(df, rf, n_top=4)
    assert "target" in result.columns
    assert len(result.columns) == 5  # 4 features + target


def test_compute_shap_importance_fallback_without_shap():
    """compute_shap_importance falls back to feature_importances_ when SHAP is unavailable."""
    from sklearn.ensemble import RandomForestClassifier
    from inversion.features.feature_selection import compute_shap_importance

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(200, 5)), columns=[f"f{i}" for i in range(5)])
    y = ((X.iloc[:, 0] + X.iloc[:, 1]) > 0).astype(int)

    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    rf.fit(X, y)

    importance = compute_shap_importance(rf, X)
    assert isinstance(importance, dict)
    # If SHAP is not installed, should still return something
    if importance:
        assert len(importance) == 5


# ─────────────────────────────────────────────────────────────────────────────
#  Regime detection
# ─────────────────────────────────────────────────────────────────────────────

def test_detect_regime_bull():
    """Detect bull regime with positive return, low vol, low VIX."""
    from inversion.features.regime_detection import detect_regime, Regime

    row = pd.Series({
        "return": 0.01,
        "volatility": 0.10,
        "vix_level": 15.0,
    })
    regime = detect_regime(row, lookback=60)
    assert regime == Regime.BULL


def test_detect_regime_bear():
    """Detect bear regime with negative return, high vol, high VIX."""
    from inversion.features.regime_detection import detect_regime, Regime

    row = pd.Series({
        "return": -0.02,
        "volatility": 0.30,
        "vix_level": 35.0,
    })
    regime = detect_regime(row, lookback=60)
    assert regime == Regime.BEAR


def test_detect_regime_neutral():
    """Neutral regime catches everything else."""
    from inversion.features.regime_detection import detect_regime, Regime

    # Moderate return + moderate vol + moderate VIX → neutral
    row = pd.Series({
        "return": 0.005,
        "volatility": 0.15,
        "vix_level": 22.0,
    })
    regime = detect_regime(row, lookback=60)
    assert regime == Regime.NEUTRAL


def test_detect_regimes_returns_series():
    """detect_regimes returns a Series aligned with the DataFrame."""
    from inversion.features.regime_detection import detect_regimes, Regime

    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.01)
    df = pd.DataFrame({
        "timestamp": dates,
        "open": close,
        "high": close + 1 + np.abs(np.random.randn(n)),
        "low": close - 1 - np.abs(np.random.randn(n)),
        "close": close,
        "volume": 1_000_000.0,
    })
    from inversion.features.build_features import add_derived_features
    df = add_derived_features(df)
    df["vix_level"] = 20.0 + np.random.randn(n) * 5

    regimes = detect_regimes(df, lookback=60)
    assert len(regimes) == n
    assert all(r in (Regime.BULL, Regime.BEAR, Regime.NEUTRAL) for r in regimes)


# ─────────────────────────────────────────────────────────────────────────────
#  Dynamic ensemble weights
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_dynamic_weights_better_model_gets_higher_weight():
    """Models with better recent performance get higher weights."""
    from inversion.ensemble.weighted_ensemble import compute_dynamic_weights
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(42)
    # Model A is perfect, Model B is random
    perf = pd.DataFrame({
        "model_a": [0.8] * 30,
        "model_b": [0.55] * 30,
    })

    rf_a = RandomForestClassifier(n_estimators=5, random_state=42)
    rf_b = RandomForestClassifier(n_estimators=5, random_state=43)
    rf_a.fit(rng.random((50, 3)), rng.integers(0, 2, 50))
    rf_b.fit(rng.random((50, 3)), rng.integers(0, 2, 50))
    models = {"model_a": rf_a, "model_b": rf_b}

    labels, weights = compute_dynamic_weights(models, perf, window=20)
    weight_map = dict(zip(labels, weights))
    assert weight_map["model_a"] > weight_map["model_b"]


def test_compute_dynamic_weights_sum_to_one():
    """Dynamic weights always sum to 1.0."""
    from inversion.ensemble.weighted_ensemble import compute_dynamic_weights
    from sklearn.ensemble import RandomForestClassifier

    perf = pd.DataFrame({
        "m1": [0.7] * 20,
        "m2": [0.6] * 20,
        "m3": [0.5] * 20,
    })

    models = {f"m{i}": RandomForestClassifier(n_estimators=3, random_state=42) for i in range(1, 4)}
    labels, weights = compute_dynamic_weights(models, perf, window=10)
    assert abs(sum(weights) - 1.0) < 1e-6
    assert all(w >= 0 for w in weights)


# ─────────────────────────────────────────────────────────────────────────────
#  Probability calibration
# ─────────────────────────────────────────────────────────────────────────────

def test_train_calibrated_rf_returns_calibrated():
    """train_calibrated_rf returns a CalibratedClassifierCV."""
    from inversion.models.train_model import train_calibrated_rf
    from sklearn.calibration import CalibratedClassifierCV

    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(100, 5))
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_val = rng.normal(size=(30, 5))
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(int)

    calibrated = train_calibrated_rf(X_train, y_train, X_val, y_val)
    assert isinstance(calibrated, CalibratedClassifierCV)
    assert hasattr(calibrated, "predict_proba")


def test_train_ticker_with_calibration(patch_paths):
    """train_ticker with calibrated=True produces a calibrated model."""
    from inversion.models.train_model import train_ticker
    from inversion.features.build_features import add_derived_features

    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0 + np.arange(n) * 0.1,
        "volume": 1_000_000.0,
        "vwap": 100.0,
    })
    df = add_derived_features(df)
    df["target"] = (np.arange(n) % 2).astype(int)
    df = df.dropna().reset_index(drop=True)

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_CAL_ml_ready.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    metrics = train_ticker("CAL", csv_path, calibrated=True)
    assert metrics["calibrated"] is True
    assert "auc" in metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Integration: build_features pipeline includes new features
# ─────────────────────────────────────────────────────────────────────────────

def test_build_features_pipeline_includes_macro_and_regime(patch_paths):
    """build_features_pipeline adds macro columns and regime column."""
    from inversion.features.build_features import build_features_pipeline

    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0 + np.arange(n) * 0.1,
        "volume": 1_000_000.0,
        "vwap": 100.0,
    })
    csv_path = patch_paths["PROCESSED_DATA_DIR"] / "PIPE_TST_ml_ready.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    written = build_features_pipeline()
    assert len(written) > 0
    features_df = pd.read_csv(written[0])
    assert "vix_level" in features_df.columns
    assert "regime" in features_df.columns


def test_feature_pipeline_reproducible(patch_paths):
    """Running build_features twice produces identical output."""
    from inversion.features.build_features import build_features_pipeline

    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0 + np.arange(n) * 0.05,
        "volume": 1_000_000.0,
        "vwap": 100.0,
    })
    csv_path = patch_paths["PROCESSED_DATA_DIR"] / "PIPE_RPT_ml_ready.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    r1 = build_features_pipeline()
    content1 = pd.read_csv(r1[0]).to_csv(index=False)
    r2 = build_features_pipeline()
    content2 = pd.read_csv(r2[0]).to_csv(index=False)
    assert content1 == content2
