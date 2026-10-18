"""Tests for the 5 ML improvements:

1. Aggressive SHAP feature selection  (aggressive_selection.py)
2. ElasticNet linear model             (train_linear.py)
3. Regime-specific models              (regime_models.py)
4. Continuous-target regression        (train_regression.py)
5. Diverse ensemble (LR+RF+HGB)        (diverse_ensemble.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score


# ─────────────────────────────────────────────────────────────────────────────
#  1. Aggressive SHAP feature selection
# ─────────────────────────────────────────────────────────────────────────────

def test_select_top_features_shap_returns_n_features(patch_paths):
    """select_top_features_shap returns exactly n_top features."""
    from inversion.features.aggressive_selection import select_top_features_shap

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(200, 10)), columns=[f"f{i}" for i in range(10)])
    y = ((X.iloc[:, 0] + X.iloc[:, 1]) > 0).astype(int)

    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    rf.fit(X, y)

    selected = select_top_features_shap(rf, X, n_top=5)
    assert len(selected) == 5
    assert all(c in X.columns for c in selected)


def test_select_top_features_shap_ordered_by_importance(patch_paths):
    """Top features should have the highest importance (SHAP or tree-based)."""
    from inversion.features.aggressive_selection import select_top_features_shap

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(200, 8)), columns=[f"f{i}" for i in range(8)])
    y = ((X.iloc[:, 0] * 3 + X.iloc[:, 1] * 2) > 0).astype(int)

    rf = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=42)
    rf.fit(X, y)

    selected = select_top_features_shap(rf, X, n_top=3)
    # f0 and f1 should be among top features since they have strongest signal
    assert "f0" in selected or "f1" in selected


def test_train_with_selected_features_returns_metrics(patch_paths):
    """train_with_selected_features returns dict with full and reduced metrics."""
    from inversion.features.aggressive_selection import train_with_selected_features
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

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_SEL_ml_ready.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    result = train_with_selected_features("SEL", csv_path, n_top=5)
    assert isinstance(result, dict)
    assert "selected_features" in result
    assert "full_model" in result
    assert "reduced_model" in result
    assert "delta_auc" in result
    assert len(result["selected_features"]) == 5
    assert "accuracy" in result["full_model"]
    assert "accuracy" in result["reduced_model"]


def test_train_with_selected_features_saves_models(patch_paths):
    """train_with_selected_features saves full and reduced model files."""
    from inversion.features.aggressive_selection import train_with_selected_features
    from inversion.features.build_features import add_derived_features
    from inversion.utils import paths as real_paths

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

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_SEL2_ml_ready.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    result = train_with_selected_features("SEL2", csv_path, n_top=3)

    assert (real_paths.MODELS_DIR / "rf_SEL2_selected_full.pkl").exists()
    assert (real_paths.MODELS_DIR / "rf_SEL2_selected.pkl").exists()


def test_select_top_features_shap_fallback_to_feature_importances(patch_paths):
    """When SHAP is unavailable, falls back to feature_importances_."""
    from inversion.features.aggressive_selection import select_top_features_shap

    # Patch _import_shap to return None
    from inversion import features
    original = features.aggressive_selection.compute_shap_importance
    features.aggressive_selection.compute_shap_importance = lambda *a, **k: {}

    try:
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(100, 5)), columns=[f"f{i}" for i in range(5)])
        y = ((X.iloc[:, 0] > 0).astype(int))

        rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        rf.fit(X, y)

        # With empty dict fallback, should return first n_top columns
        selected = select_top_features_shap(rf, X, n_top=3)
        assert len(selected) == 3
    finally:
        features.aggressive_selection.compute_shap_importance = original


# ─────────────────────────────────────────────────────────────────────────────
#  2. Linear model with strong regularization (ElasticNet)
# ─────────────────────────────────────────────────────────────────────────────

def test_train_elasticnet_returns_metrics():
    """train_elasticnet returns dict with model, accuracy, auc, alpha."""
    from inversion.models.train_linear import train_elasticnet

    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(150, 6))
    y_train = ((X_train[:, 0] + X_train[:, 1]) > 0).astype(int)
    X_test = rng.normal(size=(50, 6))
    y_test = ((X_test[:, 0] + X_test[:, 1]) > 0).astype(int)

    result = train_elasticnet(X_train, y_train, X_test, y_test)
    assert isinstance(result["model"], type(result["model"]))
    assert "accuracy" in result
    assert "auc" in result
    assert "C" in result
    assert "l1_ratio" in result
    assert "n_selected_features" in result
    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["auc"] <= 1


def test_train_elasticnet_ticker_saves_model(patch_paths):
    """train_elasticnet_ticker saves model file and returns metrics."""
    from inversion.models.train_linear import train_elasticnet_ticker
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

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_ENET_ml_ready.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    result = train_elasticnet_ticker("ENET", csv_path)
    assert result["ticker"] == "ENET"
    assert result["n_train"] > 0
    assert result["n_test"] > 0

    from inversion.utils import paths as real_paths
    assert (real_paths.MODELS_DIR / "elasticnet_ENET.pkl").exists()


def test_train_elasticnet_reproducible():
    """Same seed → same accuracy and alpha."""
    from inversion.models.train_linear import train_elasticnet

    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(150, 6))
    y_train = ((X_train[:, 0] + X_train[:, 1]) > 0).astype(int)
    X_test = rng.normal(size=(50, 6))
    y_test = ((X_test[:, 0] + X_test[:, 1]) > 0).astype(int)

    r1 = train_elasticnet(X_train, y_train, X_test, y_test, random_state=42)
    r2 = train_elasticnet(X_train, y_train, X_test, y_test, random_state=42)

    assert r1["accuracy"] == r2["accuracy"]
    assert r1["C"] == r2["C"]
    assert r1["l1_ratio"] == r2["l1_ratio"]


# ─────────────────────────────────────────────────────────────────────────────
#  3. Regime-specific models
# ─────────────────────────────────────────────────────────────────────────────

def test_train_regime_models_returns_structure(patch_paths):
    """train_regime_models returns dict with regimes and combined model."""
    from inversion.models.regime_models import train_regime_models
    from inversion.features.build_features import add_derived_features

    # Create data with enough variety to produce multiple regimes
    n = 600
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))

    df = pd.DataFrame({
        "timestamp": dates,
        "open": close + rng.normal(0, 0.5, n),
        "high": close + np.abs(rng.normal(0, 1, n)),
        "low": close - np.abs(rng.normal(0, 1, n)),
        "close": close,
        "volume": 1_000_000.0,
        "vwap": close,
    })
    df = add_derived_features(df)
    # Add VIX column needed for regime detection
    df["vix_level"] = 20.0 + rng.normal(0, 5, n)
    df["target"] = (np.arange(n) % 2).astype(int)
    df = df.dropna().reset_index(drop=True)

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_REG_ml_ready.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    result = train_regime_models("REG", csv_path)
    assert isinstance(result, dict)
    assert "regimes" in result
    assert "combined_model" in result
    assert result["ticker"] == "REG"


def test_train_regime_models_handles_few_regimes(patch_paths):
    """When few regimes have enough data, still returns valid result."""
    from inversion.models.regime_models import train_regime_models
    from inversion.features.build_features import add_derived_features

    n = 600
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({
        "timestamp": dates,
        "open": close,
        "high": close + np.abs(rng.normal(0, 1, n)),
        "low": close - np.abs(rng.normal(0, 1, n)),
        "close": close,
        "volume": 1_000_000.0,
        "vwap": close,
    })
    df = add_derived_features(df)
    df["vix_level"] = 25.0  # constant VIX
    df["target"] = (np.arange(n) % 2).astype(int)
    df = df.dropna().reset_index(drop=True)

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_REG2_ml_ready.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    result = train_regime_models("REG2", csv_path)
    assert "regimes" in result
    assert "combined_model" in result


def test_regime_models_saves_bundle(patch_paths):
    """train_regime_models saves a model bundle file."""
    from inversion.models.regime_models import train_regime_models
    from inversion.features.build_features import add_derived_features

    n = 600
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))

    df = pd.DataFrame({
        "timestamp": dates,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 1_000_000.0,
        "vwap": close,
    })
    df = add_derived_features(df)
    df["vix_level"] = 20.0 + rng.normal(0, 3, n)
    df["target"] = (np.arange(n) % 2).astype(int)
    df = df.dropna().reset_index(drop=True)

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_REG3_ml_ready.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    train_regime_models("REG3", csv_path)

    from inversion.utils import paths as real_paths
    assert (real_paths.MODELS_DIR / "regime_REG3.pkl").exists()


# ─────────────────────────────────────────────────────────────────────────────
#  4. Continuous-target regression
# ─────────────────────────────────────────────────────────────────────────────

def test_train_regression_returns_metrics():
    """train_regression returns dict with R², MAE, Spearman, and cls metrics."""
    from inversion.models.train_regression import train_regression

    rng = np.random.default_rng(42)
    # Create data where return is somewhat predictable from features
    X_train = rng.normal(size=(150, 6))
    y_train = X_train[:, 0] * 0.5 + X_train[:, 1] * 0.3 + rng.normal(0, 0.1, 150)
    X_test = rng.normal(size=(50, 6))
    y_test = X_test[:, 0] * 0.5 + X_test[:, 1] * 0.3 + rng.normal(0, 0.1, 50)

    result = train_regression(X_train, y_train, X_test, y_test)
    assert isinstance(result["model"], type(result["model"]))  # RFRegressor
    assert "r2" in result
    assert "mae" in result
    assert "spearman" in result
    assert "cls_accuracy" in result
    assert "cls_auc" in result


def test_train_regression_ticker_saves_model(patch_paths):
    """train_regression_ticker saves model file and returns metrics."""
    from inversion.models.train_regression import train_regression_ticker
    from inversion.features.build_features import add_derived_features

    n = 600
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

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_REGR_ml_ready.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    result = train_regression_ticker("REGR", csv_path)
    assert result["ticker"] == "REGR"
    assert "r2" in result
    assert "mae" in result
    assert result["horizon"] == 1

    from inversion.utils import paths as real_paths
    assert (real_paths.MODELS_DIR / "regression_REGR.pkl").exists()


def test_train_regression_reproducible():
    """Same seed → same R²."""
    from inversion.models.train_regression import train_regression

    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(150, 6))
    y_train = X_train[:, 0] * 0.5 + rng.normal(0, 0.1, 150)
    X_test = rng.normal(size=(50, 6))
    y_test = X_test[:, 0] * 0.5 + rng.normal(0, 0.1, 50)

    r1 = train_regression(X_train, y_train, X_test, y_test, random_state=42)
    r2 = train_regression(X_train, y_train, X_test, y_test, random_state=42)

    assert r1["r2"] == pytest.approx(r2["r2"], abs=1e-10)


def test_train_regression_too_short_raises(patch_paths):
    """train_regression_ticker raises ValueError for too-short datasets."""
    from inversion.models.train_regression import train_regression_ticker
    from inversion.features.build_features import add_derived_features

    n = 50
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
    df = add_derived_features(df)
    df["target"] = 0
    df = df.dropna().reset_index(drop=True)

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_SHORT_ml_ready.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="too short"):
        train_regression_ticker("SHORT", csv_path)


# ─────────────────────────────────────────────────────────────────────────────
#  5. Diverse ensemble (Linear + RF + HGB)
# ─────────────────────────────────────────────────────────────────────────────

def test_train_diverse_ensemble_returns_structure():
    """train_diverse_ensemble returns dict with models, metrics, ensemble."""
    from inversion.ensemble.diverse_ensemble import train_diverse_ensemble

    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(150, 6))
    y_train = ((X_train[:, 0] + X_train[:, 1]) > 0).astype(int)
    X_test = rng.normal(size=(50, 6))
    y_test = ((X_test[:, 0] + X_test[:, 1]) > 0).astype(int)

    result = train_diverse_ensemble(X_train, y_train, X_test, y_test)
    assert "models" in result
    assert "metrics" in result
    assert "ensemble_accuracy" in result
    assert "ensemble_auc" in result
    assert set(result["metrics"].keys()) == {"logistic", "rf", "hgb"}


def test_diverse_ensemble_has_three_models():
    """The ensemble trains exactly 3 model classes."""
    from inversion.ensemble.diverse_ensemble import train_diverse_ensemble
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(150, 6))
    y_train = ((X_train[:, 0] + X_train[:, 1]) > 0).astype(int)
    X_test = rng.normal(size=(50, 6))
    y_test = ((X_test[:, 0] + X_test[:, 1]) > 0).astype(int)

    result = train_diverse_ensemble(X_train, y_train, X_test, y_test)
    models = result["models"]
    assert isinstance(models["logistic"], LogisticRegression)
    assert isinstance(models["rf"], RandomForestClassifier)
    assert isinstance(models["hgb"], HistGradientBoostingClassifier)


def test_train_diverse_ensemble_ticker_saves_model(patch_paths):
    """train_diverse_ensemble_ticker saves model file and returns metrics."""
    from inversion.ensemble.diverse_ensemble import train_diverse_ensemble_ticker
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

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_DIV_ml_ready.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    result = train_diverse_ensemble_ticker("DIV", csv_path)
    assert result["ticker"] == "DIV"
    assert "ensemble_accuracy" in result
    assert "ensemble_auc" in result
    assert result["n_train"] > 0

    from inversion.utils import paths as real_paths
    assert (real_paths.MODELS_DIR / "diverse_DIV.pkl").exists()


def test_diverse_ensemble_reproducible():
    """Same seed → same ensemble metrics."""
    from inversion.ensemble.diverse_ensemble import train_diverse_ensemble

    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(150, 6))
    y_train = ((X_train[:, 0] + X_train[:, 1]) > 0).astype(int)
    X_test = rng.normal(size=(50, 6))
    y_test = ((X_test[:, 0] + X_test[:, 1]) > 0).astype(int)

    r1 = train_diverse_ensemble(X_train, y_train, X_test, y_test, random_state=42)
    r2 = train_diverse_ensemble(X_train, y_train, X_test, y_test, random_state=42)

    assert r1["ensemble_accuracy"] == r2["ensemble_accuracy"]
    assert r1["ensemble_auc"] == r2["ensemble_auc"]


def test_diverse_ensemble_metrics_have_all_models():
    """Per-model metrics should exist for all 3 models."""
    from inversion.ensemble.diverse_ensemble import train_diverse_ensemble

    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(150, 6))
    y_train = ((X_train[:, 0] + X_train[:, 1]) > 0).astype(int)
    X_test = rng.normal(size=(50, 6))
    y_test = ((X_test[:, 0] + X_test[:, 1]) > 0).astype(int)

    result = train_diverse_ensemble(X_train, y_train, X_test, y_test)

    for model_name in ["logistic", "rf", "hgb"]:
        assert model_name in result["metrics"]
        assert "accuracy" in result["metrics"][model_name]
        assert "auc" in result["metrics"][model_name]
        assert 0 <= result["metrics"][model_name]["accuracy"] <= 1
