"""Tests para train_lightgbm y rf_lgb_ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inversion.ensemble.rf_lgb_ensemble import (
    RF_PARAMS,
    _temporal_cv_weight,
    train_rf_lgb_ensemble,
)
from inversion.models.train_lightgbm import LGB_PARAMS, train_lgb_model, train_lgb_ticker


def _make_dataset(n: int = 600, seed: int = 42) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    from inversion.models.train_model import FEATURE_COLS

    # Usar las primeras columnas de FEATURE_COLS para que available_feature_cols las reconozca
    cols = FEATURE_COLS[:5]
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="B"),
            "close": 100 + np.cumsum(rng.normal(0, 1, n)),
        }
    )
    for c in cols:
        df[c] = rng.normal(0, 1, n)
    df["target"] = (df["close"].shift(-5) / df["close"] - 1 > 0.0).astype(int)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    return df, cols


class TestTrainLgbModel:
    def test_returns_fitted_model(self):
        df, feats = _make_dataset()
        X, y = df[feats].values, df["target"]
        model = train_lgb_model(X, y)
        assert hasattr(model, "predict_proba")
        assert model.predict(X).shape == (len(X),)

    def test_predict_proba_sums_to_one(self):
        df, feats = _make_dataset()
        X, y = df[feats].values, df["target"]
        model = train_lgb_model(X, y)
        probs = model.predict_proba(X)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_params_override(self):
        df, feats = _make_dataset()
        X, y = df[feats].values, df["target"]
        model = train_lgb_model(X, y, params={"n_estimators": 10})
        assert model.n_estimators == 10

    def test_default_params(self):
        assert LGB_PARAMS["random_state"] == 42
        assert LGB_PARAMS["verbose"] == -1
        assert LGB_PARAMS["n_estimators"] == 200


class TestTrainLgbTicker:
    def test_returns_expected_keys(self, tmp_path):
        df, feats = _make_dataset()
        csv_path = tmp_path / "test_ml_ready.csv"
        df.to_csv(csv_path, index=False)
        result = train_lgb_ticker("TEST", csv_path)
        for key in ("ticker", "accuracy", "auc", "report", "importances", "n_train", "n_test"):
            assert key in result
        assert result["ticker"] == "TEST"
        assert result["model"] == "LightGBM"

    def test_metrics_in_range(self, tmp_path):
        df, feats = _make_dataset()
        csv_path = tmp_path / "test_ml_ready.csv"
        df.to_csv(csv_path, index=False)
        result = train_lgb_ticker("TEST", csv_path)
        assert 0.0 <= result["accuracy"] <= 1.0
        assert 0.0 <= result["auc"] <= 1.0
        assert result["n_train"] > 0
        assert result["n_test"] > 0


class TestTemporalCVWeight:
    def test_weight_in_range(self):
        df, feats = _make_dataset(n=800)
        X, y = df[feats].values, df["target"].values
        w = _temporal_cv_weight(X, y, random_state=42)
        assert 0.0 <= w <= 1.0

    def test_deterministic(self):
        df, feats = _make_dataset(n=800)
        X, y = df[feats].values, df["target"].values
        w1 = _temporal_cv_weight(X, y, random_state=42)
        w2 = _temporal_cv_weight(X, y, random_state=42)
        assert w1 == w2


class TestTrainRfLgbEnsemble:
    def test_returns_expected_keys(self):
        df, feats = _make_dataset()
        X, y = df[feats], df["target"]
        result = train_rf_lgb_ensemble(X, y, random_state=42)
        for key in ("model", "rf_weight", "auc_oof"):
            assert key in result
        assert "rf" in result["model"]
        assert "lgb" in result["model"]

    def test_rf_weight_in_range(self):
        df, feats = _make_dataset()
        X, y = df[feats], df["target"]
        result = train_rf_lgb_ensemble(X, y, random_state=42)
        assert 0.0 <= result["rf_weight"] <= 1.0

    def test_explicit_weight(self):
        df, feats = _make_dataset()
        X, y = df[feats], df["target"]
        result = train_rf_lgb_ensemble(X, y, rf_weight=0.7, random_state=42)
        assert result["rf_weight"] == 0.7

    def test_rf_params(self):
        assert RF_PARAMS["n_estimators"] == 100
        assert RF_PARAMS["max_depth"] == 15
