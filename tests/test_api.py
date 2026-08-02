"""
test_api.py — Tests de la API REST multi-ticker (TMPL-002).

Usa el TestClient de FastAPI para verificar los endpoints sin levantar
servidor. El fixture patch_paths (conftest) redirige data/models a tmp_path.
"""

from __future__ import annotations

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from inversion.api.main import app
from inversion.features.build_features import add_derived_features, fit_and_save_scaler
from inversion.models.train_model import train_rf_model
from inversion.models.predict_model import FEATURE_COLS


@pytest.fixture
def client():
    return TestClient(app)


def _setup_ticker_models(sample_df, patch_paths, ticker: str = "NVDA"):
    """Entrena rf_<T>.pkl + scaler_<T>.pkl reales y deja features_<T> en interim."""
    from inversion.utils import paths

    df = add_derived_features(sample_df.copy())
    feat = df.dropna(subset=FEATURE_COLS)
    X = feat[FEATURE_COLS].values
    y = (feat["return"] > 0).astype(int).values
    fit_and_save_scaler(X, filename=f"scaler_{ticker}.pkl")
    train_rf_model(X, y, filename=f"rf_{ticker}.pkl")

    out = paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv"
    df.to_csv(out, index=False)


def test_health_ok(client, patch_paths):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_info_ok(client, patch_paths):
    resp = client.get("/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["project"] == "Stock Market Prediction"
    assert "feature_names" in body
    assert "tickers" in body


def test_predict_ok(client, patch_paths, sample_df):
    _setup_ticker_models(sample_df, patch_paths, "NVDA")
    resp = client.post("/predict", json={"ticker": "NVDA"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "NVDA"
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability_up"] <= 1
    assert 0 <= body["probability_down"] <= 1
    assert abs(body["probability_up"] + body["probability_down"] - 1) < 0.01


def test_predict_ticker_without_data_404(client, patch_paths):
    resp = client.post("/predict", json={"ticker": "ZZZ"})
    assert resp.status_code == 404


def test_predict_without_model_503(client, patch_paths):
    from inversion.utils import paths

    paths.INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": ["2020-01-01"], "close": [1.0]}).to_csv(paths.INTERIM_DATA_DIR / "features_NVDA_ml_ready.csv", index=False)
    resp = client.post("/predict", json={"ticker": "NVDA"})
    assert resp.status_code in (404, 503)


def test_process_input_returns_expected_shape(client, patch_paths, sample_df):
    from inversion.api.main import process_input

    _setup_ticker_models(sample_df, patch_paths, "NVDA")
    X = process_input("NVDA")
    assert X.shape[1] == len(FEATURE_COLS)
