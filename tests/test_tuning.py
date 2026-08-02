"""
test_tuning.py — Tests de la optimización de hiperparámetros con Optuna (TMPL-003).

Usa n_trials pequeños para que los tests sean rápidos y datos sintéticos
escritos en las rutas temporales de patch_paths.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from inversion.utils import paths
from inversion.tuning.tune_model import load_best_params, tune_ticker


@pytest.fixture
def fake_features(tmp_path, monkeypatch):
    """Escribe features_AAA_ml_ready.csv sintético en INTERIM_DATA_DIR."""
    import numpy as np

    n = 200
    rng = np.random.default_rng(42)
    cols = [
        "return",
        "volatility",
        "rsi",
        "ma_50",
        "ma_200",
        "hl_range",
        "oc_range",
        "log_volume",
        "vwap_ratio",
        "lag_1",
        "lag_5",
        "lag_20",
        "sentiment_score",
        "sentiment_ma5",
        "sentiment_vol",
    ]
    df = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    df["timestamp"] = pd.date_range("2020-01-01", periods=n)
    df["target"] = (df["return"] + df["lag_1"] > 0).astype(int)
    out = paths.INTERIM_DATA_DIR / "features_AAA_ml_ready.csv"
    df.to_csv(out, index=False)
    return out


def test_tune_ticker_guarda_json(fake_features):
    best = tune_ticker("AAA", n_trials=2)
    path = paths.ARTIFACTS_DIR / "best_params_AAA.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "n_estimators" in data["params"]
    assert isinstance(best, dict)


def test_load_best_params_returns_dict(fake_features):
    tune_ticker("AAA", n_trials=2)
    params = load_best_params("AAA")
    assert params is not None
    assert isinstance(params, dict)
    assert "max_depth" in params


def test_load_best_params_none_sin_json():
    assert load_best_params("NOEXISTE") is None
