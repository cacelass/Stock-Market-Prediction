"""
test_predict_model.py — Tests para inversion/models/predict_model.py
"""

import joblib
import numpy as np
import pytest

from inversion.features.build_features import add_derived_features, fit_and_save_scaler
from inversion.models.predict_model import predict_future
from inversion.models.train_model import available_feature_cols, train_rf_model


def _setup_model(sample_df, patch_paths):
    """Entrena modelo + scaler reales sobre features derivadas y los guarda."""
    df = add_derived_features(sample_df.copy())
    cols = available_feature_cols(df)
    feat = df.dropna(subset=cols)
    X = feat[cols].values
    y = (feat["return"] > 0).astype(int).values
    fit_and_save_scaler(X)
    train_rf_model(X, y)
    return df


def test_predict_future_returns_none_without_model(sample_df):
    """Sin modelo ni scaler guardados, predict_future debe devolver Nones."""
    result = predict_future(sample_df)
    assert all(v is None for v in result)


def test_predict_future_returns_tuple(patch_paths, sample_df):
    _setup_model(sample_df, patch_paths)
    result = predict_future(sample_df)
    assert isinstance(result, tuple)
    assert len(result) == 4


def test_predict_future_class_in_range(patch_paths, sample_df):
    _setup_model(sample_df, patch_paths)
    cls, prob_baja, prob_sube, ultimo_close = predict_future(sample_df)
    assert cls is not None
    assert prob_baja is not None
    assert prob_sube is not None
    assert cls in (0, 1)
    assert 0 <= prob_baja <= 1
    assert 0 <= prob_sube <= 1
    assert abs(prob_baja + prob_sube - 1.0) < 1e-6


def test_predict_future_returns_last_close(patch_paths, sample_df):
    _setup_model(sample_df, patch_paths)
    _, _, _, ultimo_close = predict_future(sample_df)
    assert ultimo_close == sample_df.iloc[-1]["close"]


def test_predict_future_ticker_uses_ticker_models(patch_paths, sample_df):
    """Con modelo por ticker (rf_NVDA.pkl), predict_future lo usa en vez del global."""
    _setup_model(sample_df, patch_paths)
    result = predict_future(sample_df, ticker="NVDA")
    assert all(v is not None for v in result)


def test_main_ok_ticker_returns_zero(patch_paths, sample_df, tmp_path):
    """main() con un ticker con datos devuelve 0 y escribe la predicción."""
    from inversion.models.predict_model import main

    _setup_model(sample_df, patch_paths)
    out = tmp_path / "data" / "interim" / "features_NVDA_ml_ready.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(out, index=False)
    assert main(["--ticker", "NVDA"]) == 0


def test_main_missing_ticker_returns_one(patch_paths):
    """main() con un ticker sin datos devuelve 1 (error claro)."""
    from inversion.models.predict_model import main

    assert main(["--ticker", "ZZZ"]) == 1
