"""Tests del modelo de riesgo/volatilidad (EXP-005)."""

import joblib
import numpy as np
import pandas as pd
import pytest

from inversion.models.volatility import (
    VOL_HORIZON_DAYS,
    annualize,
    make_vol_target,
    predict_volatility,
    train_vol_ticker,
)


def test_make_vol_target_requires_return():
    with pytest.raises(ValueError):
        make_vol_target(pd.DataFrame({"close": [1, 2, 3]}))


def test_make_vol_target_has_no_lookahead():
    """El vol_target en la fila t solo usa retornos futuros [t+1, t+h]."""
    n = 40
    ret = np.zeros(n)
    ret[10:20] = 0.05  # un bloque de alta volatilidad bien localizado
    df = pd.DataFrame({"return": ret})

    h = 5
    out = make_vol_target(df, horizon=h)
    # La fila 5 mira [6,10]; incluye los retornos 6..9 (0.0) y 10 (0.05).
    ventana = ret[6:11]
    assert out["vol_target"].iloc[5] == pytest.approx(ventana.std(ddof=1), rel=1e-9)
    # Una fila anterior al bloque (0) ve solo ceros → vol 0.
    assert out["vol_target"].iloc[0] == pytest.approx(0.0)
    # Las últimas h filas no tienen futuro → NaN.
    assert out["vol_target"].iloc[-h:].isna().all()


def test_annualize():
    assert annualize(0.01) == pytest.approx(0.01 * np.sqrt(252))


def test_train_vol_ticker_saves_model_and_reports_metrics(df_with_features, tmp_path):
    csv = tmp_path / "features_TST_ml_ready.csv"
    df_with_features.to_csv(csv, index=False)

    m = train_vol_ticker("TST", csv, horizon=5)
    assert {"r2", "spearman", "base_r2", "base_spearman", "n_train", "horizon"} <= set(m)
    assert m["n_train"] > 0
    assert isinstance(m["r2"], float)
    assert isinstance(m["spearman"], float)
    # El modelo queda persistido para poder predecir.
    from inversion.utils import paths

    assert (paths.MODELS_DIR / "vol_TST.pkl").exists()


def test_predict_volatility_positive_and_annualized(df_with_features, tmp_path):
    csv = tmp_path / "features_TST_ml_ready.csv"
    df_with_features.to_csv(csv, index=False)
    train_vol_ticker("TST", csv, horizon=VOL_HORIZON_DAYS)

    p = predict_volatility("TST", csv_path=csv)
    assert p["vol_daily"] > 0
    assert p["vol_annualized"] > p["vol_daily"]
    assert p["ticker"] == "TST"


def test_predict_volatility_without_model_raises(df_with_features, tmp_path):
    csv = tmp_path / "features_TST_ml_ready.csv"
    df_with_features.to_csv(csv, index=False)
    with pytest.raises(FileNotFoundError):
        predict_volatility("TST", csv_path=csv)


def test_train_vol_ticker_too_short(tmp_path):
    csv = tmp_path / "features_TINY_ml_ready.csv"
    pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=20), "return": np.zeros(20)}).to_csv(
        csv, index=False
    )
    with pytest.raises(ValueError):
        train_vol_ticker("TINY", csv)


def test_add_har_features_no_lookahead_and_columns():
    """Las features HAR usan solo el pasado y llevan las columnas esperadas."""
    from inversion.models.volatility import HAR_FEATURES, add_har_features

    n = 40
    ret = np.zeros(n)
    ret[25] = 0.10  # pico localizado
    df = pd.DataFrame({"return": ret})
    out = add_har_features(df)
    assert set(HAR_FEATURES) <= set(out.columns)
    # rv_d en t depende solo de return[t] (ventana de 1), no del futuro.
    assert out["rv_d"].iloc[24] == pytest.approx(np.log(0.0 + 1e-10))
    assert out["rv_d"].iloc[25] == pytest.approx(np.log(0.01 + 1e-10))
    # El pico (t=25) no debe influir en filas anteriores.
    assert out["rv_w"].iloc[20] == pytest.approx(np.log(0.0 + 1e-10))


def test_train_vol_ticker_uses_har_features(df_with_features, tmp_path):
    from inversion.models.volatility import HAR_FEATURES

    csv = tmp_path / "features_TST_ml_ready.csv"
    df_with_features.to_csv(csv, index=False)
    train_vol_ticker("TST", csv, horizon=5)
    from inversion.utils import paths

    bundle = joblib.load(paths.MODELS_DIR / "vol_TST.pkl")
    assert set(HAR_FEATURES) <= set(bundle["features"])
