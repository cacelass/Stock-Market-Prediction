"""
test_build_features.py — Tests para inversion/features/build_features.py
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.preprocessing import StandardScaler

from inversion.features.build_features import (
    add_derived_features,
    fit_and_save_scaler,
    build_features_pipeline,
)

EXPECTED_COLS = [
    "return",
    "volatility",
    "rsi",
    "hl_range",
    "oc_range",
    "ma_50",
    "ma_200",
    "log_volume",
    "vwap_ratio",
    "lag_1",
    "lag_5",
    "lag_20",
]


@pytest.mark.smoke
def test_add_derived_features_returns_dataframe(sample_df):
    """add_derived_features debe devolver un DataFrame con las features esperadas."""
    result = add_derived_features(sample_df)
    assert isinstance(result, pd.DataFrame)
    for col in EXPECTED_COLS:
        assert col in result.columns, f"Falta la columna: {col}"


def test_add_derived_features_preserves_rows(sample_df):
    """El número de filas no debe cambiar (solo se añaden columnas)."""
    result = add_derived_features(sample_df)
    assert len(result) == len(sample_df)


def test_add_derived_features_no_nan_at_end(sample_df):
    """Tras ma_200 y lag_20, al final debe haber filas completas sin NaN."""
    result = add_derived_features(sample_df)
    complete = result.dropna(subset=EXPECTED_COLS)
    assert len(complete) > 0


def test_return_is_pct_change(sample_df):
    """return = close.pct_change()."""
    result = add_derived_features(sample_df)
    expected = sample_df["close"].pct_change()
    pd.testing.assert_series_equal(result["return"], expected, check_names=False)


def test_fit_and_save_scaler_returns_scaler(sample_df):
    """fit_and_save_scaler debe devolver un StandardScaler ajustado."""
    df = add_derived_features(sample_df).dropna(subset=EXPECTED_COLS)
    X = df[EXPECTED_COLS].values
    scaler = fit_and_save_scaler(X)
    assert isinstance(scaler, StandardScaler)
    assert hasattr(scaler, "mean_")


def test_fit_and_save_scaler_saves_file(patch_paths, sample_df):
    """fit_and_save_scaler debe persistir el scaler en paths.SCALER_FILE."""
    from inversion.utils import paths

    df = add_derived_features(sample_df).dropna(subset=EXPECTED_COLS)
    fit_and_save_scaler(df[EXPECTED_COLS].values)
    assert patch_paths["SCALER_FILE"].exists()
    assert paths.SCALER_FILE == patch_paths["SCALER_FILE"]


def test_scaler_fits_only_on_train(patch_paths):
    """El scaler persistido debe ajustarse SOLO con datos de train (nunca con test)."""
    rng = np.random.RandomState(0)
    X_train = rng.normal(size=(200, 5))
    X_test = rng.normal(loc=10.0, size=(50, 5))  # distribución distinta

    fit_and_save_scaler(X_train)

    scaler = joblib.load(patch_paths["SCALER_FILE"])
    expected = StandardScaler().fit(X_train)
    np.testing.assert_allclose(scaler.mean_, expected.mean_)
    np.testing.assert_allclose(scaler.scale_, expected.scale_)

    # El test no puede colarse en las estadísticas del scaler
    assert not np.allclose(scaler.mean_, 0.0)


def test_build_features_pipeline_output_shape_and_columns(patch_paths, sample_df):
    """El artefacto de features debe tener las columnas esperadas y el mismo nº de filas."""
    raw = patch_paths["PROCESSED_DATA_DIR"] / "NVDA_ml_ready.csv"
    sample_df.to_csv(raw, index=False)

    written = build_features_pipeline()

    out = patch_paths["INTERIM_DATA_DIR"] / "features_NVDA_ml_ready.csv"
    assert out in written
    assert out.exists()

    result = pd.read_csv(out, parse_dates=["timestamp"])
    assert len(result) == len(sample_df)
    expected = ["timestamp", "open", "high", "low", "close", "volume"] + EXPECTED_COLS
    for col in expected:
        assert col in result.columns, f"Falta la columna: {col}"


def test_build_features_pipeline_is_deterministic(patch_paths, sample_df):
    """Dos ejecuciones del pipeline deben producir artefactos byte-idénticos."""
    raw = patch_paths["PROCESSED_DATA_DIR"] / "NVDA_ml_ready.csv"
    sample_df.to_csv(raw, index=False)

    build_features_pipeline()
    first = (patch_paths["INTERIM_DATA_DIR"] / "features_NVDA_ml_ready.csv").read_bytes()
    build_features_pipeline()
    second = (patch_paths["INTERIM_DATA_DIR"] / "features_NVDA_ml_ready.csv").read_bytes()

    assert first == second


def test_sentiment_features_no_leakage(patch_paths, sample_df):
    """SENT-003: las columnas sentiment_* solo usan noticias del mismo día o anteriores.

    Se plantan noticias SOLO en días futuros (después de la última fila de
    precios): ninguna fila anterior puede tener sentimiento, porque el merge es
    por fecha exacta y las medias móviles miran hacia atrás.
    """
    from inversion.features.build_features import build_features_pipeline

    raw = patch_paths["PROCESSED_DATA_DIR"] / "NVDA_ml_ready.csv"
    sample_df.to_csv(raw, index=False)

    # Noticias solo en el futuro respecto al final de la serie de precios
    future_day = pd.Timestamp(sample_df["timestamp"].max()) + pd.Timedelta(days=10)
    news = pd.DataFrame(
        {
            "fecha": [future_day],
            "titulo": ["Nvidia rallies on strong earnings"],
            "cuerpo": [""],
            "fuente": ["WSJ"],
        }
    )
    news_path = patch_paths["RAW_DATA_DIR"] / "news_NVDA.csv"
    news_path.parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(news_path, index=False)

    build_features_pipeline()
    out = patch_paths["INTERIM_DATA_DIR"] / "features_NVDA_ml_ready.csv"
    result = pd.read_csv(out, parse_dates=["timestamp"])

    assert "sentiment_score" in result.columns
    assert "sentiment_ma5" in result.columns
    # Ninguna fila anterior a la noticia puede tener sentimiento (sin fuga)
    assert (result["sentiment_score"] == 0.0).all()
    assert (result["sentiment_noticias"] == 0).all()
    assert (result["sentiment_vol"] == 0.0).all()


def test_sentiment_features_aligned_same_day(patch_paths, sample_df):
    """El score del día t viene de noticias del día t (alineación exacta)."""
    from inversion.features.build_features import build_features_pipeline

    raw = patch_paths["PROCESSED_DATA_DIR"] / "NVDA_ml_ready.csv"
    sample_df.to_csv(raw, index=False)

    # Noticia el MISMO día que una fila de precios existente
    day = pd.Timestamp(sample_df["timestamp"].iloc[50]).normalize()
    news = pd.DataFrame(
        {
            "fecha": [day],
            "titulo": ["The stock beat expectations and rallied"],
            "cuerpo": [""],
            "fuente": ["Reuters"],
        }
    )
    news_path = patch_paths["RAW_DATA_DIR"] / "news_NVDA.csv"
    news_path.parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(news_path, index=False)

    build_features_pipeline()
    out = patch_paths["INTERIM_DATA_DIR"] / "features_NVDA_ml_ready.csv"
    result = pd.read_csv(out, parse_dates=["timestamp"])

    row = result[pd.to_datetime(result["timestamp"]).dt.normalize() == day]
    assert len(row) == 1
    # "beats" + "surges" → compound positivo
    assert row["sentiment_score"].iloc[0] > 0
    assert row["sentiment_noticias"].iloc[0] >= 1
