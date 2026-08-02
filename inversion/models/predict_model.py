"""Predicciones multi-ticker: carga models/rf_<TICKER>.pkl + scaler_<TICKER>.pkl.

CLI: .venv/bin/python inversion/models/predict_model.py --ticker AAPL
(make predict TICKER=AAPL). Si el ticker no tiene modelo, devuelve un error
claro con instrucciones de entrenamiento.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from inversion.utils import paths
from inversion.features.build_features import add_derived_features

# Features que deben estar presentes (deben coincidir con train_model.py)
FEATURE_COLS = [
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
]

# Columnas de sentimiento: el modelo las usa si existen en el dataset (SENT-003).
SENTIMENT_COLS = ["sentiment_score", "sentiment_ma5", "sentiment_vol"]


def _load_model_and_scaler(ticker: str) -> tuple[Any, Any]:
    """Carga el modelo y scaler del ticker (o los globales como fallback)."""
    ticker = ticker.upper()
    model_path = paths.MODELS_DIR / f"rf_{ticker}.pkl"
    scaler_path = paths.MODELS_DIR / f"scaler_{ticker}.pkl"
    if model_path.exists() and scaler_path.exists():
        return joblib.load(model_path), joblib.load(scaler_path)
    if paths.MODEL_FILE.exists() and paths.SCALER_FILE.exists():
        return joblib.load(paths.MODEL_FILE), joblib.load(paths.SCALER_FILE)
    raise FileNotFoundError(
        f"No hay modelo entrenado para '{ticker}'. "
        f"Ejecuta antes el entrenamiento: .venv/bin/python inversion/models/train_model.py "
        f"(o make train). Se esperaba {model_path}."
    )


def predict_future(df: pd.DataFrame, ticker: str = "NVDA") -> tuple[int | None, float | None, float | None, float | None]:
    """
    Predice la dirección del precio para los próximos días a partir
    de los datos crudos más recientes.

    Args:
        df : DataFrame con columnas OHLCV y 'timestamp'.
        ticker : activo del catálogo (por defecto NVDA).

    Returns:
        (clase, prob_baja, prob_sube, ultimo_close)
        clase     : 0 (baja) o 1 (sube)
        prob_baja : probabilidad clase 0
        prob_sube : probabilidad clase 1
        ultimo_close : último precio de cierre conocido
    """
    try:
        model, scaler = _load_model_and_scaler(ticker)
    except FileNotFoundError:
        print("Error: Modelo o scaler no encontrados. Ejecuta main.py primero.")
        return None, None, None, None

    # Calcular features si el dataset no las trae ya (interim los incluye).
    df_feat = df.copy()
    if any(c not in df_feat.columns for c in FEATURE_COLS):
        df_feat = add_derived_features(df.copy())

    feature_cols = list(FEATURE_COLS)
    if all(c in df_feat.columns for c in SENTIMENT_COLS):
        feature_cols += SENTIMENT_COLS

    # Coger la última fila que no tenga NaN en las features
    last = df_feat.dropna(subset=feature_cols).iloc[[-1]][feature_cols]

    last_scaled = scaler.transform(last)
    pred_class = model.predict(last_scaled)[0]
    pred_probas = model.predict_proba(last_scaled)[0]
    ultimo_close = df.iloc[-1]["close"]

    return int(pred_class), float(pred_probas[0]), float(pred_probas[1]), float(ultimo_close)


def _load_ticker_data(ticker: str) -> pd.DataFrame:
    """Carga el dataset del ticker: primero el de features (interim), luego el crudo.

    El de features (data/interim/features_<T>_ml_ready.csv) ya incluye las
    columnas sentiment_* que el modelo con sentimiento necesita para predecir.
    """
    ticker = ticker.upper()
    candidate = paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv"
    if candidate.exists():
        return pd.read_csv(candidate)
    for base in (paths.RAW_DATA_DIR, paths.PROCESSED_DATA_DIR):
        candidate = base / f"{ticker}_ml_ready.csv"
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError(
        f"No hay datos para '{ticker}' en data/raw/, data/processed/ ni data/interim/. "
        f"Ejecuta .venv/bin/python inversion/data/make_dataset.py (o make data) primero."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predice la dirección del precio por ticker.")
    parser.add_argument("--ticker", default="NVDA", help="Activo del catálogo (config/tickers.yaml)")
    args = parser.parse_args(argv)

    ticker = args.ticker.upper()
    try:
        df = _load_ticker_data(ticker)
        cls, prob_baja, prob_sube, ultimo_close = predict_future(df, ticker=ticker)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1

    if cls is None:
        return 1

    direccion = "SUBE" if cls == 1 else "BAJA"
    print(f"▶ Predicción para {ticker}: {direccion}")
    print(f"   Probabilidad de subida: {prob_sube:.2%} | de bajada: {prob_baja:.2%}")
    print(f"   Último cierre conocido: {ultimo_close:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
