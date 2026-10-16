"""Predicciones multi-ticker: carga models/rf_<TICKER>.pkl + scaler_<TICKER>.pkl.

CLI: .venv/bin/python inversion/models/predict_model.py --ticker AAPL
(make predict TICKER=AAPL). Si el ticker no tiene modelo, devuelve un error
claro con instrucciones de entrenamiento.

Se puede usar el ensemble de 3 modelos con --ensemble.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from inversion.features.build_features import add_derived_features
from inversion.models import pooled

# Una sola fuente de verdad: las features y su selección viven en train_model.py.
# Duplicarlas aquí provocó desincronización (modelos entrenados con features
# que el predictor no conocía).
from inversion.models.train_model import FEATURE_COLS, available_feature_cols
from inversion.utils import paths


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


def predict_future(
    df: pd.DataFrame,
    ticker: str = "NVDA",
    use_ensemble: bool = False,
) -> tuple[int | None, float | None, float | None, float | None]:
    """
    Predice la dirección del precio para los próximos días a partir
    de los datos crudos más recientes.

    Args:
        df : DataFrame con columnas OHLCV y 'timestamp'.
        ticker : activo del catálogo (por defecto NVDA).
        use_ensemble : si True, usa el ensemble de 3 modelos en vez del RF único.

    Returns:
        (clase, prob_baja, prob_sube, ultimo_close)
        clase     : 0 (baja) o 1 (sube)
        prob_baja : probabilidad clase 0
        prob_sube : probabilidad clase 1
        ultimo_close : último precio de cierre conocido
    """
    if use_ensemble:
        try:
            from inversion.models.predict_ensemble import predict_ensemble as predict_ens
        except ImportError:
            use_ensemble = False

    try:
        route = pooled.resolve_route(ticker)
        if route == pooled.GLOBAL_TICKER:
            model, scaler, feature_cols = pooled.load_global_bundle(ticker)
        else:
            model, scaler = _load_model_and_scaler(ticker)
    except FileNotFoundError:
        print("Error: Modelo o scaler no encontrados. Ejecuta main.py primero.")
        return None, None, None, None

    # Calcular features si el dataset no las trae ya (interim los incluye).
    df_feat = df.copy()
    if any(c not in df_feat.columns for c in FEATURE_COLS):
        df_feat = add_derived_features(df.copy())

    if route == pooled.GLOBAL_TICKER:
        # El pool espera sus columnas (scale-free + one-hots constantes por ticker).
        df_feat = pooled.add_ticker_dummies(df_feat, ticker, feature_cols)
    else:
        feature_cols = available_feature_cols(df_feat)

    # Coger la última fila que no tenga NaN en las features
    last = df_feat.dropna(subset=feature_cols).iloc[[-1]][feature_cols]

    if use_ensemble:
        preds, probas = predict_ens(df, ticker=ticker)
        if len(preds) > 0:
            return int(preds[0]), float(1 - probas[0]), float(probas[0]), float(df.iloc[-1]["close"])

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
    parser.add_argument("--ensemble", action="store_true", help="Usar el ensemble de 3 modelos")
    args = parser.parse_args(argv)

    ticker = args.ticker.upper()
    try:
        df = _load_ticker_data(ticker)
        cls, prob_baja, prob_sube, ultimo_close = predict_future(df, ticker=ticker, use_ensemble=args.ensemble)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1

    if cls is None:
        return 1

    prefix = "[ensemble] " if args.ensemble else ""
    direccion = "SUBE" if cls == 1 else "BAJA"
    print(f"{prefix}▶ Predicción para {ticker}: {direccion}")
    print(f"   Probabilidad de subida: {prob_sube:.2%} | de bajada: {prob_baja:.2%}")
    print(f"   Último cierre conocido: {ultimo_close:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
