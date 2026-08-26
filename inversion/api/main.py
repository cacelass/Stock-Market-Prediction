"""main.py — App FastAPI multi-ticker (TMPL-002).

La API sirve el modelo entrenado por ticker (models/rf_<T>.pkl) sobre el
dataset de features (data/interim/features_<T>_ml_ready.csv), reutilizando
la lógica de predicción de inversion/models/predict_model.py.
"""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException

from inversion.api.schemas import HealthResponse, InfoResponse, PredictRequest, PredictResponse
from inversion.models.predict_model import _load_model_and_scaler, _load_ticker_data, predict_future
from inversion.models.train_model import FEATURE_COLS, SENTIMENT_COLS, available_feature_cols
from inversion.utils import paths

PROJECT = "Stock Market Prediction"
ML_TYPE = "supervisado"
MODEL_NAME = "RandomForest (multi-ticker)"

app = FastAPI(
    title="Stock Market Prediction API",
    description="API REST para el modelo de ML multi-ticker (rf_<TICKER>.pkl).",
    version="0.1.0",
)


def _available_tickers() -> list[str]:
    """Tickers del catálogo con modelo entrenado."""
    if not paths.MODELS_DIR.exists():
        return []
    return sorted(p.stem.replace("rf_", "") for p in paths.MODELS_DIR.glob("rf_*_base.pkl"))


def _catalog_tickers() -> list[str]:
    """Tickers del catálogo (config/tickers.yaml), en orden."""
    import yaml

    cfg = paths.PROJECT_DIR / "config" / "tickers.yaml"
    if not cfg.exists():
        return []
    with cfg.open(encoding="utf-8") as fh:
        return [str(t) for t in yaml.safe_load(fh).get("tickers", [])]


def process_input(ticker: str) -> pd.DataFrame:
    """Carga y devuelve las features del último día válido del ticker.

    Es la pieza que conecta el input con la inferencia: devuelve una fila con
    las columnas exactas que el modelo del ticker espera (FEATURE_COLS +
    SENTIMENT_COLS si el modelo se entrenó con sentimiento).
    """
    df = _load_ticker_data(ticker)
    model, scaler = _load_model_and_scaler(ticker)
    feature_cols = available_feature_cols(df)
    last = df.dropna(subset=feature_cols).iloc[[-1]][feature_cols].copy()
    return pd.DataFrame(scaler.transform(last), columns=feature_cols)


@app.get("/health", response_model=HealthResponse, tags=["Servicio"])
def health() -> HealthResponse:
    models = _available_tickers()
    return HealthResponse(
        status="ok",
        model_loaded=bool(models),
        models=models,
    )


@app.get("/info", response_model=InfoResponse, tags=["Servicio"])
def info() -> InfoResponse:
    return InfoResponse(
        project=PROJECT,
        ml_type=ML_TYPE,
        model_name=MODEL_NAME,
        tickers=_catalog_tickers(),
        feature_names=FEATURE_COLS + SENTIMENT_COLS,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Predicción"])
def predict(request: PredictRequest) -> PredictResponse:
    ticker = request.ticker.upper()
    try:
        df = _load_ticker_data(ticker)
        cls, prob_baja, prob_sube, last_close = predict_future(df, ticker=ticker)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if cls is None:
        raise HTTPException(status_code=503, detail=f"No hay modelo para '{ticker}'. Ejecuta make train.")
    return PredictResponse(
        ticker=ticker,
        prediction=int(cls),
        probability_up=round(float(prob_sube or 0.0), 4),
        probability_down=round(float(prob_baja or 0.0), 4),
        last_close=round(float(last_close or 0.0), 4),
        model_name=f"rf_{ticker}.pkl",
    )
