"""schemas.py — Modelos Pydantic para la API REST (TMPL-002)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Cuerpo de la petición POST /predict."""

    ticker: str = Field(..., description="Activo del catálogo (p.ej. NVDA)")


class PredictResponse(BaseModel):
    """Respuesta del endpoint POST /predict."""

    ticker: str
    prediction: int = Field(..., description="0 = baja, 1 = sube")
    probability_up: float = Field(..., description="Probabilidad de subida en [0, 1]")
    probability_down: float = Field(..., description="Probabilidad de bajada en [0, 1]")
    last_close: float = Field(..., description="Último cierre conocido")
    model_name: str = Field(..., description="Modelo usado (rf_<TICKER>.pkl)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    models: list[str]


class InfoResponse(BaseModel):
    project: str
    ml_type: str
    model_name: str
    tickers: list[str]
    feature_names: list[str]
