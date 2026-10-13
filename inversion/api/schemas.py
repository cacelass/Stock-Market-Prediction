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


class RiskResponse(BaseModel):
    """Respuesta del endpoint POST /risk (modelo de riesgo, EXP-005)."""

    ticker: str
    vol_daily: float = Field(..., description="Volatilidad diaria prevista (std de retornos)")
    vol_annualized: float = Field(..., description="Volatilidad anualizada (√252)")
    as_of: str = Field(..., description="Última fecha de datos usada (YYYY-MM-DD)")
    model_name: str = Field(..., description="Modelo usado (vol_<TICKER>.pkl)")


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
