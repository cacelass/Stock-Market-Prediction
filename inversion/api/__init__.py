"""inversion/api — API REST para el modelo multi-ticker (TMPL-002).

Arranca con:
    make serve
o:
    uvicorn inversion.api.main:app --reload --port 8000

Endpoints:
    GET  /health    → estado del servicio y modelos disponibles
    GET  /info      → metadata del proyecto, catálogo y features
    POST /predict   → {ticker: "NVDA"} → predicción y probabilidad
"""
