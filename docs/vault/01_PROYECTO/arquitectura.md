# Arquitectura — Stock Market Prediction

> Predicción de tendencia bursátil (sube >2% en 5 días) con RandomForest + sentimiento VADER, multi-ticker.

## Stack tecnológico

- **Lenguaje:** Python (3.12)
- **ML:** supervisado
- **API:** FastAPI (sí)
- **Tracking:** MLflow (no)
- **Dataset público:** yfinance (descarga pública en vivo)
- **Grafo de conocimiento:** graphify + obsidian vault

## Estructura del proyecto

```
inversion/
├── data/              — Datos (raw/processed/interim/external)
├── features/          — Ingeniería de features
├── models/            — Modelos entrenados (.joblib, .pkl)
├── visualization/     — Código de visualización
├── api/               — FastAPI (si use_api)
│
tests/
├── test_*.py          — Tests del proyecto
│
agents/                — Sistema multi-agente (dskit)
├── contracts.py       — Contratos de rol
├── agents/            — Implementación de agentes
├── prompts/           — Prompts de agentes
├── workspace/         — Workspace de agentes
│
vault/                 — Bóveda Obsidian del proyecto
├── 00_META/           — Metadatos e índice para IA
├── 01_PROYECTO/       — Documentación del proyecto
├── 02_DATOS/          — Documentación de datos
├── 04_VISUALIZACIONES/— Visualizaciones y grafos
├── 05_AGENTES/        — Fichas de agentes
```

## Pipeline de ML

```
data → features → train → evaluate → predict
  └── EDA (data agent)     └── análisis (ml agent)
```
