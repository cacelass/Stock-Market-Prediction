# Product Requirements Document

> Documento **generado** (`documentation update_prd`) desde el estado del proyecto: `references/00-objetivo.md`, `harness/featureslist.json` y `features/*.feature`. No lo edites a mano — se sobrescribe. Actualizado: 2026-10-13

## Objetivo

# Objetivo del proyecto

## Pregunta a responder

Con datos de mercado OHLCV+volumen y sentimiento de noticias, se puede predecir el RIESGO (volatilidad realizada a 10 dias) de un activo mejor que la persistencia simple, y servir esa senal para dimensionar posicion? La direccion a 5 dias se mantiene como senal secundaria.

## Métrica de éxito

Modelo de volatilidad: R2 fuera de muestra >= 0.15 y Spearman >= 0.40 en walk-forward de los 7 tickers. Direccion (secundaria): AUC >= 0.55.

## Datos de partida

Precios diarios OHLCV de 7 tickers (20 anos, yfinance) con features tecnicas derivadas; sentimiento de noticias NewsAPI.ai. Sin datos intradia ni fundamentales.

## Criterio de parada

Si el modelo de volatilidad no bate la persistencia en R2 en al menos 5 de 7 tickers, o el backtest con costes no supera a buy&hold, el proyecto se replantea como analitico/educativo.

## Usuarios

Uso personal y de portfolio: gestion de riesgo y dimensionado de posicion. No es asesoramiento financiero.

## Alcance y no-alcance

ENTRA: forecasting de volatilidad/riesgo, intervalos de confianza, serving por API. NO ENTRA: ejecucion de ordenes reales, datos intradia, ni prediccion de direccion como objetivo principal.

## Restricciones y riesgos

Senales debiles en direccion (probado: AUC~0.55); pocos datos efectivos por ticker; no es asesoramiento financiero.

## Alcance

**Backlog:** 46 done

| Feature | Estado | Título |
|---------|--------|--------|
| DATA-001 | done | EDA del dataset principal |
| FEAT-001 | done | Pipeline de features reproducible |
| MODEL-001 | done | Baseline entrenado y evaluado |
| QA-001 | done | Batería de calidad en verde |
| QA-002 | done | Entorno reproducible: ./init.sh termina en ENTORNO LISTO |
| DATA-002 | done | Descarga multi-ticker parametrizada |
| SENT-001 | done | Recolección de noticias financieras por ticker |
| SENT-002 | done | Analizador de sentimiento financiero (VADER) |
| SENT-003 | done | Fusión de sentimiento en el pipeline de features |
| MODEL-002 | done | Modelo multi-ticker con sentimiento y evaluación por activo |
| API-001 | done | CLI predict --ticker para cualquier activo del catálogo |
| AGENT-001 | done | Agente Python sentiment en el catálogo |
| QA-003 | done | Batería de calidad en verde con el nuevo código |
| TMPL-001 | done | Sincronizar con dskit vía copier y recuperar ficheros de repo |
| TMPL-002 | done | API REST FastAPI para el modelo multi-ticker (use_api) |
| TMPL-003 | done | Calidad de modelo: SHAP + conformal + Optuna |
| TMPL-004 | done | Monitoring de drift y rendimiento (use_monitoring) |
| TMPL-005 | done | RAG local + vault Obsidian (use_rag + graphify) |
| CI-001 | done | CI de GitHub Actions con la puerta del arnés |
| RESEARCH-001 | done | Papers de referencia sobre activos, inversión y sentimiento financiero |
| TRADE-001 | done | Señales de trading y backtesting con métricas de riesgo |
| TRADE-002 | done | Cartera multi-activo y paper trading |
| TRADE-003 | done | Evaluación honesta de viabilidad: informe de riesgo y límites |
| TRADE-004 | done | Backtest realista: costes, slippage y umbral calibrado |
| TRADE-005 | done | Features adicionales: momentum, volatilidad y estacionalidad |
| TRADE-006 | done | Señales híbridas modelo+sentimiento |
| TMPL-006 | done | Sincronizar con dskit 1.16.0: corpus de conocimiento ML y mejoras del arnés |
| SENT-004 | done | Noticias históricas para sentimiento multi-empresa |
| IMP-001 | done | Mejora modelo: importancia de features, umbral de confianza y backtest |
| IMP-002 | done | Re-tuning Optuna con las 32 features actuales y validación walk-forward |
| IMP-003 | done | Tuning honesto: Optuna optimiza sobre walk-forward en vez de un split único |
| IMP-004 | done | Los tests de tuning no deben escribir artefactos en el directorio real |
| IMP-005 | done | Modelo global pooled (todas las tickers) vs modelos por ticker |
| IMP-006 | done | Enrutado híbrido: fallback al modelo global para tickers sin señal propia |
| MOD-001 | done | Comparación honesta de clases de modelo: RF vs HistGradientBoosting y ensemble de semillas |
| SENT-005 | done | Noticias reales: fetch NewsAPI.ai en vivo y histórico GDELT |
| SENT-006 | done | Fetcher NewsAPI.ai en vivo y puntuación de noticias reales |
| DOC-001 | done | README y PRD al día con el ciclo IMP/MOD/SENT |
| FIX-001 | done | mypy estricto en verde para los módulos nuevos (CI rojo tras push) |
| EXP-001 | done | Explorar mejoras RF/HistGBM/Ensemble con walk-forward estricto |
| EXP-002 | done | Re-verificar palancas internas de mejora del modelo (estacionariedad, contexto de mercado, clase, calibración, selección, recencia) |
| EXP-003 | done | Sensibilidad al horizonte/target y baselines triviales: ¿el RF bate a momentum? |
| EXP-004 | done | ¿Los datos predicen volatilidad/riesgo mejor que dirección? (¿pivot de alcance?) |
| EXP-005 | done | Modelo de riesgo: forecasting de volatilidad (aditivo al clasificador direccional) |
| EXP-006 | done | Exponer el modelo de riesgo/volatilidad en la API REST (endpoint /risk) |
| EXP-007 | done | Modelo de volatilidad: añadir features HAR (Corsi 2009) + régimen (literatura) y medir |

