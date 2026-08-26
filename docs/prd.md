# Product Requirements Document

> Documento **generado** (`documentation update_prd`) desde el estado del proyecto: `references/00-objetivo.md`, `harness/featureslist.json` y `features/*.feature`. No lo edites a mano — se sobrescribe. Actualizado: 2026-08-26

## Objetivo

_(sin definir — ejecuta la feature SCOPE-001 del backlog, que escribe `references/00-objetivo.md` con la pregunta, la métrica de éxito y el criterio de parada)_

## Alcance

**Backlog:** 1 blocked · 36 done

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
| SENT-005 | blocked | Noticias reales: fetch NewsAPI.ai en vivo y histórico GDELT |
| SENT-006 | done | Fetcher NewsAPI.ai en vivo y puntuación de noticias reales |
