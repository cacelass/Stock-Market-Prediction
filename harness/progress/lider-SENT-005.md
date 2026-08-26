# lider · SENT-005

- **Fecha:** 2026-08-26
- **Veredicto:** ok

Fetch de NewsAPI.ai implementado y probado en vivo: 2,176 artículos reales para los 7 tickers (240-354 c/u, hasta 245 fuentes únicas), mapeados al schema estándar y puntuados con analyzer.py (compound VADER). Salidas: data/raw/news_<T>_live.csv, data/processed/noticias_puntuadas_<T>.csv, sentimiento_diario_<T>.csv, reports/cobertura_noticias_reales.csv. Clave leída de NEWSAPI_KEY env (.env ignorado por git, nunca impresa). Cobertura desigual del plan gratuito: GOOGL/AMZN/META con archivos profundos (hasta 2014/2018/2022), NVDA/TSLA/AAPL/MSFT solo días recientes. Pendiente: GDELT histórico sigue en ejecución background (/tmp/opencode/gdelt_historico.log) peleando con rate limits; al terminar, reconstruir features con sentimiento real y comparar vs sintético.
