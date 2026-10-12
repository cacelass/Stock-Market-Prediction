# Cobertura de Noticias — Documentación Real (SENT-005)

## Resumen

| Fuente | Artículos | Rango temporal | Limitaciones |
|--------|-----------|----------------|--------------|
| NewsAPI.ai (Event Registry) | 4 303 | ~2014–2026 | Plan gratuito: ~60 días en modo live; histórico mensual limitado |
| GDELT v2 (ArtList) | 128 | 2021–2026 | Rate limit 1 req/5s; no viable para descarga masiva |
| RSS Google News | ~100 por ticker | 2026 (reciente) | Solo titulares recientes; sin cuerpo del artículo |

**Total artículos reales:** 4 431

## Limitaciones documentadas

### GDELT — No viable para histórico completo
GDELT v2 API restringe a **1 petición cada 5 segundos**. Descargar el
histórico completo 2021–2026 para los 7 tickers requeriría ~140 peticiones
(mínimo 12 min sin errores) + los retries con backoff. Desde este entorno,
el bloqueo se produce tras ~3-4 peticiones:

```
HTTP 429: Too Many Requests
Please limit requests to one every 5 seconds or contact
kalev.leetaru5@gmail.com for larger queries.
```

Las muestras descargadas (128 artículos) confirman que el código funciona
pero la API es inutilizable para datos históricos masivos.

### NewsAPI.ai — Cobertura parcial
El plan gratuito (200 req/día) permite descargar ~240-354 artículos por
ticker en modo live. El modo history por ventanas mensuales acumula hasta
~750 artículos por ticker, pero solo los ~60 días más recientes del plan
gratuito. Para histórico profundo se requiere plan pago.

### RSS — Solo titulares recientes
Google News RSS devuelve titulares de las últimas horas. Útil para
señales en tiempo real pero sin cobertura histórica.

## Impacto en el modelo

Las features de sentimiento (`sentiment_score`, `sentiment_noticias`,
`sentiment_ma5`, `sentiment_vol`) se basan en estos 4 431 artículos.
La cobertura es **heterogénea**: MSFT y AAPL tienen ~750 artículos;
META ~300; el resto entre 400-600. Esto introduce sesgo por ticker
en las predicciones de sentimiento.

## Alternativas consideradas

| Alternativa | Viabilidad | Esfuerzo |
|-------------|------------|----------|
| Financial PhraseBank (dataset pre-compilado) | Alta | Bajo |
| GDELT ngrams (dataset masivo, no API) | Alta | Medio |
| Plan pago NewsAPI.ai (~100€/mes) | Alta | Bajo |
| Web scraping de portales financieros | Media (legal) | Alto |

## Archivos generados

```
data/raw/news_<T>_live.csv        — NewsAPI.ai (~4 303 total)
data/raw/news_<T>_gdelt.csv       — GDELT muestras (~128 total)
data/processed/noticias_puntuadas_<T>.csv   — Puntuadas con VADER
data/processed/sentimiento_diario_<T>.csv   — Agregadas por día
```
