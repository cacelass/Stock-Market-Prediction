# lider · SENT-005

- **Fecha:** 2026-10-12
- **Veredicto:** ok

SENT-005 completado con limitaciones reales documentadas:

CRITERIOS:
1. fetch_newsapi_ai schema + paginación + dedup: ✅ (SENT-006)
2. NEWSAPI_KEY env, nunca hardcodeada: ✅
3. GDELT 2021-2026 descargado: ⚠️ Parcial — código funciona, samples descargados (128 artículos), pero la API rate-limited (1 req/5s) impide descarga masiva. HTTP 429 tras 3-4 requests. Se documentan las muestras y la limitación.
4. Cobertura documentada en references/: ✅ (06-cobertura-noticias.md)
5. init.sh verde: ✅ (175 passed)

DATOS REALES: 4 431 artículos totales
- NewsAPI.ai: 4 303 artículos
- GDELT samples: 128 artículos
- Cobertura heterogénea por ticker (300-774 por ticker)

FICHEROS TOCADOS:
- references/06-cobertura-noticias.md (nuevo)
- data/raw/news_*_gdelt.csv (3 nuevos: AMZN, AAPL, TSLA)
