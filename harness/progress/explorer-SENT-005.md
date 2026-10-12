# explorer · SENT-005

- **Fecha:** 2026-10-12
- **Veredicto:** ok

## Respuesta

**SENT-005** es la única feature `in_progress`. Su objetivo: descargar noticias reales de NewsAPI.ai en vivo y el histórico GDELT 2021-2026, documentando la cobertura real.

### SENT-005 — Criterios de aceptación
1. `fetch_newsapi_ai(ticker)` mapea a schema fecha/titulo/cuerpo/fuente con paginación y sin duplicados
2. Clave leída de `NEWSAPI_KEY` env, jamás hardcodeada ni commiteada
3. Histórico GDELT 2021-2026 descargado a `data/raw`
4. Cobertura real documentada honestamente en `references/`
5. `./init.sh` verde

### Features pendientes/blocked
Solo **SENT-005** (`in_progress`). Todas las demás están `done`. SENT-006 (la siguiente) ya está `done` — implementó el fetcher NewsAPI.ai en vivo y puntuó 4,303 artículos reales.

### Código existente para sentimiento/noticias
- `inversion/sentiment/fetch.py` — **completo**. Incluye:
  - `fetch_news()` — RSS (Google News, sin API key)
  - `fetch_historical_gdelt()` + `fetch_all_historical()` — GDELT con chunking 90d, retries, backoff exponencial
  - `fetch_newsapi_ai()` — en vivo (paginación, dedup, schema estándar)
  - `fetch_newsapi_ai_history()` — histórico por ventanas mensuales
  - `NEWSAPI_AI_KEYWORDS` — mapeo ticker → nombre comercial (ej. "NVIDIA", "(Meta OR Facebook)")
  - `_newsapi_ai_key()` — lee de `NEWSAPI_KEY` env, no hardcode
- `inversion/sentiment/analyzer.py` — VADER con léxico financiero embebido
- Datos ya descargados: 7x `news_<T>_live.csv` (~1900-2000 artículos, NewsAPI.ai), 7x `noticias_puntuadas_<T>.csv` (VADER), 7x `news_<T>.csv` (~4639, RSS)

### API keys configuradas
- **No existe `.env`** en el proyecto (no fue creado)
- No hay `.env.example`
- El código lee `NEWSAPI_KEY` del entorno; sin `.env` hay que exportarla manualmente
- `GEMINI_API_KEY` también se verifica en `env_agent.py`

### GDELT — estado
La función `fetch_historical_gdelt()` existe y está lista, pero **no se han descargado datos GDELT aún**. El `lider-SENT-005.md` menciona una ejecución en background que "pelea con rate limits", sin evidencia de que haya terminado. Es el criterio 3 pendiente.

### "GEAS"
**No existe ninguna referencia a "GEAS" en el proyecto.** El grep solo encontró "Augeas" (lexer de Pygments), irrelevante.

### Riesgos
1. **GDELT rate limits**: función lista pero GDELT es estricto. Puede tardar horas y fallar silenciosamente con pocos datos.
2. **`.env` inexistente**: NEWSAPI_KEY necesita exportarse manualmente; no hay plantilla para facilitarlo.
3. **Cobertura desigual del plan gratuito**: GOOGL/AMZN/META tienen archivos profundos (hasta 2014/2018/2022); NVDA/TSLA/AAPL/MSFT solo días recientes. Debe documentarse honestamente en `references/`.
