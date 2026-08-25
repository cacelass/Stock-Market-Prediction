# Historial del arnés

Registro append-only de features completadas. Es el changelog del trabajo hecho
por agentes: qué se cerró, con qué evidencia y qué quedó pendiente.

Formato de cada entrada:

```
## <FEATURE-ID> — <título>

- **Cerrada:** YYYY-MM-DD
- **Verificación:** ./init.sh en verde · N tests pasando
- **Cambios:** rutas de los ficheros tocados
- **Decisiones:** lo que se eligió y por qué (solo si no es obvio desde el código)
- **Pendiente:** lo que se dejó fuera a propósito
```

---

_(sin features completadas todavía — este proyecto acaba de generarse)_

## QA-002 — Entorno reproducible: ./init.sh termina en ENTORNO LISTO

- **Cerrada:** 2026-08-01
- **Verificación:** ./init.sh en verde · 40 passed, 4 warnings in 14.55s
- **Cambios:** init.sh: PY prioriza .venv/bin/python cuando existe
- **Decisiones:** Sin uv global: el .venv es el intérprete canónico; el python del sistema (3.13) no tiene deps
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
[32m✔[0m siguiente tarea        QA-002 — Entorno reproducible: ./init.sh termina en ENTORNO LISTO [in_progress]
  [32m✔[0m pytest                 40 passed, 4 warnings in 15.71s
[1;32m━━ ENTORNO LISTO ━━[0m  1 aviso(s)
```

</details>

## AGENT-001 — Agente Python sentiment en el catálogo

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 69 passed, 16 warnings in 17.15s
- **Cambios:** agents/agents/sentiment_agent.py;agents/contracts.py;agents/prompts/sentiment_agent.md;agents/prompts/agents_reference.md
- **Decisiones:** Los 11 fallos de agents/tests/test_doctor_agent.py y test_env_agent.py son preexistentes y ajenos a esta feature: exigen el binario 'uv' ausente en la maquina (MissingDependencyError), no tocan sentiment.
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
.venv/bin/python -m agents list -> 'sentiment: Analiza el sentimiento financiero por ticker...' | run sentiment analyze --ticker NVDA -> success=true, 25 dias con noticias | run sentiment fetch --ticker NVDA -> success=true (patron RSS, requiere red) | routing NL: ask 'analiza el sentimiento de NVDA' -> selected_agent=sentiment, action=analyze | pytest agents/tests/test_contracts.py test_routing_scoring.py test_harness_agent.py -> 50 passed | ruff check -> All checks passed | mypy --strict sentiment_agent.py contracts.py -> Success | prompts-sync -> OK | init.sh --quick -> ENTORNO LISTO
```

</details>

## DATA-001 — EDA del dataset principal

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 69 passed, 16 warnings in 10.83s
- **Cambios:** inversion/data/make_dataset.py
- **Decisiones:** Bug real encontrado y arreglado: process_raw_to_processed() leia TODOS los *.csv de data/raw/ incluyendo news_NVDA.csv (SENT-001, columnas fecha/titulo/cuerpo/fuente) y fallaba con 'Missing column provided to parse_dates: timestamp'. Ahora filtra *_ml_ready.csv (datasets de precios); las noticias se ignoran.
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
.venv/bin/python -m inversion.data.make_dataset (download_catalog+process_raw_to_processed) -> 7 tickers descargados y procesados sin errores, CSV en data/raw/ y data/processed/ | run data eda_report --filename AAPL_ml_ready.csv -> success=true, 4638 filas x 20 columnas | run data detect_leakage --filename AAPL_ml_ready.csv --target_col target -> success=true, 0 columnas sospechosas | pytest tests/test_eda_leakage.py -> 2 passed | hallazgos en progress/implementer-DATA-001.md y README
```

</details>

## DATA-002 — Descarga multi-ticker parametrizada

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 69 passed, 16 warnings in 10.87s
- **Cambios:** inversion/data/make_dataset.py;config/tickers.yaml;tests/test_eda_leakage.py
- **Decisiones:** El fallo sin red es explicito: download_catalog descarga TODOS los tickers antes de escribir ningun CSV; si uno falla, RuntimeError y no se toca ningun fichero.
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
config/tickers.yaml define catalogo (AAPL AMZN GOOGL META MSFT NVDA TSLA) y rango 2007-03-01..2026-03-25 | make data (download_catalog) -> 7/7 tickers: AAPL/MSFT/GOOGL/AMZN/NVDA 4638 filas, META 3283, TSLA 3760, 0 nulos, informes de validacion en reports/data_validation/ | data eda_report AAPL_ml_ready.csv -> success=true | download_ticker lanza RuntimeError con mensaje claro si no hay red (sin escribir ficheros parciales: build_todos_antes_de_escribir)
```

</details>

## FEAT-001 — Pipeline de features reproducible

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 69 passed, 16 warnings in 15.74s
- **Cambios:** inversion/features/build_features.py
- **Decisiones:** Escalado con fit_and_save_scaler solo con datos de train; test test_scaler_fits_only_on_train lo verifica.
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
.venv/bin/python inversion/features/build_features.py -> 7 tickers procesados sin errores (data/interim/features_<T>_ml_ready.csv) | determinismo: hash md5 del contenido identico en 2 ejecuciones (67450fd2cad6cea988190f683ea5bdfc) | pytest tests/test_build_features.py -> 11 passed (shape/columnas, determinismo, escalado solo train)
```

</details>

## SENT-003 — Fusión de sentimiento en el pipeline de features

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 69 passed, 16 warnings in 15.68s
- **Cambios:** inversion/features/build_features.py;inversion/sentiment/analyzer.py;tests/test_build_features.py
- **Decisiones:** Para cumplir el criterio 'para cada ticker', descargadas noticias RSS de los 6 tickers sin corpus (AAPL MSFT GOOGL AMZN META TSLA) ademas de NVDA; la fusion es por fecha exacta con rolling solo hacia atras (sin fuga).
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
build_features produce features_<T>_ml_ready.csv con 4 columnas sentiment_* (sentiment_score, sentiment_noticias, sentiment_ma5, sentiment_vol) en los 7 tickers (24 columnas c/u) | no fuga: test_sentiment_features_no_leakage PASSED (el score del dia t solo usa noticias <= t) | determinismo: hash identico en 2 ejecuciones
```

</details>

## MODEL-001 — Baseline entrenado y evaluado

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 69 passed, 16 warnings in 15.73s
- **Cambios:** inversion/models/train_model.py
- **Decisiones:** RF max_depth=5 min_samples_leaf=50 seed=42; split temporal 80/20 sin shuffle (evita fuga temporal).
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
.venv/bin/python -m inversion.models.train_model (main) -> rf_<T>.pkl + scaler_<T>.pkl en models/ para 7 tickers, reports/resultados_<T>.csv con accuracy/auc/n_train/n_test/train_accuracy | reproducibilidad: hash md5 de models+reports identico en 2 ejecuciones (57fcbec4733992519aad8f0f5bd448dd) | check_overfitting replicando split temporal 80/20 seed 42: gap <= 0.062 en todos los tickers -> sin sobreajuste | split temporal honesto (primer 80% train, ultimo 20% test)
```

</details>

## MODEL-002 — Modelo multi-ticker con sentimiento y evaluación por activo

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 69 passed, 16 warnings in 15.81s
- **Cambios:** inversion/models/train_model.py;inversion/sentiment/analyzer.py;reports/comparacion_sentimiento.csv
- **Decisiones:** El sentimiento aporta mejora marginal: META la mas clara (acc +0.0405, auc +0.0029); NVDA no mejora (delta_auc +0.0093, delta_acc -0.0225) por corpus de 92 noticias concentradas en 2026-07/08 fuera del rango de train (limitacion del corpus, no sobreajuste).
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
main() entrena rf_<T>.pkl (con sentimiento) y rf_<T>_base.pkl (baseline sin sentimiento) para los 7 tickers | reports/comparacion_sentimiento.csv: metricas por activo + delta_accuracy/delta_auc (ej. META +0.0405 acc, AMZN +0.0041 auc) | check_overfitting: gap <= 0.091 en todos los tickers, sin sobreajuste | reproducibilidad: hash identico en 2 ejecuciones (57fcbec4733992519aad8f0f5bd448dd)
```

</details>

## QA-001 — Batería de calidad en verde

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 69 passed, 16 warnings in 16.19s
- **Cambios:** inversion/;tests/
- **Decisiones:** ruff format --check detecto 2 ficheros desformateados (formato puro, sin logica) y se formatearon.
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
ruff check inversion/ tests/ -> All checks passed | ruff format --check -> 26 files already formatted (2 reformateados con ruff format) | mypy --strict inversion/ tests/ -> Success: no issues found in 26 source files | pytest tests/ --cov=inversion --cov-fail-under=80 -> 69 passed, coverage 84.76% (>=80) | ./init.sh -> ENTORNO LISTO
```

</details>

## API-001 — CLI predict --ticker para cualquier activo del catálogo

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 69 passed, 16 warnings in 16.29s
- **Cambios:** inversion/models/predict_model.py;Makefile;tests/test_predict_model.py
- **Decisiones:** make predict ahora es 'predict: train' + ' .../predict_model.py --ticker ' con TICKER ?= NVDA. La CLI carga data/interim/features_<T>_ml_ready.csv (con sentiment_* para modelos con sentimiento); predict_future no re-deriva features si ya existen.
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
.venv/bin/python inversion/models/predict_model.py --ticker AAPL -> 'Prediccion para AAPL: BAJA | Probabilidad de subida: 42.73% | de bajada: 57.27% | Ultimo cierre: 252.3875' | --ticker NVDA -> BAJA, prob subida 48.59% | --ticker XYZ -> error claro con instrucciones (exit=1) | predict_model acepta ticker (predict_future(df, ticker=...)) | pytest tests/ -> 69 passed | mypy --strict predict_model.py -> Success | ruff -> ok
```

</details>

## RESEARCH-001 — Papers de referencia sobre activos, inversión y sentimiento financiero

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 69 passed, 16 warnings in 16.74s
- **Cambios:** reports/papers_referencia.md;README.md
- **Decisiones:** Se eligio OpenAlex (backend por defecto). El paper mas relevante para el repo: Financial News Sentiment Analysis using Modified VADER for Stock Price Prediction (2022) — valida el enfoque VADER+lexico financiero.
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
run research search --query 'stock market prediction sentiment analysis machine learning' -> success=true, 8 papers rankeados (openalex) | 3 busquedas tematicas: sentimiento VADER (5), features/indicadores (5), modelos ML (5) | informe escrito en reports/papers_referencia.md con 3 categorias (sentimiento/features/modelos) y limitaciones | referenciado en README.md seccion Resultados Honestos -> Referencias
```

</details>

## QA-003 — Batería de calidad en verde con el nuevo código

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 72 passed, 18 warnings in 17.86s
- **Cambios:** tests/test_predict_model.py;inversion/models/predict_model.py
- **Decisiones:** Anadidos 3 tests de CLI/API-001 que elevaron cobertura al 85.23%.
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
make lint equivalente: ruff check inversion/ tests/ -> All checks passed; ruff format --check -> 26 files already formatted | mypy --strict inversion/ tests/ -> Success (26 files) | pytest tests/ --cov=inversion --cov-fail-under=80 -> 72 passed, coverage 85.23% (>=80) | ./init.sh -> ENTORNO LISTO
```

</details>

## SENT-001 — Recolección de noticias financieras por ticker

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 72 passed, 18 warnings in 18.50s
- **Cambios:** inversion/sentiment/fetch.py;tests/test_sentiment.py;tests/fixtures/sample_news.csv
- **Decisiones:** Fuente RSS: Google News por ticker sin API key.
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
fetch_news usa RSS real de Google News: fetch_news('AAPL'/'MSFT'/'GOOGL'/'AMZN'/'META'/'TSLA') -> data/raw/news_<T>.csv (7 tickers descargados) | fixture offline tests/fixtures/sample_news.csv + RSS simulado (monkeypatch urlopen) | sin red: FetchError con mensaje claro + timeout 10s (URLError/OSError capturados) | pytest tests/test_sentiment.py -> 10 passed, todos offline
```

</details>

## SENT-002 — Analizador de sentimiento financiero (VADER)

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 72 passed, 18 warnings in 18.39s
- **Cambios:** inversion/sentiment/analyzer.py;inversion/sentiment/data/vader_lexicon.txt;tests/test_sentiment.py
- **Decisiones:** VADER embebido + lexico financiero (analyst, bull, bear, beat, miss...) para ajustar jerga de mercados; compound compuesto medio por dia.
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
sentiment analyze --ticker NVDA -> success=true, 25 dias con noticias y scores por dia (compound en [-1,1]) | determinismo: analyze_ticker('NVDA') x2 -> DataFrames identicos (sin aleatoriedad) | test de humo offline con sample_news.csv -> PASSED | agregacion verificable a mano: media de scores del dia | pytest tests/test_sentiment.py -> 10 passed
```

</details>

## TMPL-001 — Sincronizar con dskit vía copier y recuperar ficheros de repo

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 72 passed, 18 warnings in 19.72s
- **Cambios:** .github/workflows/ci.yml;.pre-commit-config.yaml;.editorconfig;SECURITY.md;CONTRIBUTING.md;models/artifacts/.gitkeep
- **Decisiones:** Actualizar sobre el repo sucio (284 ficheros sin commit, el usuario commitea) era arriesgado; se hizo copier update/recopy en worktree temporal sobre HEAD limpio y se portaron solo los ficheros de repo ausentes, sin pisar el trabajo customizado. El template a 76a2146 no genera nuevos ficheros en update normal (commit == HEAD) por eso se uso recopy. Hook final chmod init.sh falla solo en el worktree temporal (init.sh no existe en HEAD); no afecta al repo.
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
copier 9.17.0 instalado en .venv | copier update --trust --defaults: 'Keeping template version 1.14.0' + hook uv sync OK (scikit-learn OK) tras instalar uv 0.12.1 en .venv/bin | recopy genero los ficheros de repo ausentes y se copiaron al proyecto: .github/workflows/ci.yml, .pre-commit-config.yaml, .editorconfig, SECURITY.md, CONTRIBUTING.md, models/artifacts/.gitkeep | grep de <<<<<<< en los nuevos -> 0 | ./init.sh -> ENTORNO LISTO
```

</details>

## CI-001 — CI de GitHub Actions con la puerta del arnés

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 72 passed, 18 warnings in 19.43s
- **Cambios:** .github/workflows/ci.yml;pyproject.toml;README.md
- **Decisiones:** El workflow del template (dskit 1.14.0) ya integraba la puerta del arnes; solo faltaba el extra 'supervisado' en pyproject que referenciaba y el badge en README. El badge apunta al repo del template (cacelass/dskit) hasta que el usuario publique este repo en GitHub.
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
cicd validate_workflow --filename ci.yml -> success=true, 0 problems, warnings [] (cruce de referencias con Makefile sin errores) | el workflow ejecuta: lint (ruff), typecheck (mypy), tests con cobertura, agent tests, Harness gate (chmod +x init.sh; ./init.sh --quick), prompts-sync y agent evals -> reproduce make check + la puerta | extra 'supervisado' anadido a pyproject.toml (lo referenciaba el workflow y no existia) | badge CI anadido al README | pytest -> 72 passed | ./init.sh -> ENTORNO LISTO
```

</details>

## TMPL-002 — API REST FastAPI para el modelo multi-ticker (use_api)

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 78 passed, 21 warnings in 21.61s
- **Cambios:** inversion/api/__init__.py;inversion/api/main.py;inversion/api/schemas.py;tests/test_api.py;Makefile
- **Decisiones:** API adaptada al modelo real multi-ticker (rf_<T>.pkl + scaler_<T>.pkl + features interim con sentiment_*), no al patron generico de artifacts del template. Sin lifespan: carga perezosa por endpoint. Dependencias anadidas: fastapi, uvicorn, httpx.
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
make serve equivalente: uvicorn inversion.api.main:app --port 8000 arranca en localhost:8000 | GET /health -> 200 {'status':'ok','model_loaded':true,'models':[7 tickers]} | GET /info -> 200 con tickers del catalogo y features | POST /predict {ticker:TSLA} -> 200 {'prediction':0,'probability_up':0.3918,'probability_down':0.6082,'last_close':385.95,'model_name':'rf_TSLA.pkl'} | process_input('NVDA') devuelve fila con len(FEATURE_COLS) columnas | pytest tests/test_api.py -> 6 passed; suite completa 78 passed | ruff ok, mypy --strict Success | ./init.sh -> ENTORNO LISTO
```

</details>

## TMPL-003 — Calidad de modelo: SHAP + conformal + Optuna

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 89 passed, 21 warnings in 23.71s
- **Cambios:** inversion/tuning/tune_model.py;inversion/tuning/__init__.py;inversion/models/explain_shap.py;inversion/models/conformal.py;inversion/models/train_model.py;inversion/utils/paths.py;tests/test_tuning.py;tests/test_conformal.py;Makefile
- **Decisiones:** explain.py y calibrate.py del template son solo para redes_neuronales (ml_type de este repo es supervisado) y se renderizan vacios; SHAP implementado en explain_shap.py con TreeExplainer para RF. Optuna evalua sobre el split temporal honesto 80/20 del pipeline. Dependencias: optuna, shap, matplotlib.
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
make tune (tune_all/tune_ticker): Optuna TPESampler seed 42, 30 trials por ticker, guarda models/artifacts/best_params_<T>.json (AAPL: best_value 0.6746, n_estimators=50, max_depth=18, min_samples_leaf=13) y reports/tuning_results.csv | train_model carga los best_params (load_best_params) y los usa en train_ticker (verificado: 'usando hiperparametros de Optuna') | SHAP: shap_summary('NVDA') -> reports/figures/shap_NVDA.png (42KB) con TreeExplainer; make shap itera el catalogo | Conformal: tests/test_conformal.py 8 passed, test_empirical_coverage_meets_nominal verifica cobertura empirica >= 1-alpha (0.1) con margen 0.07 para metodos lac y aps | make check equivalente: 89 passed, cobertura 81.32%, ruff ok, mypy --strict Success (36 files) | ./init.sh -> ENTORNO LISTO
```

</details>

## TMPL-004 — Monitoring de drift y rendimiento (use_monitoring)

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 95 passed, 23 warnings in 25.09s
- **Cambios:** inversion/monitoring/drift.py;inversion/monitoring/__init__.py;tests/test_monitoring.py;Makefile
- **Decisiones:** PSI (Population Stability Index) con umbrales estandar 0.10/0.25; ventana reciente = ultimas 60 filas; referencia = primer 80% (split de train). Con datos reales el drift sale 'significativo' en features de nivel (ma_50/ma_200/log_volume) por la tendencia secular del mercado 2007->2026, no por bug: el test sintetico (sin tendencia) detecta exactamente el shift inyectado.
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
make monitor (monitor_all): 7 informes reports/monitoring/drift_<T>.json con PSI por feature y veredicto (sin_drift/moderado/significativo) + reports/monitoring/rendimiento.csv | drift sintetico: tests/test_monitoring.py -> 6 passed (test_psi_shifted_distribution_is_high, test_detect_drift_sintetico_inyectado con shift +2sigma en la ultima ventana detectado como moderado/significativo) | rendimiento vs baseline: rendimiento.csv compara accuracy/auc actual vs reports/resultados_<T>.csv con deltas (AAPL delta_acc -0.0101, delta_auc +0.0063) | make check equivalente: 95 passed, cobertura 81.86%, ruff ok, mypy --strict Success (39 files) | ./init.sh -> ENTORNO LISTO
```

</details>

## TMPL-005 — RAG local + vault Obsidian (use_rag + graphify)

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 95 passed, 23 warnings in 25.33s
- **Cambios:** agents/tools/rag_tool.py (nuevo, del template); agents/agents/rag_agent.py (nuevo); agents/tests/test_rag_agent.py (nuevo); agents/prompts/rag_agent.md, rag_workflow.md (nuevos); pyproject.toml (extras rag + rag_multilingual); Makefile (targets agentes/RAG: uv run → $(PY)); vault/ (estructura de dominios 00_META..07_REFERENCIAS del template + guia_modelos.md, Jinja resuelto; conserva exportación graphify de 2242 notas); .rag-index/ (índice ChromaDB, gitignored)
- **Decisiones:** RAG portado del template dskit v1.14.0 (use_rag activado; estaba en false en .copier-answers.yml). Embedder ONNX all-MiniLM-L6-v2 (inglés): híbrido con BM25 léxico en stdlib compensa el español. make no está instalado en el entorno: targets del Makefile verificados ejecutando su cuerpo con $(PY) (igual que QA-002 con uv); quedan correctos para entornos con make. guia_modelos.md renombrado de guiia_modelos.md del template (el criterio pedía guia_modelos). vault conserva la exportación graphify (2242 notas) + la estructura de dominios del template.
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
CRITERIO 1 — rag index: 'Índice actualizado: 8857 fragmentos de 2411 fuente(s) [onnx]. +5857 nuevos, 649 sin cambios, -0 huérfanos.' (success=true). Consulta: rag search --query 'cómo se entrena el modelo RandomForest' → success=true, '10 resultado(s). Top: [0.5947 vector] vault/...md:11 — # Entrena el RandomForest...' con score por resultado. CRITERIO 2 — vault/ con estructura de dominios: 00_META (IA_index.md + templates), 01_PROYECTO (agentes, arquitectura, guia_modelos.md, modelos, roadmap), 02_DATOS, 04_VISUALIZACIONES, 05_AGENTES (36 agentes), 06_OBSERVACIONES, 07_REFERENCIAS (24 técnicas), + .obsidian. guia_modelos.md: tabla comparativa (RandomForest → [[07_REFERENCIAS/random_forest|Breiman 2001]]...) y reglas prácticas. Jinja {{project_*}} resuelto (17 ficheros). CRITERIO 3 — prompts-check: '.venv/bin/python -m agents.prompts_sync' → '✔ Prompts y subagentes sincronizados con el código y con los contratos.' (target Makefile cambiado de uv run a $(PY) por consistencia con QA-002). skills: 37 skills copiadas a .opencode/skills/ (incluye rag_agent.md, rag_workflow.md). CRITERIO 4 — doc status: '✓ graphify — 2116 nodos / 0 aristas; ✓ RAG — 8857 fragmentos; ✓ vault — 2302 archivos'. doc search --query 'detección de drift' → success=true, '17 resultado(s) de 3 fuente(s): [graphify] inversion.monitoring..., [rag] vault/...md:11, [vault] vault/drift.py.md:16'. Puerta: ./init.sh → '━━ ENTORNO LISTO ━━ 1 aviso(s)', pytest 95 passed. ruff: All checks passed (rag_tool.py, rag_agent.py, test_rag_agent.py). pytest agents/tests/test_rag_agent.py → 50 passed.
```

</details>

## TRADE-001 — Señales de trading y backtesting con métricas de riesgo

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 118 passed, 24 warnings in 25.75s
- **Cambios:** inversion/trading/{__init__,signals,backtest,risk}.py; tests/test_{signals,backtest,risk}.py
- **Decisiones:** backtest in-sample (sin costes ni slippage) — evaluación out-of-sample es TRADE-003; señales por umbral (0.6); win rate = % de días con señal que subieron
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
pytest: 118 passed, 24 warnings in 24.79s (incluye 23 de TRADE-001); ./init.sh: ENTORNO LISTO EXIT_CODE=0; ruff: All checks passed; mypy --strict: Success
```

</details>

## TRADE-002 — Cartera multi-activo y paper trading

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 142 passed, 24 warnings in 26.33s
- **Cambios:** inversion/trading/{portfolio,paper,report_backtest}.py; tests/test_{portfolio,paper}.py; Makefile (backtest, paper-trading); README.md
- **Decisiones:** cartera = presupuesto por ticker + long/out; retorno de cartera media ponderada; fracciones de acción sin redondear (paper trading); make no está en el entorno — targets verificados por receta real
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
pytest: 142 passed, 24 warnings; ./init.sh: ENTORNO LISTO EXIT_CODE=0; backtest receta real: informe reports/backtest/ con 7 tickers + CARTERA; paper-trading receta real: 7 señales SELL/BUY/HOLD; ruff: All checks passed; cobertura 84.52%
```

</details>

## TRADE-003 — Evaluación honesta de viabilidad: informe de riesgo y límites

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 145 passed, 24 warnings in 27.54s
- **Cambios:** inversion/trading/viabilidad.py; tests/test_viabilidad.py; Makefile (viabilidad); README.md; reports/backtest/VIABILIDAD.md
- **Decisiones:** split temporal 70/30 (no walk-forward: coste alto, poco aporte); buy&hold en el mismo segmento test; conclusión honesta: la estrategia no supera a buy&hold out-of-sample
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
pytest: 145 passed, 24 warnings; ./init.sh: ENTORNO LISTO EXIT_CODE=0; viabilidad receta real regenera VIABILIDAD.md con conclusión honesta: cartera 124.69% vs buy&hold 281.70%, 2/7 tickers superan, NO invertir capital real; ruff: All checks passed
```

</details>

## TRADE-004 — Backtest realista: costes, slippage y umbral calibrado

- **Cerrada:** 2026-08-02
- **Verificación:** ./init.sh en verde · 148 passed, 24 warnings in 28.84s
- **Cambios:** inversion/trading/backtest.py (costes+slippage); inversion/trading/viabilidad.py (calibración umbral); tests/test_backtest.py, tests/test_viabilidad.py; reports/backtest/VIABILIDAD.md
- **Decisiones:** umbral por ticker calibrado en validación (último 20% del train), nunca en test; costes 0.1% comisión + 0.05% slippage por defecto; sigue sin superar a buy&hold — informe honesto
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
pytest: 148 passed, 24 warnings; ./init.sh: ENTORNO LISTO EXIT_CODE=0; viabilidad receta real regenera VIABILIDAD.md: cartera 53.59% vs buy&hold 281.70% con costes 0.1%+0.05%, impacto documentado (63.86%→53.59%); ruff: All checks passed
```

</details>

## TRADE-005 — Features adicionales: momentum, volatilidad y estacionalidad

- **Cerrada:** 2026-08-11
- **Verificación:** ./init.sh en verde · 151 passed, 24 warnings in 20.05s
- **Cambios:** inversion/features/build_features.py (momentum_10/21/63, volatility_21, RSI14, day_of_week, quarter); inversion/models/train_model.py y predict_model.py (FEATURE_COLS); tests/test_build_features.py, test_monitoring.py, test_tuning.py; reports/backtest/VIABILIDAD.md; data/interim/*.csv regenerados
- **Decisiones:** RSI aproximación simple (media móvil) documentada, no Wilder; sin fuga: momentum/volatilidad/RSI solo ventanas pasadas, calendario del día t; fixtures de monitoring/tuning actualizados al esquema nuevo; viabilidad mejora (53.59%→77.50%) pero sigue sin superar a buy&hold
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
pytest: 151 passed, 24 warnings; ./init.sh: ENTORNO LISTO EXIT_CODE=0; ruff check+format: All checks passed, 53 files formatted; viabilidad receta real: cartera 77.50% vs buy&hold 281.70% (antes 53.59% sin features nuevas); features regeneradas 7 datasets × 30 columnas; retrain 7 tickers completado
```

</details>

## TRADE-006 — Señales híbridas modelo+sentimiento

- **Cerrada:** 2026-08-11
- **Verificación:** ./init.sh en verde · 162 passed, 24 warnings in 17.60s
- **Cambios:** inversion/trading/signals.py (signal_from_probability_sentiment); inversion/trading/backtest.py (signal_fn opcional retrocompatible); inversion/trading/viabilidad.py (evaluación híbrida OOS); tests/test_signals.py, test_backtest.py, test_viabilidad.py; reports/backtest/VIABILIDAD.md
- **Decisiones:** señal híbrida: BUY si p>=thr Y sent>0; SELL si p<=1-thr Y sent<0; comparación justa (mismo test, mismo umbral, mismos costes); hallazgo: sentimiento 0.0 en todo el histórico por falta de solapamiento noticias-precios — re-evaluar cuando haya datos alineados
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
pytest: 162 passed, 24 warnings; ./init.sh: ENTORNO LISTO EXIT_CODE=0; ruff: All checks passed; viabilidad receta real: solo-modelo 77.50% vs buy&hold 281.70%; híbrida 0.00% por falta de solapamiento noticias-precios (documentado honestamente); 34 tests de la feature
```

</details>

## TMPL-006 — Sincronizar con dskit 1.16.0: corpus de conocimiento ML y mejoras del arnés

- **Cerrada:** 2026-08-11
- **Verificación:** ./init.sh en verde · 162 passed, 24 warnings in 18.51s
- **Cambios:** copier update dskit 1.14→1.16 (162 M + 92 D movidos a harness/ + 22 nuevos); harness/ (featureslist, progress, memory); docs/vault/ (vault fusionado); docs/knowledge/ (corpus ML nuevo); agents/ (arnés 1.16, formateado); .copier-answers.yml (1.16.0); README.md (layout corregido)
- **Decisiones:** layout nuevo del template 1.16: featureslist/progress/memory → harness/, vault → docs/vault/; personalizaciones conservadas (sentiment_agent, 27 features, historial, vault, tickers, trading); corpus ML indexado en vault; agents/ formateado (deuda del template)
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
pytest producto: 162 passed, 24 warnings; pytest arnés: 638 passed, 1 skipped; ./init.sh: ENTORNO LISTO EXIT_CODE=0; ruff check+format: All checks passed, 240 files formatted; harness gate: success=true; corpus 6 ficheros en docs/knowledge/ml y docs/vault/07_REFERENCIAS; personalizaciones: sentiment_agent, 27 features, historial, vault fusionado
```

</details>

## SENT-004 — Noticias históricas para sentimiento multi-empresa

- **Cerrada:** 2026-08-25
- **Verificación:** ./init.sh en verde · 170 passed, 24 warnings in 13.97s
- **Cambios:** inversion/sentiment/fetch.py, tests/test_sentiment_historical.py
- **Decisiones:** GDELT como fuente historica (gratis, sin API key), chunks de 1 anio, backoff exponencial
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
init.sh: ENTORNO LISTO, 170/170 tests pasan, 8/8 tests historicos pasan, GDELT API con chunking 2021-2026
```

</details>

## IMP-001 — Mejora modelo: importancia de features, umbral de confianza y backtest

- **Cerrada:** 2026-08-25
- **Verificación:** ./init.sh en verde · 170 passed, 24 warnings in 16.30s
- **Cambios:** inversion/models/train_model.py (helper available_feature_cols); predict_model.py, tune_model.py, drift.py, explain_shap.py, api/main.py, trading/backtest.py, trading/viabilidad.py (usan el helper); tests/test_predict_model.py, test_api.py, test_backtest.py, test_monitoring.py (fixtures con helper); notebooks/model_improvement.py (nuevo); references/05-model-improvement.md (nuevo)
- **Decisiones:** FEATURE_COLS duplicado en 6 módulos causó desync (modelos con 32 features vs predictores con 18); una sola fuente de verdad con filtro dinámico; walk-forward revela que split único 80/20 era optimista (56.6%→52.8%) y NVDA no tiene señal real
- **Pendiente:** _(nada)_

<details><summary>Evidencia</summary>

```
init.sh: 170 passed, 24 warnings in 15.16s — ENTORNO LISTO. reports/walk_forward.csv: accuracy media 0.528, auc media 0.545 (5 folds expanding window). reports/confianza_precision.csv + confianza_thresholds.csv + importancia_features.csv generados por notebooks/model_improvement.py
```

</details>
