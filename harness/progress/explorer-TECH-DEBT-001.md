# explorer · TECH-DEBT-001

- **Fecha:** 2026-10-12
- **Veredicto:** ok

## Respuesta

Audit técnico completo del proyecto Stock-Market-Prediction. Se encontraron **14 problemas** de calidad, deuda técnica y código muerto. Los más críticos son:
1. 30 ficheros binarios `.pkl` (84 MB) versionados en git sin `*.pkl` en `.gitignore`.
2. `SENTIMENT_COLS` importado en `predict_model.py` pero nunca usado dentro del fichero.
3. CI ignora silenciosamente los fallos de seguridad de bandit (`continue-on-error: true`).
4. `data/raw/data.csv` referenciado por código pero no existe en el filesystem.

## Dónde está

### Dead import
- `inversion/models/predict_model.py:24` — `SENTIMENT_COLS` se importa desde `train_model` pero nunca se usa dentro de `predict_model.py`.

### Ficheros binarios en git
- `models/` contiene 30 ficheros `.pkl` (84 MB total) + 10 JSON de `best_params_*`.
- No hay `*.pkl` ni `*.pkl` en `.gitignore`.
- `git ls-files models/` devuelve 41 ficheros versionados (30 `.pkl` + 10 JSON + 1 `.gitkeep` x2).

### Datos en el directorio equivocado
- `data/raw/` contiene ficheros `*_ml_ready.csv` (7) y `news_*_live.csv` (8) que son datos procesados, no raw. Deberían estar en `data/processed/` o `data/interim/`.
- `inversion/utils/paths.py:22` define `RAW_DATA_FILE = RAW_DATA_DIR / "data.csv"` pero ese fichero no existe.

### CI: bandit ignorado
- `.github/workflows/ci.yml:76-77`: `bandit -r inversion/ -f txt -x tests || true` + `continue-on-error: true`. Los fallos de seguridad no bloquean la CI.

### Scripts duplicados en notebooks/
- `notebooks/` contiene 2 notebooks `.ipynb` + 6 scripts `.py` (calibracion.py, derive_features.py, eda_complete.py, model_comparison.py, model_improvement.py, pooled_model.py). Estos `.py` parecen versiones extraídas de notebooks — posible duplicación de código.

### Reports: resultados de tickers no en catálogo
- `reports/resultados_CCC.csv` y `reports/resultados_TST.csv` (creados el 12 oct) parecen de tickers no presentes en el catálogo principal (AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA).
- 19 PNGs en `reports/` (>5 MB cada uno: eda_*, importancia_*): ¿se regeneran automáticamente?

### visualization/__init__.py vacío
- `inversion/visualization/__init__.py` tiene 0 bytes (ni siquiera docstring). `inversion/tuning/__init__.py` solo tiene un docstring de una línea.

## Riesgos y sorpresas

1. **Git sin LFS**: 84 MB en `.pkl` ocupan espacio de commit y history. Si el proyecto crece, el historial se inflará. O bien añadir `*.pkl` a `.gitignore` y usar Git LFS, o eliminar los modelos del historial con `git filter-repo`.

2. **Bandit silencioso**: El paso de seguridad de CI ejecuta bandit pero ignora cualquier fallo (`|| true` + `continue-on-error`). En un proyecto de trading con claves API, esto deja brechas sin alerta.

3. **SENTIMENT_COLS no usado**: `predict_model.py` importa `SENTIMENT_COLS` pero no lo usa internamente. Limpieza de imports recomendada.

4. **RAW_DATA_FILE inexistente**: `paths.py` apunta a `data/raw/data.csv` pero no existe. `make_dataset.py` maneja el error pero el proyecto no funciona sin este fichero o sin que se cambie la ruta por defecto.

5. **Datos en data/raw/**: Los ficheros `*_ml_ready.csv` en `data/raw/` violan la convención de datos (raw = sin procesar). Mezcla datos crudos con datos ML-ready.

## Sin verificar

- No verifiqué si los notebooks `.py` en `notebooks/` están sincronizados con las versiones de los `.ipynb` (pueden estar desactualizados).
- No verifiqué si los PNGs en `reports/` se regeneran automáticamente o son estáticos.
- No ejecuté `./init.sh` completo — solo verifiqué que los imports funcionan.
