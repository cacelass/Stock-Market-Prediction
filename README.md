# Stock Market Prediction

> *"Existen numerosas formas de arruinarte. De entre ellas, crear un algoritmo de machine learning que trate de predecir el valor de las acciones en bolsa para luego invertir tu dinero, es una de las más rápidas y efectivas."*

Exploración técnica de Machine Learning aplicado a series temporales financieras. El objetivo no fue buscar rentabilidad, sino enfrentarse a un problema real con alta incertidumbre y aprender de los errores — especialmente del **overfitting** y del **data leakage**.

> CI: al publicar el repositorio, sustituye la URL del badge por la de tu fork.
> [![CI](https://github.com/cacelass/dskit/actions/workflows/ci.yml/badge.svg)](https://github.com/cacelass/dskit/actions/workflows/ci.yml)

---

## El Problema: ¿Por qué es tan difícil?

### 1. La Trampa del Overfitting

El primer obstáculo fue detectar que el modelo "hacía trampa": resultados casi perfectos en entrenamiento pero nulos en test. Si el error en train es drásticamente menor que la varianza del target, es señal inequívoca de que el modelo memoriza ruido en lugar de aprender patrones.

### 2. El Caos del Mercado

Los mercados no son estacionarios; responden a ciclos económicos y eventos impredecibles (*Cisnes Negros*) que rompen cualquier patrón histórico:

- **Macroeconomía:** Crisis inmobiliaria (2008)
- **Eventos globales:** Pandemia COVID-19 (2020)
- **Comportamiento irracional:** WallStreetBets / GameStop

---

## Estrategias Implementadas

| Decisión | Justificación |
|---|---|
| **Target binario** (sube >2% en 5 días) | Más robusto al ruido que predecir el precio exacto |
| **Lags semanales y mensuales** | Capturar tendencias, no solo fluctuaciones de un día |
| **Random Forest Classifier** | Resistencia al overfitting; soporta `predict_proba` |
| **Split temporal estricto** | Sin shuffle; evita data leakage cronológico |
| **AUC-ROC + classification_report** | Más informativo que accuracy en clases desbalanceadas |
| **Baseline automático** | El pipeline informa si supera predecir siempre la clase mayoritaria |

---

## Análisis Exploratorio (EDA) — NVDA_ml_ready.csv

Informe generado con el agente `data` del arnés sobre `data/raw/NVDA_ml_ready.csv`
(4.828 filas × 20 columnas, serie diaria 2007-01-25 → 2026-04-02).

| Aspecto | Hallazgo | Acción |
|---|---|---|
| **Tipos** | 18 numéricas + `timestamp` (fecha) + `target` binario | Sin cambios; `timestamp` se parsea como fecha |
| **Nulos** | 0 nulos en todas las columnas | Nada que imputar |
| **Duplicados** | 0 filas duplicadas | — |
| **Cardinalidad** | `timestamp` 100% único (es el índice temporal, no un ID) | Excluir de features |
| **Distribuciones** | `close/high/low/open/vwap` con outliers IQR (~15%, esperable en 20 años de precios); `return/volatility` con colas moderadas | Mantener tal cual; el modelo es robusto a outliers |
| **Correlaciones fuertes** | `close~vwap`, `high~open`, `ma_50~close`, `ma_200~close` (r>0.98) | Colinealidad esperada por construcción (medias móviles/VWAP derivan de precios); no es fuga, pero conviene vigilarla en el pipeline de features |
| **Fugas con target** | Ninguna feature correlaciona ≥0.95 con `target` (máx. |r| ≈ 0.07, `volatility`) | Sin fugas detectadas: el target (sube >2% en 5d) se construye con precios futuros, pero ninguna feature usa información futura |

Reproducible con:

```bash
make data                                    # data/raw/ → data/processed/
.venv/bin/python -m agents run data eda_report --filename NVDA_ml_ready.csv --target_col target
```

---

## Descarga multi-ticker (`make data`)

`make data` descarga el catálogo de activos definido en `config/tickers.yaml`
(AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA) con `yfinance` y, por cada ticker:

- deja `data/raw/<TICKER>_ml_ready.csv` con el **mismo esquema de 20 columnas**
  que `NVDA_ml_ready.csv` (OHLCV + features derivadas + target binario
  "sube >2% en 5 días");
- genera un informe de validación (nulos, rango de fechas, tamaño) en
  `reports/data_validation/<TICKER>_validation.json`.

Para añadir un ticker o cambiar el rango de fechas, edita `config/tickers.yaml`:

```yaml
start: "2007-01-01"
end: "2026-04-02"
tickers:
  - AAPL
  - MSFT
  # ...
```

Sin conexión a internet, `make data` falla limpio con un mensaje claro y no
escribe ningún fichero.

---

## Sentimiento Financiero (módulo `inversion/sentiment`)

Recolección y análisis de sentimiento de noticias por ticker:

```bash
# 1. Descarga noticias RSS (Google News) → data/raw/news_NVDA.csv
.venv/bin/python -m inversion.sentiment.fetch --ticker NVDA

# 2. Puntúa con VADER + léxico financiero y agrega el score compuesto medio por día
.venv/bin/python -m inversion.sentiment.analyzer --ticker NVDA
```

- `fetch.py` (SENT-001): fuente RSS real, sin API key, timeout de 10s y error
  claro (`FetchError`) si no hay red. Columnas: `fecha, titulo, cuerpo, fuente`.
- `analyzer.py` (SENT-002): VADER (`nltk.sentiment.vader`) con un léxico
  financiero ligero (`bullish`, `bearish`, `beat`, `miss`, `downgrade`…).
  Scores compuestos en `[-1, 1]`, agregado diario determinista (media simple
  por fecha y ticker). El léxico VADER base viaja embebido en el repo para
  que todo funcione sin internet.
- Tests 100% offline con `tests/fixtures/sample_news.csv` (titulares anotados
  a mano con su dirección esperada).

---

## Resultados Honestos

El modelo supera de forma consistente el baseline de clase mayoritaria, pero con margen modesto. Esto es esperado dado el ruido inherente a los mercados.

**Los rendimientos pasados no garantizan rendimientos futuros.**
Este proyecto es un ejercicio académico, no una herramienta de inversión.

### Referencias

Los papers que inspiran el diseño de features, modelos y sentimiento están en
[`reports/papers_referencia.md`](reports/papers_referencia.md) (sentimiento
financiero, features técnicas y modelos ML), generados por el agente `research`.

---

## Sistema de Agentes

El repo trae un arnés de agentes autónomos (Python puro, sin SDK de proveedor)
que automatizan el ciclo de desarrollo — análisis de datos, entrenamiento,
revisión de código, tests, RAG y despliegue:

```bash
./init.sh                                  # puerta: ¿se puede trabajar?
.venv/bin/python -m agents doctor          # diagnóstico integral
.venv/bin/python -m agents run data eda_report --filename NVDA_ml_ready.csv
.venv/bin/python -m agents run review review_package
.venv/bin/python -m agents run rag search --query "detección de drift"
.venv/bin/python -m agents run knowledge status
.venv/bin/python -m agents run audit suggest_improvements
```

Las tres memorias del proyecto no se pisan: `progress/` (feature en curso,
dueño `harness`), `agents/workspace/memory/` (trayectorias de agentes, dueño
`memory`) y `vault/` (conocimiento estable, dueño `knowledge`).

---

## Estructura del Proyecto

```
├── main.py                   ← Pipeline completo (descarga → predicción)
├── pyproject.toml            ← Dependencias (uv)
├── Makefile                  ← make data / features / train / predict / serve / tune / monitor / index-rag
├── install.md                ← Instrucciones de instalación
│
├── data/raw/                 ← Datasets descargados (gitignored)
│
├── inversion/
│   ├── data/make_dataset.py  ← Descarga vía yfinance (multi-ticker)
│   ├── features/build_features.py  ← Indicadores técnicos, lags, scaler
│   ├── models/
│   │   ├── train_model.py    ← RandomForestClassifier + evaluación
│   │   ├── predict_model.py  ← Inferencia multi-ticker (--ticker)
│   │   ├── explain_shap.py   ← Informes SHAP (make shap)
│   │   └── conformal.py      ← Predicción conformal (sets con cobertura)
│   ├── sentiment/
│   │   ├── fetch.py          ← Noticias RSS por ticker → data/raw/news_<TICKER>.csv
│   │   ├── analyzer.py       ← Sentimiento VADER + léxico financiero, agregado diario
│   │   └── data/vader_lexicon.txt  ← Léxico VADER embebido (offline)
│   ├── api/
│   │   ├── main.py           ← API REST FastAPI (make serve)
│   │   └── schemas.py        ← Modelos Pydantic
│   ├── tuning/tune_model.py  ← Optimización de hiperparámetros (Optuna, make tune)
│   ├── monitoring/drift.py   ← Detección de drift y rendimiento (make monitor)
│   ├── utils/paths.py        ← Rutas del proyecto
│   └── visualization/visualize.py  ← Gráficas
│
├── agents/                   ← Sistema de agentes (27+): doctor, data, review, rag, knowledge...
├── models/                   ← Modelos .pkl + best_params (gitignored)
├── vault/                    ← Bóveda Obsidian: conocimiento estable por dominios
├── notebooks/                ← EDA
└── reports/                  ← Informes: papers, monitoring, figuras
```

---

## Cómo Reproducir

```bash
# 1. Instalar dependencias (core + extras: rag, api, tuning, monitoring)
pip install uv && uv sync --extra rag --extra api --extra tuning --extra monitoring
#   (sin uv global: usa el .venv — todos los comandos de abajo son `make`)

# 2. Ejecutar el pipeline
make data && make features && make train

# 3. Predecir (multi-ticker)
make predict TICKER=AAPL

# 4. API REST (localhost:8000, docs en /docs)
make serve

# 5. Optimizar hiperparámetros / SHAP / monitorizar drift
make tune && make shap && make monitor

# 6. Índice RAG local + consulta en lenguaje natural
make index-rag
.venv/bin/python -m agents run rag search --query "¿cómo se entrena el modelo?"

# 7. Explorar notebooks
jupyter lab notebooks/
```

Para cambiar el ticker o los parámetros, edita las constantes al inicio de `main.py`:

```python
TICKER            = "AAPL"
PREDICTION_WINDOW = 5       # días hábiles
UPSIDE_THRESHOLD  = 0.02    # +2% para clasificar como "sube"
```

---

## Stack Técnico

`Python 3.12` · `yfinance` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `joblib` · `FastAPI` · `Optuna` · `SHAP` · `ChromaDB (RAG)` · `uv`

---

## Lecciones Aprendidas

1. **Data leakage**: cualquier feature que mire al futuro produce resultados perfectos en train e inútiles en producción.
2. **Split temporal**: hacer shuffle en series temporales es data leakage encubierto.
3. **Regresión vs. clasificación**: predecir la dirección (0/1) es más abordable que predecir el precio exacto.
4. **Baseline primero**: si el modelo no supera "predecir siempre la clase mayoritaria", no sirve de nada.
5. **Regularización**: `max_depth` y `min_samples_leaf` en Random Forest son más efectivos que aumentar `n_estimators`.

---

## Licencia

MIT — libre para uso educativo y de investigación.