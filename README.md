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

# 3. Noticias reales de NewsAPI.ai — artículos recientes por ticker
set -a; . ./.env; set +a    # exporta NEWSAPI_KEY desde .env (ignorado por git)
.venv/bin/python -m inversion.sentiment.fetch --live

# 4. Histórico NewsAPI.ai por ventanas mensuales (plan gratuito ≈ 60 días)
.venv/bin/python -m inversion.sentiment.fetch --history --since 2026-07-01
```

- `fetch.py`: tres fuentes — RSS sin clave (SENT-001), GDELT histórico
  (SENT-004; **bloqueado desde algunas redes** por rate limits) y NewsAPI.ai /
  Event Registry (SENT-005/006) en modos `--live` y `--history`, con clave solo
  vía variable de entorno `NEWSAPI_KEY`. Columnas estándar:
  `fecha, titulo, cuerpo, fuente`.
- `analyzer.py` (SENT-002): VADER (`nltk.sentiment.vader`) con un léxico
  financiero ligero (`bullish`, `bearish`, `beat`, `miss`, `downgrade`…).
  Scores compuestos en `[-1, 1]`, agregado diario determinista (media simple
  por fecha y ticker). El léxico VADER base viaja embebido en el repo para
  que todo funcione sin internet.
- Cobertura real documentada en
  [`reports/cobertura_noticias_reales.csv`](reports/cobertura_noticias_reales.csv):
  ~4.300 artículos reales puntuados para los 7 tickers. El plan gratuito de
  NewsAPI.ai limita el archivo a ~60 días — el histórico profundo queda como
  trabajo pendiente (feature SENT-005 bloqueada con la razón documentada).
- Tests 100% offline con `tests/fixtures/sample_news.csv` (titulares anotados
  a mano con su dirección esperada).

---

## Trading: señales, backtest y paper trading (módulo `inversion/trading`)

Estrategia long/out por ticker a partir de las probabilidades del modelo:

```bash
# Señales de hoy para todos los tickers del catálogo
make paper-trading                 # o filtra: make paper-trading TICKERS="AAPL MSFT"

# Backtest multi-ticker + cartera → reports/backtest/backtest_report.md
make backtest                      # o filtra: make backtest TICKERS=NVDA
```

- `signals.py` (TRADE-001): probabilidad de subida → `BUY`/`HOLD`/`SELL`
  (umbral por defecto 0.6).
- `backtest.py` (TRADE-001): simula la estrategia sobre el histórico y calcula
  retorno total, CAGR, Sharpe, max drawdown y win rate, comparando con buy&hold.
- `portfolio.py` (TRADE-002): reparte capital entre los tickers del catálogo
  (`allocate_capital`, pesos configurables) y calcula el rebalanceo
  (`rebalance`, acciones objetivo o delta a operar; fracciones, sin lotes reales).
- `paper.py` (TRADE-002): cartera en papel que aplica las señales diarias con
  presupuesto por ticker y registra cada operación (fecha, ticker, señal,
  precio, cantidad, capital).

El retorno de la cartera en el informe es la media ponderada de los retornos
por ticker (presupuestos iguales por defecto, long/out independiente). Es una
simulación sin costes, slippage ni redondeo a acciones enteras.

**Viabilidad (TRADE-003):** el informe
[`reports/backtest/VIABILIDAD.md`](reports/backtest/VIABILIDAD.md) (generado
con `make viabilidad`) evalúa la estrategia con un backtest **out-of-sample** —
split temporal 70/30: re-entrena cada modelo con el primer 70% del histórico y
simula solo sobre el último 30%, nunca visto — y documenta con números si la
estrategia supera a buy&hold y sus limitaciones (costes, slippage, sobreajuste,
forward-looking bias). El backtest de `make backtest` es in-sample: sus retornos
espectaculares son sobreajuste y no deben usarse para decidir nada.

---

## Resultados Honestos

La métrica que cuenta es el **walk-forward** (reentrenamiento expanding-window,
5 folds sobre el último 40% del histórico): el split único 80/20 resultó ser
optimista (56.6% → 52.8% de accuracy real antes del tuning correcto).

Estado tras el ciclo de mejora (detalle completo en
[`references/05-model-improvement.md`](references/05-model-improvement.md)):

| Ticker | Accuracy walk-forward | AUC walk-forward | Nota |
|--------|----------------------|------------------|------|
| GOOGL | **64.5%** | **0.626** | mejor señal del catálogo |
| META | 58.2% | 0.564 | el tuning honesto la rescató |
| MSFT / AMZN | ~59.6% | 0.56-0.58 | señal moderada |
| TSLA | 56.3% | 0.564 | idiosincrásico: el pool global le falla |
| AAPL | 56.7% | 0.520 | débil |
| NVDA | — | — | sin señal propia → predice vía modelo global (pool) |

Decisiones metodológicas validadas con experimentos:

- **Optuna optimiza sobre walk-forward**, no sobre un split único (+3.6pp de
  accuracy honesta; fue la mayor mejora del ciclo y vino del protocolo).
- **RandomForest se mantiene** frente a HistGradientBoosting: gana en AUC 7/7
  tickers. La calibración isotónica se rechazó con criterio predefinido
  (mejora ECE pero cuesta −0.029 AUC): las probabilidades crudas no son
  probabilidades reales — para operar, usar las tasas empíricas por bucket.
- **Enrutado híbrido**: los tickers sin señal propia predicen vía un modelo
  global entrenado sobre el pool de todos (NVDA hoy).

**Los rendimientos pasados no garantizan rendimientos futuros.**
Este proyecto es un ejercicio académico, no una herramienta de inversión.

### Referencias

Los papers que inspiran el diseño de features, modelos y sentimiento están en
[`reports/papers_referencia.md`](reports/papers_referencia.md) (sentimiento
financiero, features técnicas y modelos ML), generados por el agente `research`.
El diario de decisiones y veredictos del modelado vive en
[`references/`](references/).

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

Las tres memorias del proyecto no se pisan: `harness/progress/` (feature en
curso, dueño `harness`), `agents/workspace/memory/` (trayectorias de agentes,
dueño `memory`) y `docs/vault/` (conocimiento estable, dueño `knowledge`).

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
│   │   ├── train_model.py    ← RandomForestClassifier + evaluación (FEATURE_COLS única)
│   │   ├── predict_model.py  ← Inferencia multi-ticker (--ticker, respeta enrutado híbrido)
│   │   ├── pooled.py         ← Modelo global del pool + routing híbrido por señal
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
├── docs/vault/                ← Bóveda Obsidian: conocimiento estable por dominios
├── notebooks/                ← EDA y experimentos: model_improvement, pooled_model, calibración...
├── references/               ← Diario de decisiones: EDA, veredictos de modelado
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
6. **El split único miente**: optimizar y reportar sobre el mismo tramo de test infla los números; el walk-forward es quien manda — y arreglar el protocolo rindió más que cualquier feature nueva.
7. **Accuracy engaña con clases desbalanceadas**: un modelo "peor" en accuracy puede tener mejor AUC; decidir siempre mirando la métrica de ordenación.
8. **Constantes duplicadas se desincronizan**: copiar FEATURE_COLS entre módulos provocó modelos entrenados con columnas que el predictor desconocía; una sola fuente de verdad con filtro dinámico.
9. **Las probabilidades crudas no están calibradas**: p=0.65 no significa 65% de aciertos; usar tasas empíricas por bucket antes de filtrar operaciones por confianza.

---

## Licencia

MIT — libre para uso educativo y de investigación.
