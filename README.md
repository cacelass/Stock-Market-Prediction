# Stock Market Prediction

> *"Existen numerosas formas de arruinarte. De entre ellas, crear un algoritmo de machine learning que trate de predecir el valor de las acciones en bolsa para luego invertir tu dinero, es una de las más rápidas y efectivas."*

Exploración técnica de Machine Learning aplicado a series temporales financieras. El objetivo no fue buscar rentabilidad, sino enfrentarse a un problema real con alta incertidumbre y aprender de los errores — especialmente del **overfitting** y del **data leakage**.

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

## Resultados Honestos

El modelo supera de forma consistente el baseline de clase mayoritaria, pero con margen modesto. Esto es esperado dado el ruido inherente a los mercados.

**Los rendimientos pasados no garantizan rendimientos futuros.**
Este proyecto es un ejercicio académico, no una herramienta de inversión.

---

## Estructura del Proyecto

```
├── main.py                   ← Pipeline completo (descarga → predicción)
├── pyproject.toml            ← Dependencias (uv)
├── install.md                ← Instrucciones de instalación
│
├── data/raw/                 ← Dataset generado por main.py (gitignored)
│
├── inversion/
│   ├── data/make_dataset.py  ← Descarga vía yfinance
│   ├── features/build_features.py  ← Indicadores técnicos, lags, scaler
│   ├── models/
│   │   ├── train_model.py    ← RandomForestClassifier + evaluación
│   │   └── predict_model.py  ← Inferencia con modelo serializado
│   ├── utils/paths.py        ← Rutas del proyecto
│   └── visualization/visualize.py  ← Gráficas
│
├── models/                   ← Modelos .pkl (gitignored)
├── notebooks/                ← EDA
└── reports/figures/          ← Gráficas generadas
```

---

## Cómo Reproducir

```bash
# 1. Instalar dependencias
pip install uv && uv sync

# 2. Ejecutar el pipeline
python main.py

# 3. Explorar notebooks
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

`Python 3.11+` · `yfinance` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `joblib` · `uv`

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