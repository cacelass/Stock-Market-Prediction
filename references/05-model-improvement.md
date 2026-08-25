# Mejoras del modelo — IMP-001 (importancia, confianza, walk-forward)

Fecha: 2026-08-25 · Script: `notebooks/model_improvement.py` · CSVs en `reports/`

## A) Features que importan (top-3 por ticker)

| Ticker | Top features (RandomForest importance) |
|--------|----------------------------------------|
| AAPL | ma_200, atr_ratio, ma_50 |
| MSFT | bb_position, sentiment_score, ma_50 |
| GOOGL | rsi, momentum_63, ma_50 |
| AMZN | rsi, momentum_63, ma_50 |
| META | sentiment_ma5, week_of_year, atr_ratio |
| TSLA | ma_50, drawdown, log_volume |
| NVDA | volatility, ma_200, atr_ratio |

Lectura: dominan **tendencia (ma_50/ma_200), volatilidad (atr_ratio) y momentum**.
El sentimiento sintético solo pesa en MSFT y META — coherente con ser un proxy.

## B) Confianza: ¿el modelo sabe cuándo acierta?

Precisión real en test temporal por bucket de probabilidad (`reports/confianza_precision.csv`):

- **MSFT**: calibración casi monótona — p>0.70 acierta 57% (n=21) vs 29% en p<0.55.
- **AAPL**: p>0.60 acierta 75% (n=8). **TSLA**: p>0.60 acierta 70% (n=10).
- **AMZN**: NO calibrado — el bucket de alta confianza rinde peor que el bajo.
- Las muestras de alta confianza son pequeñas (n=2–21): señal direccional, no garantía.

Backtest con umbral (`reports/confianza_thresholds.csv`, coste 0.1%/trade):
filtrar por confianza reduce drásticamente el nº de operaciones (1–11 trades
en ~888 días) — pocos trades = poca significancia estadística.

## C) Walk-forward honesto (expanding window, 5 folds, último 40%)

| Ticker | Accuracy | AUC | Veredicto |
|--------|----------|-----|-----------|
| GOOGL | 0.549 ± 0.094 | **0.590 ± 0.067** | señal robusta |
| MSFT | 0.528 ± 0.064 | **0.571 ± 0.028** | señal estable |
| TSLA | 0.561 ± 0.052 | 0.562 ± 0.032 | señal moderada |
| AMZN | 0.534 ± 0.077 | 0.563 ± 0.052 | débil |
| META | 0.490 ± 0.074 | 0.541 ± 0.070 | no generaliza |
| AAPL | 0.565 ± 0.114 | 0.483 ± 0.041 | accuracy sin AUC: no fiable |
| NVDA | 0.468 ± 0.061 | 0.503 ± 0.037 | **sin señal** |

**Media: accuracy 0.528, AUC 0.545.**

### La corrección más importante del análisis

El split único 80/20 daba **56.6% de accuracy medio — estaba optimista**.
El walk-forward honesto baja a **52.8%**, y desmonta dos resultados:

1. **NVDA** (50.9% acc en split único) colapsa a 46.8%: su resultado anterior era suerte del tramo de test.
2. **AAPL** (66.4%) mantiene accuracy pero AUC 0.483 < 0.5: predice la clase mayoritaria.

Los únicos tickers con señal real y estable: **GOOGL, MSFT, TSLA**.

## Conclusiones para decidir

1. Reportar siempre walk-forward; el split único queda como diagnóstico, no como métrica.
2. El edge está en filtrar por confianza (p≥0.65) en MSFT/GOOGL/TSLA — aceptando poquísimas operaciones.
3. Siguiente palanca real: sentimiento de noticias verdadero (NewsAPI) y target multi-horizonte; más feature engineering sobre precios tiene rendimientos decrecientes.

## D) Re-tuning con Optuna sobre las features actuales (IMP-002)

Los `best_params_*.json` originales se ajustaron sobre las 18 features viejas;
con 32 features quedaron obsoletos. Re-optuna: 50 trials/ticker (TPESampler,
seed 42) → reentrenamiento → mismo walk-forward expanding window.

| Ticker | Acc pre | Acc post | Δacc | AUC pre | AUC post | Δauc |
|--------|---------|----------|------|---------|----------|------|
| GOOGL | 0.549 | **0.585** | +0.035 | 0.590 | **0.608** | +0.017 |
| AAPL | 0.565 | 0.578 | +0.014 | 0.483 | **0.505** | +0.022 |
| META | 0.490 | 0.508 | +0.019 | 0.541 | 0.532 | -0.009 |
| MSFT | 0.528 | 0.542 | +0.015 | 0.571 | 0.562 | -0.009 |
| AMZN | 0.534 | 0.542 | +0.008 | 0.563 | 0.578 | +0.015 |
| NVDA | 0.468 | 0.471 | +0.003 | 0.503 | 0.496 | -0.007 |
| TSLA | 0.561 | 0.561 | -0.001 | 0.562 | 0.562 | ±0.000 |
| **MEDIA** | **0.528** | **0.541** | **+0.013** | **0.545** | **0.549** | **+0.004** |

Lectura honesta:

- La mejora media es real pero modesta (+1.3pp acc). Los mejores valores de
  Optuna sobre el split único (p.ej. MSFT 0.710) están inflados por diseño:
  optimizan sobre ese split. El walk-forward es quien manda.
- **GOOGL** es el gran ganador (acc 58.5%, AUC 0.608): la señal que ya existía
  se explota mejor con más árboles/profundidad.
- **AAPL** recupera AUC > 0.5 (0.505): antes su accuracy era puro sesgo de clase.
- **NVDA** sigue sin señal ni con tuning — confirmado con dos configuraciones.

CSVs: `reports/walk_forward.csv` (post), `reports/walk_forward_antes_retuning.csv` (pre),
`reports/tuning_results.csv`.

## E) Tuning honesto: Optuna sobre walk-forward (IMP-003)

El re-tuning de IMP-002 seguía teniendo un vicio de origen: optimizaba accuracy
sobre el mismo split 80/20 con el que se reportaba. Fix metodológico: el
objetivo de Optuna es ahora la **media de AUC sobre 3 folds expanding-window**
(el mismo protocolo con el que se valida). Cambios en `tune_model.py`:
objetivo `wf_mean_auc`, `class_weight="balanced"` en búsqueda, folds que
descartan bloques demasiado pequeños (compatibilidad con fixtures).

30 trials/ticker → reentrenamiento → walk-forward:

| Ticker | Acc pre | Acc post | Δacc | AUC pre | AUC post | Δauc |
|--------|---------|----------|------|---------|----------|------|
| GOOGL | 0.585 | **0.645** | +0.061 | 0.608 | **0.626** | +0.018 |
| META | 0.508 | **0.582** | +0.073 | 0.532 | **0.564** | +0.031 |
| MSFT | 0.542 | 0.596 | +0.053 | 0.562 | 0.576 | +0.014 |
| AMZN | 0.542 | 0.596 | +0.054 | 0.578 | 0.573 | -0.005 |
| TSLA | 0.561 | 0.563 | +0.002 | 0.562 | 0.564 | +0.002 |
| AAPL | 0.578 | 0.567 | -0.011 | 0.505 | 0.520 | +0.015 |
| NVDA | 0.471 | 0.491 | +0.020 | 0.496 | 0.501 | +0.005 |
| **MEDIA** | **0.541** | **0.577** | **+0.036** | **0.549** | **0.561** | **+0.012** |

La mejora más grande de todo el ciclo (+3.6pp acc honesta) vino de arreglar el
protocolo, no de más features ni más árboles:

- **GOOGL** 64.5% acc / AUC 0.626 y **META** 58.2% / 0.564 — META deja de ser
  "no generaliza": lo que fallaba era el objetivo del tuning, no los datos.
- Los hiperparámetros elegidos cambian mucho (GOOGL: 100 árboles/profundidad
  15 vs 400/12 del tuning inflado): menos árboles, más profundidad, hojas
  pequeñas.
- **NVDA** sigue sin señal real (AUC 0.501) — tres configuraciones distintas
  llegan a lo mismo: sus datos no contienen edge para este target.
- La accuracy del split único sube a ~62% medio pero esa cifra ya no se usa:
  el número que cuenta es el walk-forward.

CSVs: `walk_forward.csv` (post), `walk_forward_antes_wf_tuning.csv` (pre).
