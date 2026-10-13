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

## F) Modelo global pooled vs por-ticker (IMP-005)

Hipótesis: ~29.9k filas de 7 tickers comparten mecánica de mercado; un RF
global puede generalizar mejor que los individuales. Diseño: solo features
scale-free (fuera ma_50/ma_200/log_volume: niveles no comparables entre
tickers), one-hot de ticker, Optuna corto (15 trials sobre wf-AUC), folds
expanding-window globales ordenados por fecha.

| Ticker | Pool acc | Pool AUC | Indiv acc | Indiv AUC | Gana pool |
|--------|----------|----------|-----------|-----------|-----------|
| AAPL | 0.584 | 0.513 | 0.567 | 0.520 | solo acc |
| MSFT | 0.655 | 0.561 | 0.596 | 0.576 | solo acc |
| GOOGL | 0.606 | 0.580 | **0.645** | **0.626** | no |
| AMZN | 0.573 | 0.552 | **0.596** | **0.573** | no |
| META | 0.549 | 0.554 | **0.582** | **0.564** | no |
| TSLA | 0.443 | 0.488 | **0.563** | **0.564** | no |
| NVDA | **0.511** | **0.539** | 0.491 | 0.501 | **sí** |

Global pool: acc 0.560, AUC 0.569.

**Veredicto: los modelos por-ticker siguen siendo la opción correcta** — el
pool gana en AUC solo 1/7. Dos hallazgos útiles:

1. **NVDA es la excepción**: con datos propios no hay señal (AUC 0.501), pero
   el pool le presta la dinámica de las otras (AUC 0.539). Para tickers sin
   señal propia, el modelo global es un fallback razonable.
2. **TSLA es idiosincrásico**: el pool le va mal (AUC 0.488 < azar). Su
   comportamiento no transfiere — refuerza mantenerlo con modelo propio.
3. La accuracy del pool a veces gana pero con AUC perdiendo: está explotando
   el desbalance de clases, no señal. El AUC manda para decidir.

Siguiente experimento natural (no ejecutado): híbrido por ticker — usar el
modelo individual cuando su wf-AUC > 0.55 y el pool como respaldo en caso
contrario. CSVs: `reports/pooled_vs_individual.csv`, script
`notebooks/pooled_model.py`.

## G) Enrutado híbrido en producción (IMP-006)

El hallazgo de F convertido en mecanismo: `inversion/models/pooled.py`
entrena el modelo global (cutoff 80/20 temporal, params del Optuna de
IMP-005; test acc 0.577, AUC 0.564) y **deriva el routing de la evidencia**,
nunca a mano: ticker→GLOBAL si su wf-AUC individual < 0.55 y el pool le gana.
Hoy eso ocurre exactamente con NVDA (`reports/enrutado_hibrido.csv`).

- `predict_future` y `backtest_ticker` consultan la ruta antes de cargar
  modelo: NVDA predice vía pool, los otros 6 con su modelo propio.
- Sin artefactos globales (tests, repo recién clonado) el enrutado es
  identidad — comportamiento idéntico al previo; cubierto por
  `tests/test_pooled.py` (5 tests).
- Las rutas de artefactos se resuelven en tiempo de llamada, no al importar:
  lección de esta feature — una constante de módulo capturó la ruta real y
  el parcheo de tests dejó de aplicarla.
- Reentrenar el global: `uv run python -m inversion.models.pooled`. El
  routing se recalcula desde `walk_forward.csv` + `pooled_vs_individual.csv`,
  así que se mantiene honesto cuando cambian los modelos individuales.

Nota: el monitoring (drift/rendimiento) sigue midiendo los modelos
individuales; si NVDA pasa a depender del pool, su drift relevante es el del
pool — pendiente si la señal individual de NVDA cambia.

## H) Clase de modelo: RF vs HistGradientBoosting (MOD-001)

Comparación en los mismos folds walk-forward (protocolo IMP-003), 7 tickers,
GBM con regularización conservadora (`notebooks/model_comparison.py`):

| Métrica media | RF-best | HistGBM |
|---------------|---------|---------|
| AUC | **0.5695** | 0.5412 |
| Accuracy | 0.5859 | 0.5901 |
| Precision@top20% | **0.4381** | 0.4128 |

**RF gana en AUC 7/7 tickers (Δ medio +0.028)** y también en la métrica que
importa para filtrar por confianza. El GBM "gana" en accuracy media otra vez
por explotar el desbalance de clases. Veredicto con margen predefinido:
MANTENER RandomForest — la clase de modelo ya es la correcta para estos datos.
CSV: `reports/comparacion_clases_modelo.csv`.

## I) Calibración isotónica: mejora ECE pero cuesta demasiado AUC (MOD-001)

La probabilidad cruda NO está calibrada (ECE medio 0.11: cuando dice 65%
acierta menos). La calibración isotónica la corrige (ECE 0.08) pero paga
0.029 de AUC — sobre ~3k filas de entrenamiento la isotónica sobreajusta.
Criterio predefinido exige ΔECE ≤ −0.02 sin perder > 0.005 AUC: no se cumple.

**Implicación práctica**: no interpretar la p del modelo como probabilidad
real. Para decidir operar, usar las tasas empíricas por bucket de
`reports/confianza_precision.csv` (p≥0.70 en MSFT ≈ 57% real), que son las que
ya reflejan la calibración observada sin destruir ordenación.
CSV: `reports/calibracion.csv`.

## J) Re-verificación de palancas internas (EXP-002)

Fecha: 2026-10-13 · Script: `notebooks/model_levers_recheck.py`

Se re-testean seis palancas clásicas de mejora con el **mismo protocolo
honesto** (expanding-window, `_walk_forward_folds`, 5 folds, último 60%), RF
del pipeline, 7 tickers. Criterio de adopción predefinido: **ΔAUC medio > 0.01**
y consistencia (que gane en la mayoría de tickers). Media baseline RF: 0.5437.

| Hipótesis | AUC media | Δ vs base | Gana en |
|-----------|-----------|-----------|---------|
| H1 — ratios ma_50/ma_200 (estacionariedad) | 0.5406 | **−0.0032** | 1/7 |
| H2 — features de mercado (mkt return/vol, fuerza relativa) | 0.5433 | −0.0004 | 3/7 |
| H3 — HistGradientBoosting | 0.5331 | **−0.0107** | 2/7 |
| H3 — Regresión logística | 0.5388 | −0.0049 | 4/7 |
| H4 — calibración con CV interno | 0.5401 | −0.0037 | 2/7 |
| H5 — selección top-8 features | 0.5372 | −0.0066 | 1/7 |
| H5 — selección top-16 features | 0.5466 | +0.0029 | 6/7 |
| H6 — ponderación por recencia (hl=500d) | 0.5457 | +0.0019 | 5/7 |
| H2 + H6 combinado | 0.5492 | +0.0054 | 6/7 |

**Ninguna hipótesis alcanza el margen de +0.01.** Las dos leves (top-16,
recencia) y su combinación mejoran de media pero dentro de ruido de validación
(se probaron con dos estructuras de folds y el signo es el mismo, pero la
magnitud queda por debajo del umbral que el propio proyecto fija para adoptar).
Las que sí tocan el motor —estacionarizar las medias móviles, cambiar de clase
de modelo, calibrar, podar features— **empeoran** el AUC.

### Conclusión de la línea de modelado (actualizada)

Confirmado con evidencia adicional: el techo honesto del modelo con estos
datos es **AUC ≈ 0.55** y ninguna transformación interna del RF lo supera de
forma robusta. Los niveles crudos `ma_50`/`ma_200` no son un lastre (su versión
estacionaria es peor), el contexto de mercado no aporta, y la clase de modelo
ya es la correcta (H y J coinciden). Las palancas que quedan son **externas al
modelo**: más histórico/noticias reales, cambio de horizonte o de target — y
eso es una decisión de `SCOPE-001`, no una mejora de tuning.

No se adopta ningún cambio: el pipeline se mantiene tal cual, por disciplina de
evidencia (no tocar lo que no mejora de forma robusta).

## K) Sensibilidad al horizonte y baselines triviales (EXP-003)

Fecha: 2026-10-13 · Scripts: `notebooks/horizon_target_sensitivity.py`,
`notebooks/horizon_5_vs_10.py`

J dejó una duda abierta: ¿el cuello de botella es el modelo o **la definición
del problema** (horizonte de 5 días)? Y una comprobación que faltaba: ¿el RF
bate de verdad a la señal más simple, o solo la está reempaquetando?

### K.1 — ¿Otro horizonte da más señal? (target = dirección pura)

AUC media del RF por horizonte, mismos folds y features (7 tickers):

| Horizonte | AUC RF | momentum (pasado h días) | reversión | volatilidad |
|-----------|--------|--------------------------|-----------|-------------|
| 1 d | 0.5121 | 0.4856 | 0.5144 | 0.4897 |
| 3 d | 0.5166 | 0.4841 | 0.5159 | 0.4853 |
| 5 d | 0.5265 | 0.4942 | 0.5058 | 0.4757 |
| **10 d** | **0.5491** | 0.4890 | 0.5110 | 0.4677 |
| 20 d | 0.5244 | 0.5035 | 0.4965 | 0.4670 |

El 10d *parecía* claramente mejor (0.5491 vs 0.5265). Pero la comparación
pareada por fold (K.2) lo desmiente.

### K.2 — Comparación pareada 10d vs 5d (21 folds/ticker-target)

| Tipo de target | ΔAUC (10d−5d) | IC95 | gana 10d | Δprec@top-20 | Δbase-rate |
|----------------|---------------|------|----------|--------------|------------|
| dirección pura | +0.0226 | ±0.0235 | 14/21 (67%) | +0.0524 | +0.0249 |
| umbral fijo 2% (el actual) | −0.0044 | ±0.0249 | 9/21 (43%) | +0.0825 | +0.0847 |
| umbral escalado (p60) | −0.0011 | ±0.0202 | 10/21 (48%) | +0.0017 | +0.0093 |

Veredicto: **el horizonte de 10 días no es robustamente mejor.**
- Con dirección pura, el IC95 **incluye el 0** (±0.0235 sobre +0.0226) → no significativo.
- Con target de **umbral** (el que usa el proyecto) la ventaja **desaparece**:
  ΔAUC negativo y gana en <50% de folds.
- La precisión@top-quintil sube en el umbral fijo (+0.0825) pero la **tasa base
  sube casi lo mismo** (+0.0847): a 10 días simplemente hay más días "que
  suben", no es que el modelo discrimine mejor. Sin mejora operativa real.

### K.3 — ¿El RF aporta edge sobre una feature suelta? (5d)

| score | AUC |
|-------|-----|
| return | 0.4927 |
| lag_1 | 0.4986 |
| lag_5 | 0.5084 |
| lag_20 | 0.4994 |
| rsi | 0.4991 |
| volatility | 0.4757 |
| vwap_ratio | 0.4895 |
| **RF (todas las features)** | **0.5265** |

El RF **sí** añade sobre la mejor feature suelta (+0.018 sobre `lag_5`). No es
un simple momentum disfrazado: combina señales débiles en una algo mejor. Pero
el techo sigue siendo bajo — la señal individual más informativa apenas pasa
de 0.51.

### Conclusión final de la línea de modelado

Con K cerrada, el horizonte y el tipo de target **no son el cuello de botella**
(probado, no supuesto), y el RF **sí** aporta valor sobre features sueltas pero
sobre una señal intrínsecamente débil. Se mantienen el horizonte de 5 días y el
target binario >2%.

Lo único que queda por encima del ruido son datos externos (noticias reales,
más histórico) o un replanteamiento de producto en `SCOPE-001` (p.ej. dejar de
predecir dirección diaria y modelar riesgo/volatilidad, que es lo que los datos
sí parecen contener). Ninguna de las dos es una mejora de tuning: son
decisiones de alcance.
