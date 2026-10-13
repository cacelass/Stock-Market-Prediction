# Referencias 06 — Literatura de forecasting de volatilidad (EXP-007)

> Nace de la reformulación del objetivo (`references/00-objetivo.md`): la
> dirección tiene techo (AUC ≈ 0.545, EXP-002) pero la volatilidad es
> predecible (EXP-004). Antes de tocar el modelo de riesgo (`EXP-005`) se
> revisa la literatura estándar del dominio, se declara la palanca y se mide
> contra el baseline con walk-forward honesto — la misma disciplina que
> `references/05-model-improvement.md`.

## 1. El benchmark del dominio: HAR-RV (Corsi 2009)

El modelo HAR (*Heterogeneous AutoRegressive*) de Corsi es el punto de
referencia obligado para predecir volatilidad realizada. La idea: la
volatilidad agregada surge de componentes que operan a distintas escalas
temporales (diaria, semanal, mensual), y basta con esos tres lags para
reproducir la memoria larga que un AR(1) no captura.

- **Corsi, F. (2009).** *A Simple Approximate Long-Memory Model of Realized
  Volatility.* Journal of Financial Econometrics, 7(2), 174–196.
  DOI: 10.1093/jjfinec/nbp001.
- Features canónicas: `RV_d` (1 día), `RV_w` (media 5 días), `RV_m` (media 22
  días), típicamente en logaritmo. **Es exactamente lo que `EXP-007`
  implementa** en `inversion/models/volatility.py::add_har_features`.

## 2. ML contra HAR: la brecha se abre a más horizonte

- **Christensen, K., Siggaard, M., Veliyev, B. (2026).** *Machine learning
  and the term structure of realized volatility.* arXiv:2601.13014.
  Hallazgo: el ML (árboles/redes) con lags HAR bate al HAR lineal, y la
  ventaja **crece con el horizonte** — el ML aproxima mejor la persistencia
  larga que el HAR paramétrico. Justifica que nuestro horizonte sea 10 días
  (EXP-004 eligió 10d por R²/Spearman) y que el modelo sea un RF, no un HAR
  lineal.

## 3. El contrapunto: los regímenes baten al ML

- **Kilic, M. (2025).** *Linear and nonlinear econometric models against
  machine learning models: realized volatility prediction.* Finance and
  Economics Discussion Series **2025-061**, Federal Reserve Board.
  https://www.federalreserve.gov/econres/feds/files/2025061pap.pdf
  Hallazgo: sobre RV del S&P 500, los modelos con **cambio de régimen**
  (threshold-HAR, smooth-transition HAR, Markov-switching) **superan
  consistentemente** a los modelos de ML puros, y mejoran además el VaR
  out-of-sample. La no linealidad relevante no es arbitraria (como la de un
  árbol), sino **por régimen de volatilidad**.

## 4. Contexto adicional explorado (no adoptado)

- **Spillover de volatilidad entre activos** (Graph Signal Processing /
  *volatility spillover*, arXiv:2410.22706): la volatilidad de un activo
  ayuda a predecir la de otro. Cercano a las *market features* del H2 de
  EXP-002, que fueron neutras para dirección; para volatilidad no se ha
  probado todavía → candidato futuro de una feature cross-asset.
- **RV dirigida por opciones / VIX, rough volatility** (arXiv:2604.02743):
  requiere datos de opciones que el proyecto no ingiere → descartado por
  disponibilidad de datos, no por mérito.
- **HAR híbrido + indicadores técnicos con LightGBM**: la combinación
  HAR + features técnicas es lo que de facto probamos aquí (nuestro RF ya
  usa features técnicas del pipeline) — el resultado es coherente.

## 5. Qué se adoptó y qué se descartó (evidencia EXP-007)

Protocolo: walk-forward expanding-window (`_walk_forward_folds`, 3 folds),
7 tickers, target = vol realizada 10d, baseline = persistencia.

| Variante | R² medio | Spearman | ΔR² vs base |
|----------|---------:|---------:|------------:|
| base (features actuales) | 0.200 | 0.447 | — |
| **+HAR** | **0.212** | **0.453** | **+0.011** |
| +régimen (terciles) | 0.205 | 0.452 | +0.005 |
| +HAR+régimen | 0.209 | 0.457 | +0.009 |

- **Adoptado: HAR.** Mejora en **7/7 tickers** (AAPL +0.025, AMZN +0.006,
  MSFT +0.016, TSLA +0.010, GOOGL +0.003, META +0.007, NVDA +0.012).
  Consistente con Corsi (2009) y Christensen et al. (2026). Es un cambio
  pequeño pero robusto y con respaldo teórico.
- **Descartado por ahora: régimen por terciles.** Aporta poco (+0.005) y no
  se suma al HAR. La forma fuerte de la literatura (Markov-switching) es un
  modelo distinto, no una feature binaria; queda como trabajo futuro si se
  quiere perseguir la promesa de Kilic (2025).

### Conexión con `../edgebet`

El `edgebet/models/regime_detector.py` es 100% específico de apuestas
(overround, jornadas, ELO) y **no es portable como código**. Su *patrón* —
partir los datos en regímenes y evaluar/filtrar por régimen — sí es
transferible, y es lo que la variante `+régimen` explora para finanzas. No
se porta el módulo: la versión financiera (régimen de volatilidad) es una
feature, no el clasificador de fútbol.
