# EDA y Features Derivadas — Stock Market Prediction

Fecha: 2026-08-25
Estado: Análisis completado

---

## 1. Resumen de Datos

### 1.1 Datos de Precios
| Ticker | Registros | Rango de Fechas | Nulos |
|--------|-----------|-----------------|-------|
| AAPL   | 4,638     | 2007-10-17 → 2026-03-25 | 0 |
| MSFT   | 4,638     | 2007-10-17 → 2026-03-25 | 0 |
| GOOGL  | 4,638     | 2007-10-17 → 2026-03-25 | 0 |
| AMZN   | 4,638     | 2007-10-17 → 2026-03-25 | 0 |
| META   | 3,283     | 2013-03-07 → 2026-03-25 | 0 |
| TSLA   | 3,760     | 2011-04-12 → 2026-03-25 | 0 |
| NVDA   | 4,638     | 2007-10-17 → 2026-03-25 | 0 |

**Observaciones:**
- Todos los tickers tienen datos completos sin nulos
- META y TSLA tienen menos registros porque cotizaron después
- Los datos cubren ~19 años de historial

### 1.2 Datos de Noticias
| Ticker | Noticias | Rango de Fechas | Cobertura |
|--------|----------|-----------------|-----------|
| AAPL   | ~100     | Jul-Ago 2026    | ~0.01%    |
| MSFT   | ~100     | Jul-Ago 2026    | ~0.01%    |
| GOOGL  | ~100     | Jul-Ago 2026    | ~0.01%    |
| AMZN   | ~100     | Jul-Ago 2026    | ~0.01%    |
| META   | ~100     | Jul-Ago 2026    | ~0.01%    |
| TSLA   | ~100     | Jul-Ago 2026    | ~0.01%    |
| NVDA   | ~92      | Jul-Ago 2026    | ~0.01%    |

**Problema crítico:** Las noticias solo cubren ~40 días de un historial de 19 años. Esto hace que las features de sentimiento sean 0.0 en el 99%+ de los datos.

---

## 2. Estadísticas Clave

### 2.1 Retornos (AAPL como ejemplo)
- Retorno medio diario: 0.1031%
- Volatilidad diaria: 1.9602%
- Retorno anualizado: 25.98%
- Volatilidad anualizada: 31.12%
- Sharpe ratio: 0.84

### 2.2 Autocorrelación del Retorno
- Lag 1: -0.0291 (efecto reversión corto plazo)
- Lag 5: 0.0053 (prácticamente cero)
- Lag 10: -0.0106
- Lag 20: 0.0061

**Interpretación:** Los retornos diarios tienen autocorrelación prácticamente cero, lo que sugiere eficiencia de mercado a corto plazo. Esto justifica el uso de features técnicas que capturen patrones de largo plazo.

### 2.3 Balance del Target
| Ticker | Target=0 (baja) | Target=1 (sube) | Ratio 1/0 |
|--------|-----------------|-----------------|-----------|
| AAPL   | 65.2%           | 34.8%           | 0.53      |
| MSFT   | 70.7%           | 29.3%           | 0.42      |
| GOOGL  | 68.2%           | 31.8%           | 0.47      |
| AMZN   | 65.7%           | 34.3%           | 0.52      |
| META   | 64.7%           | 35.3%           | 0.54      |
| TSLA   | 57.6%           | 42.4%           | 0.74      |
| NVDA   | 59.0%           | 41.0%           | 0.69      |

**Observación:** El target está desbalanceado (~35/65). TSLA y NVDA tienen mayor proporción de días alcistas (42% y 41% respectivamente), lo que es consistente con su fuerte tendencia alcista histórica.

---

## 3. Correlaciones Importantes

### 3.1 Correlaciones Altas (esperadas)
- `close` ↔ `vwap`: 1.0000 (derivados del mismo precio)
- `close` ↔ `high`: 0.9999
- `close` ↔ `ma_50`: 0.9964
- `close` ↔ `ma_200`: 0.9893

**Nota:** Estas correlaciones son esperadas y documentadas. El modelo RandomForest es tolerante a la colinealidad.

### 3.2 Correlaciones con el Target
- La correlación máxima absoluta con el target es < 0.1
- No hay fugas de información confirmadas

---

## 4. Outliers Detectados

| Variable | Outliers | % | Interpretación |
|----------|----------|---|----------------|
| volume | 301 | 6.5% | Días de alta actividad (noticias, earnings) |
| return | 266 | 5.7% | Días de crisis o rally |
| hl_range | 257 | 5.5% | Alta volatilidad intradia |
| volatility | 228 | 4.9% | Períodos de incertidumbre |
| momentum_63 | 160 | 3.5% | Tendencias fuertes de largo plazo |

**Interpretación:** Los outliers son reales y representan eventos de mercado (crisis 2008, COVID 2020, etc.). No deben eliminarse porque contienen información valiosa.

---

## 5. Features Derivadas (48 nuevas)

### 5.1 Bandas de Bollinger (6 features)
- `ma_20`: Media móvil de 20 días
- `std_20`: Desviación estándar de 20 días
- `bollinger_upper`: Banda superior
- `bollinger_lower`: Banda inferior
- `bollinger_width`: Ancho de bandas (volatilidad relativa)
- `bollinger_position`: Posición en la banda (0-1)

**Uso:** El ancho de bandas indica volatilidad. La posición indica si el precio está sobrecomprado (>0.8) o sobrevendido (<0.2).

### 5.2 ATR (2 features)
- `atr_14`: Average True Range de 14 días
- `atr_14_pct`: ATR como porcentaje del precio

**Uso:** Mide la volatilidad intradia normalizada. Útil para stop-loss y position sizing.

### 5.3 OBV (3 features)
- `obv`: On-Balance Volume acumulado
- `obv_ma20`: Media móvil del OBV
- `obv_signal`: Señal de compra/venta (+1/-1)

**Uso:** OBV mide la presión compradora/vendedora. Cuando el OBV sube y el precio baja, es señal de acumulación.

### 5.4 Volumen (5 features)
- `volume_ma_20`: Media móvil de volumen
- `volume_ratio_20`: Volumen relativo a su media
- `volume_ma_60`: Media móvil de 60 días
- `volume_ratio_60`: Volumen relativo a 60 días
- `volume_change`: Cambio porcentual de volumen

**Uso:** El volumen relativo indica actividad anormal. Un ratio >1.5 sugiere interés institucional.

### 5.5 Momentum (9 features)
- `rsi_delta`: Aceleración del RSI
- `rsi_ma5`: Media móvil del RSI
- `rsi_signal`: Señal de compra/venta
- `close_to_ma50_ratio`: Distancia al MA50
- `close_to_ma200_ratio`: Distancia al MA200
- `ma_cross_signal`: Señal Golden/Death Cross
- `high_20d`: Máximo de 20 días
- `low_20d`: Mínimo de 20 días
- `price_position_in_range`: Posición en rango de 20 días

**Uso:** El ratio de distancia a medias móviles indica tendencia. La posición en rango indica si el precio está en máximos o mínimos recientes.

### 5.6 Volatilidad (7 features)
- `realized_vol_5/10/21/63`: Volatilidad realized de diferentes ventanas
- `vol_ratio_5_21`: Ratio de volatilidad corto/largo plazo
- `return_vol_ratio`: Sharpe instantáneo
- `high_low_ratio_5d`: Rango de 5 días

**Uso:** El ratio de volatilidades indica si la volatilidad está aumentando o disminuyendo. El Sharpe instantáneo indica retorno ajustado a riesgo.

### 5.7 Calendario (8 features)
- `month`: Mes del año
- `week_of_year`: Semana del año
- `is_month_end`: Fin de mes
- `is_month_start`: Inicio de mes
- `is_quarter_end`: Fin de trimestre
- `is_quarter_start`: Inicio de trimestre
- `sell_in_may`: Efecto Sell in May
- `is_earnings_season`: Temporada de earnings

**Uso:** El efecto "Sell in May" es conocido. Los fines de trimestre tienen mayor volatilidad por window dressing institucional.

### 5.8 Interacción (2 features)
- `return_x_volume`: Retorno con convicción
- `rsi_x_volatility`: Sobrecomprado + volátil

**Uso:** El retorno con volumen alto es más significativo. RSI alto + volatilidad alta sugiere reversión.

### 5.9 Régimen (6 features)
- `trend_regime`: 1=alcista, 0=bajista
- `volatility_regime`: 1=alta volatilidad, 0=baja
- `drawdown_current`: Drawdown relativo al máximo de 1 año
- `peak_252d`: Máximo de 1 año
- `max_drawdown_252d`: Máximo drawdown de 1 año
- `recovery_ratio`: Cuánto se ha recuperado del drawdown

**Uso:** El régimen de tendencia indica la dirección general. El drawdown actual indica cuánto ha caído desde el máximo.

---

## 6. Hallazgos Críticos

### 6.1 Problema #1: Sentimiento Muerto
- Las 4 features de sentimiento son 0.0 en el 99%+ de los datos
- Esto invalida cualquier análisis de sentimiento
- **Solución:** Ejecutar GDELT para obtener noticias 2021-2026

### 6.2 Problema #2: Sesgo de Supervivencia
- Los 7 tickers son ganadores conocidos (AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA)
- Esto sobreestima los resultados históricos
- **Solución:** Incluir tickers que han tenido problemas (GE, IBM, etc.)

### 6.3 Fortaleza: Datos Completos
- 0 nulos en todas las variables
- ~4,600 registros por ticker (excepto META/TSLA)
- 19 años de historial

### 6.4 Fortaleza: Features Derivadas
- 48 nuevas features técnicas
- Todas sin fuga de información
- Documentadas y justificadas

---

## 7. Próximos Pasos

1. **Urgente:** Ejecutar GDELT para obtener noticias históricas
2. **Alta prioridad:** Re-entrenar modelos con nuevas features
3. **Media prioridad:** Analizar importancia de features con SHAP
4. **Baja prioridad:** Incluir tickers adicionales para reducir sesgo

---

## 8. Archivos Generados

- `notebooks/eda_complete.py`: Script de EDA completo
- `notebooks/derive_features.py`: Script de derivación de features
- `reports/eda_distribuciones.png`: Distribuciones de variables
- `reports/eda_correlaciones.png`: Matriz de correlaciones
- `reports/eda_precio_volumen.png`: Precio y volumen
- `reports/eda_boxplots.png`: Boxplots de features
- `reports/eda_retorno_vs_target.png`: Retorno vs target
- `data/interim/features_*_ml_ready.csv`: Datos con features derivadas

---

## 9. Referencias

- Papers de sentiment analysis: `docs/knowledge/papers/00-trading-sentiment-ml.md`
- Feature engineering: Papers 8.1-8.7 en la base de conocimiento
- Validación temporal: Papers 9.1-9.3
- Ensemble methods: Papers 10.1-10.5
