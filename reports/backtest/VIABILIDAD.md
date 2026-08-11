# Evaluación de viabilidad — ¿la estrategia supera a buy&hold? (TRADE-003)

Generado: 2026-08-11 16:09

Backtest **out-of-sample**: por ticker se re-entrena el modelo con el primer 70% cronológico de la serie y se simula SOLO sobre el último 30% (segmento test, nunca visto por el modelo). Las métricas las calcula `run_backtest` (TRADE-001) sobre ese segmento; buy&hold se mide en el MISMO segmento test, no sobre todo el histórico.

La simulación aplica **0.1% de comisión + 0.05% de slippage** por operación (entrada y salida), y el umbral de señal se **calibra por ticker** en el último 20% del train (rejilla 0.51-0.70, se elige el de mayor Sharpe). El segmento test nunca participa en la calibración.

## Conclusión

**La estrategia NO supera a buy&hold en el segmento test (out-of-sample):** la cartera (pesos iguales) rinde 77.50% frente a 281.70% de comprar y mantener, y solo 3 de 7 tickers superan a su buy&hold.

Los retornos espectaculares del backtest in-sample de TRADE-002 (p. ej. AAPL 1.7M%, cartera 251k%) eran sobreajuste: el modelo se evaluaba sobre los mismos datos con los que se entrenó. Con evaluación honesta desaparecen. **No invertir capital real**; la estrategia queda como ejercicio académico y, como mucho, paper trading para seguir aprendiendo.

**Impacto de los costes:** con costes realistas (0.1% comisión + 0.05% slippage por operación) y el umbral calibrado por ticker, el retorno de la cartera baja de 88.85% a 77.50%.

**Señal híbrida (modelo + sentimiento, TRADE-006):** exigir que el sentimiento del día apoye a la probabilidad del modelo reduce las operaciones; en esta ventana la cartera híbrida rinde 0.00% frente a 77.50% de solo-modelo (detalle en la sección 'Señal híbrida vs solo-modelo').

## Resultados por ticker (segmento test)

| ticker | retorno estrategia | retorno buy&hold | umbral | CAGR | Sharpe | maxDD | win rate | operaciones | ventana test |
|---|---|---|---|---|---|---|---|---|---|
| AAPL | 112.98% | 111.28% | 0.58 | 15.39% | 0.65 | -33.36% | 32.28% | 1 | 2020-12-03 → 2026-03-25 |
| MSFT | 43.95% | 80.90% | 0.55 | 7.14% | 0.41 | -32.06% | 46.51% | 19 | 2020-12-03 → 2026-03-25 |
| GOOGL | 77.64% | 222.02% | 0.56 | 11.49% | 0.55 | -42.09% | 49.39% | 17 | 2020-12-03 → 2026-03-25 |
| AMZN | 84.18% | 32.87% | 0.68 | 12.26% | 0.53 | -43.49% | 33.33% | 1 | 2020-12-03 → 2026-03-25 |
| META | 37.73% | 258.52% | 0.57 | 9.11% | 0.45 | -51.46% | 55.01% | 10 | 2022-07-18 → 2026-03-25 |
| TSLA | 186.03% | 28.66% | 0.51 | 28.14% | 0.77 | -61.14% | 51.76% | 61 | 2021-12-20 → 2026-03-25 |
| NVDA | 0.00% | 1237.65% | 0.64 | 0.00% | 0.00 | 0.00% | 0.00% | 0 | 2020-12-03 → 2026-03-25 |
| CARTERA | 77.50% | 281.70% |  | 11.93% | 0.48 | -37.66% | 38.33% | 109 | 2020-12-03 → 2026-03-25 |

## Señal híbrida vs solo-modelo

La señal híbrida (`signal_from_probability_sentiment`, TRADE-006) exige que la probabilidad del modelo y el sentimiento del día t estén **alineados**: BUY si p >= umbral Y sentiment > 0, SELL si p <= 1-umbral Y sentiment < 0. Se evalúa sobre el **mismo segmento test** y con el **mismo umbral calibrado** por ticker que la estrategia solo-modelo. El sentimiento del día t se conoce el día t (sin fuga).

| ticker | solo-modelo | híbrida | Δ híbrida | operaciones híbrida |
|---|---|---|---|---|
| AAPL | 112.98% | 0.00% | -112.98% | 0 |
| MSFT | 43.95% | 0.00% | -43.95% | 0 |
| GOOGL | 77.64% | 0.00% | -77.64% | 0 |
| AMZN | 84.18% | 0.00% | -84.18% | 0 |
| META | 37.73% | 0.00% | -37.73% | 0 |
| TSLA | 186.03% | 0.00% | -186.03% | 0 |
| NVDA | 0.00% | 0.00% | +0.00% | 0 |
| **CARTERA** | **77.50%** | **0.00%** | **-77.50%** | **0** |

**La híbrida empeora a solo-modelo en esta ventana:** la cartera rinde 0.00% frente a 77.50% de solo-modelo, y solo 0 de 7 tickers mejoran. Filtrar por sentimiento descarta operaciones; en este segmento, las descartadas habrían aportado retorno.

**Nota sobre los datos de sentimiento:** en este dataset `sentiment_score` es 0.0 en todo el histórico (las noticias crudas de `data/raw/news_*.csv` solo cubren jul-ago 2026, fuera de la ventana de precios). Con la regla estricta (BUY exige sentiment > 0, SELL exige sentiment < 0) la señal híbrida nunca dispara: 0 operaciones en todos los tickers. La comparativa es degenerada — no hay sentimiento real con el que alinear el modelo — y debe re-evaluarse cuando exista cobertura de noticias solapada con los precios.

## Limitaciones

- **Costes de transacción (comisiones y spread)**: la simulación aplica un coste fijo del 0.1% de comisión + 0.05% de slippage por operación (entrada y salida). Es representativo de brokers retail, pero no modela comisiones fijas por orden ni spreads que varían con la liquidez: un activo poco líquido o un capital pequeño cambiarían materialmente el resultado.
- **Slippage**: el backtest aplica un slippage fijo del 0.05% por operación. En la práctica el slippage se desplaza contra el operador y crece cuanto menos líquido sea el activo y mayor el tamaño de la orden; un único valor fijo no captura esa variación.
- **Sobreajuste (overfitting)**: aunque este backtest es out-of-sample, el modelo y las features se eligieron mirando el histórico completo (incluido este segmento test en iteraciones previas). El proceso de selección no es a prueba de sobreajuste: cualquier resultado nuevo debe tratarse como hipótesis hasta validarse en datos realmente no vistos.
- **Forward-looking bias**: el target (sube >2% en 5 días) usa precios futuros solo para etiquetar el entrenamiento, nunca como feature. Aun así, cualquier fuga no detectada en features o en el split inflaría los resultados.
- **Cambio de régimen de mercado**: el modelo se entrena con el pasado y se evalúa en una ventana concreta. Un cambio de comportamiento del mercado (crisis, burbuja, liquidez) puede invalidar lo aprendido; el segmento test cubre un solo régimen.
- **Horizonte del target (5 días) vs. frecuencia de trading (diaria)**: el modelo predice si el precio sube >2% en 5 días, pero la simulación decide comprar/vender con la señal diaria. La desconexión entre el horizonte de la predicción y la frecuencia de las operaciones produce señales que el modelo no fue entrenado a optimizar.
- **Colas del mercado**: se opera al cierre sin modelar huecos de apertura (gaps) ni movimiento intradía; el precio de ejecución real puede diferir del cierre.
- **Capital pequeño**: se asume que se invierte cualquier fracción del capital en cada ticker. Con capital pequeño, los costes fijos por operación y el redondeo a acciones enteras cambian materialmente el resultado.
- **Métricas de cartera aproximadas**: la fila CARTERA es la media ponderada (pesos iguales) de las métricas por ticker, igual que en backtest_report.md; no es una simulación conjunta de la cartera (correlaciones, rebalanceo y sus costes no se modelan).
- **Sesgo de supervivencia**: el catálogo (AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA) son ganadores conocidos de los últimos 20 años. Elegir activos que ya se sabe que subieron sobreestima lo que habría hecho la estrategia en tiempo real.
- **Umbral de sentimiento fijo en la señal híbrida**: la señal híbrida (TRADE-006) usa sentiment_threshold = 0.0 fijo; a diferencia del umbral de probabilidad, no se calibra por ticker. Un umbral de sentimiento calibrado podría cambiar el resultado de la comparativa.
