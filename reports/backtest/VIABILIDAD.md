# Evaluación de viabilidad — ¿la estrategia supera a buy&hold? (TRADE-003)

Generado: 2026-08-02 16:17

Backtest **out-of-sample**: por ticker se re-entrena el modelo con el primer 70% cronológico de la serie y se simula SOLO sobre el último 30% (segmento test, nunca visto por el modelo). Las métricas las calcula `run_backtest` (TRADE-001) sobre ese segmento; buy&hold se mide en el MISMO segmento test, no sobre todo el histórico.

## Conclusión

**La estrategia NO supera a buy&hold en el segmento test (out-of-sample):** la cartera (pesos iguales) rinde 124.69% frente a 281.70% de comprar y mantener, y solo 2 de 7 tickers superan a su buy&hold.

Los retornos espectaculares del backtest in-sample de TRADE-002 (p. ej. AAPL 1.7M%, cartera 251k%) eran sobreajuste: el modelo se evaluaba sobre los mismos datos con los que se entrenó. Con evaluación honesta desaparecen. **No invertir capital real**; la estrategia queda como ejercicio académico y, como mucho, paper trading para seguir aprendiendo.

## Resultados por ticker (segmento test)

| ticker | retorno estrategia | retorno buy&hold | CAGR | Sharpe | maxDD | win rate | operaciones | ventana test |
|---|---|---|---|---|---|---|---|---|
| AAPL | 113.30% | 111.28% | 15.42% | 0.66 | -33.36% | 4.94% | 1 | 2020-12-03 → 2026-03-25 |
| MSFT | 67.08% | 80.90% | 10.21% | 0.51 | -37.15% | 10.00% | 1 | 2020-12-03 → 2026-03-25 |
| GOOGL | 51.10% | 222.02% | 8.13% | 0.45 | -43.63% | 38.03% | 8 | 2020-12-03 → 2026-03-25 |
| AMZN | 48.42% | 32.87% | 7.76% | 0.39 | -51.68% | 0.00% | 1 | 2020-12-03 → 2026-03-25 |
| META | 32.85% | 258.52% | 8.05% | 0.42 | -50.85% | 58.67% | 7 | 2022-07-18 → 2026-03-25 |
| TSLA | -10.93% | 28.66% | -2.69% | 0.24 | -71.69% | 35.71% | 2 | 2021-12-20 → 2026-03-25 |
| NVDA | 571.00% | 1237.65% | 43.39% | 1.01 | -60.80% | 45.45% | 2 | 2020-12-03 → 2026-03-25 |
| CARTERA | 124.69% | 281.70% | 12.89% | 0.52 | -49.88% | 27.54% | 22 | 2020-12-03 → 2026-03-25 |

## Limitaciones

- **Costes de transacción (comisiones y spread)**: la simulación entra y sale sin pagar comisiones ni spread. Con un coste de 0.1-0.3% por operación (típico en brokers retail), una estrategia con muchas operaciones pierde gran parte de su ventaja; la columna 'operaciones' permite estimar el impacto.
- **Slippage**: el backtest opera al precio de cierre exacto. En la práctica la ejecución se desplaza contra el operador, más cuanto menos líquido sea el activo y mayor el tamaño de la orden.
- **Sobreajuste (overfitting)**: aunque este backtest es out-of-sample, el modelo y las features se eligieron mirando el histórico completo (incluido este segmento test en iteraciones previas). El proceso de selección no es a prueba de sobreajuste: cualquier resultado nuevo debe tratarse como hipótesis hasta validarse en datos realmente no vistos.
- **Forward-looking bias**: el target (sube >2% en 5 días) usa precios futuros solo para etiquetar el entrenamiento, nunca como feature. Aun así, cualquier fuga no detectada en features o en el split inflaría los resultados.
- **Cambio de régimen de mercado**: el modelo se entrena con el pasado y se evalúa en una ventana concreta. Un cambio de comportamiento del mercado (crisis, burbuja, liquidez) puede invalidar lo aprendido; el segmento test cubre un solo régimen.
- **Horizonte del target (5 días) vs. frecuencia de trading (diaria)**: el modelo predice si el precio sube >2% en 5 días, pero la simulación decide comprar/vender con la señal diaria. La desconexión entre el horizonte de la predicción y la frecuencia de las operaciones produce señales que el modelo no fue entrenado a optimizar.
- **Colas del mercado**: se opera al cierre sin modelar huecos de apertura (gaps) ni movimiento intradía; el precio de ejecución real puede diferir del cierre.
- **Capital pequeño**: se asume que se invierte cualquier fracción del capital en cada ticker. Con capital pequeño, los costes fijos por operación y el redondeo a acciones enteras cambian materialmente el resultado.
- **Métricas de cartera aproximadas**: la fila CARTERA es la media ponderada (pesos iguales) de las métricas por ticker, igual que en backtest_report.md; no es una simulación conjunta de la cartera (correlaciones, rebalanceo y sus costes no se modelan).
- **Sesgo de supervivencia**: el catálogo (AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA) son ganadores conocidos de los últimos 20 años. Elegir activos que ya se sabe que subieron sobreestima lo que habría hecho la estrategia en tiempo real.
