# implementer · TRADE-004

- **Fecha:** 2026-08-02
- **Veredicto:** ok

## Qué cambié
- inversion/trading/backtest.py: run_backtest acepta cost_per_trade y slippage (fracción, default 0.0). Comprar ejecuta a close*(1+cost+slippage); vender cobra equity*(1-cost-slippage). Sin costes el comportamiento es idéntico al previo (12 tests existentes siguen pasando).
- inversion/trading/viabilidad.py: costes por defecto 0.1% comisión + 0.05% slippage por operación; calibración de umbral por ticker en el último 20% del train (rejilla 0.51-0.70, se elige el de mayor Sharpe; el test nunca participa); informe con columna umbral, frase de impacto de costes y limitaciones actualizadas.
- tests/test_backtest.py: 2 tests nuevos (costes reducen retorno; coste aplicado en buy y sell con valor exacto).
- tests/test_viabilidad.py: aserciones nuevas (comisión, slippage, umbral, Impacto de los costes) + test de _calibrate_threshold.
- reports/backtest/VIABILIDAD.md: regenerado con la receta real.

## Criterios de aceptación
1. run_backtest acepta coste y slippage: .venv/bin/python -m pytest tests/test_backtest.py -q → 12 passed.
2. viabilidad.py usa costes 0.1% + 0.05% por defecto: COST_PER_TRADE=0.001, SLIPPAGE=0.0005 en viabilidad.py y se pasan a run_backtest.
3. Umbral calibrado por ticker en validación (último 20% train), test intacto: _calibrate_threshold solo recibe val["close"]/val_probs. Umbrales: AAPL 0.60, MSFT 0.54, GOOGL 0.59, AMZN 0.61, META 0.53, TSLA 0.52, NVDA 0.61.
4. VIABILIDAD.md regenerado (salida de la receta pegada abajo) con columna umbral y frase: "con costes realistas ... el retorno de la cartera baja de 63.86% a 53.59%".
5. Tests de costes: test_costs_reduce_final_return y test_cost_applied_on_buy_and_sell.

## Evidencia
- tests/test_backtest.py + test_viabilidad.py: 16 passed
- suite completa: 148 passed
- ruff check inversion/ tests/: All checks passed
- mypy inversion/trading/: Success
- receta real: .venv/bin/python inversion/trading/viabilidad.py → CARTERA retorno=53.59% buy&hold=281.70% (antes sin costes 124.69%)

## Qué falta
- Revisión del reviewer; bump de versión y commit los hace git commit_feature (líder).
