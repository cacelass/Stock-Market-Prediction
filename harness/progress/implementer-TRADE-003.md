# implementer · TRADE-003

- **Fecha:** 2026-08-02
- **Veredicto:** ok

## Qué cambié
- inversion/trading/viabilidad.py (NUEVO): backtest out-of-sample 70/30 temporal (retrain en memoria con los mismos params que train_model, sin pisar models/). Reutiliza run_backtest (TRADE-001) y _load_ticker_data. Escribe reports/backtest/VIABILIDAD.md con conclusión data-driven, tabla por ticker + CARTERA y limitaciones.
- tests/test_viabilidad.py (NUEVO): 3 smoke tests (genera informe con secciones clave, regenera al re-ejecutar, error limpio sin datos).
- Makefile: target make viabilidad (+ .PHONY + help).
- README.md: enlace a reports/backtest/VIABILIDAD.md en la sección de Trading.

## Criterios de aceptación
1. VIABILIDAD.md documenta si la estrategia supera a buy&hold con datos reales → Conclusión: NO supera (cartera 124.69% vs buy&hold 281.70% en el segmento test; solo 2/7 tickers superan). Comando: .venv/bin/python inversion/trading/viabilidad.py
2. Métricas por ticker y de cartera → tabla con retorno, CAGR, Sharpe, maxDD, win rate, buy&hold, operaciones, ventana test + fila CARTERA.
3. Limitaciones → sección con costes, slippage, overfitting, forward-looking bias, régimen, horizonte target 5d, gaps, capital pequeño, métricas de cartera, sesgo de supervivencia.
4. make viabilidad genera/actualiza el informe → receta $(PY) $(MODULE)/trading/viabilidad.py verificada (make no está instalado en este entorno).
5. Enlazado desde README → sección Trading.

## Evidencia
- pytest tests/test_viabilidad.py: 3 passed
- pytest tests/ (suite completa): 145 passed
- ruff check + format: OK; mypy --strict inversion/ tests/: 53 files OK
- Receta viabilidad: regenera reports/backtest/VIABILIDAD.md (conclusión NO viable)

## Qué falta
- Nada de implementación. La feature queda a revisión del reviewer; el cierre (harness finish + commit) es del líder.
