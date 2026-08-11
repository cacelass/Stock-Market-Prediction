# implementer · TRADE-002

- **Fecha:** 2026-08-02
- **Veredicto:** ok

## Qué cambié
- inversion/trading/portfolio.py: load_catalog (catálogo config/tickers.yaml), equal_weights, allocate_capital (pesos normalizados), rebalance (acciones objetivo o delta comprar/vender; fracciones, sin lotes reales)
- inversion/trading/paper.py: PaperPortfolio (long/out por ticker con presupuesto de allocate_capital, señales de signals.py), Trade/PaperState dataclasses, today_signal + CLI --ticker que imprime las señales de hoy
- inversion/trading/report_backtest.py: build_report (reusa backtest_ticker), _portfolio_row (CARTERA: retorno ponderado), write_report (backtest_report.md + backtest_summary.csv), CLI --ticker
- tests/test_portfolio.py (10 tests), tests/test_paper.py (14 tests): humo — pesos suman 1, rebalanceo cambia con precio, BUY compra/SELL vende, registro crece, informe escribe md+csv
- Makefile: targets backtest y paper-trading (usa $(PY), TICKERS vacío → todo el catálogo), .PHONY y help
- README.md: sección "Trading: señales, backtest y paper trading"
- inversion/trading/__init__.py: docstring con los nuevos módulos

## Criterios de aceptación
1. portfolio.py asigna capital (pesos, rebalanceo) → tests/test_portfolio.py 10 passed
2. paper.py simula cartera con señales diarias y registra operaciones → tests/test_paper.py 14 passed
3. make backtest genera informe en reports/backtest/ → .venv/bin/python inversion/trading/report_backtest.py (receta de make): backtest_report.md + backtest_summary.csv con 7 tickers + fila CARTERA
4. make paper-trading imprime señales de hoy → .venv/bin/python inversion/trading/paper.py: 7 tickers con BUY/HOLD/SELL
5. Tests de humo → 24 passed en test_portfolio+test_paper; suite completa 142 passed, cobertura 84.52%

## Evidencia
- pytest tests/test_portfolio.py tests/test_paper.py -q → 24 passed
- pytest tests/ --cov=inversion --cov-fail-under=80 → 142 passed, 84.52%
- ruff check inversion/ tests/ → All checks passed; ruff format --check → 51 files already formatted
- mypy --strict inversion/ tests/ → Success (51 source files)
- backtest (receta make): AAPL 1722124.37%, MSFT 1292.79%, ..., CARTERA 251114.49%
- paper-trading (receta make): AAPL SELL 13.41%, MSFT BUY 60.78%, GOOGL HOLD, AMZN SELL, META HOLD, TSLA SELL, NVDA HOLD

## Qué falta
- make no está instalado en el entorno (sin root): los targets se verificaron ejecutando las recetas exactas del Makefile ($(PY) = .venv/bin/python). Un entorno con make debe confirmar make backtest / make paper-trading / make test / make lint / make typecheck.
- TRADE-003 (viabilidad) queda fuera por diseño.
