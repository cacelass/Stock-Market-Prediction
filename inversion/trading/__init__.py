"""inversion.trading — Señales, riesgo, backtesting, cartera y paper trading.

- signals.py        : convierte probabilidad de subida en decisión BUY/HOLD/SELL.
- risk.py           : position sizing y drawdown máximo de una serie de equity.
- backtest.py       : simula la estrategia sobre el histórico y calcula métricas.
- portfolio.py      : asigna capital entre los tickers del catálogo y rebalancea.
- paper.py          : cartera en papel que aplica las señales diarias y registra
                      operaciones.
- report_backtest.py: genera el informe multi-ticker + cartera en reports/backtest/.
"""
