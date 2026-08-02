# implementer · TRADE-001

- **Fecha:** 2026-08-02
- **Veredicto:** ok

## Qué cambié
- inversion/trading/__init__.py — docstring del submódulo (estilo inversion/sentiment)
- inversion/trading/signals.py — Signal(BUY/HOLD/SELL) + signal_from_probability(p, threshold=0.6); comprar si p>=umbral, vender si p<=1-umbral, HOLD en medio; ValueError si umbral fuera de (0.5,1]
- inversion/trading/risk.py — position_size(capital, risk_per_trade=0.02, stop_loss_pct=None) y max_drawdown(equity)
- inversion/trading/backtest.py — run_backtest(prices, probs, initial_capital, threshold, risk_free_rate) → BacktestResult (retorno total, CAGR, Sharpe anualizado rf configurable, max drawdown, win rate, buy&hold) + backtest_ticker() que reutiliza predict_model (_load_model_and_scaler, _load_ticker_data, FEATURE_COLS/SENTIMENT_COLS) + CLI --ticker
- tests/test_signals.py (7), tests/test_risk.py (6), tests/test_backtest.py (10) — métricas verificadas a mano

## Criterios de aceptación
1. inversion/trading/ contiene signals.py, backtest.py, risk.py → ls inversion/trading/
2. Señales prob→decisión con umbral configurable → tests/test_signals.py (7 passed)
3. Backtest produce métricas + compara vs buy&hold → tests/test_backtest.py (10 passed): retorno=21% con subida 1.1*1.1, Sharpe constante=0, drawdown serie ascendente=0, CAGR a 252d=retorno total, win rate=2/3 a mano, SELL evita caída (+20% vs buy&hold -40%)
4. Tests verifican métricas a mano → salida pytest 23 passed

## Evidencia
pytest: 23 passed in 3.66s
ruff check: All checks passed; format: 7 files already formatted
mypy --strict inversion/trading/: Success, no issues in 4 source files
CLI real: python inversion/trading/backtest.py --ticker AAPL → retorno 1722124%, buy&hold 5281%, CAGR 73.99%, Sharpe 3.31, MDD -19.59%, win rate 52.48%

## Qué falta
- No toqué Makefile (target make backtest con informe en reports/backtest/ pertenece a TRADE-002 según su criterio)
- Sin costes/slippage: simulación naive; la evaluación honesta de viabilidad es TRADE-003
