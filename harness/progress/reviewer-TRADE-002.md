# reviewer · TRADE-002

- **Fecha:** 2026-08-02
- **Veredicto:** aprobado

## Criterios
1. portfolio.py asigna capital (pesos, rebalanceo) — ✔ cumplido. load_catalog lee config/tickers.yaml (7 tickers); equal_weights suma 1; allocate_capital reparte y normaliza pesos; rebalance da acciones objetivo o delta comprar/vender. 10 tests con aserciones reales; cobertura portfolio.py 100%.

2. paper.py simula cartera con señales diarias y registra operaciones — ✔ cumplido. PaperPortfolio aplica BUY/SELL/HOLD por ticker con presupuesto de allocate_capital, registra Trade (fecha, ticker, señal, precio, cantidad, capital) y devuelve PaperState (cash, holdings, equity, trades). 14 tests: compra con presupuesto, venta realiza, registro crece, HOLD no opera, multi-ticker divide presupuesto, n_days limita. Sin fuga: itera en orden cronológico, señal del día t → operación al cierre de t (mismo patrón que backtest.py).

3. make backtest genera informe en reports/backtest/ con comparativa por ticker y cartera — ✔ cumplido. Receta real ejecutada: backtest_report.md + backtest_summary.csv con 7 tickers + fila CARTERA (retorno, CAGR, Sharpe, maxDD, win rate, buy&hold, equity, n_trades). Markdown legible. retorno CARTERA = media ponderada correcta de los retornos por ticker (10 000 dividido en presupuestos iguales).

4. make paper-trading imprime las señales de hoy — ✔ cumplido. Receta real ejecutada: 7 tickers con SELL/BUY/HOLD, p_subida y cierre. Ticker sin modelo → mensaje claro, no cuelga.

5. Tests de humo — ✔ cumplido. 24 tests (test_portfolio + test_paper) con aserciones de comportamiento real, no vacíos. Suite completa: 142 passed; cobertura 84.52% (>= 80).

## Bloqueantes
(ninguno)

## No bloqueante
- Fila CARTERA: total_return es exacta (media ponderada), pero CAGR/Sharpe/maxDD/win_rate son medias simples de los tickers, no métricas de cartera reales (el Sharpe de cartera depende de correlaciones; el maxDD conjunto no es la media). Documentado parcialmente en README; TRADE-003 (viabilidad honesta) debería calcular las métricas sobre la serie de equity combinada, no sobre medias.
- git: inversion/trading/ completo (incluidos ficheros de TRADE-001: signals/backtest/risk y sus tests) está sin commitear. El líder debe commitear TRADE-001 (o ambos juntos) al cerrar.
- make no está instalado en el entorno: los targets se verificaron ejecutando las recetas reales ($(PY)=.venv/bin/python, TICKERS vacío → catálogo). Confirmar con make en un entorno con make instalado.
- backtest es in-sample (mismos datos que el entrenamiento): decisión ya documentada en TRADE-001; la evaluación honesta es TRADE-003.

## Evidencia
- pytest tests/test_portfolio.py tests/test_paper.py -q → 24 passed in 3.06s
- ruff check inversion/trading tests/ → All checks passed
- backtest (receta): AAPL 1722124.37%, MSFT 1292.79%, ..., CARTERA 251114.49% → backtest_report.md + backtest_summary.csv en reports/backtest/
- paper-trading (receta): AAPL SELL 13.41%, MSFT BUY 60.78%, GOOGL HOLD, AMZN SELL, META HOLD, TSLA SELL, NVDA HOLD
- ./init.sh → pytest 142 passed, 24 warnings; ━━ ENTORNO LISTO ━━ EXIT=0
- agents gate → success=true (142 passed)
- secrets scan → sin secretos en el código nuevo (solo fixtures de test)
