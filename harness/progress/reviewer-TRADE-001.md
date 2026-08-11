# reviewer · TRADE-001

- **Fecha:** 2026-08-02
- **Veredicto:** ok

APROBADA

## Criterios
1. ✔ inversion/trading/ contiene signals.py, backtest.py y risk.py
   Evidencia: ls inversion/trading/ → backtest.py, risk.py, signals.py + __init__.py
2. ✔ Señales prob→decisión con umbral configurable
   Evidencia: signal_from_probability(p, threshold=0.6): BUY si p>=thr, SELL si p<=1-thr, HOLD en medio; ValueError fuera de (0.5,1]. tests/test_signals.py 7 passed (boundaries, configurabilidad, umbral inválido)
3. ✔ Backtest produce métricas y compara vs buy&hold
   Evidencia: BacktestResult con total_return, cagr, sharpe, max_drawdown, win_rate, buy_hold_return/equity. Buy&hold sobre la misma serie e índice y mismo capital → misma base de tiempo. tests/test_backtest.py 10 passed: retorno 21% (1.1*1.1), SELL evita caída (+20% vs -40%), Sharpe 9.591 a mano, MDD -20%, CAGR 252d=retorno, win rate 2/3
4. ✔ Tests verifican métricas a mano
   Evidencia: test_sharpe_hand_computed (mean/std*√252 con ddof=1), test_max_drawdown_hand_computed (-0.25), test_cagr_one_year_equals_total_return, test_win_rate_fraction_of_signal_days_that_rose. 23 passed en tests de trading

## Puerta
./init.sh → "━━ ENTORNO LISTO ━━ 118 passed, 24 warnings in 24.79s", exit 0
harness gate → success: True
pytest trading → 23 passed in 3.59s
ruff check inversion/trading/ tests/ → All checks passed
coverage → 82.4% (≥80%)
secrets scan → sin hallazgos

## Fuga de información (forward-looking bias)
No detectada. target[t] = sube>2% en 5d (make_dataset), features de la fila t usan solo datos ≤ t (rolling/pct_change/lags). probs[t] se deriva de features de la fila t; la señal entra al cierre t y la equity se valora solo con cierres ≤ t. Sin datos futuros en señal ni en valoración.

## Bloqueantes
Ninguno.

## No bloqueante
- El backtest predice in-sample (modelo entrenado sobre el mismo histórico que simula): el 1722124% del CLI es esperable por overfitting. La evaluación honesta out-of-sample es TRADE-003, que ya lo cubre explícitamente.
- win_rate mide "% de días con señal en los que el precio subió ese día", no rentabilidad de operaciones cerradas; coherente con el docstring y verificado a mano, pero semánticamente distinto del horizonte real del target (5 días). Documentar en TRADE-003.
- run_backtest (78 líneas) supera el umbral de función larga del review; es un bucle de simulación cohesivo y legible, no justifica dividirlo.
- warnings de sklearn (feature names en scaler) preexistentes, no bloquean.
