# reviewer · TRADE-005

- **Fecha:** 2026-08-11
- **Veredicto:** ok

APROBADA

## Criterios
1. ✔ build_features añade momentum (10/21/63d), volatilidad_21 (rolling 21d), RSI(14) y día semana/trimestre — diff verificado: pct_change(10/21/63), rolling(21).std(), RSI acotado, dayofweek/quarter; tests específicos test_rsi_bounded_0_100, test_momentum_no_future_leakage, test_calendar_features.
2. ✔ make features regenera datasets — receta real: .venv/bin/python inversion/features/build_features.py → 7 datasets × 30 columnas; features_AAPL_ml_ready.csv contiene las 7 columnas nuevas.
3. ✔ Tests de shape/columnas actualizados — EXPECTED_COLS con columnas nuevas, fixtures al esquema nuevo; .venv/bin/python -m pytest tests/test_build_features.py tests/test_viabilidad.py -q → 18 passed in 4.85s.
4. ✔ make train re-entrena — receta real en secuencia: .venv/bin/python inversion/models/train_model.py → 7 tickers + comparación baseline (p.ej. AAPL accuracy=0.662 auc=0.540); FEATURE_COLS idéntico (19 cols) en train_model.py y predict_model.py.
5. ✔ Informe de viabilidad regenerado y comparado — receta real: .venv/bin/python inversion/trading/viabilidad.py → CARTERA 77.50% vs buy&hold 281.70%, reproducido exactamente en VIABILIDAD.md (Generado: 2026-08-11 15:49); comparación documentada frente al informe anterior.

## Bloqueantes
Ninguno. El bloqueante de la ronda 1 (E501 tests/test_monitoring.py:20) está resuelto: la lista cols partida en una línea por columna.

## Evidencia de la puerta
- ruff check: All checks passed!
- ruff format --check: 53 files already formatted (incluye viabilidad.py, los 2 puntos pendientes resueltos)
- pytest suite: 151 passed, 24 warnings in 19.03s
- init.sh: ENTORNO LISTO (1 aviso)

## No bloqueante
- review_package: funciones largas preexistentes/aceptables (add_derived_features 84 líneas, run_backtest 86, backtest_oos_ticker 72) — no bloquea.
- Secretos: solo fixtures preexistentes en agents/tests/test_secrets_agent.py.
- TRADE-004 (viabilidad.py/backtest.py) sigue sin commitear en el working tree; el líder debe commitear en orden al cerrar.
