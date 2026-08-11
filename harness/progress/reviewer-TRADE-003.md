# reviewer · TRADE-003

- **Fecha:** 2026-08-02
- **Veredicto:** aprobado

## Criterios
1. ✔ VIABILIDAD.md documenta si la estrategia supera a buy&hold con datos reales. Conclusión: NO supera (cartera 124.69% vs buy&hold 281.70%, 2/7 tickers). Números coherentes con la tabla.
2. ✔ Métricas por ticker y de cartera: retorno estrategia, buy&hold, CAGR, Sharpe, maxDD, win rate, operaciones, ventana test por ticker + fila CARTERA.
3. ✔ Limitaciones listadas: costes, slippage, overfitting, forward-looking bias + régimen, horizonte target 5d, gaps, capital pequeño, cartera aproximada, sesgo de supervivencia.
4. ✔ make viabilidad genera/actualiza el informe. Receta $(PY) $(MODULE)/trading/viabilidad.py = .venv/bin/python inversion/trading/viabilidad.py (make no instalado; ejecuté la receta real). Regeneró VIABILIDAD.md (timestamp 16:15 → 16:17).
5. ✔ Enlazado desde README (sección Trading, líneas 143-147).

## Honestidad del backtest (propósito de la feature)
- Split temporal 70/30 real: rows ordenadas por timestamp, split = int(70% filas), sin shuffle. El segmento test no se usó para entrenar (scaler.fit en train, transform en test; modelo solo con train).
- buy&hold medido en el MISMO segmento test: run_backtest recibe solo test["close"], buy_hold = last/first del segmento.
- Conclusión no embellecida: 2/7 superan (AAPL 113.30>111.28, AMZN 48.42>32.87); el resto no. Recomienda "No invertir capital real".
- El script re-entrena con los mismos params que train_model (n_estimators=200, max_depth=5, min_samples_leaf=50, max_features=sqrt, class_weight=balanced, random_state=42).

## Ejecuciones (reales, en esta sesión)
- pytest tests/test_viabilidad.py -q → 3 passed
- ruff check inversion/trading/viabilidad.py tests/test_viabilidad.py → All checks passed
- Receta viabilidad: regeneró el informe con números idénticos; cartera 124.69% vs 281.70% buy&hold.
- ./init.sh → ENTORNO LISTO, 145 passed, 24 warnings
- harness gate → status ok, 145 passed
- Cobertura viabilidad.py → 95%
- secrets scan: 4 hallazgos SOLO en agents/tests/test_secrets_agent.py (fixtures de test con datos falsos en tmp_path). Ningún secreto en el código de la feature.

## Bloqueantes
Ninguno.

## No bloqueante
- review_package marca run_backtest como función larga (78 líneas) — pre-existente de TRADE-001, no tocada por esta feature.
- Pocas operaciones por ticker (1-2 en ~5 años) y win rates bajos (AAPL 4.94%, AMZN 0%): la estrategia casi no opera; el informe no lo oculta (columna operaciones) y refuerza la conclusión honesta.
- El split se hace sobre filas con features válidas, no calendario; META (2022) y TSLA (2021) tienen ventanas test más cortas — documentado por fila.
