# reviewer · TRADE-004

- **Fecha:** 2026-08-02
- **Veredicto:** aprobado

# Reviewer · TRADE-004 — APROBADA

## Criterios
1. ✔ run_backtest acepta coste y slippage configurable: firma cost_per_trade/slippage (default 0.0). Verificado a mano: default 0.0 reproduce equity 11000.0 exacto (comportamiento previo intacto); con costes compra a close*(1+cost+slippage) y venta cobra equity*(1-cost-slippage) → 10956.0878 = 10000*110/100.2*0.998, coincidente con el valor a mano del test.
2. ✔ viabilidad.py usa costes realistas: COST_PER_TRADE=0.001 y SLIPPAGE=0.0005, pasados a run_backtest en backtest_oos_ticker.
3. ✔ Umbral calibrado por ticker SOLO en validación: _calibrate_threshold recibe únicamente val["close"]/val_probs con val = último 20% del train (int(len(train)*(1-VALIDATION_FRACTION))); el test (último 30% cronológico) nunca participa. Rejilla 0.51-0.70, maximiza Sharpe. Umbrales por ticker documentados en VIABILIDAD.md (0.60/0.54/0.59/0.61/0.53/0.52/0.61).
4. ✔ VIABILIDAD.md regenerado: ejecuté la receta real (.venv/bin/python inversion/trading/viabilidad.py) y reproduce exactamente el informe commiteado (CARTERA 53.59%, buy&hold 281.70%). Muestra el impacto: "el retorno de la cartera baja de 63.86% a 53.59%".
5. ✔ Tests de costes: test_costs_reduce_final_return (costes ≤ sin costes con 2 operaciones) y test_cost_applied_on_buy_and_sell (valor exacto a mano). Ambos pasan.

## Puerta y calidad
- ./init.sh → ENTORNO LISTO; pytest 148 passed (últimas líneas abajo).
- harness gate → success=true.
- ruff check inversion/trading/ tests/ → All checks passed.
- Cobertura 85.8%; único fichero <60% es inversion/models/explain_shap.py (preexistente, ajeno).
- Secretos: 4 hallazgos todos en agents/tests/test_secrets_agent.py (fixtures de prueba preexistentes); ninguno en los ficheros del diff.

## Bloqueantes
Ninguno.

## No bloqueantes
- backtest_oos_ticker conserva el parámetro threshold sin uso (siempre se calibra); limpieza menor.
- NVDA con umbral calibrado 0.61 → 0 operaciones en test (0.00% vs buy&hold 1237.65%): la calibración en 20% del train es frágil para activos con pocas señales de validación. El informe lo muestra con honestidad; no bloquea, pero conviene vigilarlo en TRADE-005.
- run_backtest con 7 args/86 líneas (review_package): patrón preexistente; los 2 args nuevos son exactamente lo pedido por la feature.

## Salida real de init.sh (tail)
  ✔ featureslist.json presente
  ✔ current.md presente
  ✔ history.md presente
  ✔ subagentes lider, implementer, reviewer, explorer
  ✔ featureslist 26 features · 2 pending · 1 in_progress · 23 done · 0 blocked
  ✔ siguiente tarea TRADE-004 — Backtest realista: costes, slippage y umbral calibrado [in_progress]
  ✔ pytest 148 passed, 24 warnings in 30.12s
  ━━ ENTORNO LISTO ━━  1 aviso(s)
