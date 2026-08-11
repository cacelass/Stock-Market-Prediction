# reviewer · TMPL-006

- **Fecha:** 2026-08-11
- **Veredicto:** ok

APROBADA

## Criterios
1. ✔ copier update sin errores — .copier-answers.yml: dskit_version 1.16.0, _commit 803349f (resultado coherente; no re-ejecuté copier update --trust por ser destructivo)
2. ✔ Personalizaciones conservadas — agents/agents/sentiment_agent.py existe; harness/featureslist.json con 27 features (incl. TRADE-001..006 y TMPL-006); harness/progress/ con 26 ficheros (current/history/implementer-*/reviewer-*); docs/vault/ con dominios 00_META..07_REFERENCIAS y guia_modelos.md en 01_PROYECTO; config/tickers.yaml; inversion/trading/ con signals.py, backtest.py, risk.py, portfolio.py, paper.py, viabilidad.py; inversion/sentiment/ intacto
3. ✔ Corpus ML indexado — docs/knowledge/ml/ contiene reinforcement-learning.md, metaheuristica.md, modelos-fundacionales.md, guardarraíles.md, gestion-riesgo.md, series-temporales.md; los 6 están también en docs/vault/07_REFERENCIAS/
4. ✔ Mejoras arnés activas — harness_agent.py: parsing/umbral de μ.cert; git_agent.py: commit_atomic (plan_atomic, separación en commits); review_agent.py: severity P0-P3 + confidence + ordenación. harness gate: success=True (certainty 1.0, warnings vacíos)
5. ✔ ./init.sh ENTORNO LISTO

## Evidencia ejecutada en esta sesión
- ./init.sh tail:
    ✔ pytest                 162 passed, 24 warnings in 17.79s
    ━━ ENTORNO LISTO ━━  1 aviso(s)
- harness gate: success True
- pytest agents/tests/: 638 passed, 1 skipped (52.27s)
- ruff check agents/ tests/ inversion/: All checks passed
- ruff format --check: 240 files already formatted
- secrets scan: solo fixtures de test (test_redaction.py), no secretos reales
- Makefile: refs correctas a harness/featureslist.json; no hay featureslist.json/progress/ sueltos en la raíz

## Bloqueantes
(none)

## No bloqueante
1. README.md tiene referencias stale al layout antiguo: línea 223 `vault/` (debería ser docs/vault/) y líneas 185-187 `progress/`/`vault/` (deberían ser harness/progress/ y docs/vault/). No rompe funcionalidad pero confunde al lector. Sugerencia: actualizar al cerrar.
2. detect-secrets no instalado: el scan usó el heurístico propio, más limitado. Sugerencia: pip install detect-secrets.
