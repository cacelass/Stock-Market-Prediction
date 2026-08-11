# reviewer · TRADE-006

- **Fecha:** 2026-08-11
- **Veredicto:** ok

APROBADA

## Criterios
1. ✔ Señal combinada en signals.py: signal_from_probability_sentiment(p, sentiment, threshold=0.6, sentiment_threshold=0.0) — BUY si p>=threshold Y sentiment>0; SELL si p<=1-threshold Y sentiment<0; HOLD en el resto. Lógica simétrica correcta, valida threshold en (0.5,1] y sentiment_threshold>=0. Evidencia: código + 8 tests en tests/test_signals.py.
2. ✔ Evaluada en backtest OOS: backtest_oos_ticker corre run_backtest con signal_fn=_hybrid_signal_fn(test["sentiment_score"]) sobre el MISMO segmento test (prices/probs), MISMO best_threshold calibrado y mismos costes. run_backtest gana parámetro signal_fn retrocompatible (default None → signal_from_probability; ninguno de los callers previos lo pasa). Evidencia: receta real regenera el informe; test_backtest.py test de signal_fn (34 passed en los 3 ficheros).
3. ✔ VIABILIDAD.md compara híbrida vs solo-modelo: sección "Señal híbrida vs solo-modelo" con tabla por ticker y CARTERA, Δ y operaciones; regenerada con la receta real (CARTERA solo-modelo 77.50% vs híbrida 0.00%).
4. ✔ Tests de combinación: 8 tests de lógica (alineados BUY/SELL, desalineados HOLD, sentiment==0 estricto, umbrales configurables, errores) + integración en test_backtest.py y test_viabilidad.py.

## Verificaciones específicas
- Lógica combinada correcta: BUY p>=thr Y sent>0; SELL p<=1-thr Y sent<0 (umbral de sentimiento negado simétrico). ✔
- signal_fn retrocompatible: default None → signal_from_probability; tests previos intactos (162 passed en init.sh). ✔
- Comparación justa: misma ventana test (test["close"]/probs), mismo umbral calibrado, mismos costes; sin recalibración sobre el test. ✔
- Sin fuga temporal: merge de sentimiento por fecha exacta how="left" desde precios (pipeline SENT-003 preexistente), rolling hacia atrás; la señal híbrida lee sentiment.iloc[i] alineado por posición con probs del mismo día. ✔
- Hallazgo documentado honestamente: verifiqué los datos — sentiment_score es 0.0 en los 7 tickers (fila a fila); noticias 2026-04-15→2026-08-02 vs precios hasta 2026-03-25, sin solapamiento. Documentado en VIABILIDAD.md ("Nota sobre los datos de sentimiento", "comparativa degenerada") y en el informe del implementer ("Hallazgo honesto"). ✔
- Sigue recomendando no invertir capital real: línea de la Conclusión "**No invertir capital real**" intacta en el informe regenerado. ✔

## Bloqueantes
(ninguno)

## No bloqueante
- El implementer cita las noticias como "jul-ago 2026"; el fichero AAPL arranca 2026-04-15 (abr-ago). Imprecisión menor que no cambia la conclusión (solapamiento nulo igualmente).
- Con la señal híbrida con sentimiento real la sección de conclusión híbrida asume comparación informativa; el guard "Nota sobre los datos" la precede y cubre el caso degenerado actual. Correcto hoy.

## Evidencia ejecutada (esta sesión)
- pytest tests/test_signals.py tests/test_backtest.py tests/test_viabilidad.py -q → 34 passed
- ruff check inversion/trading tests/ → All checks passed
- mypy inversion/trading/ → Success: no issues found in 8 source files
- .venv/bin/python inversion/trading/viabilidad.py → informe regenerado; CARTERA solo-modelo 77.50% vs buy&hold 281.70%
- ./init.sh → ENTORNO LISTO (162 passed)
- secrets scan → sin hallazgos
