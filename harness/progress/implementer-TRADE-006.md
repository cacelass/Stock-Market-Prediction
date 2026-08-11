# implementer · TRADE-006

- **Fecha:** 2026-08-11
- **Veredicto:** ok

## Qué cambié
- inversion/trading/signals.py: signal_from_probability_sentiment(p, sentiment, threshold=0.6, sentiment_threshold=0.0) -> Signal. BUY si p>=threshold Y sentiment>sentiment_threshold; SELL si p<=1-threshold Y sentiment<-sentiment_threshold (variante simétrica, documentada); HOLD en el resto. Valida threshold en (0.5,1] y sentiment_threshold>=0.
- inversion/trading/backtest.py: run_backtest gana parámetro opcional signal_fn(i, p, threshold)->Signal (retrocompatible, default = signal_from_probability). Permite inyectar la señal híbrida sin romper la API.
- inversion/trading/viabilidad.py: evaluación híbrida sobre el MISMO segmento test con el MISMO umbral calibrado (solo si el dataset trae sentiment_score). OOSResult + columnas hybrid_return/hybrid_sharpe/hybrid_n_trades; sección "Señal híbrida vs solo-modelo" en VIABILIDAD.md; párrafo en la Conclusión; limitación sobre umbral de sentimiento fijo.
- tests/test_signals.py: 8 tests de lógica de combinación (BUY/HOLD/SELL alineados y desalineados, umbrales configurables, errores).
- tests/test_backtest.py: test de signal_fn con la híbrida (veta compra -> 0 operaciones).
- tests/test_viabilidad.py: smoke test de que el informe incluye la sección híbrida.

## Criterios de aceptación
1. Señal combinada: signal_from_probability_sentiment exige p>=umbral Y sentiment>0 para BUY -> tests/test_signals.py (34 passed en los 3 ficheros).
2. Evaluada en backtest OOS: viabilidad.py corre run_backtest dos veces por ticker (solo-modelo y híbrida) sobre el mismo segmento test -> VIABILIDAD.md sección nueva.
3. VIABILIDAD.md compara híbrida vs solo-modelo: sección "Señal híbrida vs solo-modelo" con retornos por ticker y CARTERA.
4. Tests de combinación: tests/test_signals.py (8 tests).

## Evidencia
- ruff: All checks passed; ruff format --check: 29 files already formatted
- mypy strict: Success: no issues found in 6 source files
- pytest tests/test_signals.py tests/test_viabilidad.py tests/test_backtest.py -q: 34 passed
- pytest tests/ -q: 162 passed
- .venv/bin/python inversion/trading/viabilidad.py -> informe regenerado; CARTERA solo-modelo 77.50% vs buy&hold 281.70% (sin cambios).

## Comparativa híbrida vs solo-modelo (out-of-sample, mismo test)
| ticker | solo-modelo | híbrida | Δ | ops híbrida |
| AAPL 112.98% | 0.00% | -112.98% | 0 |
| MSFT 43.95% | 0.00% | -43.95% | 0 |
| GOOGL 77.64% | 0.00% | -77.64% | 0 |
| AMZN 84.18% | 0.00% | -84.18% | 0 |
| META 37.73% | 0.00% | -37.73% | 0 |
| TSLA 186.03% | 0.00% | -186.03% | 0 |
| NVDA 0.00% | 0.00% | +0.00% | 0 |
| CARTERA 77.50% | 0.00% | -77.50% | 0 |

## Hallazgo honesto (importante)
sentiment_score es 0.0 en TODO el histórico de los 7 tickers: las noticias crudas (data/raw/news_*.csv) solo cubren jul-ago 2026, fuera de la ventana de precios (termina 2026-03-25). Con la regla estricta (>0 / <0) la híbrida nunca dispara: 0 operaciones en todos los tickers. La comparativa es degenerada; el informe lo documenta y recomienda re-evaluar cuando exista cobertura de noticias solapada con precios. La conclusión sigue recomendando NO invertir capital real.

## Qué falta
- Datos de sentimiento reales (noticias que solapen la ventana de precios) para que la comparativa híbrida sea informativa.
- Calibrar sentiment_threshold por ticker si se quiere (hoy fijo en 0.0, documentado como limitación).
