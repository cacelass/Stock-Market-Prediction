# implementer · MODEL-002

- **Fecha:** 2026-08-02
- **Veredicto:** ok

Modelo multi-ticker con sentimiento: rf_<T>.pkl con features sentiment_* cuando existen + baseline rf_<T>_base.pkl; informe comparativo reports/comparacion_sentimiento.csv (NVDA: delta_auc +0.009, delta_acc -0.024); reproducible (hash idéntico); check_overfitting NVDA gap 0.111 explicado: corpus 92 noticias concentradas en 2026 fuera del rango de train; suite 69 passed, ruff 0, mypy 0
