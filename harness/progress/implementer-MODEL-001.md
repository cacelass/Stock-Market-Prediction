# implementer · MODEL-001

- **Fecha:** 2026-08-02
- **Veredicto:** ok

Baseline RandomForest por ticker con split temporal 80/20, seed 42: models/rf_<T>.pkl + scaler_<T>.pkl, métricas en reports/resultados_<T>.csv, reproducible (hash idéntico), check_overfitting OK (gap 0.002-0.099, max_depth=5, min_samples_leaf=50), tests 67 passed cobertura 87%
