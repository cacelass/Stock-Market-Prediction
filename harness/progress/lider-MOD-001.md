# lider · MOD-001

- **Fecha:** 2026-08-25
- **Veredicto:** ok

Comparación RF vs HistGBM en marcha en background (/tmp/opencode/ml_queue.log, CSV incremental reports/comparacion_clases_modelo.csv). Contexto: máquina compartida con otros proyectos del usuario (load 20-27 sobre 6 cores), ejecución con n_jobs=2 y ensemble opcional desactivado por coste. Evidencia parcial (2/7): AAPL RF=0.527 vs GBM=0.488; MSFT RF=0.571 vs GBM=0.551 — RF gana en AUC en ambos; el GBM muestra accuracy mayor pero es el sesgo de clase ya documentado. Tras la comparación la cola lanza sola notebooks/calibracion.py (MOD-002 implícito: ECE raw vs isotónica, criterio de adopción predefinido ΔECE≤-0.02 y ΔAUC≥-0.005). Pendiente al retomar: leer veredictos de ambos logs, adoptar si procede, documentar secciones H/I en references/05, cerrar MOD-001 y commitear.
