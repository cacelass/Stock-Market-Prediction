# lider · IMP-003

- **Fecha:** 2026-08-25
- **Veredicto:** ok

Hallazgo: models/artifacts/best_params_AAA.json era residuo de un test de tuning (ticker sintético AAA, AUC 0.97 sobre datos falsos) escrito en el directorio real de artefactos y a punto de entrar en el commit de producto. Eliminado antes del commit. Pendiente de arreglar en código: los tests de tuning deben parchear ARTIFACTS_DIR como hacen con INTERIM/MODELS; candidato a regla si reaparece.
