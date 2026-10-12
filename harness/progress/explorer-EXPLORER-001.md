# explorer · EXPLORER-001

- **Fecha:** 2026-10-12
- **Veredicto:** ok

## Respuesta

Auditoría exhaustiva del proyecto Stock-Market-Prediction. Se identificaron **12 mejoras**, priorizadas por severidad. El proyecto está en buen estado general — la estructura es sólida, el arnés funciona, los tests pasan — pero hay varios puntos de mantenimiento que conviene atender.

## Mejoras priorizadas

### 🔴 Alta severidad

1. **Duplicados de targets en `Makefile`** — `monitor:`, `tune:` y `serve:` aparecen dos veces cada uno (líneas 217+236, 209+245, 205+264). El segundo bloque sobrescribe al primero, lo que puede causar comportamiento inesperado o confusiones al ejecutar `make monitor`, `make tune` o `make serve`. Solo el último bloque es funcional.

2. **`__future__` importado en todas las 30+ archivos de `inversion/` sin uso real** — el análisis AST muestra `__future__` como no usado en al menos 20 ficheros. No es un bug, pero es ruido que ensucia la base de código y hace pensar que se usan features de Python futuras (tipo `from __future__ import annotations`) cuando no es así.

### 🟡 Media severidad

3. **`data/processed/` tiene archivos generados que deberían ser descartados por `.gitignore`** — `feature_importance.png`, `NVDA_price_plot.png`, `predictions_plot.png` están en `data/processed/` pero solo `data/processed/*.csv` está en `.gitignore`. Los PNG escapan al gitignore y se rastrean innecesariamente.

4. **`reports/importancia_features.csv` con 218 líneas** — es el único CSV de reports que no está en `.gitignore` (los `importancia_*.csv` sí lo están). Es probable que haya sido renombrado o creado manualmente y debería renombrarse a `importancia_features.csv` (coincide con el patrón `importancia_*.csv` del gitignore) o moverse al patrón correcto para que sea ignorada.

5. **`data/interim/` contiene 7 ficheros de features (total ~25 MB)** — están en `.gitignore` pero no deberían estar en ese directorio: son datos derivados que el pipeline de features genera y consume. Podrían borrarse tras cada ejecución del pipeline para evitar confusiones, o el directorio debería limpiarse automáticamente.

6. **`news_*.csv` (GDELT) en `data/raw/` junto a `_ml_ready.csv`** — hay 6 ficheros `news_*_gdelt.csv` y 7 `news_*_live.csv` mezclados con los datos principales. El esquema de nombres no es consistente: los datos de precios son `<TICKER>_ml_ready.csv` pero las noticias son `news_<TICKER>_gdelt.csv`. Sería más limpio separarlas en subdirectorios (`data/raw/prices/` y `data/raw/news/`).

7. **`reports/data_validation/` tiene 7 JSON de validación** — son outputs de `make data`, cada ticker genera uno. Están bien pero no hay un target `make validate-data` ni referencia en el README. Si son útiles para CI, deberían integrarse; si no, podrían omitirse.

### 🔵 Baja severidad

8. **README menciona badge de CI con URL de fork** — el README dice "sustituye la URL del badge por la de tu fork" pero el badge apunta a `cacelass/dskit`. Esto es informativo y no crítico, pero debería actualizarse si el repo ya no es un fork.

9. **`inversion/trading/viabilidad.py` con 545 líneas** — es el fichero más largo del proyecto. Según los principios del proyecto ("200 líneas pueden ser 50"), debería refactorizarse.

10. **`inversion/sentiment/fetch.py` con 15 imports** — el fichero con más imports. Podría beneficiarse de una refactorización para agrupar la lógica de fetch en sub-módulos.

11. **Sin `.gitattributes`** — no hay configuración de LFS. Si los datasets o modelos crecen (actualmente ~16 MB cada uno en data/raw y ~3 MB en data/interim), conviene añadir `.gitattributes` para gestionar LFS correctamente y evitar inflar el repositorio.

12. **`reports/figures/` solo contiene 1 fichero (`shap_NVDA.png`)** — el patrón `reports/figures/` está en `.gitignore` pero el directorio existe con un fichero. Coherente pero subutilizado.

## Dónde está cada cosa

- **Makefile duplicados:** `Makefile` líneas 205-264 (serve), 217-236 (monitor), 209-245 (tune)
- **Imports `__future__` no usados:** `inversion/api/`, `inversion/data/make_dataset.py`, `inversion/features/build_features.py`, `inversion/models/`, `inversion/monitoring/drift.py`, `inversion/sentiment/`, `inversion/trading/`, `inversion/tuning/tune_model.py`
- **data/processed PNGs:** `data/processed/feature_importance.png`, `NVDA_price_plot.png`, `predictions_plot.png`
- **CSV no ignorado:** `reports/importancia_features.csv` (218 líneas, no coincide con patrón `importancia_*.csv`)
- **viabilidad.py tamaño:** `inversion/trading/viabilidad.py` — 545 líneas
- **fetch.py imports:** `inversion/sentiment/fetch.py` — 15 imports

## Riesgos y sorpresas

- Los duplicados en el Makefile son el riesgo más alto: un desarrollador que intente `make monitor` podría recibir la primera regla en vez de la segunda, con resultados distintos. Esto es difícil de detectar porque ambos targets tienen el mismo nombre.
- Los imports `__future__` no usados podrían ser un artefacto de generación automática. Si se están usando para habilitar `annotations` en Python 3.10-, su eliminación rompería la compatibilidad con versiones anteriores a 3.10 (aunque el proyecto requiere `>=3.12`).
- El fichero `reports/importancia_features.csv` no se borra automáticamente porque no coincide con el patrón de gitignore — si se genera automáticamente, el patrón debería corregirse.

## Sin verificar

- No verifiqué si las reglas duplicadas del Makefile producen comportamientos funcionalmente diferentes (el segundo bloque podría ser intencional).
- No ejecuté `./init.sh` para verificar el estado completo del arnés.
- No verifiqué si `news_*.csv` en `data/raw/` son usados por algún target del Makefile.
