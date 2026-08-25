# lider · IMP-001

- **Fecha:** 2026-08-25
- **Veredicto:** ok

Regla derivada de un fallo (patrón ttsr): commit_feature abortado por hooks pre-commit dejaba CHANGELOG duplicado en cada reintento porque DocumentationAgent._insert_changelog no era idempotente. El fallo se coló 3 veces en una sesión. Regla validada contra el historial: sí lo habría evitado (la guarda salta exactamente en ese caso). Fix: guarda de duplicados en _insert_changelog + limpieza manual del CHANGELOG. Coste: una comparación de string solo al insertar.
