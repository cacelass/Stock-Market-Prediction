# implementer · ENSEMBLE-ENHANCE

- **Fecha:** 2026-10-17
- **Veredicto:** ok

## Qué cambié
- **inversion/features/macro_features.py** (nuevo, 170 líneas): Módulo para features macroeconómicas (VIX, sector ETFs, treasury yield, DXY). Incluye fetchers desde yfinance y función add_macro_features() que integra 10 nuevas columnas.
- **inversion/features/feature_selection.py** (nuevo, 120 líneas): Selección de features basada en SHAP. Funciones compute_shap_importance(), select_top_features() y build_optimized_feature_set(). Fallback a feature_importances_ cuando SHAP no está instalado.
- **inversion/features/regime_detection.py** (nuevo, 140 líneas): Detección de régimen de mercado (bull/bear/neutral) basada en retorno, volatilidad y VIX. Funciones detect_regime(), detect_regimes() y train_regime_model().
- **inversion/features/build_features.py** (modificado): Integra add_macro_features() y add_regime_column() en el pipeline de features.
- **inversion/models/train_model.py** (modificado): Añadida train_calibrated_rf() con CalibratedClassifierCV. train_ticker() acepta parámetro calibrated=True. FEATURE_COLS ampliado con 10 columnas macro.
- **inversion/ensemble/weighted_ensemble.py** (modificado): Añadida compute_dynamic_weights() para pesos adaptativos basados en rendimiento reciente con decay exponencial.
- **inversion/features/__init__.py** (modificado): Exporta todas las nuevas funciones.
- **inversion/ensemble/__init__.py** (modificado): Exporta compute_dynamic_weights.
- **inversion/models/__init__.py** (modificado): Exporta train_calibrated_rf.
- **tests/test_ensemble.py** (modificado): +13 tests para todas las nuevas funcionalidades.

## Criterios de aceptación
1. **Macro features**: `add_macro_features()` genera 10 columnas (vix_level, vix_ma20, vix_change, sector_xlk, sector_xlf, sector_xle, treasury_yield, treasury_change, usd_index, usd_change) sin requerir red → `test_add_macro_features_no_network_returns_defaults` PASSED
2. **Feature selection**: SHAP importance y select_top_features() funcionan con fallback a feature_importances_ → `test_compute_shap_importance_fallback_without_shap` PASSED
3. **Regime detection**: detect_regime() clasifica correctamente bull/bear/neutral → `test_detect_regime_bull`, `test_detect_regime_bear`, `test_detect_regime_neutral` PASSED
4. **Probability calibration**: train_calibrated_rf() devuelve CalibratedClassifierCV → `test_train_calibrated_rf_returns_calibrated` PASSED
5. **Dynamic weights**: compute_dynamic_weights() asigna mayor peso al mejor modelo → `test_compute_dynamic_weights_better_model_gets_higher_weight` PASSED

## Evidencia
```
uv run pytest tests/test_ensemble.py -v --tb=short -k "macro or shap or regime or dynamic or calibrated or build_features"
13 passed in 15.79s

uv run pytest tests/ -q --tb=short --ignore=tests/test_monitoring_drift.py --ignore=tests/test_onnx_utils.py --ignore=tests/test_onnx_infer.py
247 passed, 2 warnings in 129.05s

./init.sh --quick → ENTORNO LISTO
uv run ruff check inversion/features/macro_features.py inversion/features/feature_selection.py inversion/features/regime_detection.py inversion/features/build_features.py inversion/models/train_model.py inversion/ensemble/weighted_ensemble.py → All checks passed!
```

## Qué falta
- La integración de SHAP feature selection en el pipeline de training (actualmente está disponible pero no se usa automáticamente)
- La integración de regime-specific models en train_ensemble (train_regime_model existe pero no se llama automáticamente)
- Los pesos dinámicos se pueden usar en predict_weighted_ensemble pero no se integra automáticamente en el flujo principal
