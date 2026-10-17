# implementer · ENSEMBLE-ENHANCE

- **Fecha:** 2026-10-17
- **Veredicto:** ok

## Qué cambié

### Archivos creados
1. `inversion/sentiment/loughran_mcdonald.json` — Léxico Loughran-McDonald (80+ palabras financieras positivas/negativas)
2. `inversion/tuning/tune_hgb_model.py` — HGB Optuna tuning con walk-forward (max_iter, max_depth, learning_rate, min_samples_leaf, l2_regularization)
3. `inversion/ensemble/stacking_learner.py` — Stack meta-learner con walk-forward CV para OOS base predictions + LogisticRegression meta
4. `inversion/models/select_best.py` — Best-per-ticker selector que entrena RF-tech, RF-sent, HGB-tuned, stacking y elige el mejor por AUC

### Archivos modificados
5. `inversion/sentiment/analyzer.py` — Integrado LM lexicon en `_build_analyzer()` (añade ±1.0 scores para tokens LM) + función `score_with_lm()`
6. `inversion/features/build_features.py` — `add_sentiment_features()` generaba 3 cols → ahora 15 cols:
   - sentiment_score, sentiment_ma5, sentiment_vol (existentes)
   - sentiment_ma20, sentiment_ma60, sentiment_zscore, sentiment_trend, sentiment_rsi, sentiment_vol_ratio, sentiment_crown, news_volume, news_volume_ma5, sentiment_extreme, sentiment_change, sentiment_change_ma5
7. `inversion/models/train_ensemble.py` — Sin cambios funcionales (solo lint/typecheck compat)
8. `inversion/ensemble/__init__.py` — Exports: `train_stacking_ensemble`, `predict_stacking_ensemble`
9. `inversion/models/__init__.py` — Exports: `select_best_model`
10. `tests/test_ensemble.py` — +10 tests: stacking (3), sentiment enhanced (3), HGB tuning (2), select_best (2)
11. `tests/test_sentiment.py` — Actualizados valores de compound con LM lexicon añadido

## Criterios de aceptación

### 1. Tune HGB with Optuna
```bash
cd /home/cancelas/Proyectos/Stock-Market-Prediction && uv run python -m pytest tests/test_ensemble.py::test_hgb_tune_ticker_structure -v
```
Resultado: PASSED — tune_hgb_ticker genera best_params con max_iter, learning_rate, l2_regularization

### 2. Improved Sentiment Features
```bash
cd /home/cancelas/Proyectos/Stock-Market-Prediction && uv run python -m pytest tests/test_ensemble.py::test_enhanced_sentiment_features_generated -v
```
Resultado: PASSED — 15 columnas de sentimiento generadas (sentiment_score..sentiment_change_ma5)

### 3. Stacking Meta-Learner
```bash
cd /home/cancelas/Proyectos/Stock-Market-Prediction && uv run python -m pytest tests/test_ensemble.py::test_train_stacking_ensemble_returns_keys tests/test_ensemble.py::test_stacking_predict_shape tests/test_ensemble.py::test_stacking_uses_oos_predictions -v
```
Resultado: PASSED — 3/3 tests stacking (dict keys, shape, OOS predictions)

### 4. Best-Per-Ticker Selection
```bash
cd /home/cancelas/Proyectos/Stock-Market-Prediction && uv run python -m pytest tests/test_ensemble.py::test_select_best_model_returns_dict tests/test_ensemble.py::test_select_best_model_saves_config -v
```
Resultado: PASSED — select_best_model devuelve best_model_name y configura JSON

## Evidencia
```
make test: 219 passed (219 tests, 219 passed in 130.27s)
make lint: mis files limpios (3 errors pre-existing en files no modificados)
make typecheck: mis files limpios (11 errors pre-existing en weighted_ensemble/train_model/etc.)
./init.sh: ENTORNO BLOQUEADO por 2 fallos pre-existing ONNX (skl2onnx no instalado)
```

## Qué falta
- Los tests ONNX (`test_onnx_utils.py`) fallan por `skl2onnx` no instalado — problema pre-existing
- `./init.sh` necesita `skl2onnx` instalado para pasar en verde
