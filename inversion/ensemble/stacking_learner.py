"""
ensemble/stacking_learner.py — Stacking ensemble con meta-learner.

Entrena un conjunto de modelos base y un meta-learner (LogisticRegression)
sobre predicciones out-of-sample generadas con walk-forward CV.

Procedimiento:
  1. Para cada fold de walk-forward, entrena los modelos base en el fold de
     train y genera predicciones en el fold de test (out-of-sample).
  2. Acumula todas las predicciones OOS de cada base → X_meta_train, y
     las etiquetas correspondientes → y_meta_train.
  3. Entrena el meta-learner (LogisticRegression) en X_meta_train, y_meta_train.
  4. Para predicción final: los modelos base se reentrenan en TODOS los datos,
     y el meta-learner combina sus predicciones.

Funciones
---------
train_stacking_ensemble — entrena + devuelve (meta_model, base_models, meta_X_test)
predict_stacking_ensemble — predice con el stacking
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def _walk_forward_folds(n: int, n_folds: int = 3, start_frac: float = 0.60) -> list[tuple[int, int]]:
    """Fronteras (train_end, test_end) de folds expanding-window."""
    min_train = 100
    min_test = 15
    start = int(n * start_frac)
    fold_size = max((n - start) // n_folds, 1)
    folds = []
    for k in range(n_folds):
        train_end = start + k * fold_size
        test_end = start + (k + 1) * fold_size if k < n_folds - 1 else n
        if train_end >= min_train and (test_end - train_end) >= min_test:
            folds.append((train_end, test_end))
    return folds


def train_stacking_ensemble(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    base_models: Dict[str, Any] | None = None,
    n_folds: int = 3,
    start_frac: float = 0.60,
) -> Dict[str, Any]:
    """Entrena base models + meta-learner via walk-forward CV.

    Base models (por defecto): RF-tech, HGB-tech, RF-sentiment.
    Meta-learner: LogisticRegression sobre predicciones out-of-sample de las bases.

    Parameters
    ----------
    X_train, y_train : datos completos de entrenamiento
    X_test, y_test : datos de test final
    base_models : dict opcional {name: model}. Si None, se crean defaults.
    n_folds : número de folds walk-forward para generar OOS predictions
    start_frac : fracción inicial para el primer fold

    Returns
    -------
    dict con claves:
        meta_model — LogisticRegression entrenado
        base_models — dict {name: model} entrenados en TODO train
        oos_train_proba — array (n_train, n_base) predicciones OOS en train
        oos_train_auc — AUC del meta-model en OOS train
        ensemble_auc — AUC del stacking en test
        ensemble_accuracy — accuracy del stacking en test
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    # Default base models
    if base_models is None:
        base_models = {
            "rf_tech": RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=50,
                max_features="sqrt", class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
            "hgb": HistGradientBoostingClassifier(
                max_iter=100, max_depth=5, learning_rate=0.1,
                random_state=42, class_weight="balanced",
            ),
            "rf_sent": RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=50,
                max_features="sqrt", class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
        }

    base_labels = list(base_models.keys())
    n_base = len(base_labels)
    n_train = len(y_train)
    folds = _walk_forward_folds(n_train, n_folds=n_folds, start_frac=start_frac)

    if not folds:
        # Fallback: training directo sin stacking
        final_models: Dict[str, Any] = {}
        for name, model in base_models.items():
            model.fit(X_train, y_train)
            final_models[name] = model
        test_proba_base = np.column_stack([final_models[label].predict_proba(X_test)[:, 1] for label in base_labels])
        meta = LogisticRegression(random_state=42, max_iter=1000)
        meta.fit(test_proba_base, y_test)
        ens_proba = meta.predict_proba(test_proba_base)[:, 1]
        return {
            "meta_model": meta,
            "base_models": final_models,
            "oos_train_proba": test_proba_base,
            "oos_train_auc": 0.0,
            "ensemble_auc": float(roc_auc_score(y_test, ens_proba)),
            "ensemble_accuracy": float(accuracy_score(y_test, (ens_proba >= 0.5).astype(int))),
        }

    # Walk-forward: generate OOS predictions for meta-training
    oos_proba = np.zeros((n_train, n_base))
    for fold_i, (train_end, test_end) in enumerate(folds):
        X_fold_train = X_train[:train_end]
        y_fold_train = y_train[:train_end]
        X_fold_test = X_train[train_end:test_end]

        for j, label in enumerate(base_labels):
            model = base_models[label].__class__(**base_models[label].get_params())
            model.fit(X_fold_train, y_fold_train)
            try:
                oos_proba[train_end:test_end, j] = model.predict_proba(X_fold_test)[:, 1]
            except (ValueError, IndexError):
                # Single class in fold or shape mismatch: assign 0.5 (neutral)
                oos_proba[train_end:test_end, j] = 0.5

    # Meta-learner on OOS predictions (train portion)
    meta_train_valid = ~np.isnan(oos_proba).any(axis=1)
    meta_X_train = oos_proba[meta_train_valid]
    meta_y_train = y_train[meta_train_valid]

    if len(meta_y_train) > 0 and len(np.unique(meta_y_train)) > 1:
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
        meta_model.fit(meta_X_train, meta_y_train)
        oos_auc = roc_auc_score(meta_y_train, meta_model.predict_proba(meta_X_train)[:, 1])
    else:
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
        meta_model.fit(meta_X_train, meta_y_train)
        oos_auc = 0.5

    # Retrain base models on ALL train data for final predictions
    retrained_models: Dict[str, Any] = {}
    for name, model in base_models.items():
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_train, y_train)
        retrained_models[name] = model_copy

    # Predict on test set with stacking
    test_proba_base = np.column_stack([retrained_models[label].predict_proba(X_test)[:, 1] for label in base_labels])
    ens_proba = meta_model.predict_proba(test_proba_base)[:, 1]

    return {
        "meta_model": meta_model,
        "base_models": retrained_models,
        "oos_train_proba": oos_proba,
        "oos_train_auc": float(oos_auc),
        "ensemble_auc": float(roc_auc_score(y_test, ens_proba)),
        "ensemble_accuracy": float(accuracy_score(y_test, (ens_proba >= 0.5).astype(int))),
    }


def predict_stacking_ensemble(
    X: np.ndarray | pd.DataFrame,
    base_models: Dict[str, Any],
    meta_model: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict using stacking ensemble.

    Parameters
    ----------
    X : features
    base_models : dict {name: model} trained on full data
    meta_model : LogisticRegression trained on base model OOS predictions

    Returns
    -------
    (predictions, probabilities) — binary predictions and stacked probability of class 1.
    """
    X = np.asarray(X)
    base_labels = list(base_models.keys())
    base_proba = np.column_stack([base_models[label].predict_proba(X)[:, 1] for label in base_labels])
    ens_proba = meta_model.predict_proba(base_proba)[:, 1]
    predictions = (ens_proba >= 0.5).astype(int)
    return predictions, ens_proba
