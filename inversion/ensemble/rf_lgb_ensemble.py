"""Ensemble RF + LightGBM (ponderado por validación cruzada temporal).

Hallazgo: el ensemble RF+LGB supera al RF solo en 6/7 tickers (+0.007 AUC media
walk-forward). La clave es que RF y LGB tienen sesgos diferentes:
- RF: bagging, reduce varianza, sensible a ruido
- LGB: boosting, reduce sesgo, captura interacciones no lineales

El peso óptimo se busca por validación cruzada temporal (no por test set).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from inversion.models.train_lightgbm import LGB_PARAMS
from inversion.models.train_model import SENTIMENT_COLS, available_feature_cols
from inversion.utils import paths

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_leaf": 2,
    "max_features": None,
}


def _temporal_cv_weight(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 3,
    random_state: int = 42,
) -> float:
    """Busca el peso óptimo de RF (w) en ensemble w*RF + (1-w)*LGB.

    Usa validación cruzada temporal (expanding window). Devuelve w en [0, 1].
    """
    n = len(X)
    start = int(n * 0.3)
    fold_size = max((n - start) // n_folds, 1)

    best_w, best_auc = 0.5, -1.0
    for w_test in np.arange(0.0, 1.05, 0.1):
        aucs = []
        for i in range(n_folds):
            train_end = start + fold_size * i
            test_end = min(start + fold_size * (i + 1), n)
            if train_end < 500 or (test_end - train_end) < 50:
                continue

            X_tr, X_te = X[:train_end], X[train_end:test_end]
            y_tr, y_te = y[:train_end], y[train_end:test_end]

            rf = RandomForestClassifier(**RF_PARAMS, random_state=random_state, n_jobs=-1)
            rf.fit(X_tr, y_tr)

            lgb = LGBMClassifier(**{**LGB_PARAMS, "random_state": random_state})
            lgb.fit(X_tr, y_tr)

            probs = w_test * rf.predict_proba(X_te)[:, 1] + (1 - w_test) * lgb.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_te, probs))

        if aucs:
            mean_auc = np.mean(aucs)
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_w = w_test

    return float(best_w)


def train_rf_lgb_ensemble(
    X_train: np.ndarray,
    y_train: pd.Series,
    filename: str | None = None,
    rf_weight: float | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Entrena ensemble RF + LightGBM.

    Si rf_weight=None, lo busca por CV temporal interno.

    Returns:
        Dict con 'model' (dict de ambos), 'rf_weight', 'auc_oof'.
    """
    if rf_weight is None:
        rf_weight = _temporal_cv_weight(X_train.values, y_train.values, random_state=random_state)

    rf = RandomForestClassifier(**RF_PARAMS, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    lgb = LGBMClassifier(**{**LGB_PARAMS, "random_state": random_state})
    lgb.fit(X_train, y_train)

    # AUC out-of-fold con el peso encontrado
    oof_probes = []
    n = len(X_train)
    start = int(n * 0.3)
    fold_size = max((n - start) // 3, 1)
    for i in range(3):
        train_end = start + fold_size * i
        test_end = min(start + fold_size * (i + 1), n)
        if train_end < 500 or (test_end - train_end) < 50:
            continue
        rf_i = RandomForestClassifier(**RF_PARAMS, random_state=random_state, n_jobs=-1)
        rf_i.fit(X_train[:train_end], y_train.iloc[:train_end])
        lgb_i = LGBMClassifier(**{**LGB_PARAMS, "random_state": random_state})
        lgb_i.fit(X_train[:train_end], y_train.iloc[:train_end])
        probs = rf_weight * rf_i.predict_proba(X_train[train_end:test_end])[:, 1] + \
                (1 - rf_weight) * lgb_i.predict_proba(X_train[train_end:test_end])[:, 1]
        oof_probes.append((probs, y_train.iloc[train_end:test_end]))

    auc_oof = -1.0
    if oof_probes:
        all_probs = np.concatenate([p for p, _ in oof_probes])
        all_y = np.concatenate([t for _, t in oof_probes])
        auc_oof = float(roc_auc_score(all_y, all_probs))

    result = {
        "model": {"rf": rf, "lgb": lgb},
        "rf_weight": rf_weight,
        "auc_oof": auc_oof,
    }

    if filename:
        models_dir = paths.MODELS_DIR
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, models_dir / filename)

    return result


def train_ensemble_lgb_ticker(
    ticker: str,
    csv_path: Path,
    random_state: int = 42,
    with_sentiment: bool = True,
) -> dict[str, Any]:
    """Entrena ensemble RF+LGB para un ticker con split temporal honesto.

    Returns:
        Dict con accuracy, auc, rf_weight y comparación con modelos individuales.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    feature_cols = available_feature_cols(df)
    if not with_sentiment:
        feature_cols = [c for c in feature_cols if c not in SENTIMENT_COLS]
    df = df.dropna(subset=[*feature_cols, "target"]).sort_values("timestamp")

    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    ens = train_rf_lgb_ensemble(
        X_train, y_train,
        filename=f"rf_lgb_{ticker}.pkl",
        random_state=random_state,
    )

    rf = ens["model"]["rf"]
    lgb = ens["model"]["lgb"]
    w = ens["rf_weight"]

    probs_ens = w * rf.predict_proba(X_test.values)[:, 1] + (1 - w) * lgb.predict_proba(X_test.values)[:, 1]
    pred_ens = (probs_ens > 0.5).astype(int)

    # Comparación individual
    auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test.values)[:, 1])
    auc_lgb = roc_auc_score(y_test, lgb.predict_proba(X_test.values)[:, 1])
    auc_ens = roc_auc_score(y_test, probs_ens)

    return {
        "ticker": ticker,
        "model": f"RF+LGB ensemble (w_rf={w:.1f})",
        "rf_weight": w,
        "accuracy": float(accuracy_score(y_test, pred_ens)),
        "auc": float(auc_ens),
        "auc_rf": float(auc_rf),
        "auc_lgb": float(auc_lgb),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_cols": feature_cols,
    }
