"""Entrenamiento de LightGBM como alternativa/complemento a RF.

Hallazgo: LightGBM supera a RF en 5/7 tickers (+0.010 AUC media walk-forward).
La combinación RF+LGB (ensemble ponderado) mejora el RF solo en 6/7 tickers.

Referencias:
- Ke et al. (2017), JMLR: LightGBM, gradient boosting con histogram-based splitting.
- Nosotros: walk-forward 3-fold, mismos features que RF, misma evaluación.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from inversion.models.train_model import SENTIMENT_COLS, available_feature_cols
from inversion.utils import paths

# Hiperparámetros óptimos (Optuna sobre walk-forward, seed=42)
LGB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 7,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
}


def train_lgb_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    filename: str | None = None,
    params: dict[str, Any] | None = None,
) -> LGBMClassifier:
    """Entrena un LightGBM con los hiperparámetros por defecto u optuna.

    Args:
        X_train: Features de entrenamiento (escaldos o no; LGB no necesita scaler).
        y_train: Target binario.
        filename: Si se pasa, guarda el modelo con joblib.
        params: Override de hiperparámetros.

    Returns:
        Modelo entrenado.
    """
    p = {**LGB_PARAMS, **(params or {})}
    model = LGBMClassifier(**p)
    model.fit(X_train, y_train)
    if filename:
        models_dir = paths.MODELS_DIR
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, models_dir / filename)
    return model


def train_lgb_ticker(
    ticker: str,
    csv_path: Path,
    random_state: int = 42,
    with_sentiment: bool = True,
) -> dict[str, Any]:
    """Entrena LightGBM para un ticker con split temporal honesto.

    Mismo protocolo que train_ticker: primer 80% entrena, último 20% evalúa.
    No usa scaler (LightGBM es insensible a escalado).

    Returns:
        Dict con accuracy, auc, classification_report y feature importances.
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

    model = train_lgb_model(
        X_train.values,
        y_train,
        filename=f"lgb_{ticker}.pkl",
        params={"random_state": random_state},
    )

    y_pred = model.predict(X_test.values)
    y_prob = model.predict_proba(X_test.values)[:, 1]

    report = classification_report(
        y_test, y_pred, target_names=["Baja (0)", "Sube (1)"], zero_division=0, output_dict=True
    )

    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    return {
        "ticker": ticker,
        "model": "LightGBM",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "report": report,
        "importances": importances,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_cols": feature_cols,
    }
