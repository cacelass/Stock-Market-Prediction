"""Diverse ensemble: Linear + RF + HistGradientBoosting.

Previous ensembles tried RF+HGB+RF-sentiment — similar model families. This
module combines three fundamentally different model classes:
  1. ElasticNet (strongly regularized linear, different inductive bias)
  2. RandomForest (non-linear, feature splitting)
  3. HistGradientBoosting (sequential boosting)

Diversity in model families is hypothesized to produce more robust probability
averages than ensembling similar models.

Functions
---------
train_diverse_ensemble : train all 3 models, average probabilities
train_diverse_ensemble_ticker : end-to-end per-ticker training
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from inversion.utils import paths


def train_diverse_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train 3 diverse models and return averaged probabilities.

    Models:
    - LogisticRegression (strong L2 regularization, C=0.01)
    - RandomForestClassifier (non-linear trees, class_weight balanced)
    - HistGradientBoostingClassifier (sequential boosting)

    Returns probability average (equal weight = 1/3 each).

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test : test data
    random_state : seed

    Returns
    -------
    dict with per-model and ensemble metrics
    """
    models: dict[str, Any] = {}

    # Model 1: Linear (strong regularization)
    lr = LogisticRegression(
        C=0.01,       # strong L2 regularization
        max_iter=5000,
        random_state=random_state,
        solver="lbfgs",
    )
    lr.fit(X_train, y_train)
    models["logistic"] = lr

    # Model 2: Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=50,
        max_features="sqrt", class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models["rf"] = rf

    # Model 3: HistGradientBoosting
    hgb = HistGradientBoostingClassifier(
        max_iter=100, max_depth=5, learning_rate=0.1,
        random_state=random_state, class_weight="balanced",
    )
    hgb.fit(X_train, y_train)
    models["hgb"] = hgb

    # Per-model metrics
    metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        try:
            auc = float(roc_auc_score(y_test, y_proba))
        except Exception:
            auc = 0.0
        acc = float(accuracy_score(y_test, y_pred))
        metrics[name] = {"accuracy": acc, "auc": auc}

    # Ensemble: equal-weight probability average
    probas = np.mean([models[n].predict_proba(X_test)[:, 1] for n in models], axis=0)
    preds = (probas >= 0.5).astype(int)

    try:
        ens_auc = float(roc_auc_score(y_test, probas))
    except Exception:
        ens_auc = 0.0
    ens_acc = float(accuracy_score(y_test, preds))

    return {
        "models": models,
        "metrics": metrics,
        "ensemble_accuracy": ens_acc,
        "ensemble_auc": ens_auc,
    }


def train_diverse_ensemble_ticker(
    ticker: str,
    csv_path: Path,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a diverse ensemble for a single ticker.

    1. Loads features, splits 80/20 temporally
    2. Scales features
    3. Trains LogisticRegression + RF + HGB
    4. Saves ensemble bundle to models/diverse_<TICKER>.pkl

    Parameters
    ----------
    ticker : ticker symbol
    csv_path : path to features CSV
    random_state : seed

    Returns
    -------
    dict with per-model, ensemble metrics, and model path
    """
    from inversion.models.train_model import available_feature_cols

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    feature_cols = available_feature_cols(df)
    df = df.dropna(subset=[*feature_cols, "target"]).sort_values("timestamp")

    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    result = train_diverse_ensemble(X_train_s, y_train, X_test_s, y_test, random_state)
    result["ticker"] = ticker
    result["n_train"] = int(len(X_train))
    result["n_test"] = int(len(X_test))
    result["feature_cols"] = feature_cols

    # Save model bundle
    save_path = paths.MODELS_DIR / f"diverse_{ticker}.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "models": result["models"],
        "scaler": scaler,
        "features": feature_cols,
    }, save_path)

    print(
        f"  Diverse ensemble {ticker}: "
        f"LR acc={result['metrics']['logistic']['accuracy']:.3f} "
        f"RF acc={result['metrics']['rf']['accuracy']:.3f} "
        f"HGB acc={result['metrics']['hgb']['accuracy']:.3f} "
        f"Ens acc={result['ensemble_accuracy']:.3f} ens_auc={result['ensemble_auc']:.3f}"
    )
    return result
