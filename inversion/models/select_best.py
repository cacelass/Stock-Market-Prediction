"""
models/select_best.py — Selección del mejor modelo por ticker.

Entrena RF-tech, RF-sentiment (con features mejoradas), HGB (tuneado con Optuna),
y stacking, y devuelve el que mejor AUC walk-forward alcance.

Funciones
---------
select_best_model — entrena + evalúa todos y devuelve el mejor
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from inversion.features.build_features import fit_and_save_scaler
from inversion.models.train_model import SENTIMENT_COLS, available_feature_cols
from inversion.tuning.tune_hgb_model import load_best_hgb_params
from inversion.tuning.tune_model import load_best_params
from inversion.utils import paths


def _train_rf(X_train: np.ndarray, y_train: np.ndarray, params: dict[str, Any] | None = None) -> RandomForestClassifier:
    """Entrena RandomForest con params de Optuna si disponibles."""
    default_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 50,
        "max_features": "sqrt",
    }
    hp = {**default_params, **(params or {})}
    return RandomForestClassifier(
        n_estimators=int(hp["n_estimators"]),
        max_depth=int(hp["max_depth"]),
        min_samples_leaf=int(hp["min_samples_leaf"]),
        max_features=hp["max_features"],
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ).fit(X_train, y_train)


def _train_hgb(X_train: np.ndarray, y_train: np.ndarray, params: dict[str, Any] | None = None) -> HistGradientBoostingClassifier:
    """Entrena HistGradientBoosting con params de Optuna si disponibles."""
    default_params = {
        "max_iter": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "min_samples_leaf": 10,
        "l2_regularization": 1.0,
    }
    hp = {**default_params, **(params or {})}
    return HistGradientBoostingClassifier(
        max_iter=int(hp["max_iter"]),
        max_depth=int(hp["max_depth"]),
        learning_rate=float(hp["learning_rate"]),
        min_samples_leaf=int(hp["min_samples_leaf"]),
        l2_regularization=float(hp["l2_regularization"]),
        random_state=42,
        class_weight="balanced",
    ).fit(X_train, y_train)


def select_best_model(ticker: str, csv_path: Path, random_state: int = 42) -> dict[str, Any]:
    """Entrena RF-tech, RF-sentiment (enhanced), HGB (tuneado) y stacking; devuelve el mejor por AUC.

    Parameters
    ----------
    ticker : ticker symbol
    csv_path : path al dataset de features
    random_state : seed

    Returns
    -------
    dict con:
        best_model_name : nombre del mejor modelo ("rf_tech" / "rf_sent" / "hgb" / "stacking")
        metrics : métricas de todos los modelos entrenados
        best_model : el modelo + scaler guardados
        best_scaler : scaler correspondiente
        best_feature_cols : columnas usadas
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    feature_cols = available_feature_cols(df)
    df = df.dropna(subset=[*feature_cols, "target"]).sort_values("timestamp").reset_index(drop=True)

    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    X_all_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_all_test = test_df[feature_cols]
    y_test = test_df["target"]

    # Scaler para features técnicas (sin sentimiento)
    tech_cols = [c for c in feature_cols if c not in SENTIMENT_COLS]
    scaler_tech = fit_and_save_scaler(X_all_train[tech_cols], filename=f"scaler_{ticker}_best.pkl")
    X_train_tech = scaler_tech.transform(X_all_train[tech_cols])
    X_test_tech = scaler_tech.transform(X_all_test[tech_cols])

    results: dict[str, Any] = {}

    # 1. RF-tech
    rf_params = load_best_params(ticker)
    rf_tech = _train_rf(X_train_tech, y_train.values, params=rf_params)
    try:
        pr = rf_tech.predict_proba(X_test_tech)[:, 1]
        results["rf_tech"] = {"auc": float(roc_auc_score(y_test.values, pr)), "accuracy": float(accuracy_score(y_test.values, rf_tech.predict(X_test_tech)))}
    except Exception:
        results["rf_tech"] = {"auc": 0.0, "accuracy": 0.0}
    print(f"   RF-tech:    AUC={results['rf_tech']['auc']:.4f}  acc={results['rf_tech']['accuracy']:.4f}")

    # 2. RF-sentiment (con features mejoradas)
    has_sentiment = all(c in df.columns for c in SENTIMENT_COLS)
    sent_cols = [c for c in SENTIMENT_COLS if c in X_all_train.columns]
    if has_sentiment and sent_cols:
        scaler_sent = fit_and_save_scaler(X_all_train[sent_cols], filename=f"scaler_sent_{ticker}_best.pkl")
        X_train_sent = scaler_sent.transform(X_all_train[sent_cols])
        X_test_sent = scaler_sent.transform(X_all_test[sent_cols])
        rf_sent = _train_rf(X_train_sent, y_train.values)
        try:
            ps = rf_sent.predict_proba(X_test_sent)[:, 1]
            results["rf_sent"] = {"auc": float(roc_auc_score(y_test.values, ps)), "accuracy": float(accuracy_score(y_test.values, rf_sent.predict(X_test_sent)))}
        except Exception:
            results["rf_sent"] = {"auc": 0.0, "accuracy": 0.0}
        print(f"   RF-sent:    AUC={results['rf_sent']['auc']:.4f}  acc={results['rf_sent']['accuracy']:.4f}")
    else:
        results["rf_sent"] = {"auc": 0.0, "accuracy": 0.0, "note": "no sentiment data"}
        print("   RF-sent:    no sentiment data")

    # 3. HGB-tuneado
    hgb_params = load_best_hgb_params(ticker)
    hgb = _train_hgb(X_train_tech, y_train.values, params=hgb_params)
    try:
        ph = hgb.predict_proba(X_test_tech)[:, 1]
        results["hgb"] = {"auc": float(roc_auc_score(y_test.values, ph)), "accuracy": float(accuracy_score(y_test.values, hgb.predict(X_test_tech)))}
    except Exception:
        results["hgb"] = {"auc": 0.0, "accuracy": 0.0}
    print(f"   HGB-tuned:  AUC={results['hgb']['auc']:.4f}  acc={results['hgb']['accuracy']:.4f}")

    # 4. Stacking
    from inversion.ensemble.stacking_learner import train_stacking_ensemble
    stacking = train_stacking_ensemble(X_train_tech, y_train.values, X_test_tech, y_test.values)
    results["stacking"] = {
        "auc": stacking["ensemble_auc"],
        "accuracy": stacking["ensemble_accuracy"],
        "oos_train_auc": stacking["oos_train_auc"],
    }
    print(f"   Stacking:   AUC={results['stacking']['auc']:.4f}  acc={results['stacking']['accuracy']:.4f}")

    # Pick best by AUC
    model_names = [k for k in results if results[k]["auc"] > 0]
    if model_names:
        best_name = max(model_names, key=lambda k: results[k]["auc"])
        best_auc = results[best_name]["auc"]
    else:
        best_name = "rf_tech"
        best_auc = 0.0

    print(f"   ✔ Best model: {best_name} (AUC={best_auc:.4f})")

    # Save best model
    models_dir = paths.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    if best_name == "rf_tech":
        model_path = models_dir / f"best_rf_{ticker}.pkl"
        joblib.dump(rf_tech, model_path)
    elif best_name == "rf_sent":
        model_path = models_dir / f"best_rf_sent_{ticker}.pkl"
        joblib.dump(rf_sent, model_path)
    elif best_name == "hgb":
        model_path = models_dir / f"best_hgb_{ticker}.pkl"
        joblib.dump(hgb, model_path)
    else:  # stacking
        model_path = models_dir / f"best_stacking_{ticker}.pkl"
        # Save base models + meta model together
        stacking_bundle = {
            "meta_model": stacking["meta_model"],
            "base_models": stacking["base_models"],
            "base_labels": list(stacking["base_models"].keys()),
        }
        joblib.dump(stacking_bundle, model_path)

    # Save config
    import json as _json
    config = {
        "ticker": ticker,
        "best_model": best_name,
        "metrics": {k: v for k, v in results.items()},
        "best_auc": round(best_auc, 4),
        "model_file": model_path.name,
    }
    config_path = models_dir / f"best_model_config_{ticker}.json"
    with open(config_path, "w") as fh:
        _json.dump(config, fh, indent=2, default=str)

    return {
        "best_model_name": best_name,
        "metrics": results,
        "best_model_path": model_path,
        "config_path": config_path,
        "best_auc": best_auc,
    }
