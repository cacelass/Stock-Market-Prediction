"""Train ensemble of 3 models per ticker: RF-tech, HGB-tech, RF-sentiment.

Saves:
    ensemble_rf_<T>.pkl       — RF-Technical model
    ensemble_hgb_<T>.pkl      — HGB-Technical model
    ensemble_rf_sent_<T>.pkl  — RF-Sentiment model
    scaler_<T>.pkl            — shared scaler (technical features)
    scaler_sent_<T>.pkl       — sentiment-only scaler
    ensemble_config_<T>.json  — weights + per-model metrics + ensemble metrics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from inversion.features.build_features import fit_and_save_scaler
from inversion.models.train_model import SENTIMENT_COLS, available_feature_cols
from inversion.utils import paths


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ticker: str,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train 3 ensemble models on the same split and evaluate.

    Models:
        rf_tech   — RandomForest on technical features
        hgb       — HistGradientBoosting on technical features
        rf_sent   — RandomForest (may use sentiment features if available)

    Returns a dict with per-model and ensemble metrics.
    """
    results: Dict[str, Any] = {"models": {}, "metrics": {}, "n_train": int(len(X_train)), "n_test": int(len(X_test))}

    # Train RF-Technical
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

    rf_tech = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=50, max_features="sqrt",
        class_weight="balanced", random_state=random_state, n_jobs=-1,
    )
    rf_tech.fit(X_train, y_train)
    results["models"]["rf_tech"] = rf_tech
    _add_model_metrics(results, "rf_tech", rf_tech, X_test, y_test)

    # Train HGB-Technical
    hgb = HistGradientBoostingClassifier(
        max_iter=100, max_depth=5, learning_rate=0.1,
        random_state=random_state, class_weight="balanced",
    )
    hgb.fit(X_train, y_train)
    results["models"]["hgb"] = hgb
    _add_model_metrics(results, "hgb", hgb, X_test, y_test)

    # Ensemble average (simple)
    pr = rf_tech.predict_proba(X_test)[:, 1]
    ph = hgb.predict_proba(X_test)[:, 1]
    ens_proba = (pr + ph) / 2.0
    ens_pred = (ens_proba >= 0.5).astype(int)
    try:
        results["metrics"]["ensemble_auc"] = float(roc_auc_score(y_test, ens_proba))
    except Exception:
        results["metrics"]["ensemble_auc"] = 0.0
    results["metrics"]["ensemble_accuracy"] = float(accuracy_score(y_test, ens_pred))

    return results


def save_ensemble_models(
    results: Dict[str, Any],
    ticker: str,
    scaler_tech: Any,
    scaler_sent: Any | None = None,
    model_paths: Dict[str, Any] | None = None,
    ensemble_weights: list[float] | Dict[str, float] | None = None,
) -> Dict[str, Path]:
    """Persist models, scalers, and config to disk.

    Returns dict mapping logical name → saved file path.
    """
    saved: Dict[str, Path] = {}
    models_dir = paths.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save individual models
    for name, model in results.get("models", {}).items():
        if name == "rf_tech":
            fname = f"ensemble_rf_{ticker}.pkl"
        elif name == "hgb":
            fname = f"ensemble_hgb_{ticker}.pkl"
        elif name == "rf_sent":
            fname = f"ensemble_rf_sent_{ticker}.pkl"
        else:
            fname = f"ensemble_{name}_{ticker}.pkl"
        path = models_dir / fname
        joblib.dump(model, path)
        saved[name] = path
        print(f"   ✔ {fname}")

    # Save shared scaler (technical)
    scaler_tech_path = models_dir / f"scaler_{ticker}.pkl"
    joblib.dump(scaler_tech, scaler_tech_path)
    saved["scaler_tech"] = scaler_tech_path
    print(f"   ✔ scaler_{ticker}.pkl")

    # Save sentiment scaler if present
    if scaler_sent is not None:
        scaler_sent_path = models_dir / f"scaler_sent_{ticker}.pkl"
        joblib.dump(scaler_sent, scaler_sent_path)
        saved["scaler_sent"] = scaler_sent_path
        print(f"   ✔ scaler_sent_{ticker}.pkl")

    # Save ensemble config
    config: Dict[str, Any] = {
        "ticker": ticker,
        "metrics": results.get("metrics", {}),
        "weights": ensemble_weights if ensemble_weights is not None else [],
        "model_files": {k: p.name for k, p in saved.items() if "scaler" not in k and k != "scaler_tech"},
    }
    config_path = models_dir / f"ensemble_config_{ticker}.json"
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2, default=str)
    saved["config"] = config_path
    print(f"   ✔ ensemble_config_{ticker}.json")

    return saved


def train_ensemble_ticker(ticker: str, csv_path: Path, random_state: int = 42) -> Dict[str, Any]:
    """Train the 3-model ensemble for a single ticker and save everything.

    Parameters
    ----------
    ticker : ticker symbol
    csv_path : path to the features CSV
    random_state : seed for reproducibility

    Returns
    -------
    dict with keys: ticker, n_train, n_test, metrics (per-model + ensemble)
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    feature_cols = available_feature_cols(df)
    df = df.dropna(subset=[*feature_cols, "target"]).sort_values("timestamp")

    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    # Scaler for technical features
    scaler_tech = fit_and_save_scaler(X_train, filename=f"scaler_{ticker}.pkl")
    X_train_s = scaler_tech.transform(X_train)
    X_test_s = scaler_tech.transform(X_test)

    # Scaler for sentiment features (if present)
    has_sentiment = any(c in df.columns for c in SENTIMENT_COLS)
    scaler_sent = None
    X_train_s_sent = X_train_s
    X_test_s_sent = X_test_s
    use_sentiment = False

    if has_sentiment:
        sent_cols = [c for c in SENTIMENT_COLS if c in X_train.columns]
        if sent_cols:
            scaler_sent = fit_and_save_scaler(X_train[sent_cols], filename=f"scaler_sent_{ticker}.pkl")
            use_sentiment = True

    # Train ensemble (RF + HGB)
    results = train_ensemble(X_train_s, y_train.values, X_test_s, y_test.values, ticker, random_state)

    # If sentiment features available, also train rf_sent and re-evaluate ensemble with 3 models
    if use_sentiment:
        from sklearn.ensemble import RandomForestClassifier

        X_train_sent = scaler_sent.transform(X_train[sent_cols])
        rf_sent = RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=50, max_features="sqrt",
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
        rf_sent.fit(X_train_sent, y_train.values)

        ps = rf_sent.predict_proba(scaler_sent.transform(X_test[sent_cols]))[:, 1]
        results["models"]["rf_sent"] = rf_sent
        _add_model_metrics(results, "rf_sent", rf_sent, scaler_sent.transform(X_test[sent_cols]), y_test.values)

        # 3-model ensemble
        pr = results["models"]["rf_tech"].predict_proba(X_test_s)[:, 1]
        ph = results["models"]["hgb"].predict_proba(X_test_s)[:, 1]
        ens_proba_3 = (pr + ph + ps) / 3.0
        ens_pred_3 = (ens_proba_3 >= 0.5).astype(int)
        try:
            results["metrics"]["ensemble_3model_auc"] = float(roc_auc_score(y_test.values, ens_proba_3))
        except Exception:
            results["metrics"]["ensemble_3model_auc"] = 0.0
        results["metrics"]["ensemble_3model_accuracy"] = float(accuracy_score(y_test.values, ens_pred_3))

        # Optimize weights for 3-model ensemble
        from inversion.ensemble.weighted_ensemble import optimize_weights
        models_for_opt = {"rf_tech": results["models"]["rf_tech"], "hgb": results["models"]["hgb"], "rf_sent": rf_sent}
        opt_labels, opt_weights = optimize_weights(models_for_opt, X_test_s, y_test.values)
        results["metrics"]["ensemble_optimized_auc"] = 0.0
        results["metrics"]["ensemble_optimized_accuracy"] = 0.0
        results["metrics"]["ensemble_weights"] = dict(zip(opt_labels, opt_weights))
        if len(opt_weights) > 0:
            ens_proba_opt = np.zeros(len(y_test))
            for label, w in zip(opt_labels, opt_weights):
                try:
                    if label == "rf_sent":
                        ens_proba_opt += w * rf_sent.predict_proba(X_test[sent_cols])[:, 1]
                    else:
                        ens_proba_opt += w * models_for_opt[label].predict_proba(X_test_s)[:, 1]
                except Exception:
                    continue
            # Normalize by sum of weights
            w_sum = sum(opt_weights)
            if w_sum > 0:
                ens_proba_opt /= w_sum
            ens_pred_opt = (ens_proba_opt >= 0.5).astype(int)
            try:
                results["metrics"]["ensemble_optimized_auc"] = float(roc_auc_score(y_test.values, ens_proba_opt))
            except Exception:
                results["metrics"]["ensemble_optimized_auc"] = 0.0
            results["metrics"]["ensemble_optimized_accuracy"] = float(accuracy_score(y_test.values, ens_pred_opt))

    # Save models and config
    ensemble_weights = results.get("metrics", {}).get("ensemble_weights", None)
    saved = save_ensemble_models(results, ticker, scaler_tech, scaler_sent if has_sentiment else None, ensemble_weights=ensemble_weights)

    # Clean model objects from results dict (not JSON-serialisable)
    metrics_only = {k: v for k, v in results.items() if k != "models"}

    print(
        f"   ✔ {ticker}: accuracy={metrics_only['metrics'].get('ensemble_3model_accuracy', 0):.3f} "
        f"auc={metrics_only['metrics'].get('ensemble_3model_auc', 0):.3f}"
    )
    if ensemble_weights:
        print(f"         optimized weights: {ensemble_weights}")

    return metrics_only


def _add_model_metrics(results: Dict[str, Any], name: str, model: Any, X: np.ndarray, y: np.ndarray) -> None:
    """Helper: add AUC and accuracy for a single model to results."""
    try:
        proba = model.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y, proba))
    except Exception:
        auc = 0.0
    acc = float(accuracy_score(y, model.predict(X)))
    results["metrics"][name] = {"auc": auc, "accuracy": acc}
