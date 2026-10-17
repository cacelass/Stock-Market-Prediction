"""Weighted ensemble: 3 models (RF-tech, HGB-tech, RF-sentiment) + optimal weights.

Trains three base models, optimises weights on a validation set, and predicts
with a weighted probability average.  Graceful degradation if some models
cannot be trained (e.g. no sentiment features).

Functions
---------
train_weighted_ensemble  — train + weight optimisation
predict_weighted_ensemble — weighted probability average
optimize_weights         — grid-search on validation set
compute_dynamic_weights  — adapt weights from recent performance
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def train_weighted_ensemble(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_val: np.ndarray | pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    use_sentiment: bool = False,
) -> Dict[str, Any]:
    """Train 3 models and find optimal weights on a validation set.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : validation data (used for weight optimisation)
    use_sentiment : if True, train the RF-sentiment model on all features
                    including sentiment columns.

    Returns
    -------
    dict with keys:
        models (list[str]) — model labels trained
        weights (list[float]) — optimised weights
        metrics (dict) — per-model and ensemble metrics on validation set
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)

    models: Dict[str, Any] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    # Model 1: RF-Technical
    try:
        rf_tech = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=50,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf_tech.fit(X_train, y_train)
        models["rf_tech"] = rf_tech
        pr = rf_tech.predict_proba(X_val)[:, 1]
        metrics["rf_tech"] = {
            "auc": float(roc_auc_score(y_val, pr)),
            "accuracy": float(accuracy_score(y_val, rf_tech.predict(X_val))),
        }
    except Exception as exc:
        print(f"  ⚠ rf_tech failed: {exc}")

    # Model 2: HGB-Technical
    try:
        hgb = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            class_weight="balanced",
        )
        hgb.fit(X_train, y_train)
        models["hgb"] = hgb
        ph = hgb.predict_proba(X_val)[:, 1]
        metrics["hgb"] = {
            "auc": float(roc_auc_score(y_val, ph)),
            "accuracy": float(accuracy_score(y_val, hgb.predict(X_val))),
        }
    except Exception as exc:
        print(f"  ⚠ hgb failed: {exc}")

    # Model 3: RF-Sentiment (only if use_sentiment and enough features)
    if use_sentiment and len(X_train.T) >= 3:
        try:
            rf_sent = RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=50,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            rf_sent.fit(X_train, y_train)
            models["rf_sent"] = rf_sent
            ps = rf_sent.predict_proba(X_val)[:, 1]
            metrics["rf_sent"] = {
                "auc": float(roc_auc_score(y_val, ps)),
                "accuracy": float(accuracy_score(y_val, rf_sent.predict(X_val))),
            }
        except Exception as exc:
            print(f"  ⚠ rf_sent failed: {exc}")

    # Optimise weights
    weight_labels, weights = optimize_weights(models, X_val, y_val)
    len(weight_labels)  # noqa: B018 — keep for clarity

    # Ensemble metric on validation
    ens_proba = np.zeros(len(y_val))
    ens_count = 0
    for label, w in zip(weight_labels, weights):
        model = models[label]
        ens_proba += w * model.predict_proba(X_val)[:, 1]
        ens_count += 1

    if ens_count > 0:
        ens_proba /= ens_count

    ens_pred = (ens_proba >= 0.5).astype(int)

    result: Dict[str, Any] = {
        "models": weight_labels,
        "weights": weights,
        "metrics": metrics,
        "ensemble_auc": float(roc_auc_score(y_val, ens_proba)) if ens_count > 0 else 0.0,
        "ensemble_accuracy": float(accuracy_score(y_val, ens_pred)) if ens_count > 0 else 0.0,
    }
    return result


def predict_weighted_ensemble(
    models: Dict[str, Any],
    X: np.ndarray | pd.DataFrame,
    weights: List[float],
    model_labels: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict using weighted ensemble average of probabilities.

    Parameters
    ----------
    models : dict of {label: model}
    X : features
    weights : list of weights corresponding to model_labels
    model_labels : which models to use (default: all available in `models`)

    Returns
    -------
    (predictions, probabilities) — binary predictions and averaged probability
    of class 1.
    """
    X = np.asarray(X)
    labels = model_labels or list(models.keys())
    w_map = dict(zip(labels, weights))

    proba = np.zeros(len(X))
    n_used = 0
    for label in labels:
        if label not in models:
            continue
        model = models[label]
        try:
            proba += w_map.get(label, 1.0) * model.predict_proba(X)[:, 1]
        except Exception:
            continue
        n_used += 1

    if n_used == 0:
        return np.zeros(len(X), dtype=int), np.zeros(len(X))

    proba /= n_used
    predictions = (proba >= 0.5).astype(int)
    return predictions, proba


def optimize_weights(
    models: Dict[str, Any],
    X_val: np.ndarray | pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    step: float = 0.1,
) -> Tuple[List[str], List[float]]:
    """Find optimal weights using grid search on validation set.

    Iterates over weight combinations from 0.0 to 1.0 in steps of `step`
    and picks the combination with the highest AUC on the validation set.
    Falls back to equal weighting if no models can produce predictions.

    Returns
    -------
    (model_labels, weights)
    """
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)
    available = [label for label in models if label in models]

    if len(available) <= 1:
        return list(available), [1.0] * len(available)

    best_auc = -1.0
    best_weights: List[float] = [1.0 / len(available)] * len(available)
    best_labels = available

    # Generate all weight combinations
    n = len(available)
    steps = int(1.0 / step) + 1
    combinations = []
    _generate_combos(n, steps, step, [], combinations)

    for combo in combinations:
        weights = list(combo) + [1.0 - sum(combo)]
        if weights[-1] < -1e-9:
            continue
        weights = [max(w, 0.0) for w in weights]

        proba = np.zeros(len(X_val))
        for i, label in enumerate(available):
            try:
                proba += weights[i] * models[label].predict_proba(X_val)[:, 1]
            except Exception:
                continue
        proba /= n  # normalise

        try:
            auc = roc_auc_score(y_val, proba)
        except Exception:
            continue

        if auc > best_auc:
            best_auc = auc
            best_weights = weights
            best_labels = available

    # Normalise weights to sum to 1
    total = sum(best_weights)
    if total > 0:
        best_weights = [w / total for w in best_weights]

    return best_labels, best_weights


def _generate_combos(n: int, steps: int, step: float, current: list, out: list) -> None:
    """Recursively generate weight combinations that sum to <= 1.0."""
    if len(current) == n - 1:
        out.append(list(current))
        return
    remaining = 1.0 - sum(current)
    for i in range(int(remaining / step) + 1):
        current.append(round(i * step, 2))
        _generate_combos(n, steps, step, current, out)
        current.pop()


def compute_dynamic_weights(
    models: Dict[str, Any],
    recent_performance: pd.DataFrame,
    model_labels: List[str] | None = None,
    window: int = 20,
    alpha: float = 0.85,
) -> Tuple[List[str], List[float]]:
    """Compute weights based on recent model performance with exponential decay.

    Models that have performed better in the recent window get higher weights.
    Recency is modelled with exponential decay (alpha < 1): more recent
    performance is weighted more heavily.

    Parameters
    ----------
    models : dict of {label: fitted model}
    recent_performance : DataFrame with columns matching model_labels,
                         each column contains daily accuracy or AUC.
    model_labels : which models to include (default: all in `models`)
    window : number of recent rows to consider
    alpha : decay factor (0.85 = strong recency bias)

    Returns
    -------
    (model_labels, weights) — normalised weights summing to 1.0
    """
    labels = model_labels or list(models.keys())
    perf = recent_performance[labels].tail(window).values

    # Exponential decay weights
    decay = np.array([alpha ** i for i in range(len(perf) - 1, -1, -1)])
    decay /= decay.sum()

    # Weighted average performance per model
    weighted_perf = np.zeros(len(labels))
    for i, label in enumerate(labels):
        col = perf[:, i]
        # Replace NaN with neutral value (0.5)
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            weighted_perf[i] = 0.5
        else:
            weighted_perf[i] = np.average(col[valid], weights=decay[valid])

    # Convert to positive weights (ensure all > 0)
    min_perf = weighted_perf.min()
    adjusted = weighted_perf - min_perf + 1e-6

    # Normalise to sum to 1
    total = adjusted.sum()
    weights = adjusted / total

    return labels, weights.tolist()
