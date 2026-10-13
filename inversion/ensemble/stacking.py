from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, roc_auc_score
except Exception:
    HistGradientBoostingClassifier = RandomForestClassifier = None
    TimeSeriesSplit = None


@dataclass
class EnsembleResult:
    accuracy: float
    auc: float
    models: List[str]
    cv_mean_auc: float
    vote_margin: float


def _cv_mean_auc(X, y, max_splits: int = 3) -> float:
    if X is None or len(X) == 0 or TimeSeriesSplit is None:
        return 0.0
    try:
        n = len(X)
        tscv = TimeSeriesSplit(n_splits=min(max_splits, max(2, n // 15 + 1 if n >= 30 else 2)))
        scores = []
        for tr, te in tscv.split(X):
            rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=1)
            hg = HistGradientBoostingClassifier(max_iter=20, max_depth=3, random_state=42)
            rf.fit(X[tr], y[tr])
            hg.fit(X[tr], y[tr])
            try:
                pr = rf.predict_proba(X[te])[:, 1]
                ph = hg.predict_proba(X[te])[:, 1]
                p = (pr + ph) / 2.0
                scores.append(roc_auc_score(y[te], p))
            except Exception:
                continue
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> Dict[str, Any]:
    if RandomForestClassifier is None or HistGradientBoostingClassifier is None:
        raise RuntimeError("sklearn required")

    rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=1)
    hg = HistGradientBoostingClassifier(max_iter=50, max_depth=4, random_state=42)

    rf.fit(X_train, y_train)
    hg.fit(X_train, y_train)

    Xv, yv = X_val, y_val
    if Xv is None:
        Xv, yv = X_train, y_train

    try:
        pr = rf.predict_proba(Xv)[:, 1]
        ph = hg.predict_proba(Xv)[:, 1]
    except Exception:
        pr = rf.predict(Xv).astype(float)
        ph = hg.predict(Xv).astype(float)
    pv = (pr + ph) / 2.0
    pred = (pv >= 0.5).astype(int)

    acc = float(accuracy_score(yv, pred))
    try:
        auc = float(roc_auc_score(yv, pv))
    except Exception:
        auc = 0.0

    vote_margin = float(np.mean(np.abs(pv - 0.5)))
    cv_auc = _cv_mean_auc(X_train, y_train)

    return {
        "accuracy": acc,
        "auc": auc,
        "models": ["rf", "histgbm"],
        "cv_mean_auc": cv_auc,
        "vote_margin": vote_margin,
        "pred_proba": pv,
    }


def save_ensemble_report(result: Dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rep = {k: v for k, v in result.items() if k != "pred_proba"}
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=2)
    return out_path


def ensemble_vote(X_train, y_train, X_val=None, y_val=None) -> EnsembleResult:
    r = train_ensemble(X_train, y_train, X_val, y_val)
    return EnsembleResult(
        accuracy=r["accuracy"],
        auc=r["auc"],
        models=r["models"],
        cv_mean_auc=r["cv_mean_auc"],
        vote_margin=r["vote_margin"],
    )
