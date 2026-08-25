"""
tuning/tune_model.py — Optimización de hiperparámetros con Optuna (TMPL-003).

Ejecutar con:
    make tune

O directamente:
    .venv/bin/python inversion/tuning/tune_model.py

Qué hace:
    1. Por cada ticker del catálogo con features (data/interim), ejecuta un
       estudio Optuna con el split temporal honesto del pipeline (80/20).
    2. Guarda los mejores params en models/artifacts/best_params_<TICKER>.json.
    3. train_model.py los carga automáticamente en el siguiente make train.
    4. Guarda un resumen en reports/tuning_results.csv.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import optuna
import pandas as pd

from inversion.models.train_model import available_feature_cols
from inversion.utils import paths

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_TRIALS = 30
RANDOM_STATE = 42

# Walk-forward para el objetivo de tuning (IMP-003): cada trial se evalúa como
# media de AUC sobre folds expanding-window. Optimizar sobre un único split
# 80/20 elegía hiperparámetros que sobreajustaban ese tramo de test concreto.
WF_FOLDS = 3
WF_START_FRAC = 0.60
WF_MIN_TRAIN = 100
WF_MIN_TEST = 15


def _load_ticker_split(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga features_<T> y devuelve X_train/X_test/y_train/y_test (split temporal 80/20)."""
    csv = paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No hay features para '{ticker}': {csv}. Ejecuta make features.")
    df = pd.read_csv(csv)
    feature_cols = available_feature_cols(df)
    X = df[feature_cols]
    y = df["target"]
    cut = int(len(df) * 0.8)
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def _load_ticker_full(ticker: str) -> tuple[pd.DataFrame, pd.Series]:
    """Carga features_<T> completas (X, y) con las columnas disponibles."""
    csv = paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No hay features para '{ticker}': {csv}. Ejecuta make features.")
    df = pd.read_csv(csv).dropna(subset=["target"]).reset_index(drop=True)
    feature_cols = available_feature_cols(df)
    df = df.dropna(subset=feature_cols)
    return df[feature_cols], df["target"]


def _walk_forward_folds(
    n: int,
    n_folds: int = WF_FOLDS,
    start_frac: float = WF_START_FRAC,
    min_train: int = WF_MIN_TRAIN,
    min_test: int = WF_MIN_TEST,
) -> list[tuple[int, int]]:
    """Fronteras (train_end, test_end) de folds expanding-window sobre n filas.

    El entrenamiento de cada fold es todo el pasado ([0, train_end)) y el test
    el bloque siguiente. Descarta folds demasiado pequeños (datasets cortos,
    p.ej. fixtures de test) en vez de fallar.
    """
    start = int(n * start_frac)
    fold_size = max((n - start) // n_folds, 1)
    folds = []
    for k in range(n_folds):
        train_end = start + k * fold_size
        test_end = start + (k + 1) * fold_size if k < n_folds - 1 else n
        if train_end >= min_train and (test_end - train_end) >= min_test:
            folds.append((train_end, test_end))
    return folds


def tune_ticker(ticker: str, n_trials: int = OPTUNA_TRIALS, n_folds: int = WF_FOLDS) -> dict[str, Any]:
    """Optimiza los hiperparámetros del RF de un ticker y guarda best_params_<T>.json.

    Objetivo (IMP-003): media de AUC sobre `n_folds` expanding-window — el
    mismo protocolo honesto con el que después se reporta. `best_value` en el
    JSON es esa media, NO la accuracy de un split único (como antes).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    X, y = _load_ticker_full(ticker)
    folds = _walk_forward_folds(len(X), n_folds=n_folds)
    if not folds:
        raise ValueError(f"'{ticker}': dataset demasiado corto ({len(X)} filas) para walk-forward.")

    def objective(trial: optuna.Trial) -> float:
        model_tpl = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
        scores = []
        for train_end, test_end in folds:
            model = RandomForestClassifier(**model_tpl)
            model.fit(X.iloc[:train_end], y.iloc[:train_end])
            proba = model.predict_proba(X.iloc[train_end:test_end])[:, 1]
            scores.append(roc_auc_score(y.iloc[train_end:test_end], proba))
        return float(sum(scores) / len(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    path = paths.ARTIFACTS_DIR / f"best_params_{ticker}.json"
    path.write_text(json.dumps({"ticker": ticker, "best_value": round(study.best_value, 4), "objective": "wf_mean_auc", "params": best}, indent=2), encoding="utf-8")
    print(f"   ✔ {ticker}: wf_auc={study.best_value:.4f} params={best} → {path.name}")
    return best


def tune_all(n_trials: int = OPTUNA_TRIALS) -> None:
    """Optimiza todos los tickers del catálogo con features y escribe tuning_results.csv."""
    rows = []
    for csv_path in sorted(paths.INTERIM_DATA_DIR.glob("features_*_ml_ready.csv")):
        ticker = csv_path.name.split("_")[1]
        try:
            best = tune_ticker(ticker, n_trials=n_trials)
            rows.append({"ticker": ticker, **best})
        except Exception as exc:  # noqa: BLE001 — continua con el resto del catálogo
            print(f"   ✘ {ticker}: {exc}")

    if rows:
        df = pd.DataFrame(rows)
        out = paths.REPORTS_DIR / "tuning_results.csv"
        df.to_csv(out, index=False)
        print(f"\n  Resumen → {out}")
        print(df.to_string(index=False))


def load_best_params(ticker: str) -> dict[str, Any] | None:
    """Carga los mejores params de Optuna para un ticker, o None si no existen."""
    path = paths.ARTIFACTS_DIR / f"best_params_{ticker}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    params = data.get("params")
    return dict(params) if isinstance(params, dict) else None


if __name__ == "__main__":
    tune_all()
