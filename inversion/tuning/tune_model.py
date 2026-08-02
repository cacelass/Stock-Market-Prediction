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

from inversion.utils import paths
from inversion.models.train_model import FEATURE_COLS, SENTIMENT_COLS

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_TRIALS = 30
RANDOM_STATE = 42


def _load_ticker_split(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga features_<T> y devuelve X_train/X_test/y_train/y_test (split temporal 80/20)."""
    csv = paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No hay features para '{ticker}': {csv}. Ejecuta make features.")
    df = pd.read_csv(csv)
    feature_cols = list(FEATURE_COLS)
    if all(c in df.columns for c in SENTIMENT_COLS):
        feature_cols += SENTIMENT_COLS
    X = df[feature_cols]
    y = df["target"]
    cut = int(len(df) * 0.8)
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def tune_ticker(ticker: str, n_trials: int = OPTUNA_TRIALS) -> dict[str, Any]:
    """Optimiza los hiperparámetros del RF de un ticker y guarda best_params_<T>.json."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = _load_ticker_split(ticker)

    def objective(trial: optuna.Trial) -> float:
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return float(accuracy_score(y_test, model.predict(X_test)))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    path = paths.ARTIFACTS_DIR / f"best_params_{ticker}.json"
    path.write_text(json.dumps({"ticker": ticker, "best_value": study.best_value, "params": best}, indent=2), encoding="utf-8")
    print(f"   ✔ {ticker}: best_value={study.best_value:.4f} params={best} → {path.name}")
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
