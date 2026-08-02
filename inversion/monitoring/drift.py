"""inversion.monitoring — Detección de drift y seguimiento de rendimiento (TMPL-004).

make monitor genera reports/monitoring/:
  - drift_<TICKER>.json  → PSI por feature entre la ventana de referencia
                           (primer 80% cronológico) y la ventana reciente
                           (últimas N filas).
  - rendimiento.csv      → accuracy/auc actual vs baseline registrado
                           (reports/resultados_<TICKER>.csv).

PSI (Population Stability Index) es la métrica estándar de drift por feature:
  PSI < 0.10  → sin cambio relevante
  0.10–0.25   → cambio moderado (revisar)
  > 0.25      → cambio significativo (acción)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inversion.utils import paths
from inversion.models.train_model import FEATURE_COLS, SENTIMENT_COLS

MONITORING_DIR = paths.PROJECT_DIR / "reports" / "monitoring"

# Umbrales de PSI estándar del sector.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25

# Tamaño de la ventana "reciente" para comparar contra la referencia.
RECENT_WINDOW = 60


def psi(feature_ref: pd.Series, feature_current: pd.Series, n_bins: int = 10) -> float:
    """PSI entre dos distribuciones de una feature (acotado a [0, inf))."""
    ref = feature_ref.dropna()
    cur = feature_current.dropna()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Bins por percentiles de la referencia; la actual se mapea a los mismos.
    bounds = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    bounds = np.unique(bounds)
    if len(bounds) < 2:
        return 0.0

    ref_bins = np.clip(np.digitize(ref, bounds[1:-1], right=True), 0, len(bounds) - 2)
    cur_bins = np.clip(np.digitize(cur, bounds[1:-1], right=True), 0, len(bounds) - 2)

    ref_pct = np.bincount(ref_bins, minlength=len(bounds) - 1) / len(ref) + 1e-6
    cur_pct = np.bincount(cur_bins, minlength=len(bounds) - 1) / len(cur) + 1e-6

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_drift(ticker: str, recent_window: int = RECENT_WINDOW) -> dict[str, Any]:
    """Compara la ventana de referencia con la reciente y devuelve el PSI por feature.

    La referencia es el primer 80% cronológico (el split de train del pipeline);
    la ventana reciente son las últimas `recent_window` filas. Se escribe el
    informe en reports/monitoring/drift_<TICKER>.json y se devuelve el dict.
    """
    csv = paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No hay features para '{ticker}': {csv}. Ejecuta make features.")

    df = pd.read_csv(csv, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    feature_cols = list(FEATURE_COLS)
    if all(c in df.columns for c in SENTIMENT_COLS):
        feature_cols += SENTIMENT_COLS

    split = int(len(df) * 0.8)
    reference = df.iloc[:split]
    current = df.iloc[-recent_window:]

    per_feature: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        per_feature[col] = {"psi": psi(reference[col], current[col])}

    max_psi = max((v["psi"] for v in per_feature.values()), default=0.0)
    if max_psi >= PSI_SIGNIFICANT:
        verdict = "significativo"
    elif max_psi >= PSI_MODERATE:
        verdict = "moderado"
    else:
        verdict = "sin_drift"

    report = {
        "ticker": ticker,
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "verdict": verdict,
        "max_psi": round(max_psi, 4),
        "features": per_feature,
    }

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    out = MONITORING_DIR / f"drift_{ticker}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def performance_vs_baseline(ticker: str) -> dict[str, Any]:
    """Compara el rendimiento actual contra el baseline registrado en reports/resultados_<T>.csv.

    El baseline registrado es la accuracy/auc que se guardó al entrenar
    (MODEL-001/MODEL-002). El rendimiento actual se mide re-evaluando el modelo
    guardado sobre el split temporal; la diferencia se reporta en
    reports/monitoring/rendimiento.csv.
    """
    import joblib
    from sklearn.metrics import accuracy_score, roc_auc_score

    ticker = ticker.upper()
    model_path = paths.MODELS_DIR / f"rf_{ticker}.pkl"
    scaler_path = paths.MODELS_DIR / f"scaler_{ticker}.pkl"
    baseline_path = paths.PROJECT_DIR / "reports" / "resultados_{ticker}.csv".format(ticker=ticker)
    if not model_path.exists() or not baseline_path.exists():
        raise FileNotFoundError(f"Falta modelo o baseline para '{ticker}'. Ejecuta make train.")

    df = pd.read_csv(paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv")
    feature_cols = list(FEATURE_COLS)
    if all(c in df.columns for c in SENTIMENT_COLS):
        feature_cols += SENTIMENT_COLS
    df = df.dropna(subset=[*feature_cols, "target"])
    split = int(len(df) * 0.8)
    X_test = df.iloc[split:][feature_cols]
    y_test = df.iloc[split:]["target"]

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    y_proba = model.predict_proba(scaler.transform(X_test))[:, 1]

    baseline = pd.read_csv(baseline_path).iloc[0]
    current = {
        "accuracy": float(accuracy_score(y_test, model.predict(scaler.transform(X_test)))),
        "auc": float(roc_auc_score(y_test, y_proba)),
    }
    delta = {
        "ticker": ticker,
        "baseline_accuracy": float(baseline["accuracy"]),
        "baseline_auc": float(baseline["auc"]),
        "current_accuracy": round(current["accuracy"], 4),
        "current_auc": round(current["auc"], 4),
        "delta_accuracy": round(current["accuracy"] - float(baseline["accuracy"]), 4),
        "delta_auc": round(current["auc"] - float(baseline["auc"]), 4),
    }
    return delta


def monitor_all() -> tuple[list[Path], Path | None]:
    """make monitor: drift + rendimiento de todos los tickers con features."""
    written: list[Path] = []
    for csv_path in sorted(paths.INTERIM_DATA_DIR.glob("features_*_ml_ready.csv")):
        ticker = csv_path.name.split("_")[1]
        try:
            report = detect_drift(ticker)
            written.append(MONITORING_DIR / f"drift_{ticker}.json")
            print(f"   ✔ {ticker}: verdict={report['verdict']} max_psi={report['max_psi']}")
        except Exception as exc:  # noqa: BLE001 — continúa con el resto
            print(f"   ✘ {ticker}: {exc}")

    perf_rows = []
    for csv_path in sorted(paths.INTERIM_DATA_DIR.glob("features_*_ml_ready.csv")):
        ticker = csv_path.name.split("_")[1]
        try:
            perf_rows.append(performance_vs_baseline(ticker))
        except Exception as exc:  # noqa: BLE001 — continúa
            print(f"   ✘ rendimiento {ticker}: {exc}")

    perf_out: Path | None = None
    if perf_rows:
        perf_out = MONITORING_DIR / "rendimiento.csv"
        pd.DataFrame(perf_rows).to_csv(perf_out, index=False)
        print(f"   ✔ rendimiento → {perf_out.name}")
    return written, perf_out


if __name__ == "__main__":
    monitor_all()
