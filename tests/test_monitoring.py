"""
test_monitoring.py — Tests de detección de drift y rendimiento (TMPL-004).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inversion.utils import paths
from inversion.monitoring.drift import detect_drift, performance_vs_baseline, psi, monitor_all


@pytest.fixture
def features_synthetic(tmp_path, monkeypatch):
    """Escribe features_BBB_ml_ready.csv con un shift de media en la última ventana."""
    n = 400
    rng = np.random.default_rng(0)
    cols = [
        "return",
        "volatility",
        "volatility_21",
        "rsi",
        "momentum_10",
        "momentum_21",
        "momentum_63",
        "ma_50",
        "ma_200",
        "hl_range",
        "oc_range",
        "log_volume",
        "vwap_ratio",
        "lag_1",
        "lag_5",
        "lag_20",
        "day_of_week",
        "quarter",
    ]
    df = pd.DataFrame(rng.normal(loc=0.0, scale=1.0, size=(n, len(cols))), columns=cols)
    df["timestamp"] = pd.date_range("2020-01-01", periods=n)
    df["target"] = (df["return"] > 0).astype(int)

    # Drift sintético: las últimas 60 filas se desplazan +2σ en 'return'.
    df.loc[n - 60 :, "return"] += 2.0

    out = paths.INTERIM_DATA_DIR / "features_BBB_ml_ready.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def test_psi_same_distribution_is_low():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0, 1, 2000))
    b = pd.Series(rng.normal(0, 1, 2000))
    assert psi(a, b) < 0.10


def test_psi_shifted_distribution_is_high():
    rng = np.random.default_rng(2)
    a = pd.Series(rng.normal(0, 1, 2000))
    b = pd.Series(rng.normal(4, 1, 2000))
    assert psi(a, b) > 0.25


def test_detect_drift_sintetico_inyectado(features_synthetic):
    """El shift +2σ inyectado en la última ventana debe detectarse como drift."""
    report = detect_drift("BBB")
    assert report["verdict"] in ("moderado", "significativo")
    assert report["features"]["return"]["psi"] > 0.10


def test_detect_drift_escribe_json(features_synthetic):
    detect_drift("BBB")
    out = paths.PROJECT_DIR / "reports" / "monitoring" / "drift_BBB.json"
    assert out.exists()


def test_performance_vs_baseline(patch_paths, sample_df):
    """Con modelo + baseline en tmp, la comparación devuelve deltas."""
    import joblib

    from inversion.features.build_features import add_derived_features, fit_and_save_scaler
    from inversion.models.train_model import train_rf_model
    from inversion.models.predict_model import FEATURE_COLS

    df = add_derived_features(sample_df.copy())
    df["target"] = (df["return"] > 0).astype(int)
    feat = df.dropna(subset=[*FEATURE_COLS, "target"])
    X = feat[FEATURE_COLS].values
    y = (feat["return"] > 0).astype(int).values
    fit_and_save_scaler(X, filename="scaler_CCC.pkl")
    train_rf_model(X, y, filename="rf_CCC.pkl")

    interim = paths.INTERIM_DATA_DIR / "features_CCC_ml_ready.csv"
    df.dropna(subset=FEATURE_COLS).to_csv(interim, index=False)
    reports = paths.PROJECT_DIR / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"accuracy": 0.5, "auc": 0.5}]).to_csv(reports / "resultados_CCC.csv", index=False)

    delta = performance_vs_baseline("CCC")
    assert "delta_accuracy" in delta
    assert "delta_auc" in delta


def test_monitor_all_writes_reports(features_synthetic):
    written, perf = monitor_all()
    assert any(p.name == "drift_BBB.json" for p in written)
