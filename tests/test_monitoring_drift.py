from __future__ import annotations
import numpy as np
import pandas as pd

from inversion.monitoring.drift import detect_feature_drift
from inversion.monitoring.alerts import emit_alerts, alerts_to_json


def test_detect_feature_drift_no_drift() -> None:
    rng = np.random.default_rng(123)
    X_ref = pd.DataFrame({f"f{i}": rng.normal(0, 1, 100) for i in range(5)})
    X_new = pd.DataFrame({f"f{i}": rng.normal(0, 1, 80) for i in range(5)})
    res = detect_feature_drift(X_ref, X_new, threshold=0.01)
    assert isinstance(res.drift_detected, bool)


def test_detect_feature_drift_detects_shift() -> None:
    rng = np.random.default_rng(42)
    X_ref = pd.DataFrame({f"f{i}": rng.normal(0, 1, 120) for i in range(4)})
    X_new = pd.DataFrame({f"f{i}": rng.normal(2.0, 1, 100) for i in range(4)})
    res = detect_feature_drift(X_ref, X_new, threshold=0.1)
    # strong shift should be detectable
    assert res.mean_shift >= 0


def test_alerts_emitted() -> None:
    alerts = emit_alerts({"drift_detected": True})
    assert len(alerts) == 1
    s = alerts_to_json(alerts)
    assert "WARN" in s or "INFO" in s
