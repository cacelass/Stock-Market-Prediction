from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple
import json
import numpy as np
import pandas as pd


@dataclass
class GateReport:
    passed: int
    filtered: int
    precision_estimate: float
    volume_impact: float


def dynamic_threshold(vol: float, base: float = 0.6, min_t: float = 0.5, max_t: float = 0.8, k: float = 0.5) -> float:
    t = base - k * vol
    return float(np.clip(t, min_t, max_t))


def conformal_gate(
    proba: np.ndarray,
    labels: np.ndarray | None = None,
    alpha: float = 0.1,
    vol_series: np.ndarray | None = None,
    base_thresh: float = 0.6,
) -> Tuple[np.ndarray, GateReport]:
    proba = np.asarray(proba).ravel()
    pred = (proba >= base_thresh).astype(int)

    passed = int(pred.sum())
    filtered = int(len(pred) - passed)

    if labels is not None and len(labels) == len(proba) and passed > 0:
        try:
            prec = float(np.mean(labels[pred == 1] == pred[pred == 1]))
        except Exception:
            prec = 0.0
    else:
        prec = 0.0

    vol_impact = float(filtered / len(pred)) if len(pred) > 0 else 0.0

    if vol_series is not None and len(vol_series) == len(proba):
        # apply dynamic per-sample approx (mean)
        v = float(np.mean(np.asarray(vol_series)))
        t = dynamic_threshold(v, base=base_thresh)
        pred2 = (proba >= t).astype(int)
        passed = int(pred2.sum())
        filtered = int(len(pred2) - passed)
        vol_impact = float(filtered / len(pred2)) if len(pred2) > 0 else 0.0

    return pred, GateReport(passed=passed, filtered=filtered, precision_estimate=prec, volume_impact=vol_impact)


def save_gate_report(rep: GateReport, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = {
        "passed": rep.passed,
        "filtered": rep.filtered,
        "precision_estimate": rep.precision_estimate,
        "volume_impact": rep.volume_impact,
    }
    with open(out_path, "w") as f:
        json.dump(d, f, indent=2)
    return out_path
