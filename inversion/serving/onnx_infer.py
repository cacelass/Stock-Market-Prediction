from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from inversion.utils import onnx as onnx_utils


@dataclass
class BenchResult:
    sklearn_ms: float
    onnx_ms: float
    n_samples: int
    speedup: float


def predict_onnx_file(
    onnx_path: str | Path,
    X: np.ndarray | pd.DataFrame,
    providers: Optional[List[str]] = None,
) -> np.ndarray:
    sess = onnx_utils.load_onnx_session(onnx_path, providers=providers)
    if isinstance(X, pd.DataFrame):
        X = X.values
    return onnx_utils.predict_onnx(sess, X)


def benchmark_latency(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    onnx_path: str | Path,
    n_runs: int = 10,
) -> BenchResult:
    if isinstance(X, pd.DataFrame):
        X = X.values
    # sklearn
    t0 = time.time()
    for _ in range(n_runs):
        try:
            model.predict_proba(X)[:, 1]
        except Exception:
            try:
                model.predict(X)
            except Exception:
                pass
    sklearn_ms = (time.time() - t0) / n_runs * 1000.0

    sess = onnx_utils.load_onnx_session(onnx_path)
    t0 = time.time()
    for _ in range(n_runs):
        onnx_utils.predict_onnx(sess, X)
    onnx_ms = (time.time() - t0) / n_runs * 1000.0

    speedup = (sklearn_ms / onnx_ms) if onnx_ms > 1e-12 else 0.0
    return BenchResult(sklearn_ms, onnx_ms, len(X), speedup)


def infer_cli(
    onnx_path: str | Path,
    csv_path: str | Path,
    out_path: str | Path | None = None,
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    # usar primeras columnas numéricas
    num = df.select_dtypes(include="number").fillna(0)
    preds = predict_onnx_file(onnx_path, num)
    res = pd.DataFrame({"pred": preds.ravel()})
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_path, index=False)
    return {"n": len(res), "mean_pred": float(res["pred"].mean())}


def infer_api_payload(payload: Dict[str, Any], onnx_path: str | Path) -> Dict[str, Any]:
    X = np.asarray(payload.get("X", []), dtype=float)
    preds = predict_onnx_file(onnx_path, X)
    return {"predictions": preds.tolist()}
