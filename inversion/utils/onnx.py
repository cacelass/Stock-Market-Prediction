from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def export_sklearn_to_onnx(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    output_path: str | Path,
    feature_names: Optional[list[str]] = None,
    opset: int = 17,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as e:  # pragma: no cover - optional
        raise ImportError(f"skl2onnx/onnxruntime no disponibles: {e}")

    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f"f{i}" for i in range(X.shape[1])]

    initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
    try:
        onx = convert_sklearn(model, initial_types=initial_type, target_opset=opset, feature_names=feature_names)
    except TypeError:
        onx = convert_sklearn(model, initial_types=initial_type, target_opset=opset)
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
    return output_path


def load_onnx_session(model_path: str | Path) -> Any:
    try:
        import onnxruntime as ort
    except Exception as e:  # pragma: no cover - optional
        raise ImportError(f"onnxruntime no disponible: {e}")
    model_path = Path(model_path)
    sess = ort.InferenceSession(str(model_path))
    return sess


def predict_onnx(
    sess: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: Optional[list[str]] = None,
) -> np.ndarray:
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f"f{i}" for i in range(X.shape[1])]

    if isinstance(X, pd.DataFrame):
        X_arr = X[feature_names].to_numpy(dtype=np.float32)
    else:
        X_arr = np.asarray(X, dtype=np.float32)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    res = sess.run([label_name], {input_name: X_arr})
    return np.array(res[0]).ravel()
