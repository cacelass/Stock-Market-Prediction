from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from inversion.utils.onnx import export_sklearn_to_onnx, load_onnx_session, predict_onnx


def test_export_and_predict_onnx(tmp_path: Path) -> None:
    X = pd.DataFrame({"a": np.random.rand(30), "b": np.random.rand(30), "c": np.random.rand(30)})
    y = (X["a"] + X["b"] > 1.0).astype(int)
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    out = tmp_path / "model.onnx"
    export_sklearn_to_onnx(model, X, out)
    assert out.exists()

    sess = load_onnx_session(out)
    pred = predict_onnx(sess, X.head(5))
    assert len(pred) == 5
    assert np.all((pred >= 0) & (pred <= 1))


def test_export_random_forest_onnx(tmp_path: Path) -> None:
    X = pd.DataFrame({"a": np.random.rand(25), "b": np.random.rand(25)})
    y = (X["a"] > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    out = tmp_path / "rf.onnx"
    export_sklearn_to_onnx(model, X, out)
    assert out.exists()

    sess = load_onnx_session(out)
    pred = predict_onnx(sess, X)
    assert len(pred) == len(X)
