from __future__ import annotations
import numpy as np
from unittest.mock import patch, MagicMock

from inversion.serving.onnx_infer import infer_cli, infer_api_payload


def test_infer_api_payload():
    with patch("inversion.serving.onnx_infer.onnx_utils.predict_onnx", return_value=np.array([0.1, 0.9])):
        with patch("inversion.serving.onnx_infer.onnx_utils.load_onnx_session", return_value="sess"):
            res = infer_api_payload({"X": [[1, 2], [3, 4]]}, "model.onnx")
            assert res["predictions"] == [0.1, 0.9]


def test_infer_cli(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    out = tmp_path / "p.csv"
    with patch("inversion.serving.onnx_infer.onnx_utils.predict_onnx", return_value=np.array([0, 1])):
        with patch("inversion.serving.onnx_infer.onnx_utils.load_onnx_session", return_value="sess"):
            res = infer_cli("m.onnx", csv, out)
            assert res["n"] == 2
            assert out.exists()
