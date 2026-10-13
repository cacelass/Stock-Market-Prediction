from __future__ import annotations
import numpy as np

from inversion.gates.conformal_gate import conformal_gate, dynamic_threshold, save_gate_report


def test_dynamic_threshold_clips():
    t = dynamic_threshold(vol=2.0, base=0.6, min_t=0.5, max_t=0.8, k=0.2)
    assert 0.5 <= t <= 0.8


def test_conformal_gate_filters():
    proba = np.array([0.3, 0.7, 0.4, 0.9])
    y = np.array([0, 1, 0, 1])
    pred, rep = conformal_gate(proba, labels=y, alpha=0.1, base_thresh=0.6)
    assert len(pred) == 4
    assert rep.passed >= 0
    assert rep.filtered >= 0
    assert 0 <= rep.volume_impact <= 1


def test_save_gate_report(tmp_path):
    proba = np.array([0.5, 0.8])
    y = np.array([0, 1])
    _, rep = conformal_gate(proba, labels=y, base_thresh=0.6)
    out = tmp_path / "gate_report.json"
    save_gate_report(rep, out)
    assert out.exists()
