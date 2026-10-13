from __future__ import annotations
import numpy as np

from inversion.ensemble.stacking import train_ensemble, ensemble_vote, save_ensemble_report


def test_train_ensemble_reproducible(tmp_path):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(200, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    Xv = rng.normal(size=(50, 6))
    yv = (Xv[:, 0] + Xv[:, 1] > 0).astype(int)

    r1 = train_ensemble(X, y, Xv, yv)
    r2 = train_ensemble(X, y, Xv, yv)
    assert r1["accuracy"] == r2["accuracy"]
    assert r1["auc"] == r2["auc"]
    assert set(r1["models"]) == {"rf", "histgbm"}


def test_ensemble_vote_and_report(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(150, 5))
    y = (X[:, 0] > 0).astype(int)
    Xv = rng.normal(size=(40, 5))
    yv = (Xv[:, 0] > 0).astype(int)

    res = ensemble_vote(X, y, Xv, yv)
    assert res.accuracy >= 0
    assert res.auc >= 0
    assert res.cv_mean_auc >= 0
    assert res.vote_margin >= 0

    r = train_ensemble(X, y, Xv, yv)
    out = tmp_path / "ensemble_report.json"
    save_ensemble_report(r, out)
    assert out.exists()
