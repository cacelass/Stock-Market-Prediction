"""test_pooled.py — Tests del enrutado híbrido global/ticker (IMP-006).

Contrato clave: sin artefactos globales el enrutado es identidad — el
comportamiento de todo el sistema debe ser idéntico al previo a IMP-006.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from inversion.models import pooled


def test_resolve_route_identity_sin_artefactos(patch_paths):
    """Sin modelo global entrenado, ningún ticker se desvía al pool."""
    assert pooled.resolve_route("NVDA") == "NVDA"
    assert pooled.resolve_route("AAPL") == "AAPL"


def test_resolve_route_desvia_segun_routing(patch_paths):
    """Con routing.json presente, el ticker desviado va a GLOBAL y el resto no."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    # Artefactos mínimos para que global_artifacts_available() sea True.
    X = pd.DataFrame({"f1": [0.0, 1.0], "f2": [1.0, 0.0]})
    arts = pooled._artifact_paths()
    joblib.dump(RandomForestClassifier(n_estimators=2).fit(X, [0, 1]), arts["model"])
    joblib.dump(X.iloc[:0], arts["scaler"])
    arts["features"].write_text(json.dumps({"features": ["f1", "f2", "tkr_MSFT", "tkr_NVDA"]}), encoding="utf-8")
    arts["routing"].write_text(
        json.dumps({"routing": {"NVDA": "GLOBAL"}}),
        encoding="utf-8",
    )

    assert pooled.resolve_route("NVDA") == "GLOBAL"
    assert pooled.resolve_route("MSFT") == "MSFT"
    assert pooled.resolve_route("GOOGL") == "GOOGL"


def test_add_ticker_dummies_fija_one_hot():
    df = pd.DataFrame({"return": [0.01, -0.02]})
    cols = ["return", "tkr_AAPL", "tkr_NVDA"]
    out = pooled.add_ticker_dummies(df, "NVDA", cols)
    assert out["tkr_NVDA"].tolist() == [1, 1]
    assert out["tkr_AAPL"].tolist() == [0, 0]
    # El original no se muta.
    assert "tkr_NVDA" not in df.columns


def test_derive_routing_solo_debil_y_ganado():
    ind = pd.DataFrame(
        {
            "ticker": ["GOOGL", "NVDA"],
            "wf_auc_media": [0.63, 0.50],
        }
    )
    cmp_ = pd.DataFrame(
        {
            "ticker": ["GOOGL", "NVDA"],
            "wf_auc": [0.58, 0.54],
            "ind_auc": [0.63, 0.50],
        }
    )
    routing = pooled._derive_routing(ind, cmp_)
    # GOOGL tiene señal propia fuerte → individual aunque el pool no le gane.
    # NVDA es débil y el pool mejora → GLOBAL.
    assert routing == {"NVDA": "GLOBAL"}


def test_load_pooled_requiere_features(patch_paths):
    with pytest.raises(FileNotFoundError):
        pooled.load_pooled()
