"""
test_train_model.py — Tests para inversion/models/train_model.py
"""

import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier

from inversion.models.train_model import (
    train_rf_model,
    evaluate_model,
    train_ticker,
    save_metrics,
)


def _make_Xy():
    """Datos sintéticos pequeños para entrenamiento rápido en tests."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=120, n_features=4, n_classes=2, random_state=42)
    return X, y


@pytest.mark.smoke
def test_train_rf_model_returns_rf():
    X, y = _make_Xy()
    model = train_rf_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")


def test_train_rf_model_saves_joblib_file(patch_paths):
    from inversion.utils import paths

    X, y = _make_Xy()
    train_rf_model(X, y)
    assert patch_paths["MODEL_FILE"].exists()
    assert paths.MODEL_FILE == patch_paths["MODEL_FILE"]


def test_train_rf_model_saves_custom_filename(patch_paths):
    from inversion.utils import paths

    X, y = _make_Xy()
    train_rf_model(X, y, filename="mi_modelo.pkl")
    assert (paths.MODELS_DIR / "mi_modelo.pkl").exists()


def test_train_rf_model_can_predict(patch_paths):
    X, y = _make_Xy()
    model = train_rf_model(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_train_rf_model_reproducible(patch_paths):
    X, y = _make_Xy()
    m1 = train_rf_model(X, y, random_state=7)
    m2 = train_rf_model(X, y, random_state=7)
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


def test_evaluate_model_returns_metrics(patch_paths):
    from sklearn.model_selection import train_test_split

    X, y = _make_Xy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = train_rf_model(X_train, y_train)
    res = evaluate_model(model, X_test, y_test)
    assert isinstance(res, dict)
    assert 0 <= res["accuracy"] <= 1
    assert 0 <= res["auc"] <= 1
    assert "report" in res


def test_train_ticker_saves_model_and_metrics(patch_paths):
    """El entrenamiento por ticker guarda modelo+scaler y devuelve métricas."""
    import pandas as pd
    from inversion.utils import paths
    from inversion.features.build_features import add_derived_features

    # Dataset sintético con target binario
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 100 + np.arange(n) * 0.1,
            "high": 100 + np.arange(n) * 0.1 + 1,
            "low": 100 + np.arange(n) * 0.1 - 1,
            "close": 100 + np.arange(n) * 0.1,
            "volume": 1_000_000.0,
            "vwap": 100 + np.arange(n) * 0.1,
        }
    )
    df = add_derived_features(df)
    df["target"] = (np.sin(np.arange(len(df)) * 0.3) > 0).astype(int)
    df = df.dropna().reset_index(drop=True)

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_TST.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    metrics = train_ticker("TST", csv_path)
    assert metrics["accuracy"] >= 0
    assert metrics["train_accuracy"] >= 0
    assert (paths.MODELS_DIR / "rf_TST.pkl").exists()
    assert (paths.MODELS_DIR / "scaler_TST.pkl").exists()

    out = save_metrics("TST", metrics)
    assert out.exists()
    import csv as _csv

    with open(out) as fh:
        row = list(_csv.DictReader(fh))[0]
    assert float(row["accuracy"]) == metrics["accuracy"]


def test_train_ticker_reproducible_with_seed(patch_paths):
    """Misma seed → mismas métricas y mismo modelo byte a byte."""
    import pandas as pd
    from inversion.utils import paths
    from inversion.features.build_features import add_derived_features

    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 100 + np.arange(n) * 0.1,
            "high": 100 + np.arange(n) * 0.1 + 1,
            "low": 100 + np.arange(n) * 0.1 - 1,
            "close": 100 + np.arange(n) * 0.1,
            "volume": 1_000_000.0,
            "vwap": 100 + np.arange(n) * 0.1,
        }
    )
    df = add_derived_features(df)
    df["target"] = (np.sin(np.arange(len(df)) * 0.3) > 0).astype(int)
    df = df.dropna().reset_index(drop=True)

    csv_path = patch_paths["INTERIM_DATA_DIR"] / "features_TST.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    m1 = train_ticker("TST", csv_path, random_state=42)
    model1 = (paths.MODELS_DIR / "rf_TST.pkl").read_bytes()
    m2 = train_ticker("TST", csv_path, random_state=42)
    model2 = (paths.MODELS_DIR / "rf_TST.pkl").read_bytes()

    assert m1["accuracy"] == m2["accuracy"]
    assert m1["train_accuracy"] == m2["train_accuracy"]
    assert model1 == model2
