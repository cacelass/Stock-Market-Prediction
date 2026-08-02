"""
test_proba.py — Smoke tests: verifican que todos los módulos son importables
y que las funciones principales existen con las firmas esperadas.
"""

import pytest
import inspect

pytestmark = pytest.mark.smoke


def test_import_data_module():
    from inversion.data import make_dataset

    assert hasattr(make_dataset, "load_data")
    assert hasattr(make_dataset, "get_yfinance_history")


def test_import_features_module():
    from inversion.features import build_features

    assert hasattr(build_features, "add_derived_features")
    assert hasattr(build_features, "fit_and_save_scaler")


def test_import_models_train():
    from inversion.models import train_model

    assert hasattr(train_model, "train_rf_model")
    assert hasattr(train_model, "evaluate_model")


def test_import_models_predict():
    from inversion.models import predict_model

    assert hasattr(predict_model, "predict_future")


def test_import_visualization():
    from inversion.visualization import visualize

    assert hasattr(visualize, "plot_price")
    assert hasattr(visualize, "plot_predictions")
    assert hasattr(visualize, "plot_feature_importance")
    assert hasattr(visualize, "plot_returns_distribution")


def test_import_utils_paths():
    from inversion.utils import paths

    assert hasattr(paths, "MODELS_DIR")
    assert hasattr(paths, "RAW_DATA_DIR")
    assert hasattr(paths, "MODEL_FILE")


def test_load_data_signature():
    """load_data debe aceptar un argumento opcional 'ticker'."""
    from inversion.data.make_dataset import load_data

    sig = inspect.signature(load_data)
    assert "ticker" in sig.parameters


def test_add_derived_features_signature():
    """add_derived_features debe aceptar un DataFrame."""
    from inversion.features.build_features import add_derived_features

    sig = inspect.signature(add_derived_features)
    assert "df" in sig.parameters


def test_train_rf_model_signature():
    from inversion.models.train_model import train_rf_model

    sig = inspect.signature(train_rf_model)
    assert "X_train" in sig.parameters
    assert "y_train" in sig.parameters


def test_predict_future_signature():
    from inversion.models.predict_model import predict_future

    sig = inspect.signature(predict_future)
    assert "df" in sig.parameters
