"""
conftest.py — Fixtures compartidas para todos los tests.
Los fixtures se adaptan automáticamente a la estructura del proyecto
(inversion/, data/, models/).
"""

import importlib

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures de datos sintéticos
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df():
    """DataFrame con columnas OHLCV (timestamp, open, high, low, close, volume)."""
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n))
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="B"),
            "open": close + np.random.randn(n),
            "high": close + 1 + np.abs(np.random.randn(n)),
            "low": close - 1 - np.abs(np.random.randn(n)),
            "close": close,
            "volume": np.random.randint(1_000_000, 5_000_000, size=n).astype(float),
        }
    )
    return df


@pytest.fixture
def df_with_features(sample_df):
    """sample_df con las features derivadas ya calculadas."""
    from inversion.features.build_features import add_derived_features

    return add_derived_features(sample_df.copy())


@pytest.fixture
def df_with_target(sample_df):
    """DataFrame con features + target binario (sube/baja)."""
    from inversion.features.build_features import add_derived_features

    df = add_derived_features(sample_df.copy())
    df["target"] = (df["return"] > 0).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fixture de parcheo de rutas (aislamiento del filesystem)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def patch_paths(monkeypatch, tmp_path):
    """
    Redirige todas las constantes de ruta del proyecto a tmp_path.
    Se aplica automáticamente a cada test para evitar escrituras en disco real.
    """
    dirs = {
        "MODELS_DIR": tmp_path / "models",
        # ARTIFACTS_DIR vive dentro de MODELS_DIR en disco; sin esta entrada,
        # los tests de tuning escribían best_params sintéticos en el
        # models/artifacts/ REAL y entraban en commits de producto (IMP-004).
        "ARTIFACTS_DIR": tmp_path / "models" / "artifacts",
        "PROCESSED_DATA_DIR": tmp_path / "data" / "processed",
        "RAW_DATA_DIR": tmp_path / "data" / "raw",
        "INTERIM_DATA_DIR": tmp_path / "data" / "interim",
        "VALIDATION_DIR": tmp_path / "reports" / "data_validation",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Ficheros derivados en inversion/utils/paths.py
    files = {
        "RAW_DATA_FILE": dirs["RAW_DATA_DIR"] / "data.csv",
        "SCALER_FILE": dirs["MODELS_DIR"] / "scaler.pkl",
        "MODEL_FILE": dirs["MODELS_DIR"] / "rf_model.pkl",
    }

    candidate_modules = [
        "inversion.utils.paths",
        "inversion.models.train_model",
        "inversion.models.predict_model",
        "inversion.features.build_features",
        "inversion.data.make_dataset",
    ]
    for mod_path in candidate_modules:
        try:
            mod = importlib.import_module(mod_path)
            for attr, val in {**dirs, **files}.items():
                if hasattr(mod, attr):
                    monkeypatch.setattr(mod, attr, val)
        except (ImportError, ModuleNotFoundError):
            pass

    return {**dirs, **files}
