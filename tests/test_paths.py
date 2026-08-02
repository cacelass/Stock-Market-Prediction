"""
test_paths.py — Tests para inversion/utils/paths.py
Común a todos los ml_type.
"""

from pathlib import Path
from inversion.utils import paths


def test_all_path_constants_are_path_objects():
    """Todas las constantes de ruta deben ser instancias de Path."""
    expected = [
        "PROJECT_DIR",
        "DATA_DIR",
        "RAW_DATA_DIR",
        "PROCESSED_DATA_DIR",
        "MODELS_DIR",
        "RAW_DATA_FILE",
        "SCALER_FILE",
        "MODEL_FILE",
    ]
    for name in expected:
        assert hasattr(paths, name), f"Falta la constante: {name}"
        assert isinstance(getattr(paths, name), Path), f"{name} debe ser Path, no {type(getattr(paths, name))}"


def test_project_dir_points_to_root():
    """PROJECT_DIR debe ser la raíz del proyecto (contiene pyproject.toml)."""
    assert (paths.PROJECT_DIR / "pyproject.toml").exists()


def test_dirs_are_under_project():
    """Los directorios principales deben colgar de PROJECT_DIR."""
    assert paths.DATA_DIR.parent == paths.PROJECT_DIR


def test_raw_and_processed_share_parent_dir():
    """RAW_DATA_DIR y PROCESSED_DATA_DIR deben vivir en el mismo dir data/."""
    assert paths.RAW_DATA_DIR.parent == paths.PROCESSED_DATA_DIR.parent


def test_raw_data_file_lives_in_raw_data_dir():
    """RAW_DATA_FILE debe estar dentro de RAW_DATA_DIR."""
    assert paths.RAW_DATA_FILE.parent == paths.RAW_DATA_DIR


def test_model_files_live_in_models_dir():
    """SCALER_FILE y MODEL_FILE deben estar dentro de MODELS_DIR."""
    assert paths.SCALER_FILE.parent == paths.MODELS_DIR
    assert paths.MODEL_FILE.parent == paths.MODELS_DIR
