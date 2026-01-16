from pathlib import Path
import os

# 1. Definir la raíz del proyecto
# __file__ es .../inversion/utils/paths.py
# parents[0] = utils
# parents[1] = inversion (paquete)
# parents[2] = inversion (carpeta raíz del proyecto)
PROJECT_DIR = Path(__file__).resolve().parents[2]

# 2. Definir directorios principales
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_DIR / "models"

# 3. Definir archivos específicos
# Eliminamos el "import config" y definimos el nombre por defecto aquí
RAW_DATA_FILE = RAW_DATA_DIR / "data.csv" 

# Nombres genéricos para los modelos (se pueden sobrescribir en el main si es necesario)
SCALER_FILE = MODELS_DIR / "scaler.pkl"
MODEL_FILE = MODELS_DIR / "rf_model.pkl"

# 4. Asegurar que existan las carpetas
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)