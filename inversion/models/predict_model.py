import joblib
import pandas as pd
from inversion.utils import paths
from inversion.features.build_features import add_derived_features

# Features que deben estar presentes (deben coincidir con main.py)
FEATURE_COLS = [
    "return", "volatility", "rsi",
    "ma_50", "ma_200",
    "hl_range", "oc_range", "log_volume",
    "vwap_ratio",
    "lag_1", "lag_5", "lag_20",
]


def predict_future(df: pd.DataFrame) -> tuple[int, float, float, float]:
    """
    Predice la dirección del precio para los próximos días a partir
    de los datos crudos más recientes.

    Args:
        df : DataFrame con columnas OHLCV y 'timestamp'.

    Returns:
        (clase, prob_baja, prob_sube, ultimo_close)
        clase     : 0 (baja) o 1 (sube)
        prob_baja : probabilidad clase 0
        prob_sube : probabilidad clase 1
        ultimo_close : último precio de cierre conocido
    """
    try:
        scaler = joblib.load(paths.SCALER_FILE)
        model  = joblib.load(paths.MODEL_FILE)
    except FileNotFoundError:
        print("Error: Modelo o scaler no encontrados. Ejecuta main.py primero.")
        return None, None, None, None

    # Calcular features sobre todo el dataset
    df_feat = add_derived_features(df.copy())

    # Coger la última fila que no tenga NaN en las features
    last = df_feat.dropna(subset=FEATURE_COLS).iloc[[-1]][FEATURE_COLS]

    last_scaled  = scaler.transform(last)
    pred_class   = model.predict(last_scaled)[0]
    pred_probas  = model.predict_proba(last_scaled)[0]
    ultimo_close = df.iloc[-1]["close"]

    return int(pred_class), float(pred_probas[0]), float(pred_probas[1]), float(ultimo_close)