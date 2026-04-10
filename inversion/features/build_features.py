import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from inversion.utils import paths


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los indicadores técnicos y lags sobre el DataFrame.

    Columnas generadas:
      - return       : retorno diario (pct_change)
      - volatility   : desv. estándar móvil 20d del retorno
      - rsi          : RSI 14 días
      - hl_range     : (high - low) / close  [normalizado]
      - oc_range     : (open - close) / close [normalizado]
      - ma_50        : media móvil 50 días
      - ma_200       : media móvil 200 días
      - log_volume   : log(1 + volume)
      - vwap_ratio   : close / vwap  (si no hay vwap real, usa (H+L+C)/3)
      - lag_1 / lag_5 / lag_20 : lags del retorno diario
    """
    df = df.sort_values("timestamp").copy()

    # Retorno diario
    df["return"] = df["close"].pct_change()

    # Volatilidad 20 días
    df["volatility"] = df["return"].rolling(window=20).std()

    # RSI 14 días (con protección contra división por cero)
    delta = df["close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Rangos normalizados por precio de cierre
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_range"] = (df["open"] - df["close"]) / df["close"]

    # Medias móviles
    df["ma_50"]  = df["close"].rolling(50).mean()
    df["ma_200"] = df["close"].rolling(200).mean()

    # Log volume
    df["log_volume"] = np.log1p(df["volume"])

    # VWAP ratio
    if "vwap" not in df.columns:
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap_ratio"] = df["close"] / df["vwap"]

    # Lags del retorno (semana, mes, trimestre aprox.)
    df["lag_1"]  = df["return"].shift(1)
    df["lag_5"]  = df["return"].shift(5)
    df["lag_20"] = df["return"].shift(20)

    return df


def fit_and_save_scaler(X, filename=None) -> StandardScaler:
    """Entrena y persiste un StandardScaler. Devuelve el scaler ajustado."""
    scaler = StandardScaler()
    scaler.fit(X)

    save_path = paths.SCALER_FILE if filename is None else paths.MODELS_DIR / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, save_path)

    return scaler