import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from inversion.utils import paths

def add_derived_features(df):
    """Calcula indicadores técnicos sobre el precio de cierre principal.
    1. Retornos diarios
    2. Volatilidad (Rolling Std Dev)
    3. RSI
    4. Rangos (High-Low, Open-Close)
    5. Medias Móviles (50 y 200 días)
    6. Log Volume
    """
    df = df.sort_values("timestamp").copy()
    
    # Usamos 'close' (que vendrá de YFinance)
    df["return"] = df["close"].pct_change()
    
    # Volatilidad (Rolling 20 días)
    df["volatility"] = df["return"].rolling(window=20).std()
    
    # RSI (14 días)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # Rangos
    df["hl_range"] = df["high"] - df["low"]
    df["oc_range"] = df["open"] - df["close"]
    
    # Medias Móviles (Simple)
    df["ma_50"] = df["close"].rolling(50).mean()
    df["ma_200"] = df["close"].rolling(200).mean() # Importante para largo plazo
    
    # Log Volume
    df["log_volume"] = np.log1p(df["volume"])
    
    return df

def calculate_technical_indicators(df):
    """Calcula los indicadores técnicos.
    Retorna el DataFrame con nuevas columnas de indicadores.
    """
    df = df.copy()
    
    # Evitar divisiones por cero o nulos al inicio
    df['return'] = df['close'].pct_change().fillna(0)
    df['log_volume'] = np.log1p(df['volume'])
    
    # Rangos
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['oc_range'] = (df['open'] - df['close']) / df['close']
    
    # Medias Móviles
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_10'] = df['close'].rolling(10).mean()
    
    # Volatilidad
    df['volatility'] = df['return'].rolling(20).std()

    # VWAP (aproximado)
    if 'vwap' not in df.columns:
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    
    # Evitar división por cero en RSI
    loss = loss.replace(0, np.nan) 
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50) # Rellenar nulos iniciales de RSI
    
    return df

def prepare_features(df, target_col):
    """
    Prepara features (X) y target (y).
    IMPORTANTE: El target es el Log Return del día siguiente.
    Retorna X, y, feature_names
    """
    # 1. Calcular indicadores
    df = calculate_technical_indicators(df)
    
    feature_names = [
        'return', 'log_volume', 'hl_range', 'oc_range', 
        'ma_5', 'ma_10', 'volatility', 'vwap', 'rsi'
    ]
    
    # 2. Crear Target: Log Return de Mañana
    # FÓRMULA CLAVE: ln(Precio_Mañana / Precio_Hoy)
    # Si esto no está así, el modelo predecirá precios gigantes
    df['target'] = np.log(df[target_col].shift(-1) / df[target_col])
    
    # 3. Limpieza final
    # Eliminamos filas donde falten indicadores o el target (última fila)
    df_clean = df.dropna()
    
    # Comprobación de seguridad (Sanity check)
    # Si el target medio es mayor a 1, algo anda mal (estamos prediciendo precios, no retornos)
    if df_clean['target'].mean() > 1:
        print(" ADVERTENCIA CRÍTICA: El target parece ser un PRECIO, no un retorno. Revisa build_features.py")
    
    X = df_clean[feature_names]
    y = df_clean['target']
    
    return X, y, feature_names

def fit_and_save_scaler(X, filename=None):
    """Entrena y guarda el escalador.
    Retorna el objeto scaler entrenado.
    """
    scaler = StandardScaler()
    scaler.fit(X)
    
    save_path = paths.SCALER_FILE if filename is None else paths.MODELS_DIR / filename
    
    # Asegurar que el directorio existe antes de guardar
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(scaler, save_path)
    return scaler