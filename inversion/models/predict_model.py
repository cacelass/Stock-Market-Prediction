# src/models/predict_model.py
import joblib
import pandas as pd
from inversion.utils import paths
from inversion.features.build_features import calculate_technical_indicators

def predict_next_price(last_row_features, scaler_name=None, model_name=None):
    """
    Predice el precio usando datos ya procesados (para sanity check).
    Carga scaler y modelo guardados.
    last_row_features: DataFrame con una sola fila de features ya calculadas.
    Retorna la predicción del precio.
    """
    try:
        scaler = joblib.load(paths.SCALER_FILE)
        model = joblib.load(paths.MODEL_FILE)
        
        # Escalar
        features_scaled = scaler.transform(last_row_features)
        
        # Predecir
        prediction = model.predict(features_scaled)
        return prediction[0]
        
    except FileNotFoundError:
        print("Error: No se encontraron los archivos del modelo o scaler.")
        return None

def predict_future(df, feature_names):
    """
    Predice el precio de MAÑANA reconstruyendo features desde los datos crudos más recientes.
    Retorna la predicción del log return y el último precio de cierre real.
    """
    # 1. Recalcular indicadores en todo el dataset (incluyendo la última fila)
    df_full = calculate_technical_indicators(df)
    
    # 2. Extraer última fila (HOY)
    # Importante: Como calculate_technical_indicators devuelve el DF completo,
    # la última fila contiene los datos de HOY con sus indicadores calculados.
    future_X = df_full.iloc[[-1]][feature_names]
    
    # 3. Cargar artefactos y predecir
    try:
        scaler = joblib.load(paths.SCALER_FILE)
        model = joblib.load(paths.MODEL_FILE)
        
        future_X_scaled = scaler.transform(future_X)
        prediction = model.predict(future_X_scaled)[0]
        
        return prediction, df.iloc[-1]['close'] # Devuelve predicción y último cierre real
        
    except FileNotFoundError:
        print("Error: Modelos no entrenados.")
        return None, None