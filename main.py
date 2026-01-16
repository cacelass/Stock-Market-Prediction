import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
import nltk
# Importaciones de tus módulos
from inversion.data.make_dataset import load_data, get_yfinance_history
from inversion.features.build_features import prepare_features, fit_and_save_scaler, add_derived_features
from inversion.models.train_model import train_rf_model, evaluate_model
# Nota: predict_next_price no lo usamos en este main simplificado, hacemos la lógica aquí
from inversion.utils import paths
from inversion.visualization.visualize import plot_price, plot_predictions, plot_feature_importance, plot_returns_distribution
# Descargar recursos de NLTK si no existen
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download("vader_lexicon")
# Añadir el directorio actual al path
sys.path.append(os.getcwd())

# -----------------------------
# CONFIGURACIÓN DEL PROYECTO
# -----------------------------
ROOT_DIR = Path.cwd()
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)


DATA_RAW = ROOT_DIR / "data/raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)
# --- CONFIGURACIÓN ---
TICKER = "NVDA" 
TARGET_COL = "close"
SCALER_NAME = f"scaler_{TICKER}.pkl"
MODEL_NAME = f"rf_{TICKER}.pkl"
# -----------------------------
   
def main():
    ticker = "NVDA"
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=20) # 20 AÑOS

    print(f"🚀 Iniciando proceso para {ticker} desde {start_date.date()} hasta {end_date.date()}...")

    # A. DESCARGAS
    df_yf = get_yfinance_history(ticker, start_date, end_date)
    # B. MERGE (FUSIÓN)
    # Usamos YFinance como esqueleto base (fechas completas)
    df_final = df_yf.copy()

    # 1. Unir Polygon (Solo nos interesa VWAP y Num_Trades si existen, el precio OHLC ya lo tenemos de YF)
    df_final["vwap"] = df_final["close"] # Fallback
    df_final["num_trades"] = 0

    # 2. Unir Earnings
    df_final["surprisePercent"] = np.nan

    # C. TRATAMIENTO Y LIMPIEZA DE DATOS (Data Cleaning)
    print(" Tratando y limpiando datos...")

    # 1. Relleno de Earnings (Forward Fill)
    # Las ganancias se reportan cada 3 meses. El impacto dura hasta el siguiente reporte.
    df_final["surprisePercent"] = df_final["surprisePercent"].ffill().fillna(0)

    # 2. Relleno de datos técnicos faltantes (VWAP/Trades de Polygon pueden tener huecos)
    # Si falta VWAP, usamos Close. Si faltan trades, ponemos 0 o media.
    df_final["vwap"] = df_final["vwap"].fillna(df_final["close"])
    df_final["num_trades"] = df_final["num_trades"].fillna(0)

    # 3. Calcular Indicadores (Features)
    # Hacemos esto DESPUÉS de asegurar que no hay huecos en OHLC
    df_final = add_derived_features(df_final)

    # 4. Limpieza de Nulos generados por indicadores (ej. MA_200 genera 200 NaNs al inicio)
    # Eliminamos las filas que no tengan suficientes datos históricos para calcular los indicadores
    df_final.dropna(subset=["ma_200", "rsi"], inplace=True)

    # 5. Creación del TARGET (Objetivo para ML)
    # Target: 1 si el precio sube > 2% en los próximos 5 días
    prediction_window = 5
    df_final["target_price"] = df_final["close"].shift(-prediction_window)
    df_final["target"] = (df_final["target_price"] > df_final["close"] * 1.02).astype(int)

    # Eliminamos las ultimas filas donde no hay target (porque hicimos shift negativo)
    df_final = df_final.iloc[:-prediction_window]

    # D. SELECCIÓN FINAL DE COLUMNAS
    cols_finales = [
        "timestamp", "open", "high", "low", "close", "volume", 
        "vwap", "num_trades",                   # De Polygon
        "return", "volatility", "rsi", "ma_50", "ma_200", # Técnicos
        "hl_range", "oc_range", "log_volume", 
        "surprisePercent",                      # Fundamental
        "target"                                # Target
    ]

    # Filtrar solo columnas existentes
    cols_existentes = [c for c in cols_finales if c in df_final.columns]
    df_final = df_final[cols_existentes]

    # E. GUARDADO
    output_file = DATA_RAW / f"{ticker}_ml_ready.csv"
    df_final.to_csv(output_file, index=False)

    print(f"✔ Proceso completado.")
    print(f"✔ Datos guardados en: {output_file}")
    print(f"✔ Dimensiones finales: {df_final.shape}")
    print(df_final.tail())

    # -----------------------------
    # SAVE
    # -----------------------------
    output_file = DATA_RAW / f"data.csv"
    df_final.to_csv(output_file, index=False)

    print(f"✔ Archivo listo: {output_file}")
    print(df_final.tail())

    print(f"\n --- INICIANDO PIPELINE PARA {TICKER} ---")

    # 1. CARGA DE DATOS
    print("\n[1/5]  Cargando datos...")
    df = load_data(TICKER)
    
    if df is None or df.empty:
        print("❌ Error: No se pudieron cargar los datos.")
        return
    print(f"      Datos cargados: {df.shape[0]} filas.")

    # Asegurar timestamp
    if "timestamp" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "timestamp"})
        else:
            # Si el índice era la fecha
            df = df.reset_index()
            if "index" in df.columns:
                df = df.rename(columns={"index": "timestamp"})

    # Convertir a datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    # 2. INGENIERÍA DE CARACTERÍSTICAS
    print("\n[2/5]   Preparando features...")
    # prepare_features devuelve features y target (log return)
    X, y, feature_names = prepare_features(df, target_col=TARGET_COL)
    print(f"      Features usadas: {len(feature_names)}")

    # 3. SPLIT Y ESCALADO
    print("\n[3/5]   Dividiendo y escalando datos...")
    split_idx = int(len(X) * 0.8)
    
    X_train_raw = X.iloc[:split_idx]
    X_test_raw = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    scaler = fit_and_save_scaler(X_train_raw, filename=SCALER_NAME)
    
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 4. ENTRENAMIENTO
    print("\n[4/5]  Entrenando modelo Random Forest...")
    rf_model = train_rf_model(X_train_scaled, y_train, filename=MODEL_NAME)
    
    mse, r2 = evaluate_model(rf_model, X_test_scaled, y_test)
    print(f"       Resultados Test -> MSE: {mse:.5f}, R²: {r2:.4f}")

    # 5. PREDICCIÓN FUTURA
    print("\n[5/5]  Generando predicción para mañana...")
    
    # Reconstruimos features para la última fila (HOY)
    df_predict = df.copy()
    
    # Calcular indicadores técnicos manualmente para asegurar que tenemos la última fila
    df_predict['return'] = df_predict['close'].pct_change()
    df_predict['log_volume'] = np.log1p(df_predict['volume'])
    df_predict['hl_range'] = (df_predict['high'] - df_predict['low']) / df_predict['close']
    df_predict['oc_range'] = (df_predict['open'] - df_predict['close']) / df_predict['close']
    df_predict['ma_5'] = df_predict['close'].rolling(5).mean()
    df_predict['ma_10'] = df_predict['close'].rolling(10).mean()
    df_predict['volatility'] = df_predict['return'].rolling(20).std()
    
    if 'vwap' not in df_predict.columns:
        df_predict['vwap'] = (df_predict['high'] + df_predict['low'] + df_predict['close']) / 3

    delta = df_predict['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df_predict['rsi'] = 100 - (100 / (1 + rs))

    # Seleccionar la última fila disponible (Datos de HOY)
    last_day_features = df_predict.iloc[[-1]][feature_names]
    
    # Escalar
    last_day_scaled = scaler.transform(last_day_features)
    
    # Predecir (El modelo devuelve Log Return)
    pred_log_return = rf_model.predict(last_day_scaled)[0]
    
    # Convertir a Precio
    last_close = df.iloc[-1]['close']
    predicted_price = last_close * np.exp(pred_log_return)
    pct_change = (np.exp(pred_log_return) - 1) * 100
    
    direction = "🟢 SUBE" if pred_log_return > 0 else "🔴 BAJA"

    print("\n" + "="*40)
    print(f" PRECIO CIERRE HOY:    ${last_close:.2f}")
    print(f" RETORNO PREDICHO:     {pct_change:+.2f}%")
    print(f" PRECIO ESTIMADO:      ${predicted_price:.2f}")
    print(f" SEÑAL:                {direction}")
    print("="*40 + "\n")

    # --- VISUALIZACIONES ---
    # Grafica precios y medias móviles
    plot_price(df, ticker=TICKER)

    # Grafica distribución de retornos
    plot_returns_distribution(df)

    # Si tienes predicciones de test
    y_pred = rf_model.predict(X_test_scaled)
    plot_predictions(df, y_test, y_pred)

    # Importancia de features del modelo Random Forest
    plot_feature_importance(rf_model, feature_names)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Cancelado.")
    except Exception as e:
        print(f"\n❌ Error: {e}")