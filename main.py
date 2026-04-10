"""
Stock Market Prediction — Pipeline principal
============================================
Ejecuta el ciclo completo:
  1. Descarga de datos (yfinance)
  2. Ingeniería de features
  3. Split temporal + escalado
  4. Entrenamiento Random Forest (clasificación binaria)
  5. Evaluación con métricas de clasificación
  6. Predicción del próximo movimiento
  7. Visualizaciones
"""

import sys
import os

# Añadir el directorio raíz al path ANTES de cualquier import local
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from inversion.data.make_dataset import get_yfinance_history
from inversion.features.build_features import add_derived_features, fit_and_save_scaler
from inversion.models.train_model import train_rf_model, evaluate_model
from inversion.visualization.visualize import (
    plot_price,
    plot_predictions,
    plot_feature_importance,
    plot_returns_distribution,
)

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent / ".env")

TICKER            = "NVDA"
YEARS_HISTORY     = 20
PREDICTION_WINDOW = 5       # días hábiles hacia adelante para el target
UPSIDE_THRESHOLD  = 0.02    # +2% para clasificar como "sube"
TRAIN_RATIO       = 0.80
SCALER_NAME       = f"scaler_{TICKER}.pkl"
MODEL_NAME        = f"rf_{TICKER}.pkl"
DATA_RAW          = Path(__file__).parent / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

# Features usadas en entrenamiento y predicción (deben coincidir con build_features.py)
FEATURE_COLS = [
    "return", "volatility", "rsi",
    "ma_50", "ma_200",
    "hl_range", "oc_range", "log_volume",
    "vwap_ratio",
    "lag_1", "lag_5", "lag_20",
]

# ─── PASO 1: DESCARGA ─────────────────────────────────────────────────────────

def download_data(ticker: str, years: int) -> pd.DataFrame:
    """Descarga histórico de yfinance y garantiza columna 'timestamp'."""
    end_date   = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)
    print(f"[1/5] Descargando {ticker} ({start_date.date()} → {end_date.date()})...")

    df = get_yfinance_history(ticker, start_date, end_date)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"      ✔ {len(df)} filas descargadas.")
    return df

# ─── PASO 2: FEATURES + TARGET ───────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica ingeniería de variables (add_derived_features) y crea el target
    de clasificación binaria:
      1  →  el precio sube más de UPSIDE_THRESHOLD en PREDICTION_WINDOW días
      0  →  el precio no lo hace
    """
    print(f"[2/5] Construyendo features (ventana: {PREDICTION_WINDOW}d, umbral: {UPSIDE_THRESHOLD:.0%})...")

    df = add_derived_features(df)

    future_close = df["close"].shift(-PREDICTION_WINDOW)
    df["target"] = (future_close > df["close"] * (1 + UPSIDE_THRESHOLD)).astype(int)

    # Eliminar filas sin indicadores suficientes y sin target
    df.dropna(subset=FEATURE_COLS + ["target"], inplace=True)
    df = df.iloc[:-PREDICTION_WINDOW]  # últimas filas donde no hay target real

    pos_rate = df["target"].mean() * 100
    print(f"      ✔ {len(df)} filas limpias. Positivos (sube): {pos_rate:.1f}%")
    return df

# ─── PASO 3: SPLIT + ESCALADO ────────────────────────────────────────────────

def split_and_scale(df: pd.DataFrame):
    """Split temporal estricto (sin shuffle) + StandardScaler."""
    print("[3/5] Dividiendo y escalando...")

    X = df[FEATURE_COLS]
    y = df["target"]

    split_idx = int(len(X) * TRAIN_RATIO)
    X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train,     y_test     = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler  = fit_and_save_scaler(X_train_raw, filename=SCALER_NAME)
    X_train = scaler.transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    print(f"      ✔ Train: {len(X_train)} muestras | Test: {len(X_test)} muestras.")
    return X_train, X_test, y_train, y_test, scaler

# ─── PASO 4: ENTRENAMIENTO ───────────────────────────────────────────────────

def train(X_train, y_train):
    print("[4/5] Entrenando modelo...")
    return train_rf_model(X_train, y_train, filename=MODEL_NAME)

# ─── PASO 5: EVALUACIÓN ──────────────────────────────────────────────────────

def evaluate(model, X_test, y_test) -> np.ndarray:
    """Imprime métricas de clasificación y devuelve las predicciones."""
    print("[5/5] Evaluando modelo...\n")

    metrics = evaluate_model(model, X_test, y_test)
    naive   = max(y_test.mean(), 1 - y_test.mean())

    print(f"  Accuracy : {metrics['accuracy']:.2%}  (baseline: {naive:.2%})")
    print(f"  AUC-ROC  : {metrics['auc']:.4f}")
    print(f"\n{metrics['report']}")

    if metrics["accuracy"] > naive:
        print("  ✔ Supera el baseline de clase mayoritaria.")
    else:
        print("  ⚠ Sin ventaja sobre el baseline — el modelo necesita revisión.")

    return model.predict(X_test)

# ─── PREDICCIÓN FUTURA ───────────────────────────────────────────────────────

def predict_next(df_raw: pd.DataFrame, df_features: pd.DataFrame, model, scaler):
    """Predice la dirección del precio para los próximos PREDICTION_WINDOW días."""
    last = df_features.dropna(subset=FEATURE_COLS).iloc[[-1]][FEATURE_COLS]
    last_scaled = scaler.transform(last)

    pred_class  = model.predict(last_scaled)[0]
    pred_probas = model.predict_proba(last_scaled)[0]
    last_close  = df_raw.iloc[-1]["close"]

    direction  = "🟢 SUBE" if pred_class == 1 else "🔴 BAJA"
    confidence = pred_probas[pred_class] * 100

    print("=" * 42)
    print(f"  PRECIO CIERRE HOY    : ${last_close:.2f}")
    print(f"  SEÑAL ({PREDICTION_WINDOW}d)          : {direction}")
    print(f"  CONFIANZA            : {confidence:.1f}%")
    print(f"  Prob. sube: {pred_probas[1]:.1%}  |  Prob. baja: {pred_probas[0]:.1%}")
    print("=" * 42)
    print("\n⚠  Esto es un experimento académico, NO asesoramiento financiero.\n")

# ─── GUARDADO ────────────────────────────────────────────────────────────────

def save_data(df: pd.DataFrame, ticker: str):
    output = DATA_RAW / f"{ticker}_ml_ready.csv"
    df.to_csv(output, index=False)
    print(f"  ✔ Dataset guardado: {output}  ({df.shape[0]}f × {df.shape[1]}c)")

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'─' * 42}")
    print(f"  PIPELINE: {TICKER}")
    print(f"{'─' * 42}\n")

    df_raw  = download_data(TICKER, YEARS_HISTORY)
    df_feat = build_features(df_raw.copy())

    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_feat)
    model  = train(X_train, y_train)
    y_pred = evaluate(model, X_test, y_test)

    predict_next(df_raw, df_feat, model, scaler)
    save_data(df_feat, TICKER)

    plot_price(df_raw, ticker=TICKER)
    plot_returns_distribution(df_raw)
    plot_predictions(df_feat, y_test, y_pred)
    plot_feature_importance(model, FEATURE_COLS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Cancelado.")
    except Exception as e:
        print(f"\n Error: {e}")