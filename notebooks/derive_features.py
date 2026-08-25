#!/usr/bin/env python3
"""
Derivación de nuevas variables técnicas y de calendario.
Stock Market Prediction — Features derivadas desde la base.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Configuración ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]


# ── Funciones de derivación de features ────────────────────────────────────
def add_bollinger_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bandas de Bollinger y衍生adas."""
    if "close" not in df.columns:
        return df

    # Media móvil de 20 días
    df["ma_20"] = df["close"].rolling(20).mean()

    # Desviación estándar de 20 días
    df["std_20"] = df["close"].rolling(20).std()

    # Bandas de Bollinger
    df["bollinger_upper"] = df["ma_20"] + 2 * df["std_20"]
    df["bollinger_lower"] = df["ma_20"] - 2 * df["std_20"]

    # Ancho de bandas (volatilidad relativa)
    df["bollinger_width"] = (df["bollinger_upper"] - df["bollinger_lower"]) / df["ma_20"]

    # Posición en la banda (0 = lower, 1 = upper)
    band_range = df["bollinger_upper"] - df["bollinger_lower"]
    df["bollinger_position"] = (df["close"] - df["bollinger_lower"]) / band_range.replace(0, np.nan)

    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range — volatilidad intradia normalizada."""
    if not all(c in df.columns for c in ["high", "low", "close"]):
        return df

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"atr_{period}"] = true_range.rolling(period).mean()

    # ATR como % del precio
    df[f"atr_{period}_pct"] = df[f"atr_{period}"] / df["close"]

    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume — presión compradora/vendedora."""
    if not all(c in df.columns for c in ["close", "volume"]):
        return df

    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])

    df["obv"] = obv
    df["obv_ma20"] = df["obv"].rolling(20).mean()
    df["obv_signal"] = np.sign(df["obv"] - df["obv_ma20"])

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de volumen."""
    if "volume" not in df.columns:
        return df

    # Volumen relativo a media móvil
    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio_20"] = df["volume"] / df["volume_ma_20"].replace(0, np.nan)

    # Volumen relativo a 60 días
    df["volume_ma_60"] = df["volume"].rolling(60).mean()
    df["volume_ratio_60"] = df["volume"] / df["volume_ma_60"].replace(0, np.nan)

    # Cambio de volumen
    df["volume_change"] = df["volume"].pct_change()

    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de momentum avanzadas."""
    if "return" not in df.columns:
        return df

    # Aceleración del RSI
    if "rsi" in df.columns:
        df["rsi_delta"] = df["rsi"] - df["rsi"].shift(1)
        df["rsi_ma5"] = df["rsi"].rolling(5).mean()
        df["rsi_signal"] = np.where(df["rsi"] > df["rsi_ma5"], 1, -1)

    # Distancia a medias móviles
    if "close" in df.columns and "ma_50" in df.columns:
        df["close_to_ma50_ratio"] = df["close"] / df["ma_50"]

    if "close" in df.columns and "ma_200" in df.columns:
        df["close_to_ma200_ratio"] = df["close"] / df["ma_200"]

    # Señal Golden/Death Cross
    if "ma_50" in df.columns and "ma_200" in df.columns:
        df["ma_cross_signal"] = np.sign(df["ma_50"] - df["ma_200"])

    # Posición en rango de 20 días
    if "high" in df.columns and "low" in df.columns:
        df["high_20d"] = df["high"].rolling(20).max()
        df["low_20d"] = df["low"].rolling(20).min()
        range_20d = df["high_20d"] - df["low_20d"]
        df["price_position_in_range"] = (df["close"] - df["low_20d"]) / range_20d.replace(0, np.nan)

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de volatilidad avanzadas."""
    if "return" not in df.columns:
        return df

    # Volatilidad realized de diferentes ventanas
    for window in [5, 10, 21, 63]:
        df[f"realized_vol_{window}"] = df["return"].rolling(window).std() * np.sqrt(252)

    # Ratio de volatilidades
    if "realized_vol_5" in df.columns and "realized_vol_21" in df.columns:
        df["vol_ratio_5_21"] = df["realized_vol_5"] / df["realized_vol_21"].replace(0, np.nan)

    # Retorno / volatilidad (Sharpe instantáneo)
    if "realized_vol_21" in df.columns:
        df["return_vol_ratio"] = df["return"] / df["realized_vol_21"].replace(0, np.nan)

    # High-Low ratio de 5 días
    if "high" in df.columns and "low" in df.columns:
        df["high_low_ratio_5d"] = df["high"].rolling(5).max() / df["low"].rolling(5).min()

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de calendario y estacionalidad."""
    if "timestamp" not in df.columns:
        return df

    ts = df["timestamp"]

    # Mes y semana del año
    df["month"] = ts.dt.month
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)

    # Efectos de calendario
    df["is_month_end"] = ts.dt.is_month_end.astype(int)
    df["is_month_start"] = ts.dt.is_month_start.astype(int)
    df["is_quarter_end"] = ts.dt.is_quarter_end.astype(int)
    df["is_quarter_start"] = ts.dt.is_quarter_start.astype(int)

    # Efecto Sell in May
    df["sell_in_may"] = df["month"].isin([5, 6, 7, 8, 9]).astype(int)

    # Earnings season (meses con reporting)
    df["is_earnings_season"] = df["month"].isin([1, 4, 7, 10]).astype(int)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de interacción entre variables."""
    if "return" not in df.columns:
        return df

    # Retorno × volumen (retornos con convicción)
    if "log_volume" in df.columns:
        df["return_x_volume"] = df["return"] * df["log_volume"]

    # RSI × volatilidad (sobrecomprado + volátil = reversión)
    if "rsi" in df.columns and "volatility" in df.columns:
        df["rsi_x_volatility"] = df["rsi"] * df["volatility"]

    # Momentum × volatilidad
    if "momentum_21" in df.columns and "volatility_21" in df.columns:
        df["momentum_x_volatility"] = df["momentum_21"] * df["volatility_21"]

    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de régimen de mercado."""
    if "close" not in df.columns:
        return df

    # Régimen de tendencia
    if "ma_200" in df.columns:
        df["trend_regime"] = np.where(df["close"] > df["ma_200"], 1, 0)

    # Régimen de volatilidad
    if "volatility" in df.columns:
        vol_median = df["volatility"].rolling(252).median()
        df["volatility_regime"] = np.where(df["volatility"] > vol_median, 1, 0)

    # Drawdown actual
    df["peak_252d"] = df["close"].rolling(252).max()
    df["drawdown_current"] = (df["close"] - df["peak_252d"]) / df["peak_252d"]

    # Ratio de recuperación
    df["max_drawdown_252d"] = df["drawdown_current"].rolling(252).min()
    df["recovery_ratio"] = df["drawdown_current"] / df["max_drawdown_252d"].replace(0, np.nan)

    return df


# ── FUNCIÓN PRINCIPAL ─────────────────────────────────────────────────────
def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye todas las features derivadas."""
    print("  Añadiendo Bandas de Bollinger...")
    df = add_bollinger_features(df)

    print("  Añadiendo ATR...")
    df = add_atr(df)

    print("  Añadiendo OBV...")
    df = add_obv(df)

    print("  Añadiendo features de volumen...")
    df = add_volume_features(df)

    print("  Añadiendo features de momentum...")
    df = add_momentum_features(df)

    print("  Añadiendo features de volatilidad...")
    df = add_volatility_features(df)

    print("  Añadiendo features de calendario...")
    df = add_calendar_features(df)

    print("  Añadiendo features de interacción...")
    df = add_interaction_features(df)

    print("  Añadiendo features de régimen...")
    df = add_regime_features(df)

    return df


# ── EJECUCIÓN ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("  DERIVACIÓN DE FEATURES — Stock Market Prediction")
    print("=" * 80)

    for ticker in TICKERS:
        print(f"\n{'─'*60}")
        print(f"  Procesando {ticker}...")
        print(f"{'─'*60}")

        # Cargar datos raw
        raw_path = DATA_RAW / f"{ticker}_ml_ready.csv"
        if not raw_path.exists():
            print(f"  ✗ No se encontró {raw_path}")
            continue

        df = pd.read_csv(raw_path, parse_dates=["timestamp"])
        print(f"  Registros iniciales: {len(df)}")
        print(f"  Columnas iniciales: {len(df.columns)}")

        # Derivar features
        df = build_derived_features(df)

        print(f"\n  Registros finales: {len(df)}")
        print(f"  Columnas finales: {len(df.columns)}")

        # Guardar en interim
        out_path = DATA_INTERIM / f"features_{ticker}_ml_ready.csv"
        df.to_csv(out_path, index=False)
        print(f"  ✓ Guardado en {out_path}")

    # ── RESUMEN ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  RESUMEN DE FEATURES DERIVADAS")
    print("=" * 80)

    # Cargar un ticker para ver las columnas
    sample_path = DATA_INTERIM / f"features_{TICKERS[0]}_ml_ready.csv"
    if sample_path.exists():
        df_sample = pd.read_csv(sample_path)

        # Categorizar features
        categories = {
            "Precio": ["open", "high", "low", "close", "volume"],
            "Técnicas originales": ["return", "volatility", "rsi", "hl_range", "oc_range",
                                   "ma_50", "ma_200", "log_volume", "vwap", "vwap_ratio",
                                   "lag_1", "lag_5", "lag_20"],
            "Derivadas: Bollinger": ["ma_20", "std_20", "bollinger_upper", "bollinger_lower",
                                    "bollinger_width", "bollinger_position"],
            "Derivadas: ATR": ["atr_14", "atr_14_pct"],
            "Derivadas: OBV": ["obv", "obv_ma20", "obv_signal"],
            "Derivadas: Volumen": ["volume_ma_20", "volume_ratio_20", "volume_ma_60",
                                  "volume_ratio_60", "volume_change"],
            "Derivadas: Momentum": ["rsi_delta", "rsi_ma5", "rsi_signal",
                                   "close_to_ma50_ratio", "close_to_ma200_ratio",
                                   "ma_cross_signal", "high_20d", "low_20d",
                                   "price_position_in_range"],
            "Derivadas: Volatilidad": ["realized_vol_5", "realized_vol_10", "realized_vol_21",
                                      "realized_vol_63", "vol_ratio_5_21", "return_vol_ratio",
                                      "high_low_ratio_5d"],
            "Derivadas: Calendario": ["month", "week_of_year", "is_month_end", "is_month_start",
                                     "is_quarter_end", "is_quarter_start", "sell_in_may",
                                     "is_earnings_season"],
            "Derivadas: Interacción": ["return_x_volume", "rsi_x_volatility", "momentum_x_volatility"],
            "Derivadas: Régimen": ["trend_regime", "volatility_regime", "drawdown_current",
                                  "peak_252d", "max_drawdown_252d", "recovery_ratio"],
            "Sentimiento": ["sentiment_score", "sentiment_noticias", "sentiment_ma5", "sentiment_vol"],
        }

        total_derived = 0
        for cat, cols in categories.items():
            existing = [c for c in cols if c in df_sample.columns]
            n_new = len(existing)
            total_derived += n_new
            print(f"  {cat:30s}: {n_new:3d} features")

        print(f"\n  {'TOTAL FEATURES':30s}: {len(df_sample.columns):3d}")
        print(f"  {'Total derivadas nuevas':30s}: {total_derived:3d}")

        # Listar features nuevas
        new_features = [c for c in df_sample.columns
                       if c not in ["timestamp", "close", "high", "low", "open", "volume",
                                   "return", "volatility", "rsi", "hl_range", "oc_range",
                                   "ma_50", "ma_200", "log_volume", "vwap", "vwap_ratio",
                                   "lag_1", "lag_5", "lag_20", "target",
                                   "volatility_21", "momentum_10", "momentum_21", "momentum_63",
                                   "day_of_week", "quarter",
                                   "sentiment_score", "sentiment_noticias", "sentiment_ma5", "sentiment_vol"]]

        print(f"\n  Features nuevas añadidas ({len(new_features)}):")
        for i, feat in enumerate(sorted(new_features), 1):
            print(f"    {i:2d}. {feat}")

    print("\n✅ Derivación de features completada.")
    print("   Datos guardados en data/interim/")
    print("   Ejecuta: uv run python notebooks/eda_complete.py para visualizar")
