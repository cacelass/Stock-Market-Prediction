from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from inversion.utils import paths
from inversion.features.macro_features import add_macro_features, MACRO_COLUMNS
from inversion.features.regime_detection import detect_regimes, Regime
from inversion.features.advanced_features import add_advanced_features, ADVANCED_COLUMNS


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los indicadores técnicos y lags sobre el DataFrame.

    Columnas generadas:
      - return       : retorno diario (pct_change)
      - volatility   : desv. estándar móvil 20d del retorno
      - volatility_21: desv. estándar móvil 21d del retorno (TRADE-005)
      - rsi          : RSI 14 días (aproximación simple, ver nota)
      - momentum_10 / momentum_21 / momentum_63 : retorno a 10/21/63 días (TRADE-005)
      - day_of_week  : día de la semana (0=lunes … 4=viernes) (TRADE-005)
      - quarter      : trimestre del año (1-4) (TRADE-005)
      - hl_range     : (high - low) / close  [normalizado]
      - oc_range     : (open - close) / close [normalizado]
      - ma_50        : media móvil 50 días
      - ma_200       : media móvil 200 días
      - log_volume   : log(1 + volume)
      - vwap_ratio   : close / vwap  (si no hay vwap real, usa (H+L+C)/3)
      - lag_1 / lag_5 / lag_20 : lags del retorno diario

    Nota RSI: se usa la aproximación simple (media móvil de ganancias y
    pérdidas a 14 días), no la suavización de Wilder; es la versión más común
    en librerías ligeras y evita el bucle recursivo de Wilder.

    El cálculo es determinista: mismo input → mismo output. Las features son
    derivadas por construcción de los precios (close~vwap, ma_50~close), por
    lo que hay colinealidad esperada; el modelo RandomForest es tolerante a
    ella y no se intenta "arreglar" aquí.

    Sin fuga: momentum, volatilidad y RSI usan solo ventanas pasadas
    (pct_change/rolling miran hacia atrás) y las features de calendario se
    derivan de la fecha del propio día t, nunca de precios futuros.
    """
    df = df.sort_values("timestamp").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Retorno diario
    df["return"] = df["close"].pct_change()

    # Volatilidad 20 días (original) y 21 días (TRADE-005)
    df["volatility"] = df["return"].rolling(window=20).std()
    df["volatility_21"] = df["return"].rolling(window=21).std()

    # RSI 14 días (aproximación simple: media de ganancias/pérdidas, no Wilder)
    # con protección contra división por cero
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Rangos normalizados por precio de cierre
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_range"] = (df["open"] - df["close"]) / df["close"]

    # Medias móviles
    df["ma_50"] = df["close"].rolling(50).mean()
    df["ma_200"] = df["close"].rolling(200).mean()

    # Log volume
    df["log_volume"] = np.log1p(df["volume"])

    # VWAP ratio
    if "vwap" not in df.columns:
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap_ratio"] = df["close"] / df["vwap"]

    # Lags del retorno (semana, mes, trimestre aprox.)
    df["lag_1"] = df["return"].shift(1)
    df["lag_5"] = df["return"].shift(5)
    df["lag_20"] = df["return"].shift(20)

    # Momentum (TRADE-005): retorno acumulado a 10/21/63 días. pct_change(n)
    # usa SOLO precios hasta t (close[t] vs close[t-n]), nunca t+k.
    df["momentum_10"] = df["close"].pct_change(10)
    df["momentum_21"] = df["close"].pct_change(21)
    df["momentum_63"] = df["close"].pct_change(63)

    # Estacionalidad (TRADE-005): día de la semana (0=lunes..4=viernes) y
    # trimestre (1-4), derivados de la fecha del día t.
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["quarter"] = df["timestamp"].dt.quarter

    return df


def fit_and_save_scaler(X: np.ndarray | pd.DataFrame, filename: str | None = None) -> StandardScaler:
    """Entrena y persiste un StandardScaler. Devuelve el scaler ajustado.

    El escalado se ajusta SOLO con los datos de train: quien llama a esta
    función pasa únicamente X_train (ver train_model.py). Nunca se ajusta con
    el conjunto de test, para no filtrar información futura.
    """
    scaler = StandardScaler()
    scaler.fit(X)

    save_path = paths.SCALER_FILE if filename is None else paths.MODELS_DIR / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, save_path)

    return scaler


def add_sentiment_features(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Fusiona el sentimiento diario (VADER + LM) con los precios, sin fuga.

    Lee data/raw/news_<TICKER>.csv (si existe), puntúa cada noticia con el
    analyzer y agrega por día. El merge es por fecha exacta: la fila del día t
    usa SOLO noticias del día t o anteriores — nunca t+k. Las medias móviles
    del sentimiento miran hacia atrás (rolling), así que ninguna feature usa
    el futuro. Si no hay noticias para el ticker, devuelve el DataFrame sin
    columnas de sentimiento.

    Genera 15 columnas de sentimiento:
      1.  sentiment_score       — daily compound (VADER)
      2.  sentiment_ma5         — 5-day moving average
      3.  sentiment_vol         — 7-day rolling std
      4.  sentiment_ma20        — 20-day moving average
      5.  sentiment_ma60        — 60-day moving average
      6.  sentiment_zscore      — z-score of daily score (w.r.t. last 20d mean/std)
      7.  sentiment_trend       — slope of 10-day linear trend
      8.  sentiment_rsi         — RSI(14) of sentiment_score
      9.  sentiment_vol_ratio   — sentiment_vol / sentiment_ma20 (normalized vol)
      10. sentiment_crown       — convolution of sentiment with price return
      11. news_volume           — count of articles per day
      12. news_volume_ma5       — 5-day average news count
      13. sentiment_extreme     — count of trailing days with |score| > 0.5
      14. sentiment_change      — daily delta of sentiment_score
      15. sentiment_change_ma5  — 5-day moving average of sentiment_change
    """
    ticker_upper = ticker.upper()
    primary = [
        paths.RAW_DATA_DIR / f"news_{ticker_upper}_live.csv",
        paths.RAW_DATA_DIR / f"news_{ticker_upper}_gdelt.csv",
    ]
    fallback = paths.RAW_DATA_DIR / f"news_{ticker_upper}.csv"
    existing = [f for f in primary if f.exists()]
    if not existing and fallback.exists():
        existing = [fallback]
    if not existing:
        return df

    from inversion.sentiment.analyzer import daily_sentiment, score_news_df

    frames = [pd.read_csv(f) for f in existing]
    all_news = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["titulo"])
    scored = score_news_df(all_news)
    daily = daily_sentiment(scored, ticker=ticker_upper)[["fecha", "compound", "noticias"]]
    daily["fecha"] = pd.to_datetime(daily["fecha"]).dt.date

    out = df.copy()
    out["_fecha"] = pd.to_datetime(out["timestamp"]).dt.date
    out = out.merge(daily, left_on="_fecha", right_on="fecha", how="left")

    out["sentiment_score"] = out["compound"].fillna(0.0)
    out["sentiment_noticias"] = out["noticias"].fillna(0).astype(int)

    # --- Enhanced sentiment features (rolling, all backward-looking) ---

    # 2. 5-day MA
    out["sentiment_ma5"] = out["sentiment_score"].rolling(5, min_periods=1).mean()
    # 3. 7-day rolling std
    out["sentiment_vol"] = out["sentiment_score"].rolling(7, min_periods=1).std().fillna(0.0)
    # 4. 20-day MA
    out["sentiment_ma20"] = out["sentiment_score"].rolling(20, min_periods=1).mean()
    # 5. 60-day MA
    out["sentiment_ma60"] = out["sentiment_score"].rolling(60, min_periods=1).mean()

    # 6. Z-score w.r.t. trailing 20-day window
    _s = out["sentiment_score"]
    _m20 = _s.rolling(20, min_periods=1).mean()
    _s20 = _s.rolling(20, min_periods=1).std().fillna(0.0)
    out["sentiment_zscore"] = ((_s - _m20) / (_s20 + 1e-9)).fillna(0.0)

    # 7. Trend: slope of 10-day linear regression (x = range(10))
    def _trend_10(s: np.ndarray | Any) -> float:
        if len(s) < 10:
            return 0.0
        # s may be ndarray (raw=True) or Series — handle both
        arr = s.values if hasattr(s, 'values') else np.asarray(s)
        if len(arr) < 10:
            return 0.0
        x = np.arange(10, dtype=float)
        y = arr[-10:].astype(float)
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    out["sentiment_trend"] = out["sentiment_score"].expanding().apply(_trend_10, raw=True).fillna(0.0)

    # 8. RSI(14) of sentiment_score
    _ds = out["sentiment_score"].diff()
    _gain = _ds.where(_ds > 0, 0).rolling(14, min_periods=1).mean()
    _loss = (-_ds.where(_ds < 0, 0)).rolling(14, min_periods=1).mean()
    out["sentiment_rsi"] = (100 - (100 / (1 + _gain / (_loss + 1e-9)))).fillna(50.0)

    # 9. Vol ratio: vol / ma20
    out["sentiment_vol_ratio"] = (out["sentiment_vol"] / (out["sentiment_ma20"].abs() + 1e-9)).fillna(0.0)

    # 10. Crown: convolution of sentiment with price return
    _ret = out["return"].fillna(0.0)
    out["sentiment_crown"] = (out["sentiment_score"] * _ret).rolling(5, min_periods=1).sum().fillna(0.0)

    # 11. News volume (alias for sentiment_noticias)
    out["news_volume"] = out["sentiment_noticias"]
    # 12. News volume MA5
    out["news_volume_ma5"] = out["sentiment_noticias"].rolling(5, min_periods=1).mean()

    # 13. Extreme days: count of trailing days with |score| > 0.5
    out["sentiment_extreme"] = out["sentiment_score"].rolling(20, min_periods=1).apply(
        lambda s: float((np.abs(np.asarray(s)) > 0.5).sum()), raw=True
    ).fillna(0.0)

    # 14. Sentiment change (daily delta)
    out["sentiment_change"] = out["sentiment_score"].diff().fillna(0.0)
    # 15. MA5 of sentiment change
    out["sentiment_change_ma5"] = out["sentiment_change"].rolling(5, min_periods=1).mean().fillna(0.0)

    return out.drop(columns=["_fecha", "fecha", "compound", "noticias"])


def add_regime_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'regime' column (bull/bear/neutral) to the DataFrame.

    Only adds the column if it isn't present already. Uses the global
    detect_regimes which relies on 'return', 'volatility' and 'vix_level'.
    """
    if "regime" not in df.columns:
        df["regime"] = detect_regimes(df, lookback=60)
    return df


def fill_macro_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values in macro features with 0.0 for early rows."""
    macro_cols = [c for c in df.columns if any(x in c for x in ['vix', 'sector_', 'treasury', 'usd_'])]
    for col in macro_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
            # Replace infinities with 0
            df[col] = df[col].replace([np.inf, -np.inf], 0.0)
    return df


def build_features_pipeline(include_advanced: bool = False) -> list[Path]:
    """Lee data/processed/*.csv y escribe data/interim/features_<nombre>.csv.

    Reproducible por diseño: mismo input → mismo output byte a byte. No hay
    aleatoriedad ni dependencia del orden de diccionarios; el CSV de salida
    queda ordenado por timestamp. Añade:
      - sentimiento diario (SENT-003)
      - features macroeconómicas (macro_features.py)
      - columna de régimen (regime_detection.py)
      - features avanzadas de microestructura (advanced_features.py,
        opcional, include_advanced=True)
    """
    paths.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths.INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for csv_file in sorted(paths.PROCESSED_DATA_DIR.glob("*.csv")):
        ticker = csv_file.stem.split("_ml_ready")[0].upper()
        df = pd.read_csv(csv_file, parse_dates=["timestamp"])
        df = add_derived_features(df).sort_values("timestamp").reset_index(drop=True)
        df = add_sentiment_features(df, ticker=ticker)
        df = add_macro_features(df, use_yf_fetch=False)
        df = add_regime_column(df)
        if include_advanced:
            df = add_advanced_features(df)
        out = paths.INTERIM_DATA_DIR / f"features_{csv_file.name}"
        df.to_csv(out, index=False)
        written.append(out)
        print(f"   ✔ {csv_file.name} → {out.name} ({df.shape[0]}f × {df.shape[1]}c)")

    return written


if __name__ == "__main__":
    build_features_pipeline()
