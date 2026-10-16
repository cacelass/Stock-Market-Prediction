"""Load ensemble models and return averaged predictions.

CLI usage:
    .venv/bin/python inversion/models/predict_ensemble.py --ticker AAPL

If a model file doesn't exist, it is skipped gracefully.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from inversion.features.build_features import add_derived_features
from inversion.models.train_model import available_feature_cols
from inversion.utils import paths


def predict_ensemble(
    df: pd.DataFrame,
    ticker: str,
    models: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ensemble models and return (predictions, probabilities).

    Parameters
    ----------
    df : DataFrame with features
    ticker : Ticker symbol
    models : List of model names to use (default: all available — rf_tech, hgb, rf_sent)

    Returns
    -------
    (predictions, probabilities) where probabilities are averaged across
    available models.
    """
    ticker = ticker.upper()
    if models is None:
        models = ["rf_tech", "hgb", "rf_sent"]

    # Try to load each model
    loaded: Dict[str, object] = {}
    scaler_tech_path = paths.MODELS_DIR / f"scaler_{ticker}.pkl"
    scaler_sent_path = paths.MODELS_DIR / f"scaler_sent_{ticker}.pkl"

    model_paths = {
        "rf_tech": paths.MODELS_DIR / f"ensemble_rf_{ticker}.pkl",
        "hgb": paths.MODELS_DIR / f"ensemble_hgb_{ticker}.pkl",
        "rf_sent": paths.MODELS_DIR / f"ensemble_rf_sent_{ticker}.pkl",
    }

    # Load models (skip if missing)
    for name, mpath in model_paths.items():
        if not mpath.exists():
            continue
        loaded[name] = joblib.load(mpath)

    if not loaded:
        # No ensemble models found — fallback to single RF
        try:
            from inversion.models.predict_model import predict_future
            cls, pb, ps, _ = predict_future(df, ticker)
            if cls is not None:
                return np.array([cls]), np.array([ps])
        except Exception:
            pass
        return np.array([]), np.array([])

    # Build features if needed
    df_feat = df.copy()
    if any(c not in df_feat.columns for c in available_feature_cols(df_feat)):
        df_feat = add_derived_features(df.copy())

    feature_cols = available_feature_cols(df_feat)
    last = df_feat.dropna(subset=feature_cols).iloc[[-1]][feature_cols]

    if not scaler_tech_path.exists():
        return np.array([]), np.array([])
    scaler_tech = joblib.load(scaler_tech_path)
    last_scaled = scaler_tech.transform(last)

    # Gather probabilities from available models
    probas: Dict[str, np.ndarray] = {}
    for name, model in loaded.items():
        try:
            if name == "rf_sent":
                # RF-sentiment needs sentiment scaler
                if scaler_sent_path.exists():
                    scaler_sent = joblib.load(scaler_sent_path)
                    sent_cols = [c for c in feature_cols if c.startswith("sentiment_")]
                    if sent_cols:
                        probas[name] = scaler_sent.transform(last[sent_cols])
                    else:
                        probas[name] = scaler_tech.transform(last)
                else:
                    probas[name] = scaler_tech.transform(last)
            else:
                probas[name] = last_scaled
        except Exception:
            continue

    # Average probabilities
    n_models = len(probas)
    if n_models == 0:
        return np.array([]), np.array([])

    avg_proba = np.zeros(n_models)
    idx = 0
    for name in probas:
        try:
            avg_proba[idx] = model.predict_proba(probas[name])[0, 1]
        except Exception:
            continue
        idx += 1

    avg_proba = avg_proba[:idx]
    if len(avg_proba) == 0:
        return np.array([]), np.array([])
    avg_proba = np.mean(avg_proba)

    predictions = np.array([1 if avg_proba >= 0.5 else 0])
    return predictions, np.array([avg_proba])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Predicci\u00f3n con ensemble de modelos.")
    parser.add_argument("--ticker", default="NVDA", help="Activo del cat\u00e1logo")
    args = parser.parse_args(argv)

    ticker = args.ticker.upper()
    try:
        df = _load_ticker_data(ticker)
    except FileNotFoundError as exc:
        print(f"\u274c {exc}")
        return 1

    preds, probas = predict_ensemble(df, ticker=ticker)

    if len(preds) == 0:
        print(f"\u274c No se encontraron modelos de ensemble para '{ticker}'.")
        print("   Ejecuta primero: .venv/bin/python inversion/models/train_ensemble.py")
        return 1

    direccion = "SUBE" if preds[0] == 1 else "BAJA"
    print(f"\u25b6 Predicci\u00f3n ensemble para {ticker}: {direccion}")
    print(f"   Probabilidad de subida: {probas[0]:.2%}")
    return 0


def _load_ticker_data(ticker: str) -> pd.DataFrame:
    ticker = ticker.upper()
    candidate = paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv"
    if candidate.exists():
        return pd.read_csv(candidate)
    for base in (paths.RAW_DATA_DIR, paths.PROCESSED_DATA_DIR):
        candidate = base / f"{ticker}_ml_ready.csv"
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError(
        f"No hay datos para '{ticker}' en data/raw/, data/processed/ ni data/interim/. "
        f"Ejecuta primero el pipeline de datos."
    )


if __name__ == "__main__":
    sys.exit(main())
