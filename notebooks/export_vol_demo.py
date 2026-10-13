"""export_vol_demo.py — datos de la demo web del modelo de volatilidad.

Genera `reports/vol_demo_data.json`: para cada ticker, la serie walk-forward
(out-of-sample) de volatilidad predicha vs realizada vs persistencia (anualizada),
y las métricas R²/Spearman. Lo consume la demo estática del sitio
cacelass.github.io (`stock-market-prediction/demo/`).

Sin backend: el navegador solo dibuja una serie ya calculada de forma honesta
(expanding-window walk-forward, sin fuga), no ejecuta el modelo.

Uso:  uv run python notebooks/export_vol_demo.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from inversion.models.train_model import available_feature_cols
from inversion.models.volatility import (
    HAR_FEATURES,
    VOL_HORIZON_DAYS,
    _prepare_frame,
    _rf_regressor,
)
from inversion.tuning.tune_model import _walk_forward_folds
from inversion.utils import paths

TICKERS = ["AAPL", "AMZN", "MSFT", "TSLA", "GOOGL", "META", "NVDA"]
ANN = 252 ** 0.5


def _series(ticker: str) -> dict:
    csv = paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv"
    df = pd.read_csv(csv, parse_dates=["timestamp"])
    df = _prepare_frame(df, VOL_HORIZON_DAYS)
    cols = available_feature_cols(df) + HAR_FEATURES
    df = df.dropna(subset=[*cols, "vol_target"]).sort_values("timestamp").reset_index(drop=True)

    X, y = df[cols].to_numpy(), df["vol_target"].to_numpy()
    persistence = df["return"].rolling(VOL_HORIZON_DAYS).std().to_numpy()

    preds, idx = [], []
    for train_end, test_end in _walk_forward_folds(len(X)):
        m = _rf_regressor(42).fit(X[:train_end], y[:train_end])
        preds.append(m.predict(X[train_end:test_end]))
        idx.extend(range(train_end, test_end))

    pred = np.concatenate(preds)
    actual = y[idx]
    persist = persistence[idx]
    dates = df["timestamp"].iloc[idx].dt.strftime("%Y-%m-%d").tolist()

    def r2(a, b):
        return float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))

    def spear(a, b):
        return float(pd.Series(a).corr(pd.Series(b), method="spearman"))

    return {
        "ticker": ticker,
        "dates": dates,
        "pred": [round(float(v) * ANN, 4) for v in pred],
        "actual": [round(float(v) * ANN, 4) for v in actual],
        "persistence": [round(float(v) * ANN, 4) if np.isfinite(v) else None for v in persist],
        "metrics": {
            "r2": round(r2(actual, pred), 3),
            "spearman": round(spear(actual, pred), 3),
            "base_r2": round(r2(actual, persist), 3),
            "base_spearman": round(spear(actual, persist), 3),
        },
    }


def main() -> None:
    data = {t: _series(t) for t in TICKERS}
    out = paths.REPORTS_DIR / "vol_demo_data.json"
    out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"→ {out}  ({kb:.0f} KB, {len(TICKERS)} tickers)")


if __name__ == "__main__":
    main()
