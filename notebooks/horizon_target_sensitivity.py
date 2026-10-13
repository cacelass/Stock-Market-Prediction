"""horizon_target_sensitivity.py — EXP-003: ¿el horizonte de 5 días es el límite?

Dos preguntas honestas que aún no se habían respondido con datos:

1. ¿Cambiar el horizonte (1/3/5/10/20 días) o el tipo de target (dirección
   pura vs umbral) da más señal? Si el AUC sube en otro horizonte, el cuello
   de botella es la definición del problema; si baja o se mantiene, es el dato.

2. ¿El RandomForest bate a un baseline trivial? Si un "sube si el pasado subió"
   (momentum) o "sube si el pasado bajó" (reversión) tiene un AUC parecido al
   del RF, entonces el modelo no aporta edge real sobre la señal más simple.

Protocolo idéntico al del proyecto (`_walk_forward_folds`, expanding window),
mismos tickers y mismos features disponibles. Scores continuos (no umbrales
de decisión) para el AUC, así que los baselines son comparables 1:1.

Uso:  uv run python notebooks/horizon_target_sensitivity.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from inversion.models.train_model import available_feature_cols
from inversion.tuning.tune_model import _walk_forward_folds
from inversion.utils import paths

TICKERS = ["AAPL", "AMZN", "MSFT", "TSLA", "GOOGL", "META", "NVDA"]
HORIZONS = [1, 3, 5, 10, 20]


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=50, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=1,
    )


def _load(ticker: str) -> pd.DataFrame:
    return pd.read_csv(paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv")


def _targets(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Un DataFrame por horizonte: features + close + target (dirección pura)."""
    out = {}
    for h in HORIZONS:
        d = df.copy()
        fwd = d["close"].shift(-h) / d["close"] - 1
        d["_fwd"] = fwd
        d["_dir"] = (fwd > 0).astype(int)
        out[h] = d.iloc[:-h].reset_index(drop=True)  # las últimas h no tienen futuro
    return out


def _auc_rf(d: pd.DataFrame, cols: list[str]) -> float:
    d = d.dropna(subset=[*cols, "_dir"]).reset_index(drop=True)
    X, y = d[cols].to_numpy(), d["_dir"].to_numpy()
    aucs = []
    for train_end, test_end in _walk_forward_folds(len(X)):
        yte = y[train_end:test_end]
        if len(set(yte)) < 2:
            continue
        m = _rf().fit(X[:train_end], y[:train_end])
        aucs.append(roc_auc_score(yte, m.predict_proba(X[train_end:test_end])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def _auc_score(d: pd.DataFrame, score_col: str) -> float:
    """AUC walk-forward de una feature suelta usada como score (sin folds de modelo)."""
    d = d.dropna(subset=[score_col, "_dir"]).reset_index(drop=True)
    s, y = d[score_col].to_numpy(), d["_dir"].to_numpy()
    aucs = []
    for train_end, test_end in _walk_forward_folds(len(s)):
        yte = y[train_end:test_end]
        if len(set(yte)) < 2:
            continue
        aucs.append(roc_auc_score(yte, s[train_end:test_end]))
    return float(np.mean(aucs)) if aucs else float("nan")


def main() -> None:
    print("=== 1) Sensibilidad al horizonte (target = dirección pura) ===")
    print(f"{'horizonte':>10} {'AUC RF':>8}  {'momentum h':>11}  {'reversión':>10}  {'volatilidad':>12}")
    per_h = {}
    for h in HORIZONS:
        rf, mom, rev, vol = [], [], [], []
        for t in TICKERS:
            d = _targets(_load(t))[h]
            d["mom"] = d["close"] / d["close"].shift(h) - 1   # retorno pasado h días
            d["rev"] = -d["mom"]
            rf.append(_auc_rf(d, available_feature_cols(d)))
            mom.append(_auc_score(d, "mom"))
            rev.append(_auc_score(d, "rev"))
            vol.append(_auc_score(d, "volatility"))
        per_h[h] = dict(rf=float(np.mean(rf)), mom=float(np.mean(mom)),
                        rev=float(np.mean(rev)), vol=float(np.mean(vol)))
        r = per_h[h]
        print(f"{h:>8}d {r['rf']:>8.4f}  {r['mom']:>11.4f}  {r['rev']:>10.4f}  {r['vol']:>12.4f}")

    print("\n=== 2) RF vs features sueltas en el horizonte actual (5d) ===")
    singles = ["return", "lag_1", "lag_5", "lag_20", "rsi", "volatility", "vwap_ratio"]
    print(f"{'score':>14} {'AUC':>8}")
    rows = {}
    for col in singles:
        vals = []
        for t in TICKERS:
            d = _targets(_load(t))[5]
            vals.append(_auc_score(d, col))
        rows[col] = float(np.mean(vals))
        print(f"{col:>14} {rows[col]:>8.4f}")
    print(f"{'RF (todas)':>14} {per_h[5]['rf']:>8.4f}")

    best_single = max(rows.values())
    print(f"\nMejor feature suelta 5d: {best_single:.4f}  | RF: {per_h[5]['rf']:.4f}"
          f"  | diferencia: {per_h[5]['rf'] - best_single:+.4f}")


if __name__ == "__main__":
    main()
