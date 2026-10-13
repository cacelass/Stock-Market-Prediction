"""har_regime_vol.py — EXP-007: ¿HAR y regímenes mejoran el modelo de volatilidad?

La literatura de forecasting de volatilidad (ver references/06) es tajante:

  - HAR-RV (Corsi 2009) es el benchmark: lags de volatilidad diaria/semanal/mensual.
  - Christensen et al. (2026): el ML con lags HAR bate a HAR y gana más a más horizonte.
  - Kilic (Fed, 2025): los modelos con CAMBIO DE RÉGIMEN (Markov/threshold) superan
    al ML puro; la persistencia y la no linealidad por régimen son la clave.

Nuestro modelo actual (inversion/models/volatility.py) usa features genéricas de
precio, NO lags HAR explícitos ni régimen. Este experimento mide si añadirlos
mejora el R²/Spearman walk-forward.

Variantes, todas con walk-forward honesto y 7 tickers:
  base    = available_feature_cols (actual)
  +HAR    = base + log RV diaria/semanal/mensual y sus ratios (Corsi)
  +REG    = base + indicador de régimen de volatilidad (terciles, calculado en train)
  +HAR+REG

Uso:  uv run python notebooks/har_regime_vol.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from inversion.models.train_model import available_feature_cols
from inversion.models.volatility import make_vol_target
from inversion.tuning.tune_model import _walk_forward_folds
from inversion.utils import paths

TICKERS = ["AAPL", "AMZN", "MSFT", "TSLA", "GOOGL", "META", "NVDA"]
HORIZON = 10


def _rf():
    return RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=40,
        max_features="sqrt", random_state=42, n_jobs=1,
    )


def _add_har(d: pd.DataFrame) -> list[str]:
    """Lags HAR de varianza realizada (Corsi 2009) sobre retornos diarios."""
    rv = d["return"].pow(2)
    d["rv_d"] = np.log(rv.rolling(1).mean() + 1e-10)
    d["rv_w"] = np.log(rv.rolling(5).mean() + 1e-10)
    d["rv_m"] = np.log(rv.rolling(22).mean() + 1e-10)
    d["rv_ratio_dw"] = d["rv_d"] - d["rv_w"]
    d["rv_ratio_wm"] = d["rv_w"] - d["rv_m"]
    return ["rv_d", "rv_w", "rv_m", "rv_ratio_dw", "rv_ratio_wm"]


def _regime_labels(train_vol: pd.Series, *targets: pd.Series) -> list[np.ndarray]:
    """Régimen por terciles de la vol reciente, con cortes del TRAIN (sin fuga)."""
    q1, q2 = np.nanquantile(train_vol, [1 / 3, 2 / 3])
    return [np.digitize(np.log(t.to_numpy() + 1e-10), [np.log(q1 + 1e-10), np.log(q2 + 1e-10)]) for t in targets]


def _evaluate(d: pd.DataFrame, cols: list[str], use_regime: bool) -> tuple[float, float]:
    d = d.dropna(subset=[*cols, "vol_target"]).sort_values("timestamp").reset_index(drop=True)
    X, y = d[cols].copy(), d["vol_target"].to_numpy()
    if use_regime:
        X["regime"] = 0.0
    preds, truth = [], []
    for train_end, test_end in _walk_forward_folds(len(X)):
        Xtr, Xte = X.iloc[:train_end].copy(), X.iloc[train_end:test_end].copy()
        if use_regime:
            vol_past = d["volatility"].iloc[:train_end]
            regs = _regime_labels(vol_past, d["volatility"].iloc[:train_end], d["volatility"].iloc[train_end:test_end])
            Xtr["regime"], Xte["regime"] = regs[0], regs[1]
        m = _rf().fit(Xtr, y[:train_end])
        preds.append(m.predict(Xte))
        truth.append(y[train_end:test_end])
    p = pd.Series(np.concatenate(preds))
    t = pd.Series(np.concatenate(truth))
    return float(r2_score(t, p)), float(t.corr(p, method="spearman"))


def main() -> None:
    agg = {k: [] for k in ("base", "+HAR", "+REG", "+HAR+REG")}
    for ticker in TICKERS:
        df = pd.read_csv(paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv", parse_dates=["timestamp"])
        df = make_vol_target(df, HORIZON).sort_values("timestamp").reset_index(drop=True)
        base_cols = available_feature_cols(df)
        har_cols = _add_har(df)

        for key, cols, reg in (
            ("base", base_cols, False),
            ("+HAR", base_cols + har_cols, False),
            ("+REG", base_cols, True),
            ("+HAR+REG", base_cols + har_cols, True),
        ):
            r2, sp = _evaluate(df, cols, reg)
            agg[key].append((r2, sp))
        print(f"  {ticker}: base R²={agg['base'][-1][0]:.3f}  +HAR R²={agg['+HAR'][-1][0]:.3f}  "
              f"+REG R²={agg['+REG'][-1][0]:.3f}  +HAR+REG R²={agg['+HAR+REG'][-1][0]:.3f}")
    print(f"{'variante':>10} {'R² medio':>9} {'Spearman':>9}")
    base_r2 = np.mean([x[0] for x in agg["base"]])
    for key, vals in agg.items():
        r2 = np.mean([x[0] for x in vals])
        sp = np.mean([x[1] for x in vals])
        print(f"{key:>10} {r2:>9.3f} {sp:>9.3f}   (ΔR²={r2 - base_r2:+.3f})")


if __name__ == "__main__":
    main()
