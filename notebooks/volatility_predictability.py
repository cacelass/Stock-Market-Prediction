"""volatility_predictability.py — EXP-004: ¿los datos contienen riesgo aunque no dirección?

La línea de modelado (secciones H–K de references/05) llegó a un techo de
AUC ≈ 0.55 prediciendo *dirección*. En finanzas la dirección es casi ruido,
pero la **volatilidad es persistente** ("clustering"): un día movido suele
seguirle otro movido. Comprobación honesta de esa intuición:

  target   = volatilidad realizada futura: std de los retornos diarios de los
             próximos h días.
  baseline = persistencia: la volatilidad de los últimos h días. Es un baseline
             fuerte y tonto; si el RF no lo bate, no hay señal explotable.
  modelo   = RandomForestRegressor sobre las mismas features que el clasificador.

Métricas walk-forward (mismos folds que el resto del proyecto): R² fuera de
muestra y Spearman (orden — lo que importa para dimensionar riesgo).

Uso:  uv run python notebooks/volatility_predictability.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from inversion.models.train_model import available_feature_cols
from inversion.tuning.tune_model import _walk_forward_folds
from inversion.utils import paths

TICKERS = ["AAPL", "AMZN", "MSFT", "TSLA", "GOOGL", "META", "NVDA"]
HORIZONS = [5, 10, 20]


def _rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=40,
        max_features="sqrt", random_state=42, n_jobs=1,
    )


def _load(ticker: str) -> pd.DataFrame:
    return pd.read_csv(paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv")


def _evaluate(df: pd.DataFrame, h: int) -> tuple[float, float, float, float]:
    """(R² RF, R² persistencia, Spearman RF, Spearman persistencia) walk-forward."""
    d = df.copy()
    # vol futura: std de los retornos de los próximos h días (shift para no mirar atrás)
    d["_y"] = d["return"].shift(-1).rolling(h).std().shift(-(h - 1))
    d["_base"] = d["return"].rolling(h).std()
    cols = available_feature_cols(d)
    d = d.dropna(subset=[*cols, "_y", "_base"]).reset_index(drop=True)
    X, y, base = d[cols].to_numpy(), d["_y"].to_numpy(), d["_base"].to_numpy()

    pred, bl, truth = [], [], []
    for train_end, test_end in _walk_forward_folds(len(X)):
        m = _rf().fit(X[:train_end], y[:train_end])
        pred.append(m.predict(X[train_end:test_end]))
        bl.append(base[train_end:test_end])
        truth.append(y[train_end:test_end])
    pred, bl, truth = np.concatenate(pred), np.concatenate(bl), np.concatenate(truth)
    return (r2_score(truth, pred), r2_score(truth, bl),
            spearmanr(truth, pred).correlation, spearmanr(truth, bl).correlation)


def main() -> None:
    print(f"{'horizonte':>10} {'R² RF':>8} {'R² persist':>11} {'Spearman RF':>12} {'Spearman persist':>18}")
    for h in HORIZONS:
        vals = np.array([_evaluate(_load(t), h) for t in TICKERS])
        mean = vals.mean(axis=0)
        print(f"{h:>8}d {mean[0]:>8.3f} {mean[1]:>11.3f} {mean[2]:>12.3f} {mean[3]:>18.3f}")

    print("\nLectura: si el RF supera a la persistencia en R² y Spearman, hay señal")
    print("de riesgo explotable aunque la dirección sea ruido.")


if __name__ == "__main__":
    main()
