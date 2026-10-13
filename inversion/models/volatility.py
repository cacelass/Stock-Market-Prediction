"""inversion/models/volatility.py — modelo de riesgo/volatilidad (EXP-005).

La línea de modelado de dirección (references/05, secciones H–K) llegó a un
techo de AUC ≈ 0.55: la dirección diaria es casi ruido. Pero la volatilidad es
persistente, y **esta misma matriz de features** la predice con R² fuera de
muestra ≈ 0.2 y Spearman ≈ 0.45 (sección L), batiendo al baseline de
persistencia. Este módulo añade ese modelo de riesgo sin tocar el clasificador:

  target  = volatilidad realizada futura: std de los retornos de los próximos
            `horizon` días (por defecto 10, el mejor según EXP-004).

Uso:
    python -m inversion.models.volatility [TICKER ...]
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from inversion.models.train_model import available_feature_cols
from inversion.tuning.tune_model import _walk_forward_folds
from inversion.utils import paths

# Horizonte del target (EXP-004: 10d maximiza R² y Spearman).
VOL_HORIZON_DAYS = 10
# Días de trading por año, para anualizar la volatilidad diaria.
TRADING_DAYS = 252


def make_vol_target(df: pd.DataFrame, horizon: int = VOL_HORIZON_DAYS) -> pd.DataFrame:
    """Añade `vol_target`: std de los retornos de los próximos `horizon` días.

    Sin fuga: la ventana es estrictamente futura respecto a cada fila. Las
    últimas `horizon` filas quedan en NaN y se descartan al entrenar/evaluar.
    """
    if "return" not in df.columns:
        raise ValueError("El dataset necesita una columna 'return'.")
    d = df.copy()
    # Momento t: ventana de retornos [t+1, t+horizon].
    fwd = d["return"].shift(-1).rolling(horizon).std().shift(-(horizon - 1))
    d["vol_target"] = fwd
    return d


def annualize(vol_daily: float) -> float:
    """Volatilidad diaria → anualizada (√252)."""
    return float(vol_daily * (TRADING_DAYS**0.5))


def _rf_regressor(random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=6,          # profundidad baja: hay poca señal, evitamos sobreajuste
        min_samples_leaf=40,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def _walk_forward_scores(X: np.ndarray, y: np.ndarray, random_state: int) -> tuple[float, float]:
    """(R², Spearman) fuera de muestra con expanding-window walk-forward."""
    preds, truth = [], []
    for train_end, test_end in _walk_forward_folds(len(X)):
        m = _rf_regressor(random_state).fit(X[:train_end], y[:train_end])
        preds.append(m.predict(X[train_end:test_end]))
        truth.append(y[train_end:test_end])
    if not preds:
        return float("nan"), float("nan")
    p = pd.Series(np.concatenate(preds))
    t = pd.Series(np.concatenate(truth))
    return float(r2_score(t, p)), float(t.corr(p, method="spearman"))


def _load_prepared(csv_path: Path, horizon: int, with_sentiment: bool) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = make_vol_target(df, horizon)
    cols = available_feature_cols(df)
    if not with_sentiment:
        cols = [c for c in cols if not c.startswith("sentiment_")]
    df = df.dropna(subset=[*cols, "vol_target"]).sort_values("timestamp").reset_index(drop=True)
    return df, cols


def train_vol_ticker(
    ticker: str,
    csv_path: Path,
    horizon: int = VOL_HORIZON_DAYS,
    random_state: int = 42,
    with_sentiment: bool = True,
) -> dict[str, Any]:
    """Entrena el modelo de volatilidad de un ticker y guarda models/vol_<T>.pkl.

    Devuelve métricas walk-forward honestas (r2, spearman) del modelo y del
    baseline de persistencia (vol de los últimos `horizon` días).
    """
    df, cols = _load_prepared(csv_path, horizon, with_sentiment)
    if len(df) < 60:
        raise ValueError(f"Dataset demasiado corto para '{ticker}': {len(df)} filas.")

    X, y = df[cols].to_numpy(), df["vol_target"].to_numpy()
    r2, spearman = _walk_forward_scores(X, y, random_state)

    # Baseline tonto y fuerte: persistencia (vol de los últimos `horizon` días).
    persistence = df["return"].rolling(horizon).std()
    mask = persistence.notna()
    base_r2 = float(r2_score(df["vol_target"][mask], persistence[mask])) if mask.sum() > 1 else float("nan")
    base_sp = float(df["vol_target"][mask].corr(persistence[mask], method="spearman")) if mask.sum() > 1 else float("nan")

    final = _rf_regressor(random_state).fit(X, y)
    save_path = paths.MODELS_DIR / f"vol_{ticker}.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": cols, "horizon": horizon}, save_path)

    return {
        "ticker": ticker,
        "horizon": horizon,
        "n_train": int(len(df)),
        "r2": r2,
        "spearman": spearman,
        "base_r2": base_r2,
        "base_spearman": base_sp,
        "beats_persistence": bool(r2 > base_r2),
    }


def predict_volatility(ticker: str, csv_path: Path | None = None) -> dict[str, Any]:
    """Carga models/vol_<T>.pkl y predice la volatilidad futura de la última fila."""
    model_file = paths.MODELS_DIR / f"vol_{ticker}.pkl"
    if not Path(model_file).exists():
        raise FileNotFoundError(f"No hay modelo de volatilidad para '{ticker}': {model_file}")
    bundle = joblib.load(model_file)
    model, cols = bundle["model"], bundle["features"]

    csv = csv_path or (paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv")
    df = pd.read_csv(csv, parse_dates=["timestamp"]).sort_values("timestamp")
    row = df.dropna(subset=cols).iloc[[-1]]
    vol_daily = float(model.predict(row[cols].to_numpy())[0])
    return {
        "ticker": ticker,
        "vol_daily": vol_daily,
        "vol_annualized": annualize(vol_daily),
        "as_of": str(row["timestamp"].iloc[0].date()),
    }


def main() -> None:
    import sys

    tickers = sys.argv[1:] or ["AAPL", "AMZN", "MSFT", "TSLA", "GOOGL", "META", "NVDA"]
    rows = []
    for t in tickers:
        csv = paths.INTERIM_DATA_DIR / f"features_{t}_ml_ready.csv"
        if not Path(csv).exists():
            print(f"  ⚠ sin features para {t}: {csv}")
            continue
        m = train_vol_ticker(t, csv)
        rows.append(m)
        print(
            f"  {t}: R²={m['r2']:.3f} (persist {m['base_r2']:.3f})  "
            f"Spearman={m['spearman']:.3f} (persist {m['base_spearman']:.3f})"
        )
    if rows:
        out = paths.PROJECT_DIR / "reports" / "resultados_volatilidad.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nMétricas → {out}")


if __name__ == "__main__":
    main()
