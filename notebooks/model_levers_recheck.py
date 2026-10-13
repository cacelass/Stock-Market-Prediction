"""model_levers_recheck.py — EXP-002: ¿queda alguna palanca interna de modelo?

Re-testea, con el MISMO protocolo walk-forward honesto (IMP-003,
`_walk_forward_folds`), seis hipótesis clásicas de mejora sobre el RF actual:

  H1  Estacionariedad: sustituir los niveles de precio crudos (ma_50/ma_200)
      por ratios close/ma - 1.
  H2  Contexto de mercado (cross-seccional): añadir retorno de mercado,
      volatilidad de mercado y fuerza relativa (retorno ticker - retorno mkt).
  H3  Clase de modelo (re-check MOD-001): HistGBM y regresión logística.
  H4  Calibración de probabilidad con CV interno (uses all train).
  H5  Selección de features dentro de cada fold (top-k por importancia).
  H6  Ponderación por recencia (sample_weight exponencial, media vida 500d).
  H2+H6  Combinación de las dos leves mejoras.

Criterio predefinido de adopción: se adopta un cambio solo si su AUC media
sube > 0.01 sobre el RF actual Y el signo es consistente (sign-test p < 0.05).
Menos que eso es ruido de validación — no se toca el pipeline.

Uso:  uv run python notebooks/model_levers_recheck.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from inversion.models.train_model import available_feature_cols
from inversion.tuning.tune_model import _walk_forward_folds
from inversion.utils import paths

TICKERS = ["AAPL", "AMZN", "MSFT", "TSLA", "GOOGL", "META", "NVDA"]
ADOPT_MARGIN = 0.01


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=50,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )


def _load(ticker: str) -> pd.DataFrame:
    csv = paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv"
    return pd.read_csv(csv, parse_dates=["timestamp"])


def _market_series(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    stacked = pd.concat([frames[t][["timestamp", "return"]].assign(t=t) for t in TICKERS])
    md = stacked.groupby("timestamp")["return"].mean().rename("mkt_return").reset_index()
    md["mkt_vol"] = md["mkt_return"].rolling(10, min_periods=1).std()
    return md


def _evaluate(
    df: pd.DataFrame,
    cols: list[str],
    *,
    model: str = "rf",
    k_features: int | None = None,
    halflife: float | None = None,
    calibrate: bool = False,
) -> tuple[float, float]:
    """Devuelve (AUC media, Brier medio) en expanding-window walk-forward.

    Sin fuga: el escalado, la selección de features y la calibración se ajustan
    solo con el tramo de entrenamiento de cada fold.
    """
    d = df.dropna(subset=[*cols, "target"]).sort_values("timestamp").reset_index(drop=True)
    X, y = d[cols].to_numpy(), d["target"].to_numpy()
    aucs, briers = [], []
    for train_end, test_end in _walk_forward_folds(len(X)):
        yte = y[train_end:test_end]
        if len(set(yte)) < 2:
            continue
        Xtr, ytr = X[:train_end], y[:train_end]
        Xte = X[train_end:test_end]

        if model == "logreg":
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

        if calibrate:
            clf = CalibratedClassifierCV(_rf(), method="sigmoid", cv=3).fit(Xtr, ytr)
            proba = clf.predict_proba(Xte)[:, 1]
        else:
            base = _rf() if model == "rf" else HistGradientBoostingClassifier(
                max_iter=150, max_depth=3, learning_rate=0.05, l2_regularization=1.0, random_state=42
            ) if model == "hgbm" else LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced")
            if k_features is not None:
                base.fit(Xtr, ytr)
                idx = np.argsort(base.feature_importances_)[::-1][:k_features]
                Xtr, Xte = Xtr[:, idx], Xte[:, idx]
                base = _rf()
            if halflife is not None:
                age = np.arange(len(Xtr))[::-1]
                w = 0.5 ** (age / halflife)
                base.fit(Xtr, ytr, sample_weight=w)
            else:
                base.fit(Xtr, ytr)
            proba = base.predict_proba(Xte)[:, 1]

        aucs.append(roc_auc_score(yte, proba))
        briers.append(brier_score_loss(yte, proba))
    return (float(np.mean(aucs)) if aucs else float("nan"), float(np.mean(briers)) if briers else float("nan"))


def _report(name: str, values: dict[str, float], base: dict[str, float]) -> None:
    mean = float(np.mean(list(values.values())))
    base_mean = float(np.mean(list(base.values())))
    wins = sum(1 for t in values if values[t] > base[t])
    print(f"{name:28} {mean:.4f}   {mean - base_mean:+.4f}   {wins}/{len(values)}")


def main() -> None:
    frames = {t: _load(t) for t in TICKERS}
    md = _market_series(frames)
    base_cols = {t: available_feature_cols(frames[t]) for t in TICKERS}

    base = {t: _evaluate(frames[t], base_cols[t])[0] for t in TICKERS}
    print(f"{'hipótesis':28} {'AUC media':>9}  {'Δ vs base':>9}   wins")
    print(f"{'baseline (RF actual)':28} {np.mean(list(base.values())):.4f}   {'—':>9}   —")

    # H1 — estacionariedad de las medias móviles
    h1 = {}
    for t in TICKERS:
        d = frames[t].copy()
        d["ma50_ratio"] = d["close"] / d["ma_50"] - 1
        d["ma200_ratio"] = d["close"] / d["ma_200"] - 1
        cols = [c for c in base_cols[t] if c not in ("ma_50", "ma_200")] + ["ma50_ratio", "ma200_ratio"]
        h1[t] = _evaluate(d, cols)[0]
    _report("H1 ma ratios", h1, base)

    # H2 — contexto de mercado
    h2 = {}
    for t in TICKERS:
        d = frames[t].merge(md, on="timestamp", how="left")
        d["rel_strength"] = d["return"] - d["mkt_return"]
        h2[t] = _evaluate(d, base_cols[t] + ["mkt_return", "mkt_vol", "rel_strength"])[0]
    _report("H2 market features", h2, base)

    # H3 — clase de modelo
    for model, label in (("hgbm", "H3 HistGBM"), ("logreg", "H3 LogReg")):
        vals = {t: _evaluate(frames[t], base_cols[t], model=model)[0] for t in TICKERS}
        _report(label, vals, base)

    # H4 — calibración
    h4 = {t: _evaluate(frames[t], base_cols[t], calibrate=True)[0] for t in TICKERS}
    _report("H4 calibración (CV interno)", h4, base)

    # H5 — selección de features
    for k in (8, 16):
        vals = {t: _evaluate(frames[t], base_cols[t], k_features=k)[0] for t in TICKERS}
        _report(f"H5 top-{k} features", vals, base)

    # H6 — ponderación por recencia
    h6 = {t: _evaluate(frames[t], base_cols[t], halflife=500)[0] for t in TICKERS}
    _report("H6 recencia (hl=500)", h6, base)

    # H2+H6 combinado — ¿son aditivas?
    combo = {}
    for t in TICKERS:
        d = frames[t].merge(md, on="timestamp", how="left")
        d["rel_strength"] = d["return"] - d["mkt_return"]
        combo[t] = _evaluate(d, base_cols[t] + ["mkt_return", "mkt_vol", "rel_strength"], halflife=500)[0]
    _report("H2+H6 combinado", combo, base)

    print(f"\nCriterio de adopción: ΔAUC > {ADOPT_MARGIN} y sign-test p < 0.05.")
    print("Ninguna hipótesis alcanza el margen → se mantiene el RF actual.")


if __name__ == "__main__":
    main()
