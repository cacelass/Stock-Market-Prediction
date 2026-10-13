"""horizon_5_vs_10.py — EXP-003b: ¿el horizonte de 10 días bate al de 5 de forma robusta?

La sensibilidad por horizonte (notebooks/horizon_target_sensitivity.py) sugirió
que el target de dirección a 10 días tiene más AUC que a 5. Antes de proponer
cambiar `TARGET_HORIZON_DAYS`, hace falta descartar que sea ruido de validación:

  - comparación PAREADA por fold (mismo fold, mismo ticker, 5d vs 10d);
  - para los tres tipos de target: dirección pura, umbral fijo 2%, umbral escalado;
  - métrica operativa: precisión@top-quintil (entre los días de mayor p para
    subir, ¿cuántos suben de verdad?).

Veredicto predefinido: se propone cambiar el horizonte solo si el 10d gana de
forma consistente (≥60% de folds) Y la precisión@top-quintil no empeora.

Uso:  uv run python notebooks/horizon_5_vs_10.py
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


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=50, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=1,
    )


def _make(df: pd.DataFrame, h: int, kind: str) -> pd.DataFrame:
    d = df.copy()
    fwd = d["close"].shift(-h) / d["close"] - 1
    if kind == "dir":
        d["_t"] = (fwd > 0).astype(int)
    elif kind == "thr2":
        d["_t"] = (fwd > 0.02).astype(int)
    elif kind.startswith("thr"):
        q = float(kind[3:])  # p.ej. thr60 -> percentil 60 del retorno futuro
        d["_t"] = (fwd > np.nanquantile(fwd, q / 100)).astype(int)
    return d.iloc[:-h].reset_index(drop=True)


def _folds(d: pd.DataFrame, cols: list[str]) -> list[tuple[float, float, float]]:
    """Devuelve [(auc, prec_top20, base_rate)] por fold walk-forward."""
    d = d.dropna(subset=[*cols, "_t"]).reset_index(drop=True)
    X, y = d[cols].to_numpy(), d["_t"].to_numpy()
    out = []
    for train_end, test_end in _walk_forward_folds(len(X)):
        yte = y[train_end:test_end]
        if len(set(yte)) < 2:
            continue
        m = _rf().fit(X[:train_end], y[:train_end])
        p = m.predict_proba(X[train_end:test_end])[:, 1]
        auc = roc_auc_score(yte, p)
        k = max(len(p) // 5, 1)
        top = np.argsort(p)[::-1][:k]
        prec = float(yte[top].mean())
        out.append((auc, prec, float(yte.mean())))
    return out


def main() -> None:
    for kind, label in (("dir", "dirección pura"), ("thr2", "umbral fijo 2%"),
                        ("thr60", "umbral escalado (p60)")):
        deltas, dprec, wins, base10 = [], [], 0, []
        for t in TICKERS:
            df = pd.read_csv(paths.INTERIM_DATA_DIR / f"features_{t}_ml_ready.csv")
            cols = available_feature_cols(df)
            f5 = _folds(_make(df, 5, kind), cols)
            f10 = _folds(_make(df, 10, kind), cols)
            n = min(len(f5), len(f10))
            for (a5, p5, b5), (a10, p10, b10) in zip(f5[:n], f10[:n]):
                deltas.append(a10 - a5)
                dprec.append(p10 - p5)
                base10.append(b10 - b5)
                wins += a10 > a5
        deltas = np.array(deltas); dprec = np.array(dprec)
        n_a = len(deltas)
        se = deltas.std(ddof=1) / np.sqrt(n_a)
        print(f"[{label}] n_folds={n_a}  ΔAUC(10d-5d)={deltas.mean():+.4f} "
              f"(±{1.96 * se:.4f} IC95)  gana 10d en {wins}/{n_a} ({wins / n_a:.0%})  "
              f"Δprec_top20={dprec.mean():+.4f}  Δbase_rate={np.mean(base10):+.4f}")

    print("\nReferencia: ΔAUC>0 y gana ≥60% de folds → cambio respaldado por evidencia.")
    print("Si el AUC sube PERO la precisión@top-quintil no, no hay mejora operativa.")


if __name__ == "__main__":
    main()
