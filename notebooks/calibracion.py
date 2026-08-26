"""calibracion.py — MOD-002: ¿las probabilidades significan lo que dicen?

El filtrado por confianza (IMP-001) solo sirve si p=0.65 implica ~65% de
aciertos. Este script mide la calibración del RF-best actual y de su versión
isotónica (CalibratedClassifierCV, ajustada solo con datos de train — sin
fuga), sobre los mismos folds walk-forward del protocolo IMP-003.

Métricas por variante:
- auc            : poder de ordenación (no debe empeorar al calibrar).
- ece            : expected calibration error (media |p_pred − frecuencia real|
                   ponderada por bins de 0.05). Más bajo = mejor calibrado.
- precision@top20: precisión entre el 20% más confiado.
- n_p65 / prec_p65: días con p≥0.65 y su acierto real.

Criterio de adopción predefinido: isotónica si ECE mejora ≥ 0.02 y el AUC no
cae más de 0.005.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from inversion.models.train_model import available_feature_cols
from inversion.tuning.tune_model import _walk_forward_folds, load_best_params
from inversion.utils import paths

N_JOBS = 2  # buen vecino: la máquina se comparte (ver model_comparison.py)
BIN_EDGES = np.linspace(0.0, 1.0, 21)
OUT_CSV = paths.REPORTS_DIR / "calibracion.csv"


def load_ticker(ticker: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv")
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    cols = available_feature_cols(df)
    df = df.dropna(subset=cols)
    return df[cols], df["target"]


def rf_factory(ticker: str):
    params = {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 50, "max_features": "sqrt"}
    params.update(load_best_params(ticker) or {})

    def make():
        return RandomForestClassifier(**params, class_weight="balanced", random_state=42, n_jobs=N_JOBS)

    return make


def ece(y: np.ndarray, p: np.ndarray) -> float:
    bins = np.digitize(p, BIN_EDGES) - 1
    total, err = len(p), 0.0
    for b in range(len(BIN_EDGES)):
        mask = bins == b
        if mask.sum() == 0:
            continue
        err += mask.sum() * abs(p[mask].mean() - y[mask].mean())
    return float(err / total)


def precision_top20(y: np.ndarray, p: np.ndarray) -> float:
    k = max(int(len(p) * 0.2), 1)
    return float(y[np.argsort(p)[-k:]].mean())


def collect(y_list: list[np.ndarray], p_list: list[np.ndarray]) -> dict:
    y = np.concatenate(y_list)
    p = np.concatenate(p_list)
    hi = p >= 0.65
    return {
        "auc": float(roc_auc_score(y, p)),
        "ece": round(ece(y, p), 4),
        "precision_top20": round(precision_top20(y, p), 4),
        "n_p65": int(hi.sum()),
        "prec_p65": round(float(y[hi].mean()), 4) if hi.any() else None,
    }


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]:
        X, y = load_ticker(ticker)
        folds = _walk_forward_folds(len(X))
        raw_y, raw_p, cal_y, cal_p = [], [], [], []

        for tr_end, te_end in folds:
            base = rf_factory(ticker)
            X_tr, y_tr = X.iloc[:tr_end], y.iloc[:tr_end]
            X_te, y_te = X.iloc[tr_end:te_end], y.iloc[tr_end:te_end]

            m_raw = base()
            m_raw.fit(X_tr, y_tr)
            raw_y.append(y_te.to_numpy())
            raw_p.append(m_raw.predict_proba(X_te)[:, 1])

            # Isotónica con CV interna SOLO sobre el tramo de train.
            m_cal = CalibratedClassifierCV(base(), cv=3, method="isotonic")
            m_cal.fit(X_tr, y_tr)
            cal_y.append(y_te.to_numpy())
            cal_p.append(m_cal.predict_proba(X_te)[:, 1])

        rows.append({"ticker": ticker, "variante": "raw", **collect(raw_y, raw_p)})
        rows.append({"ticker": ticker, "variante": "isotonica", **collect(cal_y, cal_p)})
        print(f"{ticker}: ECE raw={rows[-2]['ece']} → iso={rows[-1]['ece']} | AUC {rows[-2]['auc']:.4f}→{rows[-1]['auc']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("DONE", flush=True)

    medias = df.groupby("variante")[["auc", "ece", "precision_top20"]].mean().round(4)
    print("\n=== MEDIAS ===")
    print(medias.to_string())
    d_ece = medias.loc["isotonica", "ece"] - medias.loc["raw", "ece"]
    d_auc = medias.loc["isotonica", "auc"] - medias.loc["raw", "auc"]
    adoptar = d_ece <= -0.02 and d_auc >= -0.005
    print(f"\nΔECE={d_ece:+.4f} ΔAUC={d_auc:+.4f} → {'ADOPTAR calibración isotónica' if adoptar else 'MANTENER sin calibrar'}")


if __name__ == "__main__":
    main()
