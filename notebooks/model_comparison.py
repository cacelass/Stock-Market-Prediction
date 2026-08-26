"""model_comparison.py — MOD-001: ¿RF es la clase de modelo correcta?

Compara en los MISMOS folds walk-forward (protocolo IMP-003) tres opciones:

1. RF-best      : RandomForest con best_params_<T>.json de Optuna (estado actual).
2. HistGBM      : HistGradientBoostingClassifier (sklearn nativo, sin dependencias
                  nuevas), regularización conservadora.
3. RF-ensemble  : media de probabilidades de 5 RF con semillas distintas —
                  reduce varianza sin tocar la clase.

Métricas: AUC, accuracy y precision@top-quintil (la precisión entre el 20%
de días con mayor probabilidad de subida — lo que de verdad usa un filtro de
confianza).

Veredicto predefinido: se adopta una alternativa solo si su AUC media supera
al RF-best actual por > 0.01. Menos que eso es ruido de validación.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from inversion.models.train_model import available_feature_cols
from inversion.tuning.tune_model import load_best_params, _walk_forward_folds
from inversion.utils import paths

SEEDS = [42, 43, 44]
# El ensemble de semillas es la parte cara (3 RF por fold). Con SKIP_ENSEMBLE=1
# la comparación se limita a RF vs HistGBM: responde a "¿es correcta la clase
# de modelo?" con una fracción del coste.
SKIP_ENSEMBLE = os.environ.get("SKIP_ENSEMBLE") == "1"
REPORTS = paths.PROJECT_DIR / "reports"
ADOPTION_MARGIN = 0.01
OUT_CSV = REPORTS / "comparacion_clases_modelo.csv"
# Cap de hilos: la máquina se comparte con otros trabajos del usuario; -1
# los monopoliza y bajo carga alta acaba siendo más lento para todos.
N_JOBS = 2


def load_ticker(ticker: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv")
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    cols = available_feature_cols(df)
    df = df.dropna(subset=cols)
    return df[cols], df["target"]


def precision_at_top_quintile(y: np.ndarray, p: np.ndarray) -> float:
    """Precisión entre el 20% de filas con mayor probabilidad predicha."""
    k = max(int(len(p) * 0.2), 1)
    top = np.argsort(p)[-k:]
    return float(y[top].mean())


def evaluate(model_factory, X: pd.DataFrame, y: pd.Series, folds) -> dict:
    aucs, accs, pq5s = [], [], []
    for tr_end, te_end in folds:
        m = model_factory()
        m.fit(X.iloc[:tr_end], y.iloc[:tr_end])
        p = m.predict_proba(X.iloc[tr_end:te_end])[:, 1]
        yt = y.iloc[tr_end:te_end].to_numpy()
        aucs.append(roc_auc_score(yt, p))
        accs.append(accuracy_score(yt, (p >= 0.5).astype(int)))
        pq5s.append(precision_at_top_quintile(yt, p))
    return {
        "auc": float(np.mean(aucs)),
        "acc": float(np.mean(accs)),
        "precision_top20": float(np.mean(pq5s)),
    }


def rf_best_factory(ticker: str):
    params = {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 50, "max_features": "sqrt"}
    params.update(load_best_params(ticker) or {})

    def make():
        return RandomForestClassifier(**params, class_weight="balanced", random_state=42, n_jobs=N_JOBS)

    return make


def rf_ensemble_probs(X_tr, y_tr, X_te, params) -> np.ndarray:
    probs = []
    for seed in SEEDS:
        m = RandomForestClassifier(**params, class_weight="balanced", random_state=seed, n_jobs=N_JOBS)
        m.fit(X_tr, y_tr)
        probs.append(m.predict_proba(X_te)[:, 1])
    return np.mean(probs, axis=0)


def histgbm_factory():
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )


def main() -> None:
    rows: list[dict] = []
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Guardado incremental: cada ticker persiste al terminar, para poder
    # lanzar en segundo plano y consultar el progreso sin esperar al total.
    pd.DataFrame(columns=["ticker", "modelo", "auc", "acc", "precision_top20"]).to_csv(OUT_CSV, index=False)
    for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]:
        X, y = load_ticker(ticker)
        folds = _walk_forward_folds(len(X))
        params = {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 50, "max_features": "sqrt"}
        params.update(load_best_params(ticker) or {})

        res_rf = evaluate(rf_best_factory(ticker), X, y, folds)

        # Ensemble de semillas: mismas folds, media de probabilidades.
        res_ens = {"auc": float("nan"), "acc": float("nan"), "precision_top20": float("nan")}
        if not SKIP_ENSEMBLE:
            aucs, accs, pq5s = [], [], []
            for tr_end, te_end in folds:
                p = rf_ensemble_probs(X.iloc[:tr_end], y.iloc[:tr_end], X.iloc[tr_end:te_end], params)
                yt = y.iloc[tr_end:te_end].to_numpy()
                aucs.append(roc_auc_score(yt, p))
                accs.append(accuracy_score(yt, (p >= 0.5).astype(int)))
                pq5s.append(precision_at_top_quintile(yt, p))
            res_ens = {"auc": float(np.mean(aucs)), "acc": float(np.mean(accs)), "precision_top20": float(np.mean(pq5s))}

        res_gb = evaluate(histgbm_factory, X, y, folds)

        for name, r in (("rf_best", res_rf), ("rf_ensemble5", res_ens), ("hist_gbm", res_gb)):
            rows.append({"ticker": ticker, "modelo": name, **{k: round(v, 4) for k, v in r.items()}})
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"{ticker}: RF={res_rf['auc']:.4f} | ENS={res_ens['auc']:.4f} | GBM={res_gb['auc']:.4f}", flush=True)

    print("DONE", flush=True)

    df = pd.DataFrame(rows)
    print("\n=== MEDIAS ===")
    medias = df.groupby("modelo")[["auc", "acc", "precision_top20"]].mean().round(4)
    print(medias.to_string())

    candidatos = [m for m in ("rf_ensemble5", "hist_gbm") if m in medias.index and not np.isnan(medias.loc[m, "auc"])]
    best_alt = medias.loc[candidatos, "auc"].idxmax()
    delta = medias.loc[best_alt, "auc"] - medias.loc["rf_best", "auc"]
    verdict = f"ADOPTAR {best_alt}" if delta > ADOPTION_MARGIN else "MANTENER rf_best"
    print(f"\nVeredicto (margen {ADOPTION_MARGIN}): mejor alternativa={best_alt} Δauc={delta:+.4f} → {verdict}")
    print(f"→ {OUT_CSV}")


if __name__ == "__main__":
    main()
