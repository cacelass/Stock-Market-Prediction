"""pooled_model.py — IMP-005: modelo global pooled vs modelos por ticker.

Hipótesis: 7 tickers × ~3.5k filas = ~27k filas compartiendo mecánica de
mercado; un único RandomForest entrenado sobre todas puede generalizar mejor
que los modelos individuales, especialmente donde hay poca señal (NVDA, AAPL).

Diseño:
- Solo features scale-free (retornos, ratios, osciladores): ma_50, ma_200 y
  log_volume se excluyen porque el nivel de precio/volumen no es comparable
  entre tickers. One-hot de ticker para que el modelo sepa dónde está.
- Walk-forward honesto: filas ordenadas por fecha global; folds expanding-window
  (60% → 100%, 3 folds) — el mismo protocolo que los modelos por ticker.
- Optuna corto (15 trials) sobre la media de AUC de los folds.

Salidas: reports/pooled_walk_forward.csv (por ticker), veredicto en stdout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from inversion.models.train_model import available_feature_cols
from inversion.utils import paths

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
# Nivel absoluto de precio/volumen: no comparable entre tickers → fuera.
NON_COMPARABLE = {"ma_50", "ma_200", "log_volume"}
WF_FOLDS = 3
WF_START_FRAC = 0.60
N_TRIALS = 15
RANDOM_STATE = 42

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_pooled() -> pd.DataFrame:
    frames = []
    for ticker in TICKERS:
        csv = paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv"
        df = pd.read_csv(csv)
        cols = available_feature_cols(df)
        keep = [c for c in cols if c not in NON_COMPARABLE]
        df = df[["timestamp", "target", *keep]].copy()
        df["ticker"] = ticker
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True).dropna()
    pooled["timestamp"] = pd.to_datetime(pooled["timestamp"])
    # One-hots al final; drop_first evita colinealidad perfecta.
    dummies = pd.get_dummies(pooled["ticker"], prefix="tkr", drop_first=True).astype(int)
    pooled = pd.concat([pooled, dummies], axis=1)
    return pooled.sort_values("timestamp").reset_index(drop=True)


def wf_folds(n: int, n_folds: int = WF_FOLDS, start_frac: float = WF_START_FRAC):
    start = int(n * start_frac)
    size = (n - start) // n_folds
    return [
        (start + k * size, start + (k + 1) * size if k < n_folds - 1 else n)
        for k in range(n_folds)
    ]


def tune_pooled(X: pd.DataFrame, y: pd.Series, folds) -> dict:
    def objective(trial: optuna.Trial) -> float:
        scores = []
        for tr_end, te_end in folds:
            m = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
                max_depth=trial.suggest_int("max_depth", 4, 16),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 60),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            m.fit(X.iloc[:tr_end], y.iloc[:tr_end])
            p = m.predict_proba(X.iloc[tr_end:te_end])[:, 1]
            scores.append(roc_auc_score(y.iloc[tr_end:te_end], p))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"Optuna pooled: wf_auc={study.best_value:.4f} params={study.best_params}")
    return study.best_params


def evaluate_pooled(params: dict, X: pd.DataFrame, y: pd.Series, folds, tickers_idx: pd.Series):
    """Métricas walk-forward del modelo pool: global y por ticker."""
    rows = []
    per_ticker = {t: {"y": [], "p": []} for t in TICKERS}
    for tr_end, te_end in folds:
        m = RandomForestClassifier(**params, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X.iloc[:tr_end], y.iloc[:tr_end])
        p = m.predict_proba(X.iloc[tr_end:te_end])[:, 1]
        yt = y.iloc[tr_end:te_end].to_numpy()
        rows.append({"auc": roc_auc_score(yt, p), "acc": accuracy_score(yt, (p >= 0.5).astype(int))})
        tk = tickers_idx.iloc[tr_end:te_end].to_numpy()
        for t in TICKERS:
            mask = tk == t
            if mask.sum():
                per_ticker[t]["y"].append(yt[mask])
                per_ticker[t]["p"].append(p[mask])

    g = pd.DataFrame(rows).mean().to_dict()
    out = [{"scope": "POOLED-GLOBAL", "wf_accuracy": round(g["acc"], 4), "wf_auc": round(g["auc"], 4), "n_test": int(sum(len(x) for x in per_ticker["AAPL"]["y"]) * len(TICKERS))}]
    for t in TICKERS:
        yy = np.concatenate(per_ticker[t]["y"])
        pp = np.concatenate(per_ticker[t]["p"])
        out.append({
            "scope": t,
            "wf_accuracy": round(float(accuracy_score(yy, (pp >= 0.5).astype(int))), 4),
            "wf_auc": round(float(roc_auc_score(yy, pp)), 4),
            "n_test": int(len(yy)),
        })
    return pd.DataFrame(out)


def main() -> None:
    pooled = load_pooled()
    feature_cols = [c for c in pooled.columns if c not in ("timestamp", "target", "ticker")]
    X, y = pooled[feature_cols], pooled["target"]
    folds = wf_folds(len(X))
    print(f"Pooled: {len(X):,} filas × {len(feature_cols)} features | folds: {folds}")

    params = tune_pooled(X, y, folds)
    result = evaluate_pooled(params, X, y, folds, pooled["ticker"])

    # Comparación contra los modelos por-ticker actuales (mismo protocolo).
    per_ticker = pd.read_csv(paths.PROJECT_DIR / "reports" / "walk_forward.csv")
    per_ticker = per_ticker.rename(columns={"wf_accuracy_media": "ind_accuracy", "wf_auc_media": "ind_auc"})[
        ["ticker", "ind_accuracy", "ind_auc"]
    ]
    comp = result[result.scope != "POOLED-GLOBAL"].merge(per_ticker, left_on="scope", right_on="ticker")
    comp["gana_pool_acc"] = comp.wf_accuracy > comp.ind_accuracy
    comp["gana_pool_auc"] = comp.wf_auc > comp.ind_auc

    out = paths.REPORTS_DIR / "pooled_vs_individual.csv"
    comp.to_csv(out, index=False)
    print("\n=== POOLED vs POR-TICKER (walk-forward) ===")
    print(comp.to_string(index=False))
    print(f"\nPool gana en accuracy: {comp.gana_pool_acc.sum()}/7 | en AUC: {comp.gana_pool_auc.sum()}/7")
    print(f"Global pool: acc={result.iloc[0].wf_accuracy}, auc={result.iloc[0].wf_auc}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
