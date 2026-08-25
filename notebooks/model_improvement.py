"""model_improvement.py — IMP-001: importancia de features, confianza y walk-forward.

Tres análisis honestos sobre los modelos ya entrenados (rf_<T>.pkl):

A) Feature importance: qué features usa realmente cada RandomForest.
B) Confianza: precisión real por bucket de probabilidad en el test temporal,
   y barrido de umbral de trading con el backtest del repo (costes incluidos).
C) Walk-forward: reentrenamiento expanding-window sobre el tramo final —
   la estimación de rendimiento que no mira al futuro.

Salidas: reports/importancia_features.csv, reports/confianza_thresholds.csv,
reports/walk_forward.csv, reports/importancia_<T>.png y resumen en stdout.
"""

from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from inversion.models.train_model import available_feature_cols
from inversion.trading.backtest import run_backtest
from inversion.tuning.tune_model import load_best_params
from inversion.utils import paths

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75]
COST_PER_TRADE = 0.001  # 0.1% por operación: comisión + spread mínimo
N_FOLDS_WF = 5
REPORTS = paths.PROJECT_DIR / "reports"


def _load_split(ticker: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv")
    feature_cols = available_feature_cols(df)
    df = df.dropna(subset=[*feature_cols, "target"]).sort_values("timestamp").reset_index(drop=True)
    return df, feature_cols


def feature_importance(ticker: str, model, feature_cols: list[str]) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    imp["ticker"] = ticker
    return imp.sort_values("importance", ascending=False)


def plot_importance(imp: pd.DataFrame, ticker: str) -> None:
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], color="steelblue")
    ax.set_title(f"Top 15 features — {ticker}")
    ax.set_xlabel("Importancia")
    fig.tight_layout()
    fig.savefig(REPORTS / f"importancia_{ticker}.png", dpi=110)
    plt.close(fig)


def confidence_buckets(y_test: pd.Series, probs: np.ndarray) -> pd.DataFrame:
    """Precisión real por bucket de probabilidad predicha."""
    bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 1.01)]
    rows = []
    for lo, hi in bins:
        mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        rows.append({
            "bucket": f"[{lo:.2f},{hi:.2f})",
            "n_dias": n,
            "precision_real": round(float(y_test[mask].mean()), 4) if n else np.nan,
            "pct_dias": round(n / len(probs) * 100, 1),
        })
    out = pd.DataFrame(rows)
    out.insert(0, "ticker", "")
    return out


def threshold_backtest(df: pd.DataFrame, probs_test: np.ndarray, split: int, ticker: str) -> pd.DataFrame:
    prices = df["close"].iloc[split:].reset_index(drop=True)
    rows = []
    for th in THRESHOLDS:
        res = run_backtest(
            prices=prices,
            probs=pd.Series(probs_test),
            threshold=th,
            cost_per_trade=COST_PER_TRADE,
        )
        rows.append({
            "ticker": ticker,
            "threshold": th,
            "n_trades": res.n_trades,
            "win_rate": round(res.win_rate, 4),
            "total_return": round(res.total_return, 4),
            "sharpe": round(res.sharpe, 3),
            "max_drawdown": round(res.max_drawdown, 4),
            "buy_hold_return": round(res.buy_hold_return, 4),
        })
    return pd.DataFrame(rows)


def walk_forward(ticker: str, df: pd.DataFrame, feature_cols: list[str], n_folds: int = N_FOLDS_WF) -> dict:
    """Expanding window: cada fold entrena con todo el pasado y evalúa el bloque siguiente.

    Cubre el último ~40% del histórico. Sin shuffle, sin fuga: el modelo de
    cada fold nunca ve los días que evalúa ni los posteriores.
    """
    params = load_best_params(ticker)
    hp = {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 50, "max_features": "sqrt"}
    hp.update(params or {})

    start = int(len(df) * 0.60)
    fold_size = (len(df) - start) // n_folds
    accs, aucs = [], []
    for k in range(n_folds):
        test_lo = start + k * fold_size
        test_hi = test_hi_next = start + (k + 1) * fold_size if k < n_folds - 1 else len(df)
        train, test = df.iloc[:test_lo], df.iloc[test_lo:test_hi]
        if len(train) < 250 or len(test) < 20:
            continue
        model = RandomForestClassifier(
            n_estimators=int(hp["n_estimators"]),
            max_depth=int(hp["max_depth"]),
            min_samples_leaf=int(hp["min_samples_leaf"]),
            max_features=hp["max_features"],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train[feature_cols], train["target"])
        p = model.predict_proba(test[feature_cols])[:, 1]
        accs.append(accuracy_score(test["target"], (p >= 0.5).astype(int)))
        aucs.append(roc_auc_score(test["target"], p))
    return {
        "ticker": ticker,
        "wf_accuracy_media": round(float(np.mean(accs)), 4),
        "wf_accuracy_std": round(float(np.std(accs)), 4),
        "wf_auc_media": round(float(np.mean(aucs)), 4),
        "wf_auc_std": round(float(np.std(aucs)), 4),
        "n_folds": len(accs),
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    imp_all, conf_all, bt_all, wf_all = [], [], [], []

    for ticker in TICKERS:
        print(f"\n=== {ticker} ===")
        df, feature_cols = _load_split(ticker)
        split = int(len(df) * 0.8)
        X_test, y_test = df.iloc[split:][feature_cols], df.iloc[split:]["target"]

        # A) Importancia desde el modelo entrenado
        model = joblib.load(paths.MODELS_DIR / f"rf_{ticker}.pkl")
        imp = feature_importance(ticker, model, feature_cols)
        imp_all.append(imp)
        plot_importance(imp, ticker)
        print(f"  top-3 features: {', '.join(imp.head(3)['feature'])}")

        # B) Confianza: buckets de precisión + backtest con umbral
        scaler = joblib.load(paths.MODELS_DIR / f"scaler_{ticker}.pkl")
        probs = model.predict_proba(scaler.transform(X_test))[:, 1]

        cb = confidence_buckets(y_test.to_numpy(), probs)
        cb["ticker"] = ticker
        conf_all.append(cb)

        bt = threshold_backtest(df, probs, split, ticker)
        bt_all.append(bt)
        best = bt.loc[bt["total_return"].idxmax()]
        print(
            f"  mejor umbral={best['threshold']:.2f}: ret={best['total_return']:+.1%} "
            f"(buy&hold {best['buy_hold_return']:+.1%}), trades={best['n_trades']}, "
            f"win_rate={best['win_rate']:.1%}"
        )

        # C) Walk-forward
        wf = walk_forward(ticker, df, feature_cols)
        wf_all.append(wf)
        print(
            f"  walk-forward: acc={wf['wf_accuracy_media']:.3f}±{wf['wf_accuracy_std']:.3f} "
            f"auc={wf['wf_auc_media']:.3f}±{wf['wf_auc_std']:.3f} ({wf['n_folds']} folds)"
        )

    pd.concat(imp_all).to_csv(REPORTS / "importancia_features.csv", index=False)
    pd.concat(conf_all).to_csv(REPORTS / "confianza_precision.csv", index=False)
    pd.concat(bt_all).to_csv(REPORTS / "confianza_thresholds.csv", index=False)
    pd.DataFrame(wf_all).to_csv(REPORTS / "walk_forward.csv", index=False)

    wf_df = pd.DataFrame(wf_all)
    print("\n=== RESUMEN WALK-FORWARD (estimación honesta) ===")
    print(f"accuracy media: {wf_df['wf_accuracy_media'].mean():.3f} | auc media: {wf_df['wf_auc_media'].mean():.3f}")
    print("→ CSVs en reports/: importancia_features, confianza_precision, confianza_thresholds, walk_forward")


if __name__ == "__main__":
    main()
