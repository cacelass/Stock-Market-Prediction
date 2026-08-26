"""pooled.py — Modelo global (pool de tickers) y enrutado híbrido (IMP-006).

IMP-005 demostró que el pool pierde frente a los modelos por ticker en AUC
(1/7) excepto para tickers sin señal propia: NVDA individual wf-AUC 0.501 vs
0.539 del pool. Este módulo convierte ese hallazgo en mecanismo:

- train_and_save(): entrena el RF global sobre el pool scale-free + one-hots
  (cutoff temporal 80/20, mismo protocolo que los individuales), guarda
  modelo+scaler+features y deriva el routing comparando walk-forwards.
- resolve_route(ticker): "GLOBAL" si el routing manda ese ticker al pool;
  el propio ticker en cualquier otro caso. Sin artefactos globales devuelve
  siempre el ticker — comportamiento idéntico al previo a IMP-006.

El routing es conservador por diseño: solo desvía un ticker al pool si su
señal individual está por debajo de ROUTE_AUC_THRESHOLD Y el pool le gana a
ese ticker. Hoy eso ocurre exactamente con NVDA.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from inversion.models.train_model import available_feature_cols
from inversion.utils import paths

GLOBAL_TICKER = "GLOBAL"
# Features cuyo nivel absoluto no es comparable entre tickers.
NON_COMPARABLE_COLS = {"ma_50", "ma_200", "log_volume"}
# Hiperparámetros ganadores del Optuna de IMP-005 (wf_auc 0.5685).
POOLED_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 16,
    "min_samples_leaf": 51,
    "max_features": "sqrt",
}
ROUTE_AUC_THRESHOLD = 0.55
TRAIN_CUTOFF_FRAC = 0.80


def _artifact_paths() -> dict[str, Any]:
    """Rutas de artefactos evaluadas en tiempo de llamada.

    No son constantes de módulo a propósito: así respetan el parcheo de rutas
    de los tests (paths.MODELS_DIR → tmp_path) en vez de capturar la ruta real
    al importar.
    """
    base = paths.MODELS_DIR
    return {
        "model": base / f"rf_{GLOBAL_TICKER}.pkl",
        "scaler": base / f"scaler_{GLOBAL_TICKER}.pkl",
        "features": base / f"global_features_{GLOBAL_TICKER}.json",
        "routing": base / f"routing_{GLOBAL_TICKER}.json",
    }


def load_pooled() -> pd.DataFrame:
    """Pool de todos los tickers con features scale-free + one-hot de ticker.

    Ordenado por fecha global: los folds y el cutoff respetan el tiempo.
    """
    frames = []
    for csv in sorted(paths.INTERIM_DATA_DIR.glob("features_*_ml_ready.csv")):
        ticker = csv.name.removeprefix("features_").removesuffix("_ml_ready.csv").upper()
        df = pd.read_csv(csv)
        cols = [c for c in available_feature_cols(df) if c not in NON_COMPARABLE_COLS]
        df = df[["timestamp", "target", *cols]].copy()
        df["ticker"] = ticker
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No hay features en data/interim/. Ejecuta make features.")
    pooled = pd.concat(frames, ignore_index=True).dropna()
    pooled["timestamp"] = pd.to_datetime(pooled["timestamp"])
    dummies = pd.get_dummies(pooled["ticker"], prefix="tkr", drop_first=True).astype(int)
    return pd.concat([pooled, dummies], axis=1).sort_values("timestamp").reset_index(drop=True)


def global_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("timestamp", "target", "ticker")]


def _derive_routing(individual_wf_csv: pd.DataFrame, pooled_cmp_csv: pd.DataFrame) -> dict[str, str]:
    """Ticker→GLOBAL si su señal individual es débil y el pool le gana."""
    ind = individual_wf_csv.set_index("ticker")["wf_auc_media"]
    cmp_ = pooled_cmp_csv.set_index("ticker")
    routing: dict[str, str] = {}
    for ticker in ind.index:
        if ticker not in cmp_.index:
            continue
        weak_own = float(ind[ticker]) < ROUTE_AUC_THRESHOLD
        pool_better = float(cmp_.loc[ticker, "wf_auc"]) > float(cmp_.loc[ticker, "ind_auc"])
        if weak_own and pool_better:
            routing[ticker] = GLOBAL_TICKER
    return routing


def train_and_save() -> dict[str, Any]:
    """Entrena el modelo global, guarda artefactos y deriva el routing.

    Requiere reports/walk_forward.csv y reports/pooled_vs_individual.csv
    (productos de IMP-003/IMP-005): el routing se deriva de evidencia, nunca
    se codifica a mano.
    """
    pooled = load_pooled()
    cols = global_feature_cols(pooled)
    X, y = pooled[cols], pooled["target"]
    split = int(len(X) * TRAIN_CUTOFF_FRAC)

    scaler = StandardScaler().fit(X.iloc[:split])
    model = RandomForestClassifier(
        **POOLED_PARAMS,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(scaler.transform(X.iloc[:split]), y.iloc[:split])

    from sklearn.metrics import accuracy_score, roc_auc_score

    proba = model.predict_proba(scaler.transform(X.iloc[split:]))[:, 1]
    test_metrics = {
        "accuracy": round(float(accuracy_score(y.iloc[split:], (proba >= 0.5).astype(int))), 4),
        "auc": round(float(roc_auc_score(y.iloc[split:], proba)), 4),
    }

    artifacts = _artifact_paths()
    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts["model"])
    joblib.dump(scaler, artifacts["scaler"])
    artifacts["features"].write_text(json.dumps({"features": cols}, indent=2), encoding="utf-8")

    ind_wf = pd.read_csv(paths.REPORTS_DIR / "walk_forward.csv")
    pooled_cmp = pd.read_csv(paths.REPORTS_DIR / "pooled_vs_individual.csv")
    routing = _derive_routing(ind_wf, pooled_cmp)
    artifacts["routing"].write_text(
        json.dumps(
            {
                "threshold_individual_auc": ROUTE_AUC_THRESHOLD,
                "regla": "ticker→GLOBAL si wf_auc individual < umbral y el pool le gana",
                "routing": routing,
                "params": POOLED_PARAMS,
                "test_metrics_cutoff80": test_metrics,
                "n_train": int(split),
                "n_test": int(len(X) - split),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"   ✔ GLOBAL entrenado ({split:,} filas, {len(cols)} features): test acc={test_metrics['accuracy']} auc={test_metrics['auc']}")
    print(f"   ✔ Routing: {routing or '— (nadie desviado al pool)'} → {artifacts['routing'].name}")
    return {"routing": routing, "metrics": test_metrics}


def global_artifacts_available() -> bool:
    a = _artifact_paths()
    return all(v.exists() for v in a.values())


def resolve_route(ticker: str) -> str:
    """Devuelve 'GLOBAL' si el routing manda el ticker al pool; si no, el propio ticker.

    Sin artefactos globales (tests, repo recién clonado) no enruta nadie:
    comportamiento idéntico al previo a IMP-006.
    """
    ticker = ticker.upper()
    if ticker == GLOBAL_TICKER or not global_artifacts_available():
        return ticker
    data: Any = json.loads(_artifact_paths()["routing"].read_text(encoding="utf-8"))
    routing: dict[str, str] = dict(data.get("routing", {})) if isinstance(data, dict) else {}
    return routing.get(ticker, ticker)


def load_global_bundle(ticker: str) -> tuple[Any, Any, list[str]]:
    """Carga modelo+scaler globales y la lista de columnas con las que fue entrenado.

    Antes de transformar, el DataFrame necesita sus columnas tkr_* (constantes
    por ticker): usar add_ticker_dummies(df, ticker, feature_cols).
    """
    a = _artifact_paths()
    model = joblib.load(a["model"])
    scaler = joblib.load(a["scaler"])
    feature_cols = json.loads(a["features"].read_text(encoding="utf-8"))["features"]
    # Fija el one-hot del ticker consultado (las demás quedan a 0 por ausencia).
    ticker_dummies = [c for c in feature_cols if c.startswith("tkr_")]
    active_dummy = f"tkr_{ticker.upper()}"
    if ticker_dummies and active_dummy not in feature_cols:
        raise ValueError(f"'{ticker}' no estaba en el pool de entrenamiento ({ticker_dummies}).")
    return model, scaler, feature_cols


def add_ticker_dummies(df: pd.DataFrame, ticker: str, feature_cols: list[str]) -> pd.DataFrame:
    """Añade las columnas tkr_* constantes que el modelo global espera."""
    df = df.copy()
    for col in feature_cols:
        if col.startswith("tkr_"):
            df[col] = int(col == f"tkr_{ticker.upper()}")
    return df


def main() -> int:
    """make pooled-global: entrena el pool y deriva el routing híbrido."""
    try:
        result = train_and_save()
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1
    print(f"   Route→GLOBAL: {list(result['routing']) or 'ninguno'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
