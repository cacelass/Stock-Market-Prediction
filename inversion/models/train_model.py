from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from inversion.features.build_features import fit_and_save_scaler
from inversion.utils import paths

# Features usadas por el modelo — deben coincidir con predict_model.py
FEATURE_COLS = [
    "return",
    "volatility",
    "rsi",
    "ma_50",
    "ma_200",
    "hl_range",
    "oc_range",
    "log_volume",
    "vwap_ratio",
    "lag_1",
    "lag_5",
    "lag_20",
]

# Features de sentimiento (SENT-003) — se usan solo si el dataset las trae
SENTIMENT_COLS = ["sentiment_score", "sentiment_ma5", "sentiment_vol"]


def train_rf_model(
    X_train: Any,
    y_train: Any,
    filename: str | None = None,
    random_state: int = 42,
    params: dict[str, Any] | None = None,
) -> RandomForestClassifier:
    """
    Entrena un RandomForestClassifier y lo guarda en disco.

    Args:
        X_train      : Features de entrenamiento (array o DataFrame).
        y_train      : Target binario (0 = baja, 1 = sube).
        filename     : Nombre de archivo para el modelo (opcional).
        random_state : Semilla para reproducibilidad.
        params       : Hiperparámetros de Optuna (TMPL-003). Si es None, usa
                       los defaults del pipeline (max_depth=5, min_samples_leaf=50).

    Returns:
        Modelo entrenado.
    """
    print(f"      ...Configurando RandomForestClassifier (semilla={random_state})")

    default_params = {
        "n_estimators": 200,
        "max_depth": 5,  # profundidad baja: el mercado tiene poca señal
        "min_samples_leaf": 50,  # no divide con pocos ejemplos → generaliza más
        "max_features": "sqrt",
    }
    hp = {**default_params, **(params or {})}
    if params:
        print(f"      ...usando hiperparámetros de Optuna: {params}")

    model = RandomForestClassifier(
        n_estimators=int(hp["n_estimators"]),
        max_depth=int(hp["max_depth"]),
        min_samples_leaf=int(hp["min_samples_leaf"]),
        max_features=hp["max_features"],
        class_weight="balanced",  # compensa desbalance de clases
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    save_path = paths.MODEL_FILE if filename is None else paths.MODELS_DIR / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)

    return model


def evaluate_model(model: Any, X_test: Any, y_test: Any) -> dict[str, Any]:
    """
    Evalúa el modelo con métricas de clasificación.

    Returns:
        dict con accuracy, auc y el classification_report completo.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["Baja (0)", "Sube (1)"])

    return {"accuracy": acc, "auc": auc, "report": report}


def train_ticker(ticker: str, csv_path: Path, random_state: int = 42, with_sentiment: bool = True, model_suffix: str = "") -> dict[str, Any]:
    """Entrena el modelo de un ticker con split temporal honesto y seed fija.

    Primer 80% (orden cronológico) entrena, último 20% evalúa. Sin shuffle,
    sin fuga del futuro. Guarda models/rf_<T><suffix>.pkl (+ scaler_<T><suffix>.pkl)
    y devuelve las métricas. Si with_sentiment=True y el dataset trae columnas
    sentiment_*, se incluyen como features (MODEL-002). model_suffix permite
    guardar el baseline aparte (suffix="_base").
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    feature_cols = list(FEATURE_COLS)
    if with_sentiment and all(c in df.columns for c in SENTIMENT_COLS):
        feature_cols += SENTIMENT_COLS
    df = df.dropna(subset=[*feature_cols, "target"]).sort_values("timestamp")

    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    scaler = fit_and_save_scaler(X_train, filename=f"scaler_{ticker}{model_suffix}.pkl")
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Import perezoso: evita el ciclo train_model ↔ tuning (tune_model usa FEATURE_COLS).
    from inversion.tuning.tune_model import load_best_params

    model = train_rf_model(
        X_train_s,
        y_train,
        filename=f"rf_{ticker}{model_suffix}.pkl",
        random_state=random_state,
        params=load_best_params(ticker),
    )
    metrics = evaluate_model(model, X_test_s, y_test)
    train_acc = accuracy_score(y_train, model.predict(X_train_s))
    metrics.update(
        {"ticker": ticker, "n_train": int(len(X_train)), "n_test": int(len(X_test)), "train_accuracy": float(train_acc), "with_sentiment": with_sentiment}
    )
    return metrics


def save_metrics(ticker: str, metrics: dict[str, Any]) -> Path:
    """Persiste las métricas en reports/resultados_<TICKER>.csv."""
    reports_dir = paths.PROJECT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"resultados_{ticker}.csv"
    pd.DataFrame([{k: v for k, v in metrics.items() if k != "report"}]).to_csv(out, index=False)
    return out


def compare_sentiment(rows: list[dict[str, Any]]) -> Path:
    """Escribe reports/comparacion_sentimiento.csv con baseline vs con-sentimiento.

    El baseline se entrena SIN columnas de sentimiento y se guarda como
    rf_<T>_base.pkl para no pisar el modelo principal (MODEL-002).
    """
    reports_dir = paths.PROJECT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "comparacion_sentimiento.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main() -> None:
    """Entrena el RandomForest para cada ticker con features y registra métricas.

    Reproducible: seed fija y mismo input → mismo output. Los modelos y las
    métricas viven en models/ y reports/. Cuando hay sentimiento disponible
    (SENT-003), entrena también el baseline sin sentimiento y deja un informe
    comparativo reports/comparacion_sentimiento.csv.
    """
    paths.INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(paths.INTERIM_DATA_DIR.glob("features_*.csv"))
    if not ckpts:
        print("No hay features en data/interim/. Ejecuta 'make features' primero.")
        return

    comparison: list[dict[str, Any]] = []
    for csv_path in ckpts:
        name = csv_path.name.removeprefix("features_").removesuffix(".csv")
        ticker = name.rsplit("_ml_ready", 1)[0].upper()

        df_probe = pd.read_csv(csv_path, nrows=5)
        has_sentiment = all(c in df_probe.columns for c in SENTIMENT_COLS)

        metrics = train_ticker(ticker, csv_path, with_sentiment=True)
        save_metrics(ticker, metrics)
        print(
            f"   ✔ {ticker}: accuracy={metrics['accuracy']:.3f} auc={metrics['auc']:.3f} "
            f"(n_train={metrics['n_train']}, n_test={metrics['n_test']}) → resultados_{ticker}.csv"
        )

        if has_sentiment:
            base = train_ticker(ticker, csv_path, with_sentiment=False, model_suffix="_base")
            comparison.append(
                {
                    "ticker": ticker,
                    "baseline_accuracy": base["accuracy"],
                    "baseline_auc": base["auc"],
                    "sentiment_accuracy": metrics["accuracy"],
                    "sentiment_auc": metrics["auc"],
                    "delta_accuracy": round(metrics["accuracy"] - base["accuracy"], 4),
                    "delta_auc": round(metrics["auc"] - base["auc"], 4),
                }
            )
            print(f"       ↳ baseline sin sentimiento: accuracy={base['accuracy']:.3f} auc={base['auc']:.3f}")

    if comparison:
        out = compare_sentiment(comparison)
        print(f"   ✔ Comparación baseline vs sentimiento → {out.name}")


if __name__ == "__main__":
    main()
