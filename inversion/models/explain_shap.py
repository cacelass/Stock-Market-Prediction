"""explain_shap.py — Informes SHAP del modelo multi-ticker (TMPL-003).

Genera reports/figures/shap_<TICKER>.png con el summary plot del modelo
entrenado (rf_<T>.pkl). Ejecutar con:
    make shap
o:
    .venv/bin/python inversion/models/explain_shap.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from inversion.models.train_model import available_feature_cols
from inversion.utils import paths

FIGURES_DIR = paths.PROJECT_DIR / "reports" / "figures"


def shap_summary(ticker: str, max_display: int = 15) -> str:
    """Genera el summary plot SHAP del modelo del ticker y devuelve la ruta PNG."""
    import shap

    ticker = ticker.upper()
    model_path = paths.MODELS_DIR / f"rf_{ticker}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No hay modelo para '{ticker}': {model_path}. Ejecuta make train.")

    df = pd.read_csv(paths.INTERIM_DATA_DIR / f"features_{ticker}_ml_ready.csv")
    feature_cols = available_feature_cols(df)
    X = df[feature_cols].dropna()
    # Muestra acotada: SHAP en RF es exacto con TreeExplainer pero lento en 3.7k filas.
    sample = X.sample(n=min(300, len(X)), random_state=42)
    sample = sample.reset_index(drop=True)

    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Clasificación binaria: shap_values es [valores_clase0, valores_clase1].
    values = shap_values[1] if isinstance(shap_values, list) else shap_values

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"shap_{ticker}.png"
    plt.figure(figsize=(12, 8))
    shap.summary_plot(values, sample, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out)


def shap_all() -> list[str]:
    """Genera el informe SHAP de todos los tickers con modelo entrenado."""
    written = []
    for model_path in sorted(paths.MODELS_DIR.glob("rf_*_base.pkl")):
        ticker = model_path.name.replace("rf_", "").replace("_base.pkl", "")
        try:
            written.append(shap_summary(ticker))
            print(f"   ✔ shap_{ticker}.png")
        except Exception as exc:  # noqa: BLE001 — continúa con el resto
            print(f"   ✘ {ticker}: {exc}")
    return written


if __name__ == "__main__":
    shap_all()
