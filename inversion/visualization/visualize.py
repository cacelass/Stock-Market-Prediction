import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any

import numpy as np
import pandas as pd


def plot_price(df: pd.DataFrame, ticker: str = "", output_dir: str = "./data/processed") -> None:
    """Grafica el precio de cierre con medias móviles (MA50 y MA200)."""
    if "timestamp" not in df.columns:
        df = df.reset_index()

    plt.figure(figsize=(14, 6))
    plt.plot(df["timestamp"], df["close"], label="Close", color="steelblue", linewidth=1)
    if "ma_50" in df.columns:
        plt.plot(df["timestamp"], df["ma_50"], label="MA 50", color="orange", linewidth=1.2)
    if "ma_200" in df.columns:
        plt.plot(df["timestamp"], df["ma_200"], label="MA 200", color="green", linewidth=1.2)

    plt.title(f"{ticker} — Precio de cierre y medias móviles")
    plt.xlabel("Fecha")
    plt.ylabel("Precio ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{output_dir}/{ticker}_price_plot.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✔ Gráfico de precio guardado en {filename}")


def plot_predictions(df: pd.DataFrame, y_true: Any, y_pred: Any, title: str = "Señales: Real vs Predicción", output_dir: str = "./data/processed") -> None:
    """
    Compara la clase real (0/1) con la predicha en el periodo de test.
    Muestra aciertos en verde y errores en rojo.
    """
    if "timestamp" not in df.columns:
        df = df.reset_index()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    dates = df["timestamp"].values[-n:]
    correct = y_true == y_pred

    plt.figure(figsize=(14, 4))
    plt.scatter(dates[correct], y_true[correct], c="green", s=8, label="Acierto", alpha=0.6)
    plt.scatter(dates[~correct], y_true[~correct], c="red", s=8, label="Error", alpha=0.6)
    plt.yticks([0, 1], ["Baja (0)", "Sube (1)"])
    plt.title(title)
    plt.xlabel("Fecha")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{output_dir}/predictions_plot.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✔ Gráfico de predicciones guardado en {filename}")


def plot_feature_importance(model: Any, feature_names: list[str], top_n: int = 15, output_dir: str = "./data/processed") -> None:
    """Visualiza las features más importantes del modelo."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[sorted_idx], y=np.array(feature_names)[sorted_idx], palette="viridis")
    plt.title(f"Top {top_n} features por importancia")
    plt.xlabel("Importancia")
    plt.ylabel("Feature")
    plt.tight_layout()

    filename = f"{output_dir}/feature_importance.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✔ Gráfico de importancia de features guardado en {filename}")


def plot_returns_distribution(df: pd.DataFrame, output_dir: str = "./data/processed") -> None:
    """
    Grafica la distribución del retorno diario para detectar sesgos u outliers.
    Usa la columna 'return' (continua), no el target binario.
    """
    if "return" not in df.columns:
        print("⚠ No existe columna 'return' en el DataFrame. Omitiendo gráfico.")
        return

    plt.figure(figsize=(10, 5))
    sns.histplot(df["return"].dropna(), bins=60, kde=True, color="purple")
    plt.title("Distribución de retornos diarios")
    plt.xlabel("Retorno diario")
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{output_dir}/returns_distribution.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✔ Gráfico de distribución de retornos guardado en {filename}")
