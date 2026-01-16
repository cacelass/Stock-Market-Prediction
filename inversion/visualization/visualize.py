# visualize.py
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_price(df, ticker="", output_dir="./data/processed"):
    """
    Grafica el precio de cierre con medias móviles y guarda la figura.
    df debe contener al menos 'timestamp' y 'close'.
    """
    if 'timestamp' not in df.columns:
        df = df.reset_index()  # Si el índice es datetime, lo ponemos como columna

    plt.figure(figsize=(14,6))
    plt.plot(df['timestamp'], df['close'], label='Close', color='blue')
    if 'ma_50' in df.columns:
        plt.plot(df['timestamp'], df['ma_50'], label='MA50', color='orange')
    if 'ma_200' in df.columns:
        plt.plot(df['timestamp'], df['ma_200'], label='MA200', color='green')
    plt.title(f"{ticker} - Precio de cierre y medias móviles")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.legend()
    plt.grid(True)

    filename = f"{output_dir}/{ticker}_price_plot.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✔ Gráfico de precio guardado en {filename}")

def plot_predictions(df, y_true, y_pred, title="Predicciones vs Real", output_dir="./data/processed"):
    """
    Compara valores reales vs predicciones y guarda la figura.
    df debe contener al menos 'timestamp' para el eje x.
    """
    if 'timestamp' not in df.columns:
        df = df.reset_index()

    plt.figure(figsize=(14,6))
    plt.plot(df['timestamp'].iloc[:len(y_true)], y_true, label='Real', color='blue')
    plt.plot(df['timestamp'].iloc[:len(y_pred)], y_pred, label='Predicción', color='red')
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Log Return")
    plt.legend()
    plt.grid(True)

    filename = f"{output_dir}/predictions_plot.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✔ Gráfico de predicciones guardado en {filename}")

def plot_feature_importance(model, feature_names, top_n=15, output_dir="./data/processed"):
    """
    Visualiza la importancia de features de un modelo tipo Random Forest.
    args:
        model: Modelo entrenado con atributo feature_importances_
        feature_names: Lista de nombres de features
        top_n: Número de features más importantes a mostrar
    """
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10,6))
    sns.barplot(x=importances[sorted_idx], y=np.array(feature_names)[sorted_idx])
    plt.title("Top {} Features por importancia".format(top_n))
    plt.xlabel("Importancia")
    plt.ylabel("Feature")

    filename = f"{output_dir}/feature_importance.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✔ Gráfico de importancia de features guardado en {filename}")

def plot_returns_distribution(df, target_col="target", output_dir="./data/processed"):
    """
    Grafica la distribución de los retornos (target) para ver sesgos o outliers.
    args:
        df: DataFrame con columna target_col
        target_col: Nombre de la columna objetivo
    """
    plt.figure(figsize=(10,5))
    sns.histplot(df[target_col], bins=50, kde=True, color='purple')
    plt.title(f"Distribución de {target_col}")
    plt.xlabel("Log Return")
    plt.ylabel("Frecuencia")
    plt.grid(True)

    filename = f"{output_dir}/{target_col}_distribution.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✔ Gráfico de distribución de retornos guardado en {filename}")
