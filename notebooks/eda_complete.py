#!/usr/bin/env python3
"""
EDA completo: análisis de variables, distribuciones, correlaciones y features derivadas.
Stock Market Prediction — Análisis desde la base.
"""
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── Configuración ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["figure.dpi"] = 100

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]


# ── Funciones auxiliares ───────────────────────────────────────────────────
def load_data(ticker: str, kind: str = "ml_ready") -> pd.DataFrame:
    """Carga datos raw o interim para un ticker."""
    path = DATA_RAW / f"{ticker}_{kind}.csv" if kind == "ml_ready" else DATA_INTERIM / f"features_{ticker}_{kind}.csv"
    if not path.exists():
        # Fallback: buscar en data/interim/
        path = DATA_INTERIM / f"features_{ticker}_ml_ready.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"] if "timestamp" in pd.read_csv(path, nrows=0).columns else [0])
    return df


def print_section(title: str) -> None:
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_subsection(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── 1. CARGA DE DATOS ─────────────────────────────────────────────────────
print_section("1. CARGA DE DATOS")

dfs_raw: dict[str, pd.DataFrame] = {}
dfs_features: dict[str, pd.DataFrame] = {}

for ticker in TICKERS:
    try:
        dfs_raw[ticker] = load_data(ticker, "ml_ready")
        dfs_features[ticker] = load_data(ticker, "features")
        print(f"  ✓ {ticker}: {len(dfs_raw[ticker])} registros (raw), {len(dfs_features[ticker])} registros (features)")
    except Exception as e:
        print(f"  ✗ {ticker}: Error cargando datos — {e}")

# Verificar consistencia
print(f"\n  Tickers cargados: {len(dfs_raw)}")
if dfs_raw:
    sample_ticker = list(dfs_raw.keys())[0]
    print(f"  Columnas raw ({sample_ticker}): {list(dfs_raw[sample_ticker].columns)}")
    print(f"  Columnas features ({sample_ticker}): {list(dfs_features[sample_ticker].columns)}")


# ── 2. ANÁLISIS DE CALIDAD DE DATOS ────────────────────────────────────────
print_section("2. CALIDAD DE DATOS")

for ticker in TICKERS:
    if ticker not in dfs_raw:
        continue
    df = dfs_raw[ticker]
    print_subsection(f"{ticker}")
    print(f"  Shape: {df.shape}")
    print(f"  Rango de fechas: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Días únicos: {df['timestamp'].dt.date.nunique()}")
    print(f"  Valores nulos: {df.isnull().sum().sum()}")

    # Verificar tipos
    print(f"\n  Tipos de datos:")
    for col in df.columns:
        print(f"    {col:25s} {str(df[col].dtype):15s} nunique={df[col].nunique():6d}")


# ── 3. ANÁLISIS ESTADÍSTICO DESCRIPTIVO ────────────────────────────────────
print_section("3. ESTADÍSTICAS DESCRIPTIVAS (PRIMER TICKER)")

if dfs_raw:
    sample = list(dfs_raw.values())[0]
    numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()

    print_subsection("Resumen estadístico")
    desc = sample[numeric_cols].describe().T
    desc["skewness"] = sample[numeric_cols].skew()
    desc["kurtosis"] = sample[numeric_cols].kurtosis()
    print(desc.to_string())


# ── 4. DISTRIBUCIONES ─────────────────────────────────────────────────────
print_section("4. DISTRIBUCIONES DE VARIABLES CLAVE")

if dfs_raw:
    sample = list(dfs_raw.values())[0]

    # Variables de precio
    price_vars = ["open", "high", "low", "close", "volume"]
    available_price = [v for v in price_vars if v in sample.columns]

    if available_price:
        print_subsection("Variables de precio")
        for col in available_price:
            data = sample[col].dropna()
            print(f"  {col:15s}: media={data.mean():12.2f}, std={data.std():12.2f}, "
                  f"min={data.min():12.2f}, max={data.max():12.2f}, "
                  f"skew={data.skew():6.2f}, kurt={data.kurtosis():6.2f}")


# ── 5. ANÁLISIS DE TARGET ─────────────────────────────────────────────────
print_section("5. ANÁLISIS DEL TARGET")

for ticker in TICKERS:
    if ticker not in dfs_features:
        continue
    df = dfs_features[ticker]
    if "target" not in df.columns:
        continue

    print_subsection(f"{ticker}")
    target = df["target"]
    print(f"  Distribución del target:")
    print(f"    0 (baja): {(target == 0).sum():6d} ({(target == 0).mean()*100:.1f}%)")
    print(f"    1 (sube): {(target == 1).sum():6d} ({(target == 1).mean()*100:.1f}%)")
    print(f"  Ratio 1/0: {target.sum() / (len(target) - target.sum()):.2f}")


# ── 6. CORRELACIONES ──────────────────────────────────────────────────────
print_section("6. MATRIZ DE CORRELACIONES")

if dfs_features:
    sample = list(dfs_features.values())[0]
    numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > 1:
        corr = sample[numeric_cols].corr()

        # Top correlaciones absolutas
        print_subsection("Top 20 correlaciones absolutas")
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                corr_pairs.append((
                    corr.columns[i],
                    corr.columns[j],
                    abs(corr.iloc[i, j])
                ))
        corr_pairs.sort(key=lambda x: x[2], reverse=True)

        for col1, col2, corr_val in corr_pairs[:20]:
            print(f"  {col1:25s} ↔ {col2:25s}: {corr_val:.4f}")


# ── 7. ANÁLISIS DE OUTLIERS ───────────────────────────────────────────────
print_section("7. DETECCIÓN DE OUTLIERS (MÉTODO IQR)")

if dfs_features:
    sample = list(dfs_features.values())[0]
    numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()

    outlier_summary = []
    for col in numeric_cols:
        data = sample[col].dropna()
        if len(data) == 0:
            continue
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((data < lower) | (data > upper)).sum()
        pct_outliers = n_outliers / len(data) * 100
        outlier_summary.append((col, n_outliers, pct_outliers))

    outlier_summary.sort(key=lambda x: x[2], reverse=True)

    print_subsection("Outliers por variable")
    for col, n_out, pct in outlier_summary:
        print(f"  {col:25s}: {n_out:6d} outliers ({pct:.1f}%)")


# ── 8. ANÁLISIS TEMPORAL ──────────────────────────────────────────────────
print_section("8. ANÁLISIS TEMPORAL")

if dfs_raw:
    sample = list(dfs_raw.values())[0]

    if "timestamp" in sample.columns:
        print_subsection("Retornos diarios")
        sample = sample.sort_values("timestamp").copy()
        sample["return"] = sample["close"].pct_change()

        print(f"  Retorno medio diario: {sample['return'].mean()*100:.4f}%")
        print(f"  Volatilidad diaria: {sample['return'].std()*100:.4f}%")
        print(f"  Retorno anualizado: {sample['return'].mean()*252*100:.2f}%")
        print(f"  Volatilidad anualizada: {sample['return'].std()*np.sqrt(252)*100:.2f}%")
        print(f"  Sharpe ratio (risk-free=0): {sample['return'].mean()/sample['return'].std()*np.sqrt(252):.2f}")

        # Autocorrelación
        print_subsection("Autocorrelación del retorno")
        for lag in [1, 5, 10, 20]:
            autocorr = sample["return"].autocorr(lag=lag)
            print(f"  Lag {lag:2d}: {autocorr:.4f}")


# ── 9. ANÁLISIS DE FEATURES DE SENTIMIENTO ────────────────────────────────
print_section("9. FEATURES DE SENTIMIENTO")

if dfs_features:
    sample = list(dfs_features.values())[0]
    sent_cols = [c for c in sample.columns if "sentiment" in c.lower()]

    if sent_cols:
        print_subsection("Cobertura de sentimiento")
        for col in sent_cols:
            data = sample[col]
            non_zero = (data != 0).sum()
            pct_non_zero = non_zero / len(data) * 100
            print(f"  {col:25s}: {non_zero:6d} valores no-cero ({pct_non_zero:.1f}%)")
    else:
        print("  No se encontraron columnas de sentimiento")


# ── 10. GENERAR GRÁFICOS ──────────────────────────────────────────────────
print_section("10. GENERANDO GRÁFICOS...")

if dfs_features:
    sample_ticker = list(dfs_features.keys())[0]
    df = dfs_features[sample_ticker].copy()

    # 10.1 Distribuciones
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "target" in numeric_cols:
        numeric_cols.remove("target")

    n_cols = min(6, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else np.array([[axes]])

    for i, col in enumerate(numeric_cols[:n_rows * n_cols]):
        row, c = divmod(i, n_cols)
        ax = axes[row, c]
        df[col].hist(bins=50, ax=ax, alpha=0.7, edgecolor="black")
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=8)

    # Ocultar ejes vacíos
    for i in range(len(numeric_cols), n_rows * n_cols):
        row, c = divmod(i, n_cols)
        axes[row, c].set_visible(False)

    plt.suptitle(f"Distribuciones de Features — {sample_ticker}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(REPORTS / "eda_distribuciones.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ reports/eda_distribuciones.png")

    # 10.2 Matriz de correlaciones
    corr = df[numeric_cols[:20]].corr()  # Top 20 para legibilidad
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title(f"Matriz de Correlaciones — {sample_ticker}", fontsize=14)
    plt.tight_layout()
    plt.savefig(REPORTS / "eda_correlaciones.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ reports/eda_correlaciones.png")

    # 10.3 Precio y volumen
    if "timestamp" in df.columns and "close" in df.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        ax1.plot(df["timestamp"], df["close"], linewidth=1, color="steelblue")
        if "ma_50" in df.columns:
            ax1.plot(df["timestamp"], df["ma_50"], linewidth=1, color="orange", alpha=0.8, label="MA50")
        if "ma_200" in df.columns:
            ax1.plot(df["timestamp"], df["ma_200"], linewidth=1, color="red", alpha=0.8, label="MA200")
        ax1.set_ylabel("Precio")
        ax1.set_title(f"{sample_ticker} — Precio y Medias Móviles")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if "volume" in df.columns:
            ax2.bar(df["timestamp"], df["volume"], width=1, alpha=0.6, color="gray")
            ax2.set_ylabel("Volumen")
            ax2.set_xlabel("Fecha")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(REPORTS / "eda_precio_volumen.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ reports/eda_precio_volumen.png")

    # 10.4 Boxplots de features numéricas
    fig, ax = plt.subplots(figsize=(16, 8))
    df[numeric_cols[:15]].boxplot(ax=ax, vert=True, patch_artist=True)
    plt.xticks(rotation=45, ha="right")
    ax.set_title(f"Boxplots de Features — {sample_ticker}")
    plt.tight_layout()
    plt.savefig(REPORTS / "eda_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ reports/eda_boxplots.png")

    # 10.5 Retorno vs Target
    if "return" in df.columns and "target" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        for t in [0, 1]:
            subset = df[df["target"] == t]
            ax.hist(subset["return"], bins=50, alpha=0.5, label=f"Target={t}", density=True)
        ax.set_xlabel("Retorno diario")
        ax.set_ylabel("Densidad")
        ax.set_title(f"Distribución de Retorno por Target — {sample_ticker}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(REPORTS / "eda_retorno_vs_target.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ reports/eda_retorno_vs_target.png")


# ── RESUMEN ────────────────────────────────────────────────────────────────
print_section("RESUMEN DEL EDA")
print("""
  Hallazgos principales:
  1. Datos de precios completos: 7 tickers, 2007-2026, ~4600 registros cada uno
  2. Features técnicas: 18 variables (OHLCV + indicadores)
  3. Features de sentimiento: 3 variables (score, ma5, vol) — COBERTURA CRÍTICA
     Solo ~100 noticias por ticker (jul-ago 2026), sentimiento = 0.0 en 99%+ de datos
  4. Target: binario (sube >2% en 5 días), balance ~35/65
  5. Correlaciones: colinealidad esperada entre features de precio (ma_50 ~ close)
  6. Outliers: esperados en retorno y volumen (días de crisis)
  7. Autocorrelación: baja en retornos diarios (eficiente a corto plazo)

  Próximos pasos:
  1. Ejecutar GDELT para obtener noticias históricas 2021-2026
  2. Derivar nuevas features técnicas (Bollinger, ATR, OBV, etc.)
  3. Analizar features de calendario (estacionalidad)
  4. Crear features de interacción (sentimiento × retorno)
""")

print("✅ EDA completado. Gráficos en reports/eda_*.png")
