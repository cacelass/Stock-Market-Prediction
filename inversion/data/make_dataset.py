import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from inversion.features.build_features import add_derived_features
from inversion.utils import paths
import yfinance as yf

# Catálogo multi-ticker: qué descarga `make data` y en qué rango de fechas.
TICKERS_CONFIG = paths.PROJECT_DIR / "config" / "tickers.yaml"
# Informes de validación por ticker (nulos, rango de fechas, tamaño).
VALIDATION_DIR = paths.PROJECT_DIR / "reports" / "data_validation"

# Target binario: "sube >2% en 5 días hábiles".
TARGET_HORIZON_DAYS = 5
TARGET_THRESHOLD = 0.02

# Esquema de los datasets ml_ready (idéntico a data/raw/NVDA_ml_ready.csv).
ML_READY_COLUMNS = [
    "timestamp",
    "close",
    "high",
    "low",
    "open",
    "volume",
    "return",
    "volatility",
    "rsi",
    "hl_range",
    "oc_range",
    "ma_50",
    "ma_200",
    "log_volume",
    "vwap",
    "vwap_ratio",
    "lag_1",
    "lag_5",
    "lag_20",
    "target",
]


def get_yfinance_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Descarga 20 años de historia diaria usando Yahoo Finance
    Retorna un DataFrame con columnas:
    ['timestamp', 'open', 'high', 'low', 'close', 'volume']"""
    print(f"   ⬇ Descargando YFinance para {ticker}...")
    try:
        # Descarga
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

        # Aplanar MultiIndex si existe (corrección común en versiones recientes de yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        # Renombrar columnas a minúsculas estándar
        df = df.rename(columns={"Date": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

        # Asegurar formato fecha
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        print(f"   ⚠ Error con YFinance: {e}")
        return pd.DataFrame()


def load_data(ticker: str | None = None) -> pd.DataFrame | None:
    """Carga los datos crudos desde el archivo CSV.
    Usa la ruta definida en paths.py"""
    try:
        # Usa la variable RAW_DATA_FILE definida en paths.py
        df = pd.read_csv(paths.RAW_DATA_FILE, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en {paths.RAW_DATA_FILE}")
        return None
    except Exception as e:
        print(f"❌ Error inesperado cargando datos: {e}")
        return None


def load_catalog(config_path: Path | None = None) -> dict[str, Any]:
    """Carga el catálogo de tickers y el rango de fechas desde un fichero YAML.

    El formato esperado es el de config/tickers.yaml: claves `start`, `end` y
    `tickers` (lista). Lanza ValueError si el fichero falta o no define tickers.
    """
    path = config_path or TICKERS_CONFIG
    if not path.exists():
        raise ValueError(f"No se encontró el fichero de catálogo: {path}")
    with path.open("r", encoding="utf-8") as fh:
        catalog = yaml.safe_load(fh)
    if not isinstance(catalog, dict) or not catalog.get("tickers"):
        raise ValueError(f"El catálogo '{path}' no define 'tickers' (¿fichero vacío o mal formado?)")
    for key in ("start", "end"):
        if not catalog.get(key):
            raise ValueError(f"El catálogo '{path}' no define '{key}'")
    return catalog


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Descarga el OHLCV diario de un ticker.

    Lanza RuntimeError si la descarga viene vacía (p.ej. sin internet o ticker
    inexistente) en vez de devolver silenciosamente un DataFrame vacío.
    """
    df = get_yfinance_history(ticker, start, end)
    if df.empty:
        raise RuntimeError(f"No se pudo descargar '{ticker}' ({start} → {end}). ¿Hay conexión a internet? Yahoo Finance puede estar caído o bloqueado.")
    return df


def build_ml_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el dataset ml_ready con el mismo esquema que NVDA_ml_ready.csv.

    Features derivadas (add_derived_features) + target binario "sube >2% en
    5 días". Las últimas TARGET_HORIZON_DAYS filas no tienen ventana futura,
    así que se descartan (el target sería desconocido).
    """
    df = add_derived_features(df)
    future_return = df["close"].shift(-TARGET_HORIZON_DAYS) / df["close"] - 1
    has_future = future_return.notna()
    df = df.loc[has_future].copy()
    df["target"] = (future_return[has_future] > TARGET_THRESHOLD).astype(int)
    # Descarta las filas de calentamiento (ventanas móviles NaN) para dejar un
    # dataset sin nulos, igual que data/raw/NVDA_ml_ready.csv.
    df = df.dropna().reset_index(drop=True)
    return df[ML_READY_COLUMNS]


def validate_dataset(df: pd.DataFrame, ticker: str) -> dict[str, Any]:
    """Informe de validación de un dataset ml_ready: nulos, rango y tamaño."""
    return {
        "ticker": ticker,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "nulls": int(df.isnull().sum().sum()),
        "nulls_by_column": {str(col): int(v) for col, v in df.isnull().sum().items()},
        "date_min": str(pd.to_datetime(df["timestamp"]).min().date()),
        "date_max": str(pd.to_datetime(df["timestamp"]).max().date()),
        "target_mean": round(float(df["target"].mean()), 4),
    }


def save_validation_report(report: dict[str, Any]) -> Path:
    """Escribe el informe de validación de un ticker en reports/data_validation/."""
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    out = VALIDATION_DIR / f"{report['ticker']}_validation.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def download_catalog(config_path: Path | None = None) -> list[Path]:
    """Descarga todos los tickers del catálogo y deja data/raw/<TICKER>_ml_ready.csv.

    Primero descarga y construye TODOS los datasets; solo escribe los CSV si
    todas las descargas han ido bien. Si una falla (p.ej. sin internet), lanza
    RuntimeError sin haber tocado ningún fichero.
    """
    catalog = load_catalog(config_path)
    start = str(catalog["start"])
    end = str(catalog["end"])
    tickers = [str(t) for t in catalog["tickers"]]

    datasets: list[tuple[str, pd.DataFrame, dict[str, Any]]] = []
    for ticker in tickers:
        raw = download_ticker(ticker, start, end)
        ml_ready = build_ml_ready(raw)
        report = validate_dataset(ml_ready, ticker)
        datasets.append((ticker, ml_ready, report))
        print(f"   ✔ {ticker}: {report['rows']} filas · {report['nulls']} nulos · {report['date_min']} → {report['date_max']}")

    paths.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for ticker, ml_ready, report in datasets:
        out = paths.RAW_DATA_DIR / f"{ticker}_ml_ready.csv"
        ml_ready.to_csv(out, index=False)
        written.append(out)
        save_validation_report(report)
        print(f"   ✔ Guardado {out.name} + informe de validación")

    return written


def process_raw_to_processed() -> list[Path]:
    """Procesa los CSV de data/raw/ y los escribe ordenados en data/processed/.

    Lee todos los ficheros `*_ml_ready.csv` de paths.RAW_DATA_DIR (los
    datasets de precios del catálogo), ordena por timestamp (fecha ascendente)
    y los guarda con el mismo nombre en paths.PROCESSED_DATA_DIR. Los ficheros
    de noticias (`news_*.csv`) no son datasets de precios y se ignoran.
    Devuelve la lista de ficheros escritos.
    """
    paths.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for raw_file in sorted(paths.RAW_DATA_DIR.glob("*_ml_ready.csv")):
        df = pd.read_csv(raw_file, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        out = paths.PROCESSED_DATA_DIR / raw_file.name
        df.to_csv(out, index=False)
        written.append(out)
        print(f"   ✔ {raw_file.name} → {out.name} ({df.shape[0]}f × {df.shape[1]}c)")

    return written


if __name__ == "__main__":
    try:
        download_catalog()
    except (RuntimeError, ValueError, yaml.YAMLError) as exc:
        print(f"❌ Error en la descarga multi-ticker: {exc}")
        raise SystemExit(1) from exc
    process_raw_to_processed()
