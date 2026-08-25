"""
fetch.py — Recolección de noticias financieras por ticker (SENT-001, SENT-004).

Descarga titulares de una fuente RSS pública (Google News, sin API key) y los
guarda en data/raw/news_<TICKER>.csv con columnas fecha, titulo, cuerpo, fuente.

SENT-004: Añade fetch_historical_gdelt() para noticias históricas desde GDELT.

Sin red la función no cuelga: aplica un timeout y lanza FetchError con un
mensaje claro.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from inversion.utils import paths

# Fuente RSS real: Google News por ticker (pública, sin API key).
RSS_URL = "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
RSS_TIMEOUT_SECONDS = 10

# GDELT Doc API para noticias históricas (SENT-004).
GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_TIMEOUT_SECONDS = 30
GDELT_MAX_RETRIES = 5
GDELT_BACKOFF_BASE = 4  # Segundos base para backoff exponencial (más conservador)
GDELT_RATE_LIMIT_DELAY = 5.0  # Segundos entre peticiones (GDELT es estricto con rate limits)
GDELT_CHUNK_DAYS = 90  # Chunks de 3 meses en vez de 1 año

# Columnas del CSV de salida, en este orden.
CSV_COLUMNS = ["fecha", "titulo", "cuerpo", "fuente"]


class FetchError(RuntimeError):
    """La descarga falló (sin red, timeout o RSS inválido)."""


def _parse_rss(xml_bytes: bytes) -> pd.DataFrame:
    """Convierte el XML de un canal RSS en un DataFrame con CSV_COLUMNS."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise FetchError(f"La fuente RSS devolvió XML inválido: {exc}") from exc

    channel = root.find("channel")
    if channel is None:
        raise FetchError("La fuente RSS no contiene un canal válido")

    rows = []
    for item in channel.findall("item"):
        rows.append(
            {
                "fecha": item.findtext("pubDate") or "",
                "titulo": item.findtext("title") or "",
                "cuerpo": item.findtext("description") or "",
                "fuente": item.findtext("source") or "unknown",
            }
        )
    if not rows:
        raise FetchError("La fuente RSS no devolvió ninguna noticia")

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True, errors="coerce")
    return df


def fetch_news(ticker: str, out_path: Path | None = None) -> Path:
    """Descarga noticias de <ticker> desde RSS y escribe data/raw/news_<TICKER>.csv.

    Devuelve la ruta del fichero escrito. Sin red o con RSS inválido lanza
    FetchError con un mensaje claro; nunca se queda colgado (timeout de 10s).
    """
    url = RSS_URL.format(ticker=ticker.upper())
    try:
        with urllib.request.urlopen(url, timeout=RSS_TIMEOUT_SECONDS) as resp:
            xml_bytes = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        raise FetchError(f"No se pudieron descargar noticias para {ticker.upper()}: sin red o la fuente RSS no responde ({exc}).") from exc

    df = _parse_rss(xml_bytes)
    out = out_path or (paths.RAW_DATA_DIR / f"news_{ticker.upper()}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def fetch_historical_gdelt(
    ticker: str,
    start_date: str = "2021-01-01",
    end_date: str = "2026-01-01",
    out_path: Path | None = None,
) -> Path:
    """Descarga noticias históricas desde GDELT para un ticker (SENT-004).

    Divide el rango de fechas en chunks de ~1 año para evitar timeouts.
    Implementa rate limiting (1 req/s) y reintentos con backoff exponencial.

    Args:
        ticker: Símbolo bursátil (p.ej. NVDA).
        start_date: Fecha de inicio en formato YYYY-MM-DD.
        end_date: Fecha de fin en formato YYYY-MM-DD.
        out_path: Ruta de salida opcional. Si es None, usa data/raw/news_<TICKER>.csv.

    Returns:
        Ruta del fichero CSV escrito.

    Raises:
        FetchError: Si la descarga falla tras agotar reintentos.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_articles = []
    current = start

    while current < end:
        chunk_end = min(current + timedelta(days=GDELT_CHUNK_DAYS), end)
        chunk_start_str = current.strftime("%Y%m%d")
        chunk_end_str = chunk_end.strftime("%Y%m%d")

        print(f"  Descargando {ticker.upper()} desde {chunk_start_str} hasta {chunk_end_str}...")

        # Construir query para GDELT
        query = f'"{ticker.upper()}" AND sourcecountry:US'
        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": 250,
            "format": "json",
            "startdatetime": chunk_start_str + "000000",
            "enddatetime": chunk_end_str + "235959",
        }

        # Construir URL con parámetros (usar urlencode para codificar correctamente)
        url = f"{GDELT_BASE_URL}?{urllib.parse.urlencode(params)}"

        # Descargar con reintentos
        articles = _fetch_gdelt_chunk(url, ticker)
        all_articles.extend(articles)

        # Avanzar al siguiente chunk
        current = chunk_end + timedelta(days=1)

        # Rate limiting
        time.sleep(GDELT_RATE_LIMIT_DELAY)

    if not all_articles:
        raise FetchError(f"No se encontraron noticias históricas para {ticker.upper()}")

    # Crear DataFrame
    df = pd.DataFrame(all_articles, columns=CSV_COLUMNS)
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True, errors="coerce")

    # Eliminar duplicados por título
    df = df.drop_duplicates(subset=["titulo"], keep="first")

    # Guardar CSV
    out = out_path or (paths.RAW_DATA_DIR / f"news_{ticker.upper()}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    return out


def _fetch_gdelt_chunk(url: str, ticker: str) -> list[list[str]]:
    """Descarga un chunk de noticias desde GDELT con reintentos.

    Returns:
        Lista de listas con [fecha, titulo, cuerpo, fuente].

    Raises:
        FetchError: Si la descarga falla tras agotar reintentos.
    """
    for attempt in range(GDELT_MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=GDELT_TIMEOUT_SECONDS) as resp:
                data = resp.read()

            # GDELT devuelve JSON
            result = json.loads(data)
            articles = result.get("articles", [])

            rows = []
            for article in articles:
                # Extraer campos del artículo GDELT
                title = article.get("title", "")
                body = article.get("excerpt", "")
                source = article.get("domain", "unknown")
                date_str = article.get("seendate", "")

                # GDELT usa formato YYYYMMDDHHMMSS
                if date_str and len(date_str) >= 8:
                    try:
                        date_obj = datetime.strptime(date_str[:8], "%Y%m%d")
                        fecha = date_obj.strftime("%Y-%m-%d")
                    except ValueError:
                        fecha = date_str[:10]
                else:
                    fecha = ""

                # Filtrar por ticker en título o cuerpo
                if ticker.upper() in title.upper() or ticker.upper() in body.upper():
                    rows.append([fecha, title, body, source])

            return rows

        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            if attempt < GDELT_MAX_RETRIES - 1:
                wait_time = GDELT_BACKOFF_BASE ** (attempt + 1)
                print(f"    Error en intento {attempt + 1}/{GDELT_MAX_RETRIES}: {exc}. Reintentando en {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise FetchError(f"Error descargando noticias históricas para {ticker.upper()} tras {GDELT_MAX_RETRIES} intentos: {exc}") from exc

    return []


def fetch_all_historical(
    start_date: str = "2021-01-01",
    end_date: str = "2026-01-01",
    tickers_path: Path | None = None,
) -> dict[str, Path]:
    """Descarga noticias históricas para todos los tickers en config/tickers.yaml.

    Args:
        start_date: Fecha de inicio en formato YYYY-MM-DD.
        end_date: Fecha de fin en formato YYYY-MM-DD.
        tickers_path: Ruta al fichero tickers.yaml. Si es None, usa config/tickers.yaml.

    Returns:
        Diccionario {ticker: ruta_csv} con las rutas de los ficheros descargados.
    """
    tickers_file = tickers_path or (paths.PROJECT_DIR / "config" / "tickers.yaml")

    with open(tickers_file) as f:
        config = yaml.safe_load(f)

    tickers = config.get("tickers", [])
    results = {}

    for ticker in tickers:
        print(f"\n{'=' * 60}")
        print(f"Procesando {ticker}...")
        print(f"{'=' * 60}")

        try:
            out_path = fetch_historical_gdelt(ticker, start_date, end_date)
            results[ticker] = out_path
            n = len(pd.read_csv(out_path))
            print(f"✔ {n} noticias para {ticker} → {out_path}")
        except FetchError as e:
            print(f"✗ Error para {ticker}: {e}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga noticias financieras por ticker")
    parser.add_argument("--ticker", default="NVDA", help="Ticker (p.ej. NVDA)")
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Descargar noticias históricas desde GDELT (SENT-004)",
    )
    parser.add_argument(
        "--start",
        default="2021-01-01",
        help="Fecha de inicio para modo histórico (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default="2026-01-01",
        help="Fecha de fin para modo histórico (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    if args.historical:
        print(f"Descargando noticias históricas desde {args.start} hasta {args.end}...")
        results = fetch_all_historical(args.start, args.end)
        print(f"\n✔ Descarga completada para {len(results)} tickers")
    else:
        out = fetch_news(args.ticker)
        n = len(pd.read_csv(out))
        print(f"✔ {n} noticias para {args.ticker.upper()} → {out}")


if __name__ == "__main__":
    main()
