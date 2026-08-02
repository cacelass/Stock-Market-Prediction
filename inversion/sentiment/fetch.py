"""
fetch.py — Recolección de noticias financieras por ticker (SENT-001).

Descarga titulares de una fuente RSS pública (Google News, sin API key) y los
guarda en data/raw/news_<TICKER>.csv con columnas fecha, titulo, cuerpo, fuente.

Sin red la función no cuelga: aplica un timeout y lanza FetchError con un
mensaje claro.
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from inversion.utils import paths

# Fuente RSS real: Google News por ticker (pública, sin API key).
RSS_URL = "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
RSS_TIMEOUT_SECONDS = 10

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga noticias RSS por ticker")
    parser.add_argument("--ticker", default="NVDA", help="Ticker (p.ej. NVDA)")
    args = parser.parse_args()

    out = fetch_news(args.ticker)
    n = len(pd.read_csv(out))
    print(f"✔ {n} noticias para {args.ticker.upper()} → {out}")


if __name__ == "__main__":
    main()
