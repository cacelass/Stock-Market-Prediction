"""
analyzer.py — Analizador de sentimiento financiero con VADER (SENT-002).

Clasifica titulares/cuerpos con nltk VADER (SentimentIntensityAnalyzer) más un
léxico financiero ligero (jerga de mercados) y agrega el score compuesto medio
por (fecha, ticker). Sin aleatoriedad: mismo input → mismo output.

El léxico VADER base viaja embebido en inversion/sentiment/data/vader_lexicon.txt
para que los tests sean 100% offline; si ese fichero falta, se usa (o descarga)
el de nltk en el primer uso.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer, VaderConstants

from inversion.sentiment.fetch import fetch_news
from inversion.utils import paths

# Léxico financiero ligero: jerga de mercados que el VADER base no conoce.
# Formato VADER: score en [-4, +4] (positivo = alcista).
FINANCIAL_LEXICON: dict[str, float] = {
    "bullish": 2.5,
    "bearish": -2.5,
    "outperform": 2.2,
    "underperform": -2.2,
    "upgrade": 2.0,
    "downgrade": -2.0,
    "beat": 1.8,
    "miss": -1.8,
    "rally": 2.3,
    "selloff": -2.0,
    "surge": 2.0,
    "plunge": -2.2,
    "soar": 2.2,
    "tumble": -2.0,
    "skyrocket": 2.5,
    "crash": -3.0,
    "boom": 2.0,
    "slump": -2.0,
    "record": 1.5,
    "weakness": -1.2,
    "strong": 1.5,
    "weak": -1.2,
    "profit": 1.5,
    "loss": -1.5,
    "growth": 1.5,
    "decline": -1.5,
    "exceed": 1.5,
    "fell": -1.5,
    "raised": 1.3,
    "cut": -1.3,
    "guidance": 0.4,
    "earnings": 0.3,
    "revenue": 0.3,
    "eps": 0.3,
}

# Léxico VADER base embebido (copia de nltk_data) para uso 100% offline.
BUNDLED_LEXICON = Path(__file__).resolve().parent / "data" / "vader_lexicon.txt"

_ANALYZER: SentimentIntensityAnalyzer | None = None


def _build_analyzer() -> SentimentIntensityAnalyzer:
    """Devuelve un SentimentIntensityAnalyzer con el léxico financiero añadido.

    Prioridad: léxico embebido en el repo (offline y determinista) → léxico
    VADER de nltk ya descargado → descarga en el primer uso (necesita red).
    Si nada es posible, lanza RuntimeError con un mensaje claro.
    """
    if BUNDLED_LEXICON.exists():
        analyzer = SentimentIntensityAnalyzer.__new__(SentimentIntensityAnalyzer)
        analyzer.lexicon_file = BUNDLED_LEXICON.read_text(encoding="utf-8")
        analyzer.lexicon = analyzer.make_lex_dict()
        analyzer.constants = VaderConstants()
    else:
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            try:
                nltk.download("vader_lexicon", quiet=True)
            except Exception as exc:
                raise RuntimeError(
                    "VADER lexicon no disponible: sin red y falta el léxico embebido "
                    f"{BUNDLED_LEXICON}. Restaura ese fichero o descarga nltk_data "
                    "(python -m nltk.downloader vader_lexicon)."
                ) from exc
        analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(FINANCIAL_LEXICON)
    return analyzer


def get_analyzer() -> SentimentIntensityAnalyzer:
    """Analyzer VADER con caché: se construye una sola vez y se reutiliza."""
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = _build_analyzer()
    return _ANALYZER


def score_text(text: str) -> float:
    """Score compuesto VADER de un texto, acotado en [-1, 1]."""
    return float(get_analyzer().polarity_scores(text)["compound"])


def score_news_df(df: pd.DataFrame, text_cols: list[str] | None = None) -> pd.DataFrame:
    """Añade columnas neg/neu/pos/compound (VADER) al DataFrame de noticias.

    El texto se forma concatenando las columnas indicadas (por defecto
    titulo y cuerpo). No modifica el DataFrame original.
    """
    analyzer = get_analyzer()
    cols = text_cols or ["titulo", "cuerpo"]
    out = df.copy()
    text = out[cols].fillna("").apply(lambda row: " ".join(row), axis=1)
    scores = text.map(analyzer.polarity_scores)
    out["neg"] = scores.map(lambda s: float(s["neg"]))
    out["neu"] = scores.map(lambda s: float(s["neu"]))
    out["pos"] = scores.map(lambda s: float(s["pos"]))
    out["compound"] = scores.map(lambda s: float(s["compound"]))
    return out


def daily_sentiment(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Agrega el score compuesto medio por (fecha, ticker). Sin aleatoriedad.

    Si el DataFrame no trae columna 'ticker', se usa el parámetro ticker.
    Añade 'noticias' (número de titulares por grupo) para verificar a mano
    la media: compound = mean de los compound de cada noticia.
    """
    out = df.copy()
    if "ticker" not in out.columns:
        out["ticker"] = ticker
    out["fecha"] = pd.to_datetime(out["fecha"], utc=True)
    out["fecha_dia"] = out["fecha"].dt.date
    daily = (
        out.groupby(["fecha_dia", "ticker"], sort=True)["compound"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"fecha_dia": "fecha", "mean": "compound", "size": "noticias"})
    )
    return daily


def analyze_ticker(ticker: str) -> pd.DataFrame:
    """Puntúa las noticias de <ticker> y devuelve el agregado diario.

    Si data/raw/news_<TICKER>.csv no existe, lo descarga primero con fetch_news.
    """
    csv_path = paths.RAW_DATA_DIR / f"news_{ticker.upper()}.csv"
    if not csv_path.exists():
        fetch_news(ticker)
    df = pd.read_csv(csv_path)
    scored = score_news_df(df)
    return daily_sentiment(scored, ticker=ticker.upper())


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentimiento financiero VADER por ticker")
    parser.add_argument("--ticker", default="NVDA", help="Ticker (p.ej. NVDA)")
    args = parser.parse_args()

    daily = analyze_ticker(args.ticker)
    print(daily.to_string(index=False))


if __name__ == "__main__":
    main()
