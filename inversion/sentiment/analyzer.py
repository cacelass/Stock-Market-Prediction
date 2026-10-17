"""
analyzer.py — Analizador de sentimiento financiero con VADER + Loughran-McDonald.

Clasifica titulares/cuerpos con nltk VADER (SentimentIntensityAnalyzer) más un
léxico financiero ligero (jerga de mercados) y un léxico Loughran-McDonald
específico de finanzas. Agrega el score compuesto medio por (fecha, ticker).
Sin aleatoriedad: mismo input → mismo output.

El léxico VADER base viaja embebido en inversion/sentiment/data/vader_lexicon.txt
para que los tests sean 100% offline; si ese fichero falta, se usa (o descarga)
el de nltk en el primer uso.

El léxico Loughran-McDonald (LM) se carga desde loughran_mcdonald.json como
diccionario {positive: [...], negative: [...]} y se añade al analizador VADER
como tokens con scores moderados (+1.0 / -1.0) para evitar dominar VADER.
"""

import argparse
import json
from pathlib import Path

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer, VaderConstants

from inversion.sentiment.fetch import fetch_news
from inversion.utils import paths

# Léxico financiero ligero (heredado): jerga de mercados que el VADER base no conoce.
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

# Ruta al léxico Loughran-McDonald embebido en el repo.
LM_LEXICON_PATH = Path(__file__).resolve().parent / "loughran_mcdonald.json"

# Scores moderados para LM (no dominan VADER, solo matizan).
LM_POSITIVE_SCORE = 1.0
LM_NEGATIVE_SCORE = -1.0

# Léxico VADER base embebido (copia de nltk_data) para uso 100% offline.
BUNDLED_LEXICON = Path(__file__).resolve().parent / "data" / "vader_lexicon.txt"

_ANALYZER: SentimentIntensityAnalyzer | None = None


def _build_analyzer() -> SentimentIntensityAnalyzer:
    """Devuelve un SentimentIntensityAnalyzer con VADER + LM léxicos añadidos.

    Prioridad: léxico embebido en el repo (offline y determinista) → léxico
    VADER de nltk ya descargado → descarga en el primer uso (necesita red).
    Si nada es posible, lanza RuntimeError con un mensaje claro.

    Añade:
      - FINANCIAL_LEXICON   : jerga de mercados (score ±1.5 a ±3.0)
      - Loughran-McDonald   : tokens financieros genéricos (score ±1.0)
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

    # Léxico financiero ligero (heredado)
    analyzer.lexicon.update(FINANCIAL_LEXICON)

    # Léxico Loughran-McDonald (si existe el fichero en el repo)
    if LM_LEXICON_PATH.exists():
        lm_data = json.loads(LM_LEXICON_PATH.read_text(encoding="utf-8"))
        for word in lm_data.get("positive", []):
            analyzer.lexicon[word] = LM_POSITIVE_SCORE
        for word in lm_data.get("negative", []):
            analyzer.lexicon[word] = LM_NEGATIVE_SCORE

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


def score_with_lm(text: str) -> dict[str, float]:
    """Score de texto usando solo el léxico Loughran-McDonald.

    Devuelve {"lm_positive", "lm_negative", "lm_delta"} donde delta = positive - negative.
    Útil para tener una señal de sentimiento independiente de VADER.
    """
    analyzer = get_analyzer()
    lm_positive = 0
    lm_negative = 0
    for word in text.lower().split():
        word = "".join(c for c in word if c.isalpha())
        score = analyzer.lexicon.get(word, 0)
        if score > 0:
            lm_positive += 1
        elif score < 0:
            lm_negative += 1
    return {
        "lm_positive": float(lm_positive),
        "lm_negative": float(lm_negative),
        "lm_delta": float(lm_positive - lm_negative),
    }


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
    # Buscar noticias reales: _live.csv (NewsAPI.ai) y _gdelt.csv (GDELT samples).
    # Fallback a news_<TICKER>.csv (para tests / retrocompatibilidad).
    # Si ninguna existe, intenta descargar RSS como último recurso.
    ticker_upper = ticker.upper()
    csv_path_live = paths.RAW_DATA_DIR / f"news_{ticker_upper}_live.csv"
    csv_path_gdelt = paths.RAW_DATA_DIR / f"news_{ticker_upper}_gdelt.csv"
    csv_path_fallback = paths.RAW_DATA_DIR / f"news_{ticker_upper}.csv"
    if csv_path_live.exists() and csv_path_gdelt.exists():
        df = pd.concat([pd.read_csv(csv_path_live), pd.read_csv(csv_path_gdelt)], ignore_index=True).drop_duplicates(subset=["titulo"])
    elif csv_path_live.exists():
        df = pd.read_csv(csv_path_live)
    elif csv_path_gdelt.exists():
        df = pd.read_csv(csv_path_gdelt)
    elif csv_path_fallback.exists():
        df = pd.read_csv(csv_path_fallback)
    else:
        fetch_news(ticker)
        df = pd.read_csv(paths.RAW_DATA_DIR / f"news_{ticker_upper}.csv")
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
