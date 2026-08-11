"""
agents.agents.sentiment_agent — Análisis de sentimiento financiero por ticker.

Conoce que el módulo inversion/sentiment/ (SENT-001 fetch, SENT-002 analyzer)
descarga noticias RSS a data/raw/news_<TICKER>.csv y las puntúa con VADER +
léxico financiero, agregando por día. Este agente expone esas dos operaciones
de forma determinista; no modifica modelos ni datasets de precios.
"""

from __future__ import annotations

from collections.abc import Callable

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent


@register_agent
class SentimentAgent(BaseAgent):
    name = "sentiment"
    description = (
        "Analiza el sentimiento financiero por ticker: descarga noticias (RSS) "
        "y las puntúa con VADER + léxico financiero, agregando scores por día. "
        "Determinista: mismo input → mismo output."
    )
    capabilities = [
        "sentimiento",
        "sentiment",
        "noticias",
        "news",
        "vader",
        "titulares",
        "score de sentimiento",
        "analisis de sentimiento",
        "scores",
        "sentimiento financiero",
        "clima del mercado",
    ]

    def action_aliases(self) -> dict[str, list[str]]:
        return {
            "analyze": ["analiza el sentimiento", "sentimiento de", "score de", "scores de"],
            "fetch": ["descarga noticias", "trae noticias", "rss"],
        }

    def actions(self) -> dict[str, Callable[..., AgentResult]]:
        return {
            "analyze": self.analyze,
            "fetch": self.fetch,
        }

    def analyze(self, *, ticker: str) -> AgentResult:
        """Puntúa las noticias de <ticker> y devuelve el agregado diario."""
        from inversion.sentiment.analyzer import analyze_ticker

        try:
            daily = analyze_ticker(ticker)
        except Exception as exc:  # noqa: BLE001 — error claro al usuario
            return AgentResult(
                False,
                self.name,
                "analyze",
                f"No se pudo analizar el sentimiento de '{ticker}': {exc}",
            )
        if daily.empty:
            return AgentResult(
                False,
                self.name,
                "analyze",
                f"No hay noticias para {ticker.upper()} con las que puntuar.",
            )
        rows = daily.head(10).to_dict(orient="records")
        n_days = int(len(daily))
        return AgentResult(
            True,
            self.name,
            "analyze",
            f"Sentimiento de {ticker.upper()} completo: {n_days} día(s) con noticias. Últimas filas: {rows}",
            data={"ticker": ticker.upper(), "n_days": n_days, "ultimas_filas": rows},
        )

    def fetch(self, *, ticker: str) -> AgentResult:
        """Descarga noticias RSS de <ticker> a data/raw/news_<TICKER>.csv."""
        from inversion.sentiment.fetch import FetchError, fetch_news

        try:
            out = fetch_news(ticker)
        except FetchError as exc:
            return AgentResult(False, self.name, "fetch", str(exc))
        return AgentResult(
            True,
            self.name,
            "fetch",
            f"Noticias de {ticker.upper()} guardadas en {out}",
            data={"ticker": ticker.upper(), "path": str(out)},
        )
