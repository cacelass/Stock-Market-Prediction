"""
test_sentiment.py — Tests para inversion/sentiment (SENT-001 fetch, SENT-002 analyzer).

Todos los tests son offline: el fetch usa un RSS simulado y el analyzer usa el
sample offline de tests/fixtures/sample_news.csv + el léxico VADER embebido.
"""

from __future__ import annotations

import datetime
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_news.csv"

# RSS de ejemplo para simular la fuente sin red.
SAMPLE_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item>
    <title>Nvidia rallies on strong earnings</title>
    <pubDate>Mon, 27 Jul 2026 20:21:00 GMT</pubDate>
    <description>Analysts bullish</description>
    <source>WSJ</source>
  </item>
  <item>
    <title>Nvidia stock tumbles</title>
    <pubDate>Tue, 28 Jul 2026 10:00:00 GMT</pubDate>
    <description>Weak outlook</description>
    <source>Reuters</source>
  </item>
</channel></rss>
"""


class _FakeResponse:
    """Respuesta falsa de urllib para simular la fuente RSS sin red."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        pass

    def read(self) -> bytes:
        return self._body


def _sample_as_news_csv(patch_paths: dict) -> Path:
    """Deja el sample offline como si lo hubiera escrito fetch_news."""
    target = patch_paths["RAW_DATA_DIR"] / "news_NVDA.csv"
    pd.read_csv(FIXTURE).to_csv(target, index=False)
    return target


# ─────────────────────────────────────────────────────────────────────────────
# SENT-001 — fetch
# ─────────────────────────────────────────────────────────────────────────────


def test_fetch_news_writes_csv(monkeypatch, patch_paths):
    """fetch_news debe escribir data/raw/news_NVDA.csv con el schema esperado."""
    from inversion.sentiment.fetch import fetch_news

    def fake_urlopen(url: str, timeout: float) -> _FakeResponse:
        return _FakeResponse(SAMPLE_RSS)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    out = fetch_news("nvda")  # minúsculas → se normaliza a NVDA

    expected = patch_paths["RAW_DATA_DIR"] / "news_NVDA.csv"
    assert out == expected
    assert expected.exists()
    df = pd.read_csv(expected)
    assert list(df.columns) == ["fecha", "titulo", "cuerpo", "fuente"]
    assert len(df) == 2
    assert df["titulo"].iloc[0] == "Nvidia rallies on strong earnings"
    assert df["fuente"].iloc[1] == "Reuters"


def test_fetch_news_without_network_raises_clear_error(monkeypatch, patch_paths):
    """Sin red, fetch_news debe lanzar FetchError con mensaje claro — no colgar."""
    from inversion.sentiment.fetch import FetchError, fetch_news

    def fake_urlopen(url: str, timeout: float) -> _FakeResponse:
        raise urllib.error.URLError("no network")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(FetchError, match="sin red"):
        fetch_news("NVDA")


def test_fetch_news_invalid_rss_raises(monkeypatch, patch_paths):
    """Un RSS no parseable debe lanzar FetchError, no romper en silencio."""
    from inversion.sentiment.fetch import FetchError, fetch_news

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout: _FakeResponse(b"this is not xml"),
    )

    with pytest.raises(FetchError, match="XML"):
        fetch_news("NVDA")


def test_offline_sample_has_expected_schema():
    """Fixture/sample offline: schema y contenido esperados para los tests."""
    df = pd.read_csv(FIXTURE)
    assert list(df.columns) == ["fecha", "titulo", "cuerpo", "fuente", "label"]
    assert len(df) >= 6
    assert df["label"].isin(["positiva", "negativa", "neutral"]).all()
    assert pd.to_datetime(df["fecha"]).notna().all()


# ─────────────────────────────────────────────────────────────────────────────
# SENT-002 — analyzer
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_smoke_analyze_ticker_on_offline_sample(patch_paths):
    """Test de humo: analiza el sample offline de SENT-001 de principio a fin."""
    from inversion.sentiment.analyzer import analyze_ticker

    _sample_as_news_csv(patch_paths)

    daily = analyze_ticker("NVDA")

    assert set(daily.columns) >= {"fecha", "ticker", "compound", "noticias"}
    assert len(daily) == 3  # 28, 29 y 30 de julio de 2026
    assert daily["compound"].between(-1, 1).all()
    assert (daily["noticias"] > 0).all()


def test_daily_aggregation_reproducible(patch_paths):
    """El agregado diario es determinista: mismo input → mismo output."""
    from inversion.sentiment.analyzer import analyze_ticker

    _sample_as_news_csv(patch_paths)

    d1 = analyze_ticker("NVDA")
    d2 = analyze_ticker("NVDA")
    pd.testing.assert_frame_equal(d1, d2)


def test_daily_aggregation_matches_manual_mean():
    """El agregado diario es la media simple de los compound del día.

    Valores fijados a mano sobre el sample offline (determinista):
      - 28-jul: (0.4404 + 0.7184 + 0.0772) / 3 =  0.412000
      - 29-jul: (-0.3818 - 0.8481 - 0.4215) / 3 = -0.550467
      - 30-jul: (0.3612 + 0.0000) / 2 =            0.180600
    """
    from inversion.sentiment.analyzer import daily_sentiment, score_news_df

    scored = score_news_df(pd.read_csv(FIXTURE))
    daily = daily_sentiment(scored, ticker="NVDA")

    expected = {
        datetime.date(2026, 7, 28): (0.412000, 3),
        datetime.date(2026, 7, 29): (-0.550467, 3),
        datetime.date(2026, 7, 30): (0.180600, 2),
    }
    for fecha, (compound, noticias) in expected.items():
        row = daily[(daily["fecha"] == fecha) & (daily["ticker"] == "NVDA")]
        assert len(row) == 1
        assert row["compound"].iloc[0] == pytest.approx(compound, abs=1e-4)
        assert row["noticias"].iloc[0] == noticias


def test_scores_direction_on_manual_labels():
    """La dirección del score coincide con la anotación manual del sample."""
    from inversion.sentiment.analyzer import score_news_df

    scored = score_news_df(pd.read_csv(FIXTURE))

    for _, row in scored.iterrows():
        if row["label"] == "positiva":
            assert row["compound"] > 0, row["titulo"]
        elif row["label"] == "negativa":
            assert row["compound"] < 0, row["titulo"]
        else:
            assert abs(row["compound"]) < 0.5, row["titulo"]


def test_scores_bounded_in_unit_interval():
    """Todos los scores compuestos están acotados en [-1, 1]."""
    from inversion.sentiment.analyzer import score_news_df

    scored = score_news_df(pd.read_csv(FIXTURE))
    assert scored["compound"].between(-1, 1).all()
    for col in ("neg", "neu", "pos"):
        assert scored[col].between(0, 1).all()


def test_financial_lexicon_changes_sentiment():
    """El léxico financiero puntúa jerga que el VADER base deja en neutral."""
    from inversion.sentiment.analyzer import score_text

    assert score_text("The stock beat expectations") > 0
    assert score_text("Analysts downgrade the stock") < 0
    assert score_text("bullish outlook") > 0
    assert score_text("bearish outlook") < 0
