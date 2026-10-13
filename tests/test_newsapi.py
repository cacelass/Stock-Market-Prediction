from __future__ import annotations
import os
from unittest.mock import patch, MagicMock

from inversion.sentiment.newsapi.fetcher import fetch_news, articles_to_df, aggregate_daily


def test_fetch_news_requires_key():
    with patch.dict(os.environ, {}, clear=True):
        try:
            fetch_news("TSLA", days=1, max_pages=1)
            assert False
        except ValueError:
            assert True


def test_fetch_news_dedup_and_pages():
    with patch.dict(os.environ, {"NEWSAPI_KEY": "k"}):
        resp1 = {
            "articles": {
                "results": [
                    {"title": "A", "body": "t", "url": "u1", "date": "2025-01-01"},
                    {"title": "B", "body": "t", "url": "u2", "date": "2025-01-01"},
                ]
            }
        }
        resp2 = {
            "articles": {
                "results": [
                    {"title": "A", "body": "t", "url": "u1", "date": "2025-01-01"},  # dup
                ]
            }
        }
        mock = MagicMock()
        mock.json.side_effect = [resp1, resp2]
        mock.raise_for_status = lambda: None
        with patch("inversion.sentiment.newsapi.fetcher.requests.get", return_value=mock):
            arts = fetch_news("TSLA", days=2, max_pages=2)
            assert len(arts) == 2
            df = articles_to_df(arts)
            assert set(df.columns) >= {"titulo", "texto", "fecha"}
            agg = aggregate_daily("TSLA", arts)
            assert "fecha" in agg.columns
