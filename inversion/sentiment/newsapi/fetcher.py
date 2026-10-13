from __future__ import annotations
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import hashlib

import pandas as pd
import requests


NEWSAPI_ENDPOINT = "https://eventregistry.org/api/v1/article/getArticles"


@dataclass
class Article:
    title: str
    text: str
    date: str
    source: str
    url: str


def _dedup_key(a: Article) -> str:
    h = hashlib.sha256((a.url or a.title or "").encode("utf-8")).hexdigest()
    return h


def _normalize_date(d: str | datetime) -> str:
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    # try parse
    try:
        dt = pd.to_datetime(d)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(d)[:10]


def fetch_news(
    query: str,
    api_key: Optional[str] = None,
    days: int = 7,
    max_pages: int = 1,
) -> List[Article]:
    key = api_key or os.getenv("NEWSAPI_KEY") or os.getenv("NEWSAPI_AI_KEY")
    # Aceptar también NEWSAPI_TOKEN si se usa
    if not key:
        key = os.getenv("NEWSAPI_TOKEN")
    if not key:
        raise ValueError("NEWSAPI_KEY no configurada")

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    articles: List[Article] = []
    seen = set()
    # EventRegistry/NewsAPI.ai v1 format aproximado? Simplificado
    for page in range(1, max_pages + 1):
        try:
            params = {
                "apiKey": key,
                "keyword": query,
                "lang": "es,en",
                "dateStart": start.strftime("%Y-%m-%d"),
                "dateEnd": end.strftime("%Y-%m-%d"),
                "page": page,
            }
            r = requests.get("https://eventregistry.org/api/v1/article/getArticles", params=params, timeout=30)
            r.raise_for_status()
            j = r.json()
            items = j.get("articles", {}).get("results", []) or []
            if not items:
                break
            for it in items:
                title = it.get("title", "") or ""
                text = it.get("body", "") or it.get("snippet", "") or title
                url = it.get("url", "") or ""
                date = it.get("date", "") or it.get("publishedAt", "")
                source = it.get("source", {}).get("title", "") or it.get("source", {}).get("name", "") or ""
                a = Article(title=title, text=text, date=_normalize_date(date), source=source, url=url)
                k = _dedup_key(a)
                if k not in seen:
                    seen.add(k)
                    articles.append(a)
            if len(items) < 20:
                break
        except Exception:
            break
    return articles


def articles_to_df(articles: List[Article]) -> pd.DataFrame:
    return pd.DataFrame([{"titulo": a.title, "texto": a.text, "fecha": a.date, "source": a.source, "url": a.url} for a in articles])


def aggregate_daily(ticker: str, articles: List[Article]) -> pd.DataFrame:
    df = articles_to_df(articles)
    if df.empty:
        return df[["fecha"]].iloc[0:0]
    # simple mean por fecha
    g = df.groupby("fecha").size().reset_index(name="count")
    return g
