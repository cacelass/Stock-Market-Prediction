from __future__ import annotations
import pandas as pd
from typing import Optional

from inversion.sentiment import compute_daily_sentiment


def add_sentiment_features(
    df: pd.DataFrame,
    news_df: Optional[pd.DataFrame] = None,
    text_col: str = "text",
    date_col: str = "date",
    ticker_col: str = "ticker",
    join_on: Optional[list[str]] = None,
) -> pd.DataFrame:
    out = df.copy()
    if news_df is None or news_df.empty:
        # add neutral columns to avoid missing
        for c in ["sentiment_compound", "sentiment_pos", "sentiment_neg"]:
            if c not in out.columns:
                out[c] = 0.0
        return out

    sdf = compute_daily_sentiment(news_df, text_col=text_col, date_col=date_col, ticker_col=ticker_col)
    if sdf.empty:
        for c in ["sentiment_compound", "sentiment_pos", "sentiment_neg"]:
            if c not in out.columns:
                out[c] = 0.0
        return out

    keys = join_on or ([date_col, ticker_col] if ticker_col in out.columns and ticker_col in sdf.columns else [date_col])
    out = out.merge(sdf, on=keys, how="left")
    for c in ["sentiment_compound", "sentiment_pos", "sentiment_neg"]:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
    return out
