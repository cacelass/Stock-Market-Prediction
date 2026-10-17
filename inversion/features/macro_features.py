"""Macroeconomic features for stock prediction.

Adds external market signals: VIX level, sector ETF returns, treasury yield,
and USD index (DXY). Data is fetched from yfinance (VIX, sector ETFs, DXY)
or from FRED (treasury yield).

Functions
---------
add_macro_features : augment a price DataFrame with macro indicators
fetch_vix : download VIX data from yfinance (^VIX)
fetch_sector_data : download sector ETF returns (XLK, XLF, XLE, …)
fetch_treasury_yield : download 10-year Treasury yield from yfinance (or FRED)
fetch_usd_index : download USD index (DXY) from yfinance
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

# ── Fetch helpers ────────────────────────────────────────────────────────────

def fetch_vix(ticker: str = "^VIX") -> pd.DataFrame:
    """Download VIX index data via yfinance.

    Returns DataFrame with columns: timestamp, vix_close.
    """
    if yf is None:
        return pd.DataFrame(columns=["timestamp", "vix_close"])
    try:
        vix = yf.Ticker(ticker)
        hist = vix.history(period="5y")
        if hist.empty:
            return pd.DataFrame(columns=["timestamp", "vix_close"])
        return hist[["Close"]].rename(columns={"Close": "vix_close"}).reset_index()
    except Exception:  # pragma: no cover — network
        return pd.DataFrame(columns=["timestamp", "vix_close"])


def fetch_sector_data(
    tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Download sector ETF adjusted close prices.

    Default tickers: XLK (tech), XLF (financials), XLE (energy),
    XLV (healthcare), XLI (industrials).

    Returns a wide DataFrame: timestamp + one column per ETF.
    """
    if yf is None:
        cols = ["timestamp"] + (tickers or ["XLK", "XLF", "XLE", "XLV", "XLI"])
        return pd.DataFrame(columns=cols)
    symbols = tickers or ["XLK", "XLF", "XLE", "XLV", "XLI"]
    data = yf.download(symbols, period="5y", progress=False)["Close"]
    if data.empty:
        return pd.DataFrame(columns=["timestamp"] + symbols)
    data = data.reset_index().rename(columns={"Date": "timestamp"})
    return data


def fetch_treasury_yield(ticker: str = "^TNX") -> pd.DataFrame:
    """Download 10-year Treasury yield proxy (^TNX on yfinance).

    Returns DataFrame with columns: timestamp, treasury_yield.
    """
    if yf is None:
        return pd.DataFrame(columns=["timestamp", "treasury_yield"])
    try:
        tny = yf.Ticker(ticker)
        hist = tny.history(period="5y")
        if hist.empty:
            return pd.DataFrame(columns=["timestamp", "treasury_yield"])
        return hist[["Close"]].rename(columns={"Close": "treasury_yield"}).reset_index()
    except Exception:  # pragma: no cover — network
        return pd.DataFrame(columns=["timestamp", "treasury_yield"])


def fetch_usd_index(ticker: str = "DX-Y.NYB") -> pd.DataFrame:
    """Download USD Index (DXY) via yfinance.

    Returns DataFrame with columns: timestamp, usd_index.
    """
    if yf is None:
        return pd.DataFrame(columns=["timestamp", "usd_index"])
    try:
        dx = yf.Ticker(ticker)
        hist = dx.history(period="5y")
        if hist.empty:
            return pd.DataFrame(columns=["timestamp", "usd_index"])
        return hist[["Close"]].rename(columns={"Close": "usd_index"}).reset_index()
    except Exception:  # pragma: no cover — network
        return pd.DataFrame(columns=["timestamp", "usd_index"])


# ── Feature builder ──────────────────────────────────────────────────────────

MACRO_COLUMNS = [
    "vix_level",
    "vix_ma20",
    "vix_change",
    "sector_xlk",
    "sector_xlf",
    "sector_xle",
    "treasury_yield",
    "treasury_change",
    "usd_index",
    "usd_change",
]


def add_macro_features(
    df: pd.DataFrame,
    vix_data: Optional[pd.DataFrame] = None,
    sector_data: Optional[pd.DataFrame] = None,
    treasury_data: Optional[pd.DataFrame] = None,
    usd_data: Optional[pd.DataFrame] = None,
    use_yf_fetch: bool = True,
) -> pd.DataFrame:
    """Add macroeconomic features to a price DataFrame.

    Parameters
    ----------
    df : DataFrame with at least a 'timestamp' column (datetime or date).
    vix_data : Optional pre-fetched VIX data. Columns: timestamp, vix_close.
    sector_data : Optional pre-fetched sector ETF data. Columns: timestamp + one per ETF.
    treasury_data : Optional pre-fetched treasury yield. Columns: timestamp, treasury_yield.
    usd_data : Optional pre-fetched USD index. Columns: timestamp, usd_index.
    use_yf_fetch : If True and data is missing, attempt to download via yfinance.
                   Set to False in tests where network is unavailable.

    Returns
    -------
    df with additional columns:
        vix_level, vix_ma20, vix_change,
        sector_xlk, sector_xlf, sector_xle,
        treasury_yield, treasury_change,
        usd_index, usd_change
    """
    df = df.sort_values("timestamp").copy()

    # 1. VIX
    if vix_data is None and use_yf_fetch:
        vix_data = fetch_vix()
    if vix_data is not None and not vix_data.empty:
        vix_data = vix_data.rename(columns={"timestamp": "timestamp"})
        vix_data["timestamp"] = pd.to_datetime(vix_data["timestamp"]).dt.normalize()
        vix_merge = vix_data.groupby("timestamp")["vix_close"].mean().reset_index()
        df = df.merge(vix_merge, left_on=pd.to_datetime(df["timestamp"]).dt.normalize(), right_on="timestamp", how="left")
        df["vix_level"] = df["vix_close"].ffill().values
        df["vix_ma20"] = df["vix_level"].rolling(20, min_periods=1).mean().values
        df["vix_change"] = df["vix_level"].diff().fillna(0.0).values
        df = df.drop(columns=["vix_close", "timestamp_y"])
    else:
        for c in MACRO_COLUMNS[:3]:
            if c not in df.columns:
                df[c] = 0.0

    # 2. Sector ETF returns
    if sector_data is None and use_yf_fetch:
        sector_data = fetch_sector_data()
    if sector_data is not None and not sector_data.empty:
        sector_data["timestamp"] = pd.to_datetime(sector_data["timestamp"]).dt.normalize()
        for col in ["XLK", "XLF", "XLE"]:
            if col in sector_data.columns:
                prefix = f"sector_{col.lower()}"
                df = df.merge(sector_data[["timestamp", col]], on="timestamp", how="left")
                df[f"{prefix}_close"] = df[col].ffill().values
                df[prefix] = df[f"{prefix}_close"].pct_change().fillna(0.0).values
                df = df.drop(columns=[col, f"{prefix}_close"])
            else:
                if prefix not in df.columns:
                    df[prefix] = 0.0
    else:
        for prefix in ["sector_xlk", "sector_xlf", "sector_xle"]:
            if prefix not in df.columns:
                df[prefix] = 0.0

    # 3. Treasury yield
    if treasury_data is None and use_yf_fetch:
        treasury_data = fetch_treasury_yield()
    if treasury_data is not None and not treasury_data.empty:
        treasury_data["timestamp"] = pd.to_datetime(treasury_data["timestamp"]).dt.normalize()
        treasury_merge = treasury_data.groupby("timestamp")["treasury_yield"].mean().reset_index()
        df = df.merge(treasury_merge, left_on=pd.to_datetime(df["timestamp"]).dt.normalize(), right_on="timestamp", how="left")
        df["treasury_yield"] = df["treasury_yield_x"].ffill().values
        df["treasury_change"] = df["treasury_yield"].diff().fillna(0.0).values
        df = df.drop(columns=["treasury_yield_x", "timestamp_y"])
    else:
        for c in MACRO_COLUMNS[6:8]:
            if c not in df.columns:
                df[c] = 0.0

    # 4. USD Index (DXY)
    if usd_data is None and use_yf_fetch:
        usd_data = fetch_usd_index()
    if usd_data is not None and not usd_data.empty:
        usd_data["timestamp"] = pd.to_datetime(usd_data["timestamp"]).dt.normalize()
        usd_merge = usd_data.groupby("timestamp")["usd_index"].mean().reset_index()
        df = df.merge(usd_merge, left_on=pd.to_datetime(df["timestamp"]).dt.normalize(), right_on="timestamp", how="left")
        df["usd_index"] = df["usd_index_x"].ffill().values
        df["usd_change"] = df["usd_index"].diff().fillna(0.0).values
        df = df.drop(columns=["usd_index_x", "timestamp_y"])
    else:
        for c in MACRO_COLUMNS[8:10]:
            if c not in df.columns:
                df[c] = 0.0

    return df
