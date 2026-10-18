"""Advanced feature engineering — market microstructure & higher-order stats.

Generates features that capture:
  1. Interaction terms: return * volatility, rsi * momentum
  2. Fractal dimension (Hurst exponent) — measures market memory
  3. Entropy features — measures predictability
  4. Roll's implied spread — measures illiquidity
  5. Amihud illiquidity ratio
  6. Price impact coefficient
  7. VWAP deviation from different windows
  8. Volume-price correlation
  9. Order flow imbalance proxy (close-open / high-low)
  10. Realized volatility ratios (20d/63d, etc.)
  11. GARCH-like volatility features (squared returns, abs returns)
  12. Skewness and kurtosis of returns
  13. Higher-order moments (vol of vol, etc.)
  14. Rolling statistics for multiple windows (5, 10, 20, 60)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Interaction terms
# ─────────────────────────────────────────────────────────────────────────────

INTERACTION_COLS: List[str] = []


def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Add pairwise interaction terms between key features.

    Generates:
      - return * volatility          — direction × magnitude of moves
      - rsi * momentum_10            — momentum strength at extreme RSI
      - hl_range * log_volume        — intraday range × activity
      - oc_range * return            — gap direction × daily return
      - volatility * log_volume      — activity-weighted volatility
      - momentum_10 * momentum_21    — short × medium momentum alignment
    """
    out = df.copy()

    interactions: Dict[str, str] = {
        "ret_vol": "return * volatility",
        "rsi_mom10": "rsi * momentum_10",
        "hl_vol": "hl_range * log_volume",
        "oc_ret": "oc_range * return",
        "vol_vol": "volatility * log_volume",
        "mom10_21": "momentum_10 * momentum_21",
    }

    for col, expr in interactions.items():
        try:
            out[col] = out.eval(expr)
            INTERACTION_COLS.append(col)
        except Exception:
            pass

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Fractal dimension / Hurst exponent
# ─────────────────────────────────────────────────────────────────────────────

HURST_COLS: List[str] = []


def add_hurst_exponent(
    df: pd.DataFrame,
    windows: List[int] | None = None,
    min_periods: int = 10,
) -> pd.DataFrame:
    """Add Hurst exponent features measuring market memory.

    H > 0.5 → persistent (trending), H < 0.5 → anti-persistent (mean-reverting),
    H ≈ 0.5 → random walk.

    Computed via R/S analysis on returns within each rolling window.
    """
    if windows is None:
        windows = [20, 60]

    out = df.copy()
    out["return"] = out["return"].fillna(0.0)

    def _hurst(series: np.ndarray) -> float:
        """Compute Hurst exponent via R/S analysis."""
        if len(series) < min_periods:
            return 0.5  # default: random walk
        arr = np.asarray(series)
        n = len(arr)
        if n < min_periods:
            return 0.5

        # Detrended R/S: divide into chunks, compute R/S for each
        chunk_sizes = []
        for size in range(10, n // 2):
            chunk_sizes.append(size)

        if not chunk_sizes:
            return 0.5

        # Use a few representative chunk sizes for efficiency
        chunk_sizes = [int(n * f) for f in (0.1, 0.2, 0.3, 0.5)]
        chunk_sizes = [s for s in chunk_sizes if s >= min_periods]
        if not chunk_sizes:
            return 0.5

        log_rs = []
        log_n = []
        for size in chunk_sizes:
            n_chunks = n // size
            if n_chunks < 1:
                continue

            # Mean-subtracted chunks
            means = np.mean(arr[: n_chunks * size].reshape(n_chunks, size), axis=1)
            centered = arr[: n_chunks * size].reshape(n_chunks, size) - means[:, None]

            # Cumulative deviations
            cum = np.cumsum(centered, axis=1)
            std = np.std(arr[: n_chunks * size].reshape(n_chunks, size), axis=1)
            std = np.where(std == 0, 1e-9, std)

            # R/S
            r = np.max(cum, axis=1) - np.min(cum, axis=1)
            rs = r / std
            rs = np.where(rs == 0, 1e-9, rs)

            log_rs.append(np.log(np.mean(rs)))
            log_n.append(np.log(size))

        if len(log_rs) < 2:
            return 0.5

        # Linear fit: log(R/S) = H * log(n) + C
        slope, _ = np.polyfit(log_n, log_rs, 1)
        return float(np.clip(slope, 0.0, 1.0))

    for w in windows:
        col = f"hurst_{w}d"
        HURST_COLS.append(col)
        out[col] = (
            out["return"]
            .rolling(w, min_periods=min_periods)
            .apply(_hurst, raw=True)
            .fillna(0.5)
        )

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Entropy features
# ─────────────────────────────────────────────────────────────────────────────

ENTROPY_COLS: List[str] = []


def add_entropy_features(
    df: pd.DataFrame,
    bins: int = 10,
    windows: List[int] | None = None,
) -> pd.DataFrame:
    """Add entropy features measuring predictability.

    Shannon entropy of returns distribution — lower entropy = more predictable.
    Also computes the entropy of volume distribution as an activity measure.
    """
    if windows is None:
        windows = [20, 60]

    out = df.copy()
    out["return"] = out["return"].fillna(0.0)

    def _shannon_entropy(series: np.ndarray, n_bins: int) -> float:
        """Compute Shannon entropy of a series."""
        if len(series) == 0:
            return 0.0
        hist, _ = np.histogram(np.asarray(series), bins=n_bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    for w in windows:
        for name, series_name in [("ret_entropy", "return"), ("vol_entropy", "log_volume")]:
            col = f"{name}_{w}d"
            ENTROPY_COLS.append(col)
            out[col] = (
                out[series_name]
                .rolling(w, min_periods=max(w // 2, 5))
                .apply(_shannon_entropy, raw=True, args=(bins,))
                .fillna(0.0)
            )

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Roll's implied spread
# ─────────────────────────────────────────────────────────────────────────────

ROLL_SPREAD_COL = "rolls_spread"


def add_rolls_spread(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add Roll's (1984) implied spread measure.

    Spread ≈ 2 * sqrt(-cov(delta_price_t, delta_price_{t-1})).
    Only non-negative covariances are valid; negative ones indicate
    overbouncing and are set to 0 (no spread).
    """
    out = df.copy()
    deltas = out["close"].diff()

    cov = deltas.rolling(window, min_periods=5).cov(deltas.shift(1))
    out[ROLL_SPREAD_COL] = 2.0 * np.sqrt(np.maximum(0, -cov))
    out[ROLL_SPREAD_COL] = out[ROLL_SPREAD_COL].fillna(0.0)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Amihud illiquidity ratio
# ─────────────────────────────────────────────────────────────────────────────

AMIHUD_COL = "amihud_illiquidity"


def add_amihud_illiquidity(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add Amihud (2002) illiquidity measure.

    Illiquidity = mean(|return| / dollar_volume). Higher = more illiquid.
    """
    out = df.copy()
    out["return"] = out["return"].fillna(0.0)

    # Dollar volume in millions for scale
    dollar_vol = out["volume"] * out["close"] / 1e6
    dollar_vol = dollar_vol.replace(0, 1e-9)

    ratio = (out["return"].abs() / dollar_vol).rolling(window, min_periods=5).mean()
    out[AMIHUD_COL] = ratio.fillna(0.0)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Price impact coefficient
# ─────────────────────────────────────────────────────────────────────────────

PRICE_IMPACT_COL = "price_impact"


def add_price_impact(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add price impact coefficient.

    Regresses |return| on volume within rolling window. The slope tells
    how much price moves per unit of volume — higher = more impact.
    """
    out = df.copy()
    out["return"] = out["return"].fillna(0.0)
    ret_abs = out["return"].abs()
    vol = out["volume"]

    # Rolling regression: |return| ~ volume
    def _price_impact(series: np.ndarray) -> float:
        """Simple linear regression slope."""
        # series has ret_abs and vol interleaved as columns if raw=False,
        # but with raw=True it's 2D. We'll handle differently.
        return 0.0

    out[PRICE_IMPACT_COL] = ret_abs.rolling(window, min_periods=5).apply(
        lambda s: 0.0, raw=True
    )

    # More efficient: use rolling cov/var
    cov_v = ret_abs.rolling(window, min_periods=5).cov(vol)
    var_v = vol.rolling(window, min_periods=5).var()
    out[PRICE_IMPACT_COL] = (cov_v / (var_v + 1e-9)).fillna(0.0)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  VWAP deviation from different windows
# ─────────────────────────────────────────────────────────────────────────────

VWAP_DEV_COLS: List[str] = []


def add_vwap_deviation(
    df: pd.DataFrame,
    windows: List[int] | None = None,
) -> pd.DataFrame:
    """Add VWAP deviation features at multiple windows.

    close / VWAP_ma_n — how far current price is from its own VWAP average.
    """
    if windows is None:
        windows = [5, 20]

    out = df.copy()
    if "vwap" not in out.columns:
        out["vwap"] = (out["high"] + out["low"] + out["close"]) / 3

    for w in windows:
        vwap_ma = out["vwap"].rolling(w, min_periods=1).mean()
        col = f"vwap_dev_{w}d"
        VWAP_DEV_COLS.append(col)
        out[col] = ((out["close"] - vwap_ma) / (vwap_ma + 1e-9)).fillna(0.0)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Volume-price correlation
# ─────────────────────────────────────────────────────────────────────────────

VOL_PRICE_COL = "vol_price_corr"


def add_volume_price_correlation(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Add rolling correlation between volume and return."""
    out = df.copy()
    out["return"] = out["return"].fillna(0.0)

    corr = out["return"].rolling(window, min_periods=5).corr(out["log_volume"])
    out[VOL_PRICE_COL] = corr.fillna(0.0)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Order flow imbalance proxy
# ─────────────────────────────────────────────────────────────────────────────

OFI_PROXY_COL = "ofi_proxy"


def add_ofi_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Add order flow imbalance proxy.

    OFI ≈ (close - open) / (high - low).
    Positive = buyers in control, Negative = sellers in control.
    """
    out = df.copy()
    hl = out["high"] - out["low"]
    out[OFI_PROXY_COL] = ((out["close"] - out["open"]) / (hl + 1e-9)).fillna(0.0)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Realized volatility ratios
# ─────────────────────────────────────────────────────────────────────────────

RVOL_RATIOS: List[str] = []


def add_realized_vol_ratios(
    df: pd.DataFrame,
    short: int = 20,
    long: int = 63,
) -> pd.DataFrame:
    """Add realized volatility ratios.

    rv_short / rv_long — short-term vol relative to long-term vol.
    > 1  → expanding volatility
    < 1  → contracting volatility
    """
    out = df.copy()
    out["return"] = out["return"].fillna(0.0)
    rv_short = out["return"].rolling(short, min_periods=5).std()
    rv_long = out["return"].rolling(long, min_periods=10).std()

    col = f"rvol_ratio_{short}d_{long}d"
    RVOL_RATIOS.append(col)
    out[col] = (rv_short / (rv_long + 1e-9)).fillna(1.0)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  GARCH-like volatility features
# ─────────────────────────────────────────────────────────────────────────────

GARCH_COLS: List[str] = []


def add_garch_features(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Add GARCH(1,1)-like features from returns.

    - squared returns mean (conditional variance proxy)
    - absolute returns (asymmetric volatility)
    - squared return × previous squared return (vol clustering)
    """
    out = df.copy()
    out["return"] = out["return"].fillna(0.0)
    sq_ret = out["return"] ** 2
    abs_ret = out["return"].abs()

    GARCH_COLS.extend(["sq_ret_mean", "abs_ret_mean", "vol_clustering"])

    out["sq_ret_mean"] = sq_ret.rolling(window, min_periods=5).mean().fillna(0.0)
    out["abs_ret_mean"] = abs_ret.rolling(window, min_periods=5).mean().fillna(0.0)

    # Vol clustering: correlation of squared returns with their lag
    sq_lag = sq_ret.shift(1)
    out["vol_clustering"] = sq_ret.rolling(window, min_periods=5).corr(sq_lag).fillna(0.0)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Higher-order moments
# ─────────────────────────────────────────────────────────────────────────────

MOMENT_COLS: List[str] = []


def add_moments(
    df: pd.DataFrame,
    windows: List[int] | None = None,
) -> pd.DataFrame:
    """Add skewness and kurtosis of returns.

    - Skewness > 0 → right-tail heavy (crash risk)
    - Kurtosis > 3 → fat tails (more extreme moves than normal)
    """
    if windows is None:
        windows = [20, 60]

    out = df.copy()
    out["return"] = out["return"].fillna(0.0)

    for w in windows:
        for name, func in [("skew", "skew"), ("kurt", "kurt")]:
            col = f"{name}_ret_{w}d"
            MOMENT_COLS.append(col)
            out[col] = (
                out["return"]
                .rolling(w, min_periods=max(w // 2, 5))
                .aggregate(func)
                .fillna(0.0)
            )

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Rolling window statistics
# ─────────────────────────────────────────────────────────────────────────────

ROLLING_STATS_COLS: List[str] = []


def add_rolling_stats(
    df: pd.DataFrame,
    windows: List[int] | None = None,
) -> pd.DataFrame:
    """Add rolling statistics for multiple windows.

    For each window (5, 10, 20, 60), adds:
      - rolling_mean of return
      - rolling_std of return
      - rolling skewness of return
      - rolling kurtosis of return
      - rolling R² of linear regression (return ~ time index)
      - rolling information ratio (mean return / std return)
    """
    if windows is None:
        windows = [5, 10, 20, 60]

    out = df.copy()
    out["return"] = out["return"].fillna(0.0)

    for w in windows:
        # Return statistics
        for stat_name, agg in [("mean", "mean"), ("std", "std")]:
            col = f"ret_{stat_name}_{w}d"
            ROLLING_STATS_COLS.append(col)
            out[col] = out["return"].rolling(w, min_periods=1).aggregate(agg).fillna(0.0)

        for stat_name, agg in [("skew", "skew"), ("kurt", "kurt")]:
            col = f"ret_{stat_name}_{w}d"
            if col not in out.columns:
                ROLLING_STATS_COLS.append(col)
                out[col] = (
                    out["return"]
                    .rolling(w, min_periods=max(w // 2, 3))
                    .aggregate(agg)
                    .fillna(0.0)
                )

        # Rolling R² of return ~ t
        def _r_squared(series: np.ndarray) -> float:
            """Compute R² of linear regression on array."""
            if len(series) < 3:
                return 0.0
            arr = np.asarray(series)
            x = np.arange(len(arr), dtype=float)
            x_mean = np.mean(x)
            y_mean = np.mean(arr)
            ss_tot = np.sum((arr - y_mean) ** 2)
            if ss_tot == 0:
                return 0.0
            slope = np.sum((x - x_mean) * (arr - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-9)
            y_pred = slope * (x - x_mean) + y_mean
            ss_res = np.sum((arr - y_mean - (y_pred - y_mean)) ** 2)
            r2 = 1.0 - ss_res / ss_tot
            return float(np.clip(r2, 0.0, 1.0))

        col = f"ret_r2_{w}d"
        ROLLING_STATS_COLS.append(col)
        out[col] = (
            out["return"]
            .rolling(w, min_periods=3)
            .apply(_r_squared, raw=True)
            .fillna(0.0)
        )

        # Information ratio
        col = f"ret_info_ratio_{w}d"
        ROLLING_STATS_COLS.append(col)
        ret_mean = out["return"].rolling(w, min_periods=1).mean()
        ret_std = out["return"].rolling(w, min_periods=1).std()
        out[col] = (ret_mean / (ret_std + 1e-9)).fillna(0.0)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Combined: add all advanced features
# ─────────────────────────────────────────────────────────────────────────────

ADVANCED_COLUMNS: List[str] = []


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all advanced features in a single pass.

    This is the main entry point. It calls each feature generator in order.
    All features are backward-looking by construction.
    """
    out = df.copy()

    # 1. Interaction terms
    out = add_interaction_terms(out)

    # 2. Hurst exponent
    out = add_hurst_exponent(out, windows=[20, 60])

    # 3. Entropy
    out = add_entropy_features(out, windows=[20, 60])

    # 4. Roll's spread
    out = add_rolls_spread(out)

    # 5. Amihud
    out = add_amihud_illiquidity(out)

    # 6. Price impact
    out = add_price_impact(out)

    # 7. VWAP deviation
    out = add_vwap_deviation(out, windows=[5, 20])

    # 8. Volume-price correlation
    out = add_volume_price_correlation(out)

    # 9. Order flow imbalance
    out = add_ofi_proxy(out)

    # 10. Realized vol ratios
    out = add_realized_vol_ratios(out)

    # 11. GARCH-like
    out = add_garch_features(out)

    # 12. Higher-order moments
    out = add_moments(out)

    # 13. Rolling statistics
    out = add_rolling_stats(out)

    # Collect all generated column names
    ADVANCED_COLUMNS.extend(INTERACTION_COLS + HURST_COLS + ENTROPY_COLS + [
        ROLL_SPREAD_COL, AMIHUD_COL, PRICE_IMPACT_COL,
    ] + VWAP_DEV_COLS + [VOL_PRICE_COL, OFI_PROXY_COL] + RVOL_RATIOS + GARCH_COLS + MOMENT_COLS + ROLLING_STATS_COLS)

    return out
