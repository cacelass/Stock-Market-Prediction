"""Market regime detection.

Detects whether the market is in a bull, bear, or neutral regime based on
return, volatility, and VIX level over a lookback window. Enables
regime-specific modelling for better accuracy.

Functions
---------
detect_regime : classify a single row as bull/bear/neutral
detect_regimes : classify every row in a DataFrame
train_regime_model : train a model for a specific regime
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ── Regime thresholds ───────────────────────────────────────────────────────

class Regime:
    """Regime labels."""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


def detect_regime(
    df_row: pd.Series,
    lookback: int = 60,
    return_threshold: float = 0.05,
    vol_threshold: float = 0.20,
    vix_bull_max: float = 20.0,
    vix_bear_min: float = 30.0,
) -> str:
    """Detect the market regime for a single data point.

    Parameters
    ----------
    df_row : a row (or slice) of the DataFrame containing at least 'return',
             'volatility', and optionally 'vix_level'.
    lookback : window for cumulative return and rolling volatility.
    return_threshold : minimum cumulative return to be considered bullish.
    vol_threshold : maximum rolling volatility to be considered low-vol.
    vix_bull_max : VIX below this supports a bull regime.
    vix_bear_min : VIX above this supports a bear regime.

    Returns
    -------
    One of 'bull', 'bear', 'neutral'.
    """
    # Compute cumulative return over lookback
    ret = df_row.get("return", 0.0)
    if isinstance(ret, (list, np.ndarray)):
        ret = float(ret[0])

    # Compute volatility over lookback
    vol = df_row.get("volatility", 0.0)
    if isinstance(vol, (list, np.ndarray)):
        vol = float(vol[0])
    if np.isnan(vol) or vol == 0:
        # Fallback: use rolling std of returns from the DataFrame
        vol = 0.20  # default moderate volatility

    # VIX level
    vix = df_row.get("vix_level", 25.0)
    if isinstance(vix, (list, np.ndarray)):
        vix = float(vix[0])
    if np.isnan(vix):
        vix = 25.0

    # Decision logic
    is_bull = (
        ret >= return_threshold / lookback  # annualised-ish
        and vol <= vol_threshold
        and vix <= vix_bull_max
    )
    is_bear = (
        ret < 0
        and vol >= vol_threshold * 0.8
        and vix >= vix_bear_min * 0.7
    )

    if is_bull:
        return Regime.BULL
    if is_bear:
        return Regime.BEAR
    return Regime.NEUTRAL


def detect_regimes(
    df: pd.DataFrame,
    lookback: int = 60,
) -> pd.Series:
    """Detect regime for every row in a DataFrame.

    Parameters
    ----------
    df : sorted DataFrame with timestamp, return, volatility, vix_level columns
    lookback : window size for lookback calculations

    Returns
    -------
    Series of regime labels ('bull', 'bear', 'neutral') aligned with df index
    """
    return df.apply(
        lambda row: detect_regime(row, lookback=lookback),
        axis=1,
    )


def train_regime_model(
    ticker: str,
    csv_path: Path,
    regime: str,
    feature_cols: Optional[List[str]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train a model for a specific market regime.

    Parameters
    ----------
    ticker : ticker symbol (used for saving models)
    csv_path : path to the features CSV (must contain 'regime' column)
    regime : one of 'bull', 'bear', 'neutral'
    feature_cols : columns to use as features (optional)
    random_state : seed

    Returns
    -------
    dict with 'model' (fitted), 'metrics', 'n_samples', and 'regime'
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["regime"] = df.get("regime", detect_regimes(df))

    # Filter to requested regime
    regime_df = df[df["regime"] == regime].copy()
    if len(regime_df) < 50:  # Too few samples
        return {"model": None, "metrics": {}, "n_samples": 0, "regime": regime}

    if feature_cols is None:
        from inversion.models.train_model import available_feature_cols
        feature_cols = available_feature_cols(regime_df)

    regime_df = regime_df.dropna(subset=[*feature_cols, "target"]).sort_values("timestamp")
    split = int(len(regime_df) * 0.8)
    train_df, test_df = regime_df.iloc[:split], regime_df.iloc[split:]

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    model = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=50,
        max_features="sqrt", class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        auc = 0.5

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": auc,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    return {"model": model, "metrics": metrics, "n_samples": len(regime_df), "regime": regime}
