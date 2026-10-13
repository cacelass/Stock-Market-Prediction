"""
Optimización de portafolio Markowitz para gestión de stakes.
Maximiza Sharpe ratio con restricciones de concentración y VaR.

ESTADO (RISK-002, 2026-09-15): herramienta SOLO investigación/offline.
NO está cableada al pipeline de produccion: el scan usa apply_risk_management
(edgebet/tracking/bankroll.py: Kelly+drawdown + VaR/CVaR) con kelly.py
(KELLY_FRACTION=0.10, MAX_EXPOSURE_PCT=0.15). Unicos importers: tests.
NO activar en produccion sin comparacion OOS vs apply_risk_management —
precedente LIT-007 (kelly_market.py): el staking multi-mercado con covarianza
no supera al quarter-Kelly OOS (ROI -2.66% identico, ver harness/history.md).
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

MAX_CONCENTRATION: float = 0.30
MIN_EDGE_PORTFOLIO: float = 0.02
RISK_FREE_RATE: float = 0.0


def _portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple:
    ret = weights @ mu
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 1e-8 else 0.0
    return ret, vol, sharpe


def _neg_sharpe(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    return -_portfolio_stats(weights, mu, cov)[2]


def optimal_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    max_concentration: float = MAX_CONCENTRATION,
    allow_short: bool = False,
) -> dict:
    n = len(expected_returns)
    if n == 1:
        vol = float(np.sqrt(cov_matrix[0, 0]))
        return {
            "weights": np.array([1.0]),
            "expected_return": float(expected_returns[0]),
            "volatility": vol,
            "sharpe_ratio": float(expected_returns[0] / max(vol, 1e-8)),
            "converged": True,
            "n_assets": 1,
        }
    if expected_returns.max() <= 0:
        w = np.zeros(n)
        w[np.argmax(expected_returns)] = min(1.0, max_concentration)
        w /= w.sum()
        ret, vol, sharpe = _portfolio_stats(w, expected_returns, cov_matrix)
        return {"weights": w, "expected_return": ret, "volatility": vol, "sharpe_ratio": sharpe, "converged": True, "n_assets": n}
    mc = max(max_concentration, 1.0 / n + 0.01)
    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
    ]
    bounds = [(0.0, mc)] * n if not allow_short else [(0.0, 1.0)] * n
    best_result = None
    best_sharpe = -np.inf
    for seed in range(5):
        w0 = np.random.default_rng(seed).dirichlet(np.ones(n), 1)[0]
        w0 = np.clip(w0, 0.0, mc)
        w0 /= w0.sum()
        result = minimize(
            _neg_sharpe, w0, args=(expected_returns, cov_matrix),
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-9},
        )
        if result is None:
            continue
        _, _, sharpe = _portfolio_stats(result.x, expected_returns, cov_matrix)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_result = result
    w = best_result.x if best_result is not None else np.ones(n) / n
    ret, vol, sharpe = _portfolio_stats(w, expected_returns, cov_matrix)
    return {
        "weights": w,
        "expected_return": ret,
        "volatility": vol,
        "sharpe_ratio": sharpe,
        "converged": best_result is not None and best_result.success,
        "n_assets": n,
    }


def efficient_frontier(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_points: int = 20,
    max_concentration: float = MAX_CONCENTRATION,
) -> list[dict]:
    n = len(expected_returns)
    if n <= 1:
        return []
    mc = max(max_concentration, 1.0 / n + 0.01)
    ret_min = expected_returns.min()
    ret_max = expected_returns.max()
    target_returns = np.linspace(ret_min, ret_max, n_points)
    frontier = []
    bounds = [(0.0, mc)] * n
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, t=target: float(w @ expected_returns - t)},
        ]
        best_result = None
        best_vol = np.inf
        for seed in range(5):
            w0 = np.random.default_rng(seed + 42).dirichlet(np.ones(n), 1)[0]
            w0 = np.clip(w0, 0.0, mc)
            w0 /= w0.sum()
            result = minimize(
                lambda w: float(np.sqrt(w @ cov_matrix @ w)),
                w0, method="SLSQP", bounds=bounds, constraints=constraints,
                options={"maxiter": 2000, "ftol": 1e-9},
            )
            if result.success:
                vol = float(np.sqrt(result.x @ cov_matrix @ result.x))
                if vol < best_vol:
                    best_vol = vol
                    best_result = result
        if best_result is not None:
            w = best_result.x
            vol = best_vol
            sharpe = (target - RISK_FREE_RATE) / vol if vol > 1e-8 else 0.0
            frontier.append({"return": target, "volatility": vol, "sharpe": sharpe, "weights": w})
    return frontier


def portfolio_from_signals(
    signals: list[dict],
    bankroll: float = 10000.0,
    max_concentration: float = MAX_CONCENTRATION,
    std_scale: float = 1.5,
    same_league_corr: float = 0.10,
    max_exposure_pct: float = 0.15,
) -> dict:
    if not signals:
        return {"n_signals": 0, "weights": np.array([]), "stakes": {}, "expected_return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0}
    n = len(signals)
    names = [s.get("name", f"bet_{i}") for i, s in enumerate(signals)]
    edges = np.array([max(s.get("edge", 0.0), 0.001) for s in signals], dtype=float)
    stds = edges / std_scale
    mu = edges
    cov = np.diag(stds ** 2)
    for i in range(n):
        for j in range(i + 1, n):
            same_league = signals[i].get("league") and signals[j].get("league") and signals[i]["league"] == signals[j]["league"]
            same_day = signals[i].get("date") and signals[j].get("date") and signals[i]["date"] == signals[j]["date"]
            corr = same_league_corr if (same_league and same_day) else 0.0
            cov[i, j] = corr * stds[i] * stds[j]
            cov[j, i] = cov[i, j]
    opt = optimal_portfolio(mu, cov, max_concentration)
    stakes = {names[i]: float(opt["weights"][i] * bankroll * max_exposure_pct) for i in range(n)}
    return {
        "n_signals": n,
        "weights": opt["weights"],
        "stakes": stakes,
        "expected_return": opt["expected_return"],
        "volatility": opt["volatility"],
        "sharpe_ratio": opt["sharpe_ratio"],
        "converged": opt["converged"],
    }


def allocate_stakes(
    bets_df: pd.DataFrame,
    bankroll: float,
    max_concentration: float = MAX_CONCENTRATION,
) -> pd.DataFrame:
    if bets_df.empty or "bet_signal" not in bets_df.columns:
        return bets_df
    active = bets_df[bets_df["bet_signal"]].copy()
    if active.empty:
        bets_df["markowitz_weight"] = 0.0
        bets_df["markowitz_stake"] = 0.0
        return bets_df
    # Build signals dict from columns directly — no iterrows overhead
    match_ids = [str(idx) for idx in active.index]
    edges = active["edge"].values if "edge" in active.columns else np.full(len(active), 0.0)
    odds_vals = active["odds_market"].values if "odds_market" in active.columns else np.full(len(active), 2.0)
    leagues = active["league"].values if "league" in active.columns else np.full(len(active), "")
    dates = active["date"].values if "date" in active.columns else np.full(len(active), "")
    signals = [
        {"name": n, "edge": float(e), "odds": float(o), "league": str(league), "date": str(d)}
        for n, e, o, league, d in zip(match_ids, edges, odds_vals, leagues, dates)
    ]
    result = portfolio_from_signals(signals, bankroll, max_concentration)
    active["markowitz_weight"] = result["weights"]
    active["markowitz_stake"] = result["stakes"].values()
    bets_df["markowitz_weight"] = 0.0
    bets_df["markowitz_stake"] = 0.0
    bets_df.loc[active.index, "markowitz_weight"] = active["markowitz_weight"]
    bets_df.loc[active.index, "markowitz_stake"] = active["markowitz_stake"]
    return bets_df
