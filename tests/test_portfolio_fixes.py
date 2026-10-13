"""Tests for edgebet/risk/portfolio.py — BUG-006 fix and portfolio_from_signals.

Covers optimal_portfolio with single/double assets, portfolio_from_signals
with edge cases, and equal-weight fallback when optimizer fails.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from inversion.risk.portfolio import (
    optimal_portfolio,
    portfolio_from_signals,
)


# ── optimal_portfolio single/two assets ────────────────────────────────────

class TestOptimalPortfolioFixes:
    """BUG-006 fix: single asset must not crash the optimizer."""

    def test_single_asset(self):
        """Single asset → weight=1.0, no optimizer needed."""
        mu = np.array([0.05])
        cov = np.array([[0.01]])
        res = optimal_portfolio(mu, cov)
        assert res["n_assets"] == 1
        assert res["weights"][0] == pytest.approx(1.0)
        assert res["sharpe_ratio"] == pytest.approx(0.05 / 0.1)

    def test_single_asset_zero_vol(self):
        """Single asset with very small volatility → positive large sharpe (vol floored at 1e-8)."""
        mu = np.array([0.05])
        cov = np.array([[1e-18]])
        res = optimal_portfolio(mu, cov)
        # vol = sqrt(1e-18) = 1e-9, floored to 1e-8 → sharpe = 0.05/1e-8 = 5e6
        assert res["sharpe_ratio"] == pytest.approx(0.05 / 1e-8)

    def test_two_assets(self):
        """Two assets → weights sum to 1.0 and converge."""
        mu = np.array([0.05, 0.08])
        cov = np.array([[0.01, 0.001], [0.001, 0.02]])
        res = optimal_portfolio(mu, cov)
        assert res["n_assets"] == 2
        assert abs(res["weights"].sum() - 1.0) < 1e-4
        assert res["sharpe_ratio"] > 0

    def test_all_negative_returns(self):
        """All negative returns → picks the least-negative, capped."""
        mu = np.array([-0.05, -0.10])
        cov = np.array([[0.01, 0.0], [0.0, 0.01]])
        res = optimal_portfolio(mu, cov)
        assert abs(res["weights"].sum() - 1.0) < 1e-4
        assert res["converged"] is True

    def test_max_concentration_enforced(self):
        """Weights should not exceed max_concentration (with enough assets for mc to be binding)."""
        # Use 4 assets so mc = max(0.30, 1/4+0.01) = max(0.30, 0.26) = 0.30
        mu = np.array([0.10, 0.01, 0.01, 0.01])
        cov = np.diag([0.01, 0.01, 0.01, 0.01])
        res = optimal_portfolio(mu, cov, max_concentration=0.30)
        assert res["weights"].max() <= 0.30 + 1e-6


# ── portfolio_from_signals ─────────────────────────────────────────────────

class TestPortfolioFromSignals:
    """Tests for portfolio_from_signals edge cases."""

    def test_empty_signals(self):
        """Empty signals → n_signals=0, empty weights."""
        res = portfolio_from_signals([])
        assert res["n_signals"] == 0
        assert len(res["weights"]) == 0
        assert res["stakes"] == {}

    def test_zero_edges(self):
        """All edges are zero (below min_edge 0.001 floor) → still works."""
        signals = [
            {"name": "bet_A", "edge": 0.0, "league": "EPL", "date": "2026-01-01"},
            {"name": "bet_B", "edge": 0.0, "league": "EPL", "date": "2026-01-01"},
        ]
        res = portfolio_from_signals(signals)
        assert res["n_signals"] == 2
        assert len(res["weights"]) == 2
        assert abs(res["weights"].sum() - 1.0) < 1e-4

    def test_with_new_parameters(self):
        """portfolio_from_signals with std_scale, same_league_corr, max_exposure_pct."""
        signals = [
            {"name": "bet_A", "edge": 0.05, "league": "EPL", "date": "2026-01-01"},
            {"name": "bet_B", "edge": 0.08, "league": "EPL", "date": "2026-01-01"},
        ]
        res = portfolio_from_signals(
            signals,
            bankroll=10000,
            std_scale=2.0,
            same_league_corr=0.20,
            max_exposure_pct=0.20,
        )
        assert res["n_signals"] == 2
        # Stakes should be positive and use the new max_exposure_pct
        stakes = list(res["stakes"].values())
        assert all(s > 0 for s in stakes)

    def test_stakes_proportional_to_bankroll(self):
        """Stakes should scale with bankroll."""
        signals = [
            {"name": "bet_A", "edge": 0.05},
            {"name": "bet_B", "edge": 0.08},
        ]
        r1 = portfolio_from_signals(signals, bankroll=10000, max_exposure_pct=0.15)
        r2 = portfolio_from_signals(signals, bankroll=20000, max_exposure_pct=0.15)
        # Doubling bankroll should double stakes
        for name in r1["stakes"]:
            assert r2["stakes"][name] == pytest.approx(r1["stakes"][name] * 2.0, rel=1e-4)

    def test_single_signal(self):
        """Single signal → weight=1.0."""
        signals = [{"name": "solo", "edge": 0.05}]
        res = portfolio_from_signals(signals)
        assert res["n_signals"] == 1
        assert res["weights"][0] == pytest.approx(1.0)

    def test_same_league_correlation(self):
        """Same-league same-day signals should have correlated covariance."""
        signals = [
            {"name": "A", "edge": 0.05, "league": "EPL", "date": "2026-01-01"},
            {"name": "B", "edge": 0.05, "league": "EPL", "date": "2026-01-01"},
        ]
        res = portfolio_from_signals(signals, same_league_corr=0.50)
        # Both have equal edge → equal weights
        assert res["weights"][0] == pytest.approx(res["weights"][1], abs=0.05)
        assert abs(res["weights"].sum() - 1.0) < 1e-4


# ── Fallback when optimizer fails ─────────────────────────────────────────

class TestOptimizerFallback:
    """Verify equal-weight fallback when minimize returns None."""

    def test_fallback_to_equal_weight(self):
        """If minimize returns result with None, fallback to equal weights."""
        with patch("inversion.risk.portfolio.minimize", return_value=None):
            mu = np.array([0.05, 0.08])
            cov = np.array([[0.01, 0.001], [0.001, 0.02]])
            res = optimal_portfolio(mu, cov)
            # With best_result=None, fallback uses ones(n)/n
            assert res["weights"][0] == pytest.approx(0.5)
            assert res["weights"][1] == pytest.approx(0.5)
