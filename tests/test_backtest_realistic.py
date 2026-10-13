from __future__ import annotations
import numpy as np
import pandas as pd

from inversion.backtest.realistic import backtest_realistic


def test_backtest_realistic_metrics() -> None:
    r = pd.Series([0.01, -0.005, 0.02, 0.0])
    m = backtest_realistic(r, fees_bps=10, slippage_bps=5)
    assert m.n_trades == 3
    assert m.max_drawdown >= 0
    assert m.sharpe == m.sharpe  # finite
    assert m.sortino >= 0 or m.sortino == float("inf")


def test_backtest_realistic_costs_reduce_return() -> None:
    r = pd.Series([0.01, 0.01])
    m = backtest_realistic(r, fees_bps=0, slippage_bps=0)
    m2 = backtest_realistic(r, fees_bps=100, slippage_bps=0)
    assert m.net_return >= m2.net_return
