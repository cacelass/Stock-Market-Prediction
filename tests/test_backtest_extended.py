from __future__ import annotations
import numpy as np
import pandas as pd

from inversion.backtest.realistic import backtest_realistic


def test_backtest_with_costs_slippage():
    r = pd.Series([0.01, 0.02, -0.01])
    m = backtest_realistic(r, fees_bps=10, slippage_bps=5)
    assert m.n_trades == 3
    assert m.net_return < m.gross_return + 1e-12
    assert m.max_drawdown >= 0


def test_drawdown_control():
    r = pd.Series([-0.5, -0.1])
    m = backtest_realistic(r, fees_bps=0, slippage_bps=0, max_dd_limit=0.3)
    assert m.drawdown_control_triggered is True
