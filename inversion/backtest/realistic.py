from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    n_trades: int
    gross_return: float
    net_return: float
    max_drawdown: float
    sharpe: float
    sortino: float
    drawdown_control_triggered: bool = False


def backtest_realistic(
    returns: pd.Series | np.ndarray,
    fees_bps: float = 10.0,
    slippage_bps: float = 5.0,
    risk_free: float = 0.0,
    max_dd_limit: float = 0.5,
) -> BacktestMetrics:
    r = np.asarray(returns, dtype=float).ravel()
    cost = (fees_bps + slippage_bps) / 10000.0
    net = r - cost

    n_trades = int((np.abs(r) > 1e-18).sum())
    gross_return = float(np.sum(r))
    net_return = float(np.sum(net))

    cum = np.cumsum(net)
    # max drawdown from running peak (starting at 0)
    peak = 0.0
    max_dd = 0.0
    cur = 0.0
    for x in net:
        cur += x
        if cur > peak:
            peak = cur
        dd = peak - cur
        if dd > max_dd:
            max_dd = dd

    mean = np.mean(net)
    std = np.std(net, ddof=1) if len(net) > 1 else (np.std(net) + 1e-18)
    sharpe = float((mean - risk_free) / std) if std > 1e-18 else 0.0

    neg = net[net < 0]
    if len(neg) > 1:
        down_std = np.std(neg, ddof=1)
    elif len(neg) == 1:
        down_std = abs(neg[0])
    else:
        down_std = abs(mean) if mean < 0 else 1e-18
    down_std = down_std if down_std > 1e-18 else 1e-18
    sortino = float((mean - risk_free) / down_std)

    dd_trigger = max_dd > max_dd_limit

    return BacktestMetrics(
        n_trades=n_trades,
        gross_return=gross_return,
        net_return=net_return,
        max_drawdown=float(max_dd),
        sharpe=sharpe,
        sortino=sortino,
        drawdown_control_triggered=bool(dd_trigger),
    )
