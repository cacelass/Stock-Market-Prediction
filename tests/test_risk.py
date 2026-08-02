"""test_risk.py — Position sizing y drawdown (TRADE-001)."""

import pandas as pd
import pytest

from inversion.trading.risk import max_drawdown, position_size


def test_position_size_fixed_fraction():
    # 2% por defecto de 1000 = 20
    assert position_size(1000) == 20.0
    assert position_size(1000, risk_per_trade=0.1) == 100.0


def test_position_size_with_stop_loss():
    # Arriesgar el 2% de 1000 (20) con stop al 10% → invertir 200
    assert position_size(1000, risk_per_trade=0.02, stop_loss_pct=0.1) == 200.0


def test_max_drawdown_increasing_series_is_zero():
    equity = pd.Series([100.0, 110.0, 121.0, 133.1])
    assert max_drawdown(equity) == 0.0


def test_max_drawdown_flat_series_is_zero():
    assert max_drawdown(pd.Series([100.0, 100.0, 100.0])) == 0.0


def test_max_drawdown_hand_computed():
    # Pico 120, valle 90 → 90/120 - 1 = -0.25
    equity = pd.Series([100.0, 120.0, 100.0, 90.0])
    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_max_drawdown_after_new_high():
    # Nuevo pico 130; caída posterior a 104 → 104/130 - 1 = -0.20
    equity = pd.Series([100.0, 130.0, 110.0, 104.0])
    assert max_drawdown(equity) == pytest.approx(-0.20)
