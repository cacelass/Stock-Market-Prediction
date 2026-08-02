"""test_signals.py — Conversión de probabilidad a señal (TRADE-001)."""

import pytest

from inversion.trading.signals import DEFAULT_THRESHOLD, Signal, signal_from_probability


def test_buy_when_probability_above_threshold():
    assert signal_from_probability(0.7) is Signal.BUY


def test_sell_when_probability_below_one_minus_threshold():
    assert signal_from_probability(0.3) is Signal.SELL


def test_hold_in_the_middle():
    assert signal_from_probability(0.5) is Signal.HOLD


def test_boundary_buy_at_threshold():
    assert signal_from_probability(DEFAULT_THRESHOLD) is Signal.BUY


def test_boundary_sell_at_one_minus_threshold():
    assert signal_from_probability(1 - DEFAULT_THRESHOLD) is Signal.SELL


def test_threshold_configurable():
    assert signal_from_probability(0.75, threshold=0.8) is Signal.HOLD
    assert signal_from_probability(0.85, threshold=0.8) is Signal.BUY
    assert signal_from_probability(0.1, threshold=0.8) is Signal.SELL


def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        signal_from_probability(0.5, threshold=0.5)
    with pytest.raises(ValueError):
        signal_from_probability(0.5, threshold=1.5)
