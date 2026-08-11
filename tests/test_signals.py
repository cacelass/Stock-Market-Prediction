"""test_signals.py — Conversión de probabilidad a señal (TRADE-001) y señal híbrida (TRADE-006)."""

import pytest

from inversion.trading.signals import (
    DEFAULT_THRESHOLD,
    Signal,
    signal_from_probability,
    signal_from_probability_sentiment,
)


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


# ─────────────────────────────────────────────────────────────────────────────
# Señal híbrida modelo + sentimiento (TRADE-006)
# ─────────────────────────────────────────────────────────────────────────────


def test_hybrid_buy_when_probability_and_sentiment_aligned():
    assert signal_from_probability_sentiment(0.8, sentiment=0.5) is Signal.BUY
    assert signal_from_probability_sentiment(0.6, sentiment=0.1) is Signal.BUY


def test_hybrid_hold_when_probability_high_but_sentiment_against():
    # p alta + sentimiento negativo: el sentimiento veta la compra → HOLD.
    assert signal_from_probability_sentiment(0.8, sentiment=-0.5) is Signal.HOLD


def test_hybrid_sell_when_probability_and_sentiment_aligned_down():
    assert signal_from_probability_sentiment(0.3, sentiment=-0.5) is Signal.SELL
    assert signal_from_probability_sentiment(0.4, sentiment=-0.1) is Signal.SELL


def test_hybrid_hold_when_probability_low_but_sentiment_positive():
    # p baja + sentimiento positivo: el sentimiento veta la venta → HOLD.
    assert signal_from_probability_sentiment(0.3, sentiment=0.5) is Signal.HOLD


def test_hybrid_hold_in_the_middle():
    assert signal_from_probability_sentiment(0.5, sentiment=0.5) is Signal.HOLD


def test_hybrid_requires_strict_sentiment_inequality():
    # sentiment == 0 no es > 0 ni < 0: con el umbral por defecto no hay señal.
    assert signal_from_probability_sentiment(0.8, sentiment=0.0) is Signal.HOLD
    assert signal_from_probability_sentiment(0.3, sentiment=0.0) is Signal.HOLD


def test_hybrid_sentiment_threshold_configurable():
    # sentiment_threshold=0.5: comprar exige sentiment > 0.5.
    assert signal_from_probability_sentiment(0.8, sentiment=0.3, sentiment_threshold=0.5) is Signal.HOLD
    assert signal_from_probability_sentiment(0.8, sentiment=0.7, sentiment_threshold=0.5) is Signal.BUY
    # vender exige sentiment < -0.5.
    assert signal_from_probability_sentiment(0.2, sentiment=-0.3, sentiment_threshold=0.5) is Signal.HOLD
    assert signal_from_probability_sentiment(0.2, sentiment=-0.7, sentiment_threshold=0.5) is Signal.SELL


def test_hybrid_threshold_configurable():
    # threshold=0.8: p=0.7 ya no es BUY aunque el sentimiento sea positivo.
    assert signal_from_probability_sentiment(0.7, sentiment=0.5, threshold=0.8) is Signal.HOLD
    assert signal_from_probability_sentiment(0.85, sentiment=0.5, threshold=0.8) is Signal.BUY
    assert signal_from_probability_sentiment(0.1, sentiment=-0.5, threshold=0.8) is Signal.SELL


def test_hybrid_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        signal_from_probability_sentiment(0.5, sentiment=0.0, threshold=0.5)
    with pytest.raises(ValueError):
        signal_from_probability_sentiment(0.5, sentiment=0.0, threshold=1.5)
    with pytest.raises(ValueError):
        signal_from_probability_sentiment(0.5, sentiment=0.0, sentiment_threshold=-0.1)
