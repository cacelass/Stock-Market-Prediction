"""test_backtest.py — Métricas del backtest verificadas a mano (TRADE-001)."""

import numpy as np
import pandas as pd
import pytest

from inversion.trading.backtest import BacktestResult, backtest_ticker, run_backtest


def test_always_buy_rising_price_returns_price_gain():
    # Entra en 100 y se mantiene: 1.1 * 1.1 - 1 = 21%
    prices = pd.Series([100.0, 110.0, 121.0])
    probs = pd.Series([1.0, 1.0, 1.0])
    result = run_backtest(prices, probs, initial_capital=10_000.0)
    assert result.total_return == pytest.approx(0.21)
    assert result.final_equity == pytest.approx(12_100.0)
    assert result.n_trades == 1


def test_always_buy_matches_buy_and_hold():
    prices = pd.Series([100.0, 110.0, 121.0])
    probs = pd.Series([1.0, 1.0, 1.0])
    result = run_backtest(prices, probs)
    assert result.total_return == pytest.approx(result.buy_hold_return)
    pd.testing.assert_series_equal(result.equity, result.buy_hold_equity)


def test_constant_price_zero_metrics():
    prices = pd.Series([100.0, 100.0, 100.0, 100.0])
    probs = pd.Series([1.0, 1.0, 1.0, 1.0])
    result = run_backtest(prices, probs)
    assert result.total_return == 0.0
    assert result.sharpe == 0.0
    assert result.max_drawdown == 0.0


def test_sell_signal_exits_before_drop():
    # Entra en 100, la SELL del día 2 cierra en 120; el día 3 cae a 60
    # pero ya estamos en cash. +20% de la estrategia vs -40% de buy&hold.
    prices = pd.Series([100.0, 120.0, 60.0])
    probs = pd.Series([0.9, 0.1, 0.1])
    result = run_backtest(prices, probs)
    assert result.total_return == pytest.approx(0.20)
    assert result.buy_hold_return == pytest.approx(-0.40)
    assert result.n_trades == 1


def test_hold_keeps_position_without_reentry():
    # HOLD en el día 2 mantiene la posición desde 100; el BUY del día 3
    # no re-entra (ya estamos en largo) → 21% con una sola operación.
    prices = pd.Series([100.0, 110.0, 121.0])
    probs = pd.Series([0.9, 0.5, 0.9])
    result = run_backtest(prices, probs, threshold=0.6)
    assert result.n_trades == 1
    assert result.total_return == pytest.approx(0.21)


def test_win_rate_fraction_of_signal_days_that_rose():
    # Días con señal (no HOLD): 1, 2, 3. Subieron: 110>100 ✓, 90>110 ✗, 120>90 ✓ → 2/3
    prices = pd.Series([100.0, 110.0, 90.0, 120.0])
    probs = pd.Series([1.0, 1.0, 1.0, 1.0])
    result = run_backtest(prices, probs)
    assert result.win_rate == pytest.approx(2 / 3)


def test_max_drawdown_propagated():
    # Equity [10000, 9000, 8000] → 8000/10000 - 1 = -20%
    prices = pd.Series([100.0, 90.0, 80.0])
    probs = pd.Series([1.0, 1.0, 1.0])
    result = run_backtest(prices, probs)
    assert result.max_drawdown == pytest.approx(-0.20)


def test_cagr_one_year_equals_total_return():
    # 253 precios = 252 días de trading → exponente 252/252 = 1 → CAGR = retorno total = 10%
    prices = pd.Series(np.linspace(100.0, 110.0, 253))
    probs = pd.Series([1.0] * 253)
    result = run_backtest(prices, probs)
    assert result.total_return == pytest.approx(0.10)
    assert result.cagr == pytest.approx(0.10)


def test_sharpe_hand_computed():
    # Equity [10000, 11000, 10500, 11500] → retornos [0.1, -0.0454545, 0.0952381]
    # mean = 0.0499279, std(ddof=1) = 0.0826379
    # sharpe = 0.0499279 / 0.0826379 * √252 ≈ 9.591
    prices = pd.Series([100.0, 110.0, 105.0, 115.0])
    probs = pd.Series([1.0, 1.0, 1.0, 1.0])
    result = run_backtest(prices, probs)
    assert result.sharpe == pytest.approx(9.591, abs=1e-2)


def test_backtest_ticker_end_to_end(patch_paths, sample_df):
    """Con modelo+scaler guardados (patrón de test_predict_model), el backtest corre."""
    from inversion.features.build_features import add_derived_features, fit_and_save_scaler
    from inversion.models.predict_model import FEATURE_COLS
    from inversion.models.train_model import train_rf_model

    df = add_derived_features(sample_df.copy())
    feat = df.dropna(subset=FEATURE_COLS)
    X = feat[FEATURE_COLS].values
    y = (feat["return"] > 0).astype(int).values
    fit_and_save_scaler(X, filename="scaler_AAPL.pkl")
    train_rf_model(X, y, filename="rf_AAPL.pkl")

    out = patch_paths["INTERIM_DATA_DIR"] / "features_AAPL_ml_ready.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    result = backtest_ticker("AAPL")
    assert isinstance(result, BacktestResult)
    assert result.initial_capital == 10_000.0
    assert np.isfinite(result.total_return)
    assert np.isfinite(result.sharpe)
