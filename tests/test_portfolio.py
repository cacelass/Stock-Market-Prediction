"""test_portfolio.py — Asignación de capital y rebalanceo (TRADE-002)."""

import pytest

from inversion.trading.portfolio import allocate_capital, equal_weights, load_catalog, rebalance


def test_equal_weights_sum_to_one():
    weights = equal_weights(["AAPL", "MSFT", "NVDA"])
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(v == pytest.approx(1 / 3) for v in weights.values())


def test_allocate_capital_equal_weights_splits_evenly():
    alloc = allocate_capital(10_000.0, tickers=["AAPL", "MSFT"])
    assert alloc["AAPL"] == pytest.approx(5_000.0)
    assert alloc["MSFT"] == pytest.approx(5_000.0)
    assert sum(alloc.values()) == pytest.approx(10_000.0)


def test_allocate_capital_custom_weights():
    alloc = allocate_capital(10_000.0, weights={"AAPL": 0.75, "MSFT": 0.25})
    assert alloc["AAPL"] == pytest.approx(7_500.0)
    assert alloc["MSFT"] == pytest.approx(2_500.0)


def test_allocate_capital_normalizes_weights():
    # pesos que no suman 1 se normalizan: 2/3 y 1/3 de 9000
    alloc = allocate_capital(9_000.0, weights={"AAPL": 2.0, "MSFT": 1.0})
    assert alloc["AAPL"] == pytest.approx(6_000.0)
    assert alloc["MSFT"] == pytest.approx(3_000.0)


def test_rebalance_target_shares():
    # 50/50 de 10000: 5000 a 100 → 50 acciones; 5000 a 200 → 25
    target = rebalance({"AAPL": 0.5, "MSFT": 0.5}, {"AAPL": 100.0, "MSFT": 200.0}, 10_000.0)
    assert target["AAPL"] == pytest.approx(50.0)
    assert target["MSFT"] == pytest.approx(25.0)


def test_rebalance_changes_when_price_changes():
    t1 = rebalance({"AAPL": 1.0}, {"AAPL": 100.0}, 10_000.0)
    t2 = rebalance({"AAPL": 1.0}, {"AAPL": 200.0}, 10_000.0)
    assert t1["AAPL"] == pytest.approx(100.0)
    assert t2["AAPL"] == pytest.approx(50.0)
    assert t2["AAPL"] != t1["AAPL"]


def test_rebalance_delta_buys_and_sells():
    # objetivo 100 acciones, tenemos 60 → comprar 40
    delta = rebalance({"AAPL": 1.0}, {"AAPL": 100.0}, 10_000.0, current_shares={"AAPL": 60.0})
    assert delta["AAPL"] == pytest.approx(40.0)
    # objetivo 50 acciones, tenemos 80 → vender 30
    delta2 = rebalance({"AAPL": 1.0}, {"AAPL": 200.0}, 10_000.0, current_shares={"AAPL": 80.0})
    assert delta2["AAPL"] == pytest.approx(-30.0)


def test_allocate_capital_defaults_to_catalog():
    # sin tickers ni pesos → catálogo de config/tickers.yaml
    alloc = allocate_capital(7_000.0)
    assert sum(alloc.values()) == pytest.approx(7_000.0)
    assert set(alloc) >= {"AAPL", "NVDA"}


def test_load_catalog_reads_yaml():
    assert "NVDA" in load_catalog()


def test_allocate_capital_raises_on_non_positive_weights():
    with pytest.raises(ValueError):
        allocate_capital(10_000.0, weights={"AAPL": 0.0, "MSFT": 0.0})
