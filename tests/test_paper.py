"""test_paper.py — Paper trading con señales diarias (TRADE-002)."""

import pandas as pd
import pytest

from inversion.trading.paper import PaperPortfolio, PaperState
from inversion.trading.signals import Signal


def _single_ticker_data():
    """4 días: BUY (día 1), HOLD, SELL (día 3), HOLD."""
    prices = pd.Series([100.0, 110.0, 120.0, 120.0])
    probs = pd.Series([0.9, 0.5, 0.1, 0.5])
    return {"AAPL": (prices, probs)}


def test_buy_signal_buys_with_budget():
    # solo BUY: día 1 compra 100 acciones a 100 con todo el presupuesto y mantiene
    prices = pd.Series([100.0, 110.0, 120.0])
    probs = pd.Series([0.9, 0.9, 0.9])
    state = PaperPortfolio(initial_capital=10_000.0).run({"AAPL": (prices, probs)})
    assert state.holdings["AAPL"] == pytest.approx(100.0)
    assert state.cash == pytest.approx(0.0)


def test_sell_signal_sells_and_realizes():
    portfolio = PaperPortfolio(initial_capital=10_000.0)
    state = portfolio.run(_single_ticker_data())
    # día 3 SELL → vende las 100 acciones a 120
    assert state.holdings == {}
    assert state.cash == pytest.approx(12_000.0)
    assert state.equity == pytest.approx(12_000.0)


def test_trades_registry_grows_with_signal_days():
    portfolio = PaperPortfolio(initial_capital=10_000.0)
    state = portfolio.run(_single_ticker_data())
    assert len(state.trades) == 2
    assert len(portfolio.trades) == 2
    assert state.trades[0].signal is Signal.BUY
    assert state.trades[1].signal is Signal.SELL
    trade = state.trades[0]
    assert trade.ticker == "AAPL"
    assert trade.price == pytest.approx(100.0)
    assert trade.shares == pytest.approx(100.0)
    assert trade.capital == pytest.approx(10_000.0)


def test_hold_does_not_trade():
    prices = pd.Series([100.0, 110.0, 120.0])
    probs = pd.Series([0.5, 0.5, 0.5])
    state = PaperPortfolio(initial_capital=10_000.0).run({"AAPL": (prices, probs)})
    assert state.trades == []
    assert state.holdings == {}
    assert state.cash == pytest.approx(10_000.0)


def test_multi_ticker_splits_budget():
    data = {
        "AAPL": (pd.Series([100.0, 110.0, 120.0, 120.0]), pd.Series([0.9, 0.5, 0.1, 0.5])),
        "MSFT": (pd.Series([200.0, 210.0, 220.0, 220.0]), pd.Series([0.9, 0.5, 0.1, 0.5])),
    }
    state = PaperPortfolio(initial_capital=10_000.0).run(data)
    # presupuesto 5000 por ticker: AAPL 50 acciones a 100 vendidas a 120 → 6000
    # MSFT 25 acciones a 200 vendidas a 220 → 5500 → equity 11500
    assert state.equity == pytest.approx(11_500.0)


def test_n_days_limits_history():
    # con n_days=2 el BUY (día 1) queda fuera → nunca se opera
    state = PaperPortfolio(initial_capital=10_000.0).run(_single_ticker_data(), n_days=2)
    assert state.trades == []
    assert state.cash == pytest.approx(10_000.0)


def test_run_returns_paper_state():
    state = PaperPortfolio(initial_capital=10_000.0).run(_single_ticker_data())
    assert isinstance(state, PaperState)


def test_report_writes_markdown_and_csv(tmp_path):
    """Smoke del informe (TRADE-002): write_report crea md + csv."""
    from inversion.trading.report_backtest import write_report

    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "CARTERA"],
            "total_return": [0.10, 0.08],
            "cagr": [0.09, 0.07],
            "sharpe": [1.2, 1.0],
            "max_drawdown": [-0.15, -0.12],
            "win_rate": [0.55, 0.53],
            "buy_hold_return": [0.12, 0.10],
            "final_equity": [11_000.0, 10_800.0],
            "n_trades": [3, 5],
        }
    )
    md_path = write_report(df, tmp_path)
    assert md_path.exists()
    assert (tmp_path / "backtest_summary.csv").exists()
    content = md_path.read_text()
    assert "AAPL" in content and "CARTERA" in content


def test_report_end_to_end_with_fake_backtest(monkeypatch, tmp_path):
    """build_report + main: reutiliza backtest_ticker y añade la fila CARTERA."""
    from inversion.trading import report_backtest
    from inversion.trading.backtest import run_backtest

    def fake_backtest(ticker, initial_capital=10_000.0, threshold=0.6, risk_free_rate=0.0):
        prices = pd.Series([100.0, 110.0], index=pd.date_range("2024-01-01", periods=2))
        probs = pd.Series([0.9, 0.9], index=prices.index)
        return run_backtest(prices, probs, initial_capital=initial_capital)

    monkeypatch.setattr(report_backtest, "backtest_ticker", fake_backtest)
    df = report_backtest.build_report(tickers=["AAPL", "MSFT"])
    assert list(df["ticker"]) == ["AAPL", "MSFT"]

    rc = report_backtest.main(["--out", str(tmp_path)])
    assert rc == 0
    content = (tmp_path / "backtest_report.md").read_text()
    assert "CARTERA" in content
    assert "AAPL" in content and "MSFT" in content


def test_report_skips_ticker_without_model(monkeypatch):
    from inversion.trading import report_backtest

    def missing(ticker, **kwargs):
        raise FileNotFoundError(f"no hay modelo para {ticker}")

    monkeypatch.setattr(report_backtest, "backtest_ticker", missing)
    df = report_backtest.build_report(tickers=["AAPL", "MSFT"])
    assert df.empty


def test_today_signal_uses_predict_future(monkeypatch):
    from inversion.trading import paper

    monkeypatch.setattr(paper, "_load_ticker_data", lambda ticker: None)
    monkeypatch.setattr(paper, "predict_future", lambda df, ticker: (1, 0.2, 0.8, 123.0))
    sig, prob, close = paper.today_signal("AAPL")
    assert sig is Signal.BUY
    assert prob == pytest.approx(0.8)
    assert close == pytest.approx(123.0)


def test_today_signal_missing_model(monkeypatch):
    from inversion.trading import paper

    monkeypatch.setattr(paper, "_load_ticker_data", lambda ticker: None)
    monkeypatch.setattr(paper, "predict_future", lambda df, ticker: (None, None, None, None))
    assert paper.today_signal("AAPL") == (None, None, None)


def test_paper_main_prints_today_signals(monkeypatch, capsys):
    from inversion.trading import paper

    monkeypatch.setattr(paper, "today_signal", lambda ticker, threshold: (Signal.BUY, 0.8, 123.0))
    rc = paper.main(["--ticker", "AAPL"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "AAPL" in out and "BUY" in out


def test_paper_main_reports_missing_data(monkeypatch, capsys):
    from inversion.trading import paper

    def missing(ticker, threshold):
        raise FileNotFoundError("no hay datos")

    monkeypatch.setattr(paper, "today_signal", missing)
    rc = paper.main(["--ticker", "AAPL"])
    assert rc == 0
    assert "✘ AAPL" in capsys.readouterr().out
