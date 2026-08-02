"""Paper trading: cartera en papel que aplica las señales diarias (TRADE-002).

Reutiliza signals.signal_from_probability para convertir probabilidades en
señales y portfolio.allocate_capital para repartir el capital inicial entre
los tickers (long/out independiente por ticker con su presupuesto):

- BUY  → comprar con todo el presupuesto del ticker si no hay posición.
- SELL → vender toda la posición al cierre.
- HOLD → mantener (posición o cash).

Cada operación se registra con fecha, ticker, señal, precio, cantidad y
capital implicado. El estado final (cash + tenencias valoradas al último
cierre) devuelve la valoración de la cartera.

CLI: .venv/bin/python inversion/trading/paper.py [--ticker T [T ...]]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import pandas as pd

from inversion.models.predict_model import _load_ticker_data, predict_future
from inversion.trading.portfolio import allocate_capital, load_catalog
from inversion.trading.signals import DEFAULT_THRESHOLD, Signal, signal_from_probability


@dataclass
class Trade:
    """Operación registrada por la cartera en papel."""

    date: pd.Timestamp
    ticker: str
    signal: Signal
    price: float
    shares: float
    capital: float


@dataclass
class PaperState:
    """Estado final de la cartera tras la simulación."""

    cash: float
    holdings: dict[str, float]
    equity: float
    trades: list[Trade] = field(default_factory=list)


class PaperPortfolio:
    """Cartera en papel multi-ticker con presupuesto por ticker."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        weights: dict[str, float] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.weights = weights
        self.threshold = threshold
        self.cash = self.initial_capital
        self.holdings: dict[str, float] = {}
        self.trades: list[Trade] = []

    def run(
        self,
        data: dict[str, tuple[pd.Series, pd.Series]],
        n_days: int | None = None,
    ) -> PaperState:
        """Aplica las señales diarias sobre el histórico de cada ticker.

        Args:
            data : {ticker: (prices, probs)} alineados por índice.
            n_days : si se indica, solo se usan los últimos n_days días.

        Returns:
            PaperState con cash, tenencias, valoración y el registro de
            operaciones (también accesible en self.trades).
        """
        tickers = list(data)
        budgets = allocate_capital(self.initial_capital, tickers=tickers, weights=self.weights)
        self.cash = self.initial_capital
        self.holdings = {}
        self.trades = []
        last_prices: dict[str, float] = {}

        for ticker in tickers:
            prices, probs = data[ticker]
            common = prices.index.intersection(probs.index)
            prices = prices.reindex(common)
            probs = probs.reindex(common)
            if n_days is not None:
                prices = prices.iloc[-n_days:]
                probs = probs.iloc[-n_days:]
            if len(prices) == 0:
                continue
            self._run_ticker(ticker, prices, probs, budgets[ticker])
            last_prices[ticker] = float(prices.iloc[-1])

        equity = self.cash + sum(sh * last_prices[t] for t, sh in self.holdings.items())
        return PaperState(
            cash=self.cash,
            holdings=dict(self.holdings),
            equity=equity,
            trades=list(self.trades),
        )

    def _run_ticker(
        self,
        ticker: str,
        prices: pd.Series,
        probs: pd.Series,
        budget: float,
    ) -> None:
        """Long/out sobre un ticker con su presupuesto, registrando operaciones."""
        ticker_cash = budget
        shares = 0.0
        for date, (price, prob) in zip(prices.index, zip(prices.values, probs.values)):
            sig = signal_from_probability(float(prob), self.threshold)
            if sig is Signal.BUY and shares == 0.0:
                shares = ticker_cash / float(price)
                self.trades.append(Trade(date, ticker, sig, float(price), shares, ticker_cash))
                self.cash -= ticker_cash
                ticker_cash = 0.0
            elif sig is Signal.SELL and shares > 0.0:
                proceeds = shares * float(price)
                self.trades.append(Trade(date, ticker, sig, float(price), shares, proceeds))
                self.cash += proceeds
                ticker_cash = proceeds
                shares = 0.0
        if shares > 0.0:
            self.holdings[ticker] = shares


def today_signal(
    ticker: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[Signal | None, float | None, float | None]:
    """Señal de hoy para un ticker: (señal, prob_subida, último cierre).

    Devuelve (None, None, None) si el ticker no tiene modelo entrenado.
    """
    df = _load_ticker_data(ticker)
    _, _, prob_sube, close = predict_future(df, ticker=ticker)
    if prob_sube is None or close is None:
        return None, None, None
    return signal_from_probability(prob_sube, threshold), float(prob_sube), float(close)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paper trading: señales de hoy.")
    parser.add_argument("--ticker", nargs="*", default=None, help="Ticker(s); por defecto todo el catálogo")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Umbral de señal")
    args = parser.parse_args(argv)

    tickers = [t.upper() for t in args.ticker] if args.ticker else load_catalog()
    print("▶ Señales de hoy (paper trading)")
    for ticker in tickers:
        try:
            sig, prob, close = today_signal(ticker, args.threshold)
        except FileNotFoundError as exc:
            print(f"   ✘ {ticker}: {exc}")
            continue
        if sig is None:
            print(f"   ✘ {ticker}: sin modelo entrenado — ejecuta make train")
            continue
        print(f"   {ticker:<6} {sig.value:<5} (p_subida={prob:.2%}, cierre={close:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
