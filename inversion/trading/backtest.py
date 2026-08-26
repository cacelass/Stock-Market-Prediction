"""Backtesting: simulación paso a paso de la estrategia long/out por ticker.

Carga el modelo y el scaler reutilizando predict_model (API-001) y convierte
las probabilidades históricas en señales con signals.py. Métricas: retorno
total, CAGR, Sharpe anualizado, max drawdown y win rate, comparadas contra
buy&hold (mantener el activo todo el periodo).

CLI: .venv/bin/python inversion/trading/backtest.py --ticker AAPL
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from inversion.models import pooled
from inversion.models.predict_model import _load_model_and_scaler, _load_ticker_data
from inversion.models.train_model import available_feature_cols
from inversion.trading.risk import max_drawdown
from inversion.trading.signals import DEFAULT_THRESHOLD, Signal, signal_from_probability

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    """Métricas del backtest y serie de equity."""

    initial_capital: float
    final_equity: float
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    buy_hold_return: float
    buy_hold_equity: pd.Series
    equity: pd.Series
    n_trades: int


def _daily_sharpe(equity: pd.Series, risk_free_rate: float) -> float:
    """Sharpe anualizado de la serie de equity (rf diaria = rf / 252).

    Una serie constante tiene Sharpe 0 (no hay volatilidad que premiar).
    """
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    std = excess.std()
    if std == 0.0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def run_backtest(
    prices: pd.Series,
    probs: pd.Series,
    initial_capital: float = 10_000.0,
    threshold: float = DEFAULT_THRESHOLD,
    risk_free_rate: float = 0.0,
    cost_per_trade: float = 0.0,
    slippage: float = 0.0,
    signal_fn: Callable[[int, float, float], Signal] | None = None,
) -> BacktestResult:
    """Simula la estrategia long/out sobre un histórico.

    Args:
        prices : serie de precios de cierre, alineada con probs.
        probs : probabilidad de subida por día (lo que predice el modelo).
        initial_capital : capital inicial de la cuenta.
        threshold : umbral de señal (ver signals.DEFAULT_THRESHOLD).
        risk_free_rate : tasa libre de riesgo anual (por defecto 0).
        cost_per_trade : comisión fraccional por operación (p.ej. 0.001 = 0.1%),
            pagada al comprar y al vender. Por defecto 0 (sin costes).
        slippage : degradación fraccional de ejecución por operación (p.ej.
            0.0005 = 0.05%), aplicada en el mismo sentido que el coste.
        signal_fn : función de señal opcional con contrato
            (i, p, threshold) -> Signal, donde i es la posición del día en
            prices/probs. Por defecto usa signal_from_probability; permite
            inyectar señales que dependen de más datos por día (p.ej. la
            híbrida modelo+sentimiento de TRADE-006).

    Reglas de la simulación:
        - BUY  → entrar en largo al cierre del día si no hay posición.
        - SELL → cerrar la posición al cierre del día.
        - HOLD → mantener la posición actual (o seguir en cash).
    La equity se valora al cierre de cada día; cuando hay posición es
    cash * precio_dia / precio_entrada. Con costes, comprar ejecuta a
    close * (1 + cost + slippage) y vender cobra equity * (1 - cost - slippage).

    win_rate: % de días con señal (BUY o SELL) en los que el precio subió.
    """
    n = len(prices)
    if n == 0:
        raise ValueError("prices no puede estar vacío")
    if len(probs) != n:
        raise ValueError("prices y probs deben tener la misma longitud")

    if signal_fn is None:
        signals = [signal_from_probability(float(p), threshold) for p in probs]
    else:
        signals = [signal_fn(i, float(p), threshold) for i, p in enumerate(probs)]
    equity = np.empty(n, dtype=float)
    cash = float(initial_capital)
    entry_price: float | None = None
    n_trades = 0

    for t in range(n):
        sig = signals[t]
        close = float(prices.iloc[t])
        if sig is Signal.BUY and entry_price is None:
            entry_price = close * (1.0 + cost_per_trade + slippage)
            n_trades += 1
        if sig is Signal.SELL and entry_price is not None:
            equity[t] = cash * close * (1.0 - cost_per_trade - slippage) / entry_price
            cash = equity[t]
            entry_price = None
        elif entry_price is not None:
            equity[t] = cash * close / entry_price
        else:
            equity[t] = cash

    equity_series = pd.Series(equity, index=prices.index)
    final_equity = float(equity_series.iloc[-1])
    total_return = final_equity / initial_capital - 1.0

    n_days = len(equity_series) - 1
    cagr = (1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0 if n_days > 0 else 0.0

    signal_days = [t for t in range(1, n) if signals[t] is not Signal.HOLD]
    wins = [t for t in signal_days if float(prices.iloc[t]) > float(prices.iloc[t - 1])]
    win_rate = len(wins) / len(signal_days) if signal_days else 0.0

    buy_hold_equity = prices / float(prices.iloc[0]) * initial_capital
    buy_hold_return = float(prices.iloc[-1]) / float(prices.iloc[0]) - 1.0

    return BacktestResult(
        initial_capital=initial_capital,
        final_equity=final_equity,
        total_return=total_return,
        cagr=cagr,
        sharpe=_daily_sharpe(equity_series, risk_free_rate),
        max_drawdown=max_drawdown(equity_series),
        win_rate=win_rate,
        buy_hold_return=buy_hold_return,
        buy_hold_equity=buy_hold_equity,
        equity=equity_series,
        n_trades=n_trades,
    )


def backtest_ticker(
    ticker: str,
    df: pd.DataFrame | None = None,
    initial_capital: float = 10_000.0,
    threshold: float = DEFAULT_THRESHOLD,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Backtest de un ticker del catálogo con su modelo entrenado.

    Reutiliza la carga de predict_model (modelo + scaler + datos) en lugar de
    reimplementarla. Los datos con features ya traen la probabilidad por día:
    se escala cada fila válida y se predice con predict_proba.
    """
    if df is None:
        df = _load_ticker_data(ticker)

    # IMP-006: si el routing manda el ticker al pool, se usa el modelo global
    # con sus columnas (scale-free + one-hots constantes por ticker).
    route = pooled.resolve_route(ticker)
    if route == pooled.GLOBAL_TICKER:
        model, scaler, feature_cols = pooled.load_global_bundle(ticker)
        df = pooled.add_ticker_dummies(df, ticker, feature_cols)
    else:
        model, scaler = _load_model_and_scaler(ticker)
        feature_cols = available_feature_cols(df)

    mask = df[feature_cols].notna().all(axis=1)
    X = df.loc[mask, feature_cols]
    probs = pd.Series(model.predict_proba(scaler.transform(X))[:, 1], index=df.index[mask])
    prices = df["close"].reindex(probs.index)

    return run_backtest(prices, probs, initial_capital, threshold, risk_free_rate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest de la estrategia long/out por ticker.")
    parser.add_argument("--ticker", default="NVDA", help="Activo del catálogo (config/tickers.yaml)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Umbral de señal")
    args = parser.parse_args(argv)

    result = backtest_ticker(args.ticker, threshold=args.threshold)
    print(f"▶ Backtest {args.ticker.upper()} (threshold={args.threshold})")
    print(f"   Retorno total: {result.total_return:.2%} | buy&hold: {result.buy_hold_return:.2%}")
    print(f"   CAGR: {result.cagr:.2%} | Sharpe: {result.sharpe:.2f}")
    print(f"   Max drawdown: {result.max_drawdown:.2%} | win rate: {result.win_rate:.2%}")
    print(f"   Equity final: {result.final_equity:,.2f} ({result.n_trades} operaciones)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
