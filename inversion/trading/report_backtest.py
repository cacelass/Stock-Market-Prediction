"""Informe de backtest multi-ticker + cartera → reports/backtest/ (TRADE-002).

Reutiliza backtest_ticker (TRADE-001) para las métricas por ticker y
portfolio.allocate_capital para la comparativa de cartera: con presupuestos
iguales y long/out independiente por ticker, el retorno de la cartera es la
media ponderada (peso = fracción de capital) de los retornos por ticker.

CLI: .venv/bin/python inversion/trading/report_backtest.py [--ticker T [T ...]]
(make backtest → todo el catálogo; make backtest TICKERS=AAPL MSFT → filtro)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from inversion.trading.backtest import backtest_ticker
from inversion.trading.portfolio import allocate_capital, load_catalog
from inversion.utils import paths

DEFAULT_CAPITAL = 10_000.0


def build_report(
    tickers: list[str] | None = None,
    initial_capital: float = DEFAULT_CAPITAL,
) -> pd.DataFrame:
    """Backtest de cada ticker → DataFrame resumen con una fila por ticker."""
    tickers = tickers or load_catalog()
    rows = []
    for ticker in tickers:
        try:
            result = backtest_ticker(ticker, initial_capital=initial_capital)
        except FileNotFoundError as exc:
            print(f"   ✘ {ticker}: {exc}", file=sys.stderr)
            continue
        rows.append(
            {
                "ticker": ticker,
                "total_return": result.total_return,
                "cagr": result.cagr,
                "sharpe": result.sharpe,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "buy_hold_return": result.buy_hold_return,
                "final_equity": result.final_equity,
                "n_trades": result.n_trades,
            }
        )
    return pd.DataFrame(rows)


def _portfolio_row(df: pd.DataFrame, initial_capital: float) -> dict[str, float | int | str]:
    """Fila 'CARTERA': retorno ponderado por el capital asignado por ticker."""
    alloc = allocate_capital(initial_capital, tickers=list(df["ticker"]))
    weights = {t: alloc[t] / initial_capital for t in alloc}
    total_return = sum(weights[t] * (1.0 + r) for t, r in zip(df["ticker"], df["total_return"])) - 1.0
    return {
        "ticker": "CARTERA",
        "total_return": total_return,
        "cagr": float(df["cagr"].mean()),
        "sharpe": float(df["sharpe"].mean()),
        "max_drawdown": float(df["max_drawdown"].mean()),
        "win_rate": float(df["win_rate"].mean()),
        "buy_hold_return": float(df["buy_hold_return"].mean()),
        "final_equity": initial_capital * (1.0 + total_return),
        "n_trades": int(df["n_trades"].sum()),
    }


def write_report(df: pd.DataFrame, out_dir: Path) -> Path:
    """Escribe backtest_report.md (legible) y backtest_summary.csv en out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "backtest_summary.csv", index=False)

    lines = [
        "# Informe de backtest (TRADE-002)",
        "",
        f"Generado: {pd.Timestamp.now():%Y-%m-%d %H:%M}",
        "",
        "| ticker | retorno | CAGR | Sharpe | maxDD | win rate | buy&hold | equity final | operaciones |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['ticker']} | {row['total_return']:.2%} | {row['cagr']:.2%} | {row['sharpe']:.2f} "
            f"| {row['max_drawdown']:.2%} | {row['win_rate']:.2%} | {row['buy_hold_return']:.2%} "
            f"| {row['final_equity']:,.2f} | {row['n_trades']} |"
        )
    md_path = out_dir / "backtest_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest multi-ticker + cartera → reports/backtest/")
    parser.add_argument("--ticker", nargs="*", default=None, help="Ticker(s); por defecto todo el catálogo")
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--out", type=Path, default=None, help="Directorio de salida (por defecto reports/backtest/)")
    args = parser.parse_args(argv)

    tickers = [t.upper() for t in args.ticker] if args.ticker else None
    print(f"▶ Backtest: {', '.join(tickers) if tickers else 'todo el catálogo'}")
    df = build_report(tickers, initial_capital=args.initial_capital)
    if df.empty:
        print("   No hay tickers con modelo entrenado. Ejecuta make train primero.", file=sys.stderr)
        return 1

    df = pd.concat([df, pd.DataFrame([_portfolio_row(df, args.initial_capital)])], ignore_index=True)
    out_dir = args.out if args.out is not None else paths.REPORTS_DIR / "backtest"
    md_path = write_report(df, out_dir)
    print(f"   Informe: {md_path}")
    print(f"   Resumen: {out_dir / 'backtest_summary.csv'}")
    for _, row in df.iterrows():
        print(
            f"   {row['ticker']:<8} retorno={row['total_return']:>8.2%}  "
            f"CAGR={row['cagr']:>8.2%}  Sharpe={row['sharpe']:>6.2f}  "
            f"maxDD={row['max_drawdown']:>8.2%}  win={row['win_rate']:>7.2%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
