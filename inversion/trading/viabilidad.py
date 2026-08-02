"""Evaluación honesta de viabilidad — backtest out-of-sample (TRADE-003).

El backtest de TRADE-002 (reports/backtest/backtest_report.md) es IN-SAMPLE:
backtest_ticker aplica sobre TODO el histórico un modelo entrenado con ese
mismo histórico, por eso arroja retornos absurdos (AAPL 1.7M%, cartera 251k%).
Eso es sobreajuste, no rendimiento real.

Este módulo re-entrena cada modelo con el primer 70% cronológico de la serie
(split temporal, sin shuffle) y backtestea SOLO el último 30%: el segmento
test nunca lo vio el modelo. Con esos números escribe
reports/backtest/VIABILIDAD.md: ¿supera la estrategia a buy&hold con una
evaluación honesta?

Se eligió un split simple 70/30 en vez de walk-forward por simplicidad y
coste: un solo retrain por ticker (segundos) y un test de ~30% del histórico
(~5 años). El walk-forward re-entrenaría en cada ventana y añade poco a la
pregunta de viabilidad que responde este informe.

CLI: .venv/bin/python inversion/trading/viabilidad.py [--ticker T [T ...]]
(make viabilidad → todo el catálogo)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from inversion.models.predict_model import FEATURE_COLS, SENTIMENT_COLS, _load_ticker_data
from inversion.trading.backtest import run_backtest
from inversion.trading.portfolio import load_catalog
from inversion.trading.signals import DEFAULT_THRESHOLD
from inversion.utils import paths

TEST_FRACTION = 0.3  # último 30% cronológico = segmento test out-of-sample
DEFAULT_CAPITAL = 10_000.0
RANDOM_STATE = 42

LIMITATIONS = [
    "- **Costes de transacción (comisiones y spread)**: la simulación entra y sale sin pagar "
    "comisiones ni spread. Con un coste de 0.1-0.3% por operación (típico en brokers retail), "
    "una estrategia con muchas operaciones pierde gran parte de su ventaja; la columna "
    "'operaciones' permite estimar el impacto.",
    "- **Slippage**: el backtest opera al precio de cierre exacto. En la práctica la ejecución "
    "se desplaza contra el operador, más cuanto menos líquido sea el activo y mayor el tamaño "
    "de la orden.",
    "- **Sobreajuste (overfitting)**: aunque este backtest es out-of-sample, el modelo y las "
    "features se eligieron mirando el histórico completo (incluido este segmento test en "
    "iteraciones previas). El proceso de selección no es a prueba de sobreajuste: cualquier "
    "resultado nuevo debe tratarse como hipótesis hasta validarse en datos realmente no vistos.",
    "- **Forward-looking bias**: el target (sube >2% en 5 días) usa precios futuros solo para "
    "etiquetar el entrenamiento, nunca como feature. Aun así, cualquier fuga no detectada en "
    "features o en el split inflaría los resultados.",
    "- **Cambio de régimen de mercado**: el modelo se entrena con el pasado y se evalúa en una "
    "ventana concreta. Un cambio de comportamiento del mercado (crisis, burbuja, liquidez) puede "
    "invalidar lo aprendido; el segmento test cubre un solo régimen.",
    "- **Horizonte del target (5 días) vs. frecuencia de trading (diaria)**: el modelo predice "
    "si el precio sube >2% en 5 días, pero la simulación decide comprar/vender con la señal "
    "diaria. La desconexión entre el horizonte de la predicción y la frecuencia de las "
    "operaciones produce señales que el modelo no fue entrenado a optimizar.",
    "- **Colas del mercado**: se opera al cierre sin modelar huecos de apertura (gaps) ni "
    "movimiento intradía; el precio de ejecución real puede diferir del cierre.",
    "- **Capital pequeño**: se asume que se invierte cualquier fracción del capital en cada "
    "ticker. Con capital pequeño, los costes fijos por operación y el redondeo a acciones "
    "enteras cambian materialmente el resultado.",
    "- **Métricas de cartera aproximadas**: la fila CARTERA es la media ponderada (pesos "
    "iguales) de las métricas por ticker, igual que en backtest_report.md; no es una simulación "
    "conjunta de la cartera (correlaciones, rebalanceo y sus costes no se modelan).",
    "- **Sesgo de supervivencia**: el catálogo (AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA) son "
    "ganadores conocidos de los últimos 20 años. Elegir activos que ya se sabe que subieron "
    "sobreestima lo que habría hecho la estrategia en tiempo real.",
]


@dataclass
class OOSResult:
    """Métricas out-of-sample de un ticker (segmento test, nunca visto)."""

    ticker: str
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    buy_hold_return: float
    n_trades: int
    n_test_days: int
    test_start: str
    test_end: str


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Features del modelo: las técnicas + sentimiento si el dataset lo trae."""
    cols = list(FEATURE_COLS)
    if all(c in df.columns for c in SENTIMENT_COLS):
        cols += SENTIMENT_COLS
    return cols


def _train_oos_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """RandomForest con los mismos parámetros que train_model, sin guardar en disco."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=50,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def backtest_oos_ticker(
    ticker: str,
    test_fraction: float = TEST_FRACTION,
    initial_capital: float = DEFAULT_CAPITAL,
    threshold: float = DEFAULT_THRESHOLD,
) -> OOSResult | None:
    """Backtest out-of-sample de un ticker.

    Entrena con el primer (1 - test_fraction) del histórico y simula SOLO
    sobre el último test_fraction (segmento test temporal). Reutiliza
    run_backtest (TRADE-001) para la simulación y las métricas.
    """
    try:
        df = _load_ticker_data(ticker)
    except FileNotFoundError as exc:
        print(f"   ✘ {ticker}: {exc}", file=sys.stderr)
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feature_cols = _feature_cols(df)
    rows = df.dropna(subset=[*feature_cols, "target"]).sort_values("timestamp").reset_index(drop=True)
    if len(rows) < 10:
        print(f"   ✘ {ticker}: pocas filas válidas ({len(rows)})", file=sys.stderr)
        return None

    split = int(len(rows) * (1.0 - test_fraction))
    train, test = rows.iloc[:split], rows.iloc[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    model = _train_oos_model(X_train, train["target"])

    X_test = scaler.transform(test[feature_cols])
    probs = pd.Series(model.predict_proba(X_test)[:, 1], index=test.index)
    prices = test["close"]

    result = run_backtest(prices, probs, initial_capital, threshold)
    return OOSResult(
        ticker=ticker.upper(),
        total_return=result.total_return,
        cagr=result.cagr,
        sharpe=result.sharpe,
        max_drawdown=result.max_drawdown,
        win_rate=result.win_rate,
        buy_hold_return=result.buy_hold_return,
        n_trades=result.n_trades,
        n_test_days=len(test),
        test_start=str(test["timestamp"].iloc[0].date()),
        test_end=str(test["timestamp"].iloc[-1].date()),
    )


def build_report(
    tickers: list[str] | None = None,
    test_fraction: float = TEST_FRACTION,
    initial_capital: float = DEFAULT_CAPITAL,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Backtest out-of-sample de cada ticker → DataFrame resumen por ticker."""
    tickers = tickers or load_catalog()
    rows = []
    for ticker in tickers:
        result = backtest_oos_ticker(ticker, test_fraction, initial_capital, threshold)
        if result is None:
            continue
        rows.append(
            {
                "ticker": result.ticker,
                "total_return": result.total_return,
                "cagr": result.cagr,
                "sharpe": result.sharpe,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "buy_hold_return": result.buy_hold_return,
                "n_trades": result.n_trades,
                "n_test_days": result.n_test_days,
                "test_start": result.test_start,
                "test_end": result.test_end,
            }
        )
    return pd.DataFrame(rows)


def _portfolio_row(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Fila 'CARTERA': media ponderada (pesos iguales) de las métricas por ticker."""
    n = float(len(df))
    if n == 0:
        raise ValueError("no hay filas para construir la cartera")
    return {
        "ticker": "CARTERA",
        "total_return": float(df["total_return"].mean()),
        "cagr": float(df["cagr"].mean()),
        "sharpe": float(df["sharpe"].mean()),
        "max_drawdown": float(df["max_drawdown"].mean()),
        "win_rate": float(df["win_rate"].mean()),
        "buy_hold_return": float(df["buy_hold_return"].mean()),
        "n_trades": int(df["n_trades"].sum()),
        "n_test_days": int(df["n_test_days"].sum()),
        "test_start": str(df["test_start"].min()),
        "test_end": str(df["test_end"].max()),
    }


def _conclusion_lines(df: pd.DataFrame) -> list[str]:
    """Conclusión con números: ¿supera la estrategia a buy&hold out-of-sample?"""
    per_ticker = df[df["ticker"] != "CARTERA"]
    cartera = df[df["ticker"] == "CARTERA"].iloc[0]
    n = len(per_ticker)
    n_beat = int((per_ticker["total_return"] > per_ticker["buy_hold_return"]).sum())
    cartera_ret = float(cartera["total_return"])
    cartera_bh = float(cartera["buy_hold_return"])
    cartera_beats = cartera_ret > cartera_bh
    majority_beats = n_beat > n / 2

    lines = ["## Conclusión", ""]
    if cartera_beats and majority_beats:
        lines += [
            f"**La estrategia SUPERA a buy&hold en el segmento test:** la cartera "
            f"(pesos iguales) rinde {cartera_ret:.2%} frente a {cartera_bh:.2%} de "
            f"buy&hold, y {n_beat} de {n} tickers superan a su buy&hold.",
            "",
            "La ventaja es real en esta ventana concreta, pero el backtest no modela "
            "costes, slippage ni cambios de régimen (ver Limitaciones), y el modelo se "
            "eligió mirando el histórico. Recomendación prudente: seguir solo en paper "
            "trading hasta que la ventaja se reproduzca en datos realmente nuevos.",
        ]
    else:
        lines += [
            f"**La estrategia NO supera a buy&hold en el segmento test "
            f"(out-of-sample):** la cartera (pesos iguales) rinde {cartera_ret:.2%} "
            f"frente a {cartera_bh:.2%} de comprar y mantener, y solo {n_beat} de "
            f"{n} tickers superan a su buy&hold.",
            "",
            "Los retornos espectaculares del backtest in-sample de TRADE-002 "
            "(p. ej. AAPL 1.7M%, cartera 251k%) eran sobreajuste: el modelo se "
            "evaluaba sobre los mismos datos con los que se entrenó. Con evaluación "
            "honesta desaparecen. **No invertir capital real**; la estrategia queda "
            "como ejercicio académico y, como mucho, paper trading para seguir "
            "aprendiendo.",
        ]
    return lines


def write_report(df: pd.DataFrame, out_dir: Path) -> Path:
    """Escribe reports/backtest/VIABILIDAD.md con la evaluación honesta."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_pct = 100.0 * (1.0 - TEST_FRACTION)

    lines: list[str] = [
        "# Evaluación de viabilidad — ¿la estrategia supera a buy&hold? (TRADE-003)",
        "",
        f"Generado: {pd.Timestamp.now():%Y-%m-%d %H:%M}",
        "",
        f"Backtest **out-of-sample**: por ticker se re-entrena el modelo con el "
        f"primer {train_pct:.0f}% cronológico de la serie y se simula SOLO sobre el "
        f"último {100.0 * TEST_FRACTION:.0f}% (segmento test, nunca visto por el "
        f"modelo). Las métricas las calcula `run_backtest` (TRADE-001) sobre ese "
        f"segmento; buy&hold se mide en el MISMO segmento test, no sobre todo el "
        f"histórico.",
        "",
    ]
    lines += _conclusion_lines(df)
    lines += [
        "",
        "## Resultados por ticker (segmento test)",
        "",
        "| ticker | retorno estrategia | retorno buy&hold | CAGR | Sharpe | maxDD | win rate | operaciones | ventana test |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['ticker']} | {row['total_return']:.2%} | {row['buy_hold_return']:.2%} "
            f"| {row['cagr']:.2%} | {row['sharpe']:.2f} | {row['max_drawdown']:.2%} "
            f"| {row['win_rate']:.2%} | {row['n_trades']} | {row['test_start']} → {row['test_end']} |"
        )
    lines += ["", "## Limitaciones", ""]
    lines += LIMITATIONS

    md_path = out_dir / "VIABILIDAD.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest out-of-sample y evaluación de viabilidad → reports/backtest/VIABILIDAD.md")
    parser.add_argument("--ticker", nargs="*", default=None, help="Ticker(s); por defecto todo el catálogo")
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION, help="Fracción final de la serie usada como test")
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--out", type=Path, default=None, help="Directorio de salida (por defecto reports/backtest/)")
    args = parser.parse_args(argv)

    tickers = [t.upper() for t in args.ticker] if args.ticker else None
    out_dir = args.out if args.out is not None else paths.REPORTS_DIR / "backtest"
    print(f"▶ Viabilidad out-of-sample (test_fraction={args.test_fraction:.0%}): {', '.join(tickers) if tickers else 'todo el catálogo'}")

    df = build_report(tickers, test_fraction=args.test_fraction, initial_capital=args.initial_capital)
    if df.empty:
        print("   No hay datos con features para ningún ticker. Ejecuta make data && make features primero.", file=sys.stderr)
        return 1

    df = pd.concat([df, pd.DataFrame([_portfolio_row(df)])], ignore_index=True)
    md_path = write_report(df, out_dir)
    print(f"   Informe: {md_path}")
    for _, row in df.iterrows():
        print(
            f"   {row['ticker']:<8} retorno={row['total_return']:>8.2%}  "
            f"buy&hold={row['buy_hold_return']:>8.2%}  Sharpe={row['sharpe']:>6.2f}  "
            f"maxDD={row['max_drawdown']:>8.2%}  win={row['win_rate']:>7.2%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
