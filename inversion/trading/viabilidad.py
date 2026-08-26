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

Costes y calibración (TRADE-004): la simulación aplica 0.1% de comisión +
0.05% de slippage por operación, y el umbral de señal se calibra por ticker
en un segmento de validación (último 20% del train): se prueba la rejilla
0.51-0.70 y se elige el umbral con mejor Sharpe. El test nunca participa en
la calibración.

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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from inversion.models.predict_model import _load_ticker_data
from inversion.models.train_model import available_feature_cols
from inversion.trading.backtest import run_backtest
from inversion.trading.portfolio import load_catalog
from inversion.trading.signals import DEFAULT_THRESHOLD, Signal, signal_from_probability_sentiment
from inversion.utils import paths

TEST_FRACTION = 0.3  # último 30% cronológico = segmento test out-of-sample
VALIDATION_FRACTION = 0.2  # último 20% del train = segmento de validación (calibración)
DEFAULT_CAPITAL = 10_000.0
RANDOM_STATE = 42

# Costes realistas por operación (entrada y salida), broker retail típico.
COST_PER_TRADE = 0.001  # 0.1% comisión
SLIPPAGE = 0.0005  # 0.05% degradación de ejecución

# Umbrales candidatos para calibrar por ticker (señal BUY con p >= umbral).
THRESHOLD_GRID = np.arange(0.51, 0.71, 0.01)

LIMITATIONS = [
    "- **Costes de transacción (comisiones y spread)**: la simulación aplica un coste "
    "fijo del 0.1% de comisión + 0.05% de slippage por operación (entrada y salida). "
    "Es representativo de brokers retail, pero no modela comisiones fijas por orden "
    "ni spreads que varían con la liquidez: un activo poco líquido o un capital "
    "pequeño cambiarían materialmente el resultado.",
    "- **Slippage**: el backtest aplica un slippage fijo del 0.05% por operación. En la "
    "práctica el slippage se desplaza contra el operador y crece cuanto menos líquido sea "
    "el activo y mayor el tamaño de la orden; un único valor fijo no captura esa variación.",
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
    "- **Umbral de sentimiento fijo en la señal híbrida**: la señal híbrida (TRADE-006) usa "
    "sentiment_threshold = 0.0 fijo; a diferencia del umbral de probabilidad, no se calibra por "
    "ticker. Un umbral de sentimiento calibrado podría cambiar el resultado de la comparativa.",
]


@dataclass
class OOSResult:
    """Métricas out-of-sample de un ticker (segmento test, nunca visto)."""

    ticker: str
    total_return: float
    total_return_no_costs: float
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    buy_hold_return: float
    threshold: float
    n_trades: int
    n_test_days: int
    test_start: str
    test_end: str
    # Señal híbrida modelo+sentimiento (TRADE-006) sobre el MISMO segmento test.
    # None si el dataset no trae columnas de sentimiento.
    hybrid_return: float | None = None
    hybrid_sharpe: float | None = None
    hybrid_n_trades: int | None = None


def _hybrid_signal_fn(sentiment: pd.Series) -> Callable[[int, float, float], Signal]:
    """Señal híbrida por posición: alinea el sentimiento del día i con su probs.

    run_backtest inyecta la señal como (i, p, threshold); aquí se usa i para
    leer el sentimiento del MISMO día (misma posición en el segmento test),
    sin tocar el futuro. El sentimiento del día t se conoce el día t.
    """

    def fn(i: int, p: float, threshold: float) -> Signal:
        return signal_from_probability_sentiment(p, float(sentiment.iloc[i]), threshold)

    return fn


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Features del modelo: delega en la fuente única (train_model)."""
    return available_feature_cols(df)


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


def _calibrate_threshold(
    prices: pd.Series,
    probs: pd.Series,
    initial_capital: float,
    cost_per_trade: float,
    slippage: float,
) -> float:
    """Elige el umbral que maximiza Sharpe en el segmento de validación.

    Recorre THRESHOLD_GRID con run_backtest y se queda con el umbral de mayor
    Sharpe (empate → el más alto, menos operaciones y menos costes). El
    segmento test nunca se usa aquí: solo validación.
    """
    best = DEFAULT_THRESHOLD
    best_score = -np.inf
    for threshold in THRESHOLD_GRID:
        result = run_backtest(
            prices,
            probs,
            initial_capital,
            threshold,
            cost_per_trade=cost_per_trade,
            slippage=slippage,
        )
        if (result.sharpe, threshold) > (best_score, best):
            best_score, best = result.sharpe, threshold
    return float(best)


def backtest_oos_ticker(
    ticker: str,
    test_fraction: float = TEST_FRACTION,
    initial_capital: float = DEFAULT_CAPITAL,
    threshold: float = DEFAULT_THRESHOLD,
    cost_per_trade: float = COST_PER_TRADE,
    slippage: float = SLIPPAGE,
) -> OOSResult | None:
    """Backtest out-of-sample de un ticker.

    Entrena con el primer (1 - test_fraction) del histórico y simula SOLO
    sobre el último test_fraction (segmento test temporal). El umbral de
    señal se calibra por ticker en el último VALIDATION_FRACTION del train
    (maximizando Sharpe) y nunca toca el test. Reutiliza run_backtest
    (TRADE-001) para la simulación y las métricas.
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

    # Segmento de validación: último VALIDATION_FRACTION del train.
    val_start = int(len(train) * (1.0 - VALIDATION_FRACTION))
    val = train.iloc[val_start:]
    X_val = scaler.transform(val[feature_cols])
    val_probs = pd.Series(model.predict_proba(X_val)[:, 1], index=val.index)
    best_threshold = _calibrate_threshold(val["close"], val_probs, initial_capital, cost_per_trade, slippage)

    X_test = scaler.transform(test[feature_cols])
    probs = pd.Series(model.predict_proba(X_test)[:, 1], index=test.index)
    prices = test["close"]

    result = run_backtest(
        prices,
        probs,
        initial_capital,
        best_threshold,
        cost_per_trade=cost_per_trade,
        slippage=slippage,
    )
    # Mismo umbral sin costes: aísla el impacto de los costes para el informe.
    result_no_costs = run_backtest(prices, probs, initial_capital, best_threshold)

    # Señal híbrida modelo+sentimiento (TRADE-006): misma ventana test, mismo
    # umbral calibrado, pero exigiendo que el sentimiento del día apoye a la
    # probabilidad. Solo si el dataset trae sentimiento.
    hybrid_return = hybrid_sharpe = None
    hybrid_n_trades: int | None = None
    if "sentiment_score" in test.columns:
        hybrid = run_backtest(
            prices,
            probs,
            initial_capital,
            best_threshold,
            cost_per_trade=cost_per_trade,
            slippage=slippage,
            signal_fn=_hybrid_signal_fn(test["sentiment_score"]),
        )
        hybrid_return, hybrid_sharpe, hybrid_n_trades = hybrid.total_return, hybrid.sharpe, hybrid.n_trades

    return OOSResult(
        ticker=ticker.upper(),
        total_return=result.total_return,
        total_return_no_costs=result_no_costs.total_return,
        cagr=result.cagr,
        sharpe=result.sharpe,
        max_drawdown=result.max_drawdown,
        win_rate=result.win_rate,
        buy_hold_return=result.buy_hold_return,
        threshold=best_threshold,
        n_trades=result.n_trades,
        n_test_days=len(test),
        test_start=str(test["timestamp"].iloc[0].date()),
        test_end=str(test["timestamp"].iloc[-1].date()),
        hybrid_return=hybrid_return,
        hybrid_sharpe=hybrid_sharpe,
        hybrid_n_trades=hybrid_n_trades,
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
                "total_return_no_costs": result.total_return_no_costs,
                "cagr": result.cagr,
                "sharpe": result.sharpe,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "buy_hold_return": result.buy_hold_return,
                "threshold": result.threshold,
                "n_trades": result.n_trades,
                "n_test_days": result.n_test_days,
                "test_start": result.test_start,
                "test_end": result.test_end,
                "hybrid_return": result.hybrid_return,
                "hybrid_sharpe": result.hybrid_sharpe,
                "hybrid_n_trades": result.hybrid_n_trades,
            }
        )
    return pd.DataFrame(rows)


def _fmt_threshold(value: float | str) -> str:
    """Umbral formateado a 2 decimales ('' para la fila CARTERA)."""
    return "" if isinstance(value, str) else f"{value:.2f}"


def _portfolio_row(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Fila 'CARTERA': media ponderada (pesos iguales) de las métricas por ticker."""
    n = float(len(df))
    if n == 0:
        raise ValueError("no hay filas para construir la cartera")
    return {
        "ticker": "CARTERA",
        "total_return": float(df["total_return"].mean()),
        "total_return_no_costs": float(df["total_return_no_costs"].mean()),
        "cagr": float(df["cagr"].mean()),
        "sharpe": float(df["sharpe"].mean()),
        "max_drawdown": float(df["max_drawdown"].mean()),
        "win_rate": float(df["win_rate"].mean()),
        "buy_hold_return": float(df["buy_hold_return"].mean()),
        "threshold": "",
        "n_trades": int(df["n_trades"].sum()),
        "n_test_days": int(df["n_test_days"].sum()),
        "test_start": str(df["test_start"].min()),
        "test_end": str(df["test_end"].max()),
        # Media de las métricas híbridas; NaN si ningún ticker trae sentimiento.
        "hybrid_return": float(df["hybrid_return"].mean()),
        "hybrid_sharpe": float(df["hybrid_sharpe"].mean()),
        "hybrid_n_trades": int(df["hybrid_n_trades"].sum()),
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
            "cambios de régimen (ver Limitaciones), y el modelo se eligió mirando el "
            "histórico. Recomendación prudente: seguir solo en paper trading hasta "
            "que la ventaja se reproduzca en datos realmente nuevos.",
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
    cartera_ret_no_costs = float(cartera["total_return_no_costs"])
    lines += [
        "",
        f"**Impacto de los costes:** con costes realistas (0.1% comisión + 0.05% "
        f"slippage por operación) y el umbral calibrado por ticker, el retorno de "
        f"la cartera baja de {cartera_ret_no_costs:.2%} a {cartera_ret:.2%}.",
    ]
    if pd.notna(cartera["hybrid_return"]):
        lines += [
            "",
            f"**Señal híbrida (modelo + sentimiento, TRADE-006):** exigir que el "
            f"sentimiento del día apoye a la probabilidad del modelo reduce las "
            f"operaciones; en esta ventana la cartera híbrida rinde "
            f"{float(cartera['hybrid_return']):.2%} frente a "
            f"{cartera_ret:.2%} de solo-modelo (detalle en la sección "
            f"'Señal híbrida vs solo-modelo').",
        ]
    return lines


def _hybrid_section_lines(df: pd.DataFrame) -> list[str]:
    """Sección 'Señal híbrida vs solo-modelo': misma ventana test, mismo umbral."""
    per_ticker = df[df["ticker"] != "CARTERA"]
    cartera = df[df["ticker"] == "CARTERA"].iloc[0]
    solo = float(cartera["total_return"])
    hib = float(cartera["hybrid_return"])
    n = len(per_ticker)
    n_beat = int((per_ticker["hybrid_return"] > per_ticker["total_return"]).sum())

    lines = [
        "## Señal híbrida vs solo-modelo",
        "",
        "La señal híbrida (`signal_from_probability_sentiment`, TRADE-006) exige "
        "que la probabilidad del modelo y el sentimiento del día t estén "
        "**alineados**: BUY si p >= umbral Y sentiment > 0, SELL si p <= 1-umbral "
        "Y sentiment < 0. Se evalúa sobre el **mismo segmento test** y con el "
        "**mismo umbral calibrado** por ticker que la estrategia solo-modelo. El "
        "sentimiento del día t se conoce el día t (sin fuga).",
        "",
        "| ticker | solo-modelo | híbrida | Δ híbrida | operaciones híbrida |",
        "|---|---|---|---|---|",
    ]
    for _, row in per_ticker.iterrows():
        delta = float(row["hybrid_return"]) - float(row["total_return"])
        lines.append(f"| {row['ticker']} | {row['total_return']:.2%} | {row['hybrid_return']:.2%} | {delta:+.2%} | {int(row['hybrid_n_trades'])} |")
    delta = hib - solo
    lines.append(f"| **CARTERA** | **{solo:.2%}** | **{hib:.2%}** | **{delta:+.2%}** | **{int(cartera['hybrid_n_trades'])}** |")
    lines += [""]
    if hib > solo:
        lines.append(
            f"**La híbrida mejora a solo-modelo en esta ventana:** la cartera rinde "
            f"{hib:.2%} frente a {solo:.2%} de solo-modelo, y {n_beat} de {n} tickers "
            f"mejoran. Filtrar por sentimiento descarta operaciones que habrían sido "
            f"perdedoras."
        )
    else:
        lines.append(
            f"**La híbrida empeora a solo-modelo en esta ventana:** la cartera rinde "
            f"{hib:.2%} frente a {solo:.2%} de solo-modelo, y solo {n_beat} de {n} "
            f"tickers mejoran. Filtrar por sentimiento descarta operaciones; en este "
            f"segmento, las descartadas habrían aportado retorno."
        )
    if (per_ticker["hybrid_n_trades"] == 0).all():
        lines += [
            "",
            "**Nota sobre los datos de sentimiento:** en este dataset `sentiment_score` "
            "es 0.0 en todo el histórico (las noticias crudas de `data/raw/news_*.csv` "
            "solo cubren jul-ago 2026, fuera de la ventana de precios). Con la regla "
            "estricta (BUY exige sentiment > 0, SELL exige sentiment < 0) la señal "
            "híbrida nunca dispara: 0 operaciones en todos los tickers. La comparativa "
            "es degenerada — no hay sentimiento real con el que alinear el modelo — y "
            "debe re-evaluarse cuando exista cobertura de noticias solapada con los "
            "precios.",
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
        f"La simulación aplica **{COST_PER_TRADE:.1%} de comisión + "
        f"{SLIPPAGE:.2%} de slippage** por operación (entrada y salida), y el "
        f"umbral de señal se **calibra por ticker** en el último "
        f"{100.0 * VALIDATION_FRACTION:.0f}% del train (rejilla "
        f"{THRESHOLD_GRID[0]:.2f}-{THRESHOLD_GRID[-1]:.2f}, se elige el de mayor "
        f"Sharpe). El segmento test nunca participa en la calibración.",
        "",
    ]
    lines += _conclusion_lines(df)
    lines += [
        "",
        "## Resultados por ticker (segmento test)",
        "",
        "| ticker | retorno estrategia | retorno buy&hold | umbral | CAGR | Sharpe | maxDD | win rate | operaciones | ventana test |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['ticker']} | {row['total_return']:.2%} | {row['buy_hold_return']:.2%} "
            f"| {_fmt_threshold(row['threshold'])} | {row['cagr']:.2%} | {row['sharpe']:.2f} "
            f"| {row['max_drawdown']:.2%} | {row['win_rate']:.2%} | {row['n_trades']} "
            f"| {row['test_start']} → {row['test_end']} |"
        )
    if df["hybrid_return"].notna().any():
        lines += ["", *_hybrid_section_lines(df)]
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
            f"buy&hold={row['buy_hold_return']:>8.2%}  umbral={_fmt_threshold(row['threshold']):<5}  "
            f"Sharpe={row['sharpe']:>6.2f}  maxDD={row['max_drawdown']:>8.2%}  "
            f"win={row['win_rate']:>7.2%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
