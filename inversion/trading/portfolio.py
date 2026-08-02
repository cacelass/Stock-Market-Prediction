"""Cartera multi-activo: asignación de capital y rebalanceo (TRADE-002).

Trabaja con el catálogo de config/tickers.yaml y con cantidades
fraccionarias de acciones: no se redondea a lotes reales (ver `rebalance`).
"""

from __future__ import annotations

import yaml

from inversion.utils import paths


def load_catalog() -> list[str]:
    """Tickers del catálogo (config/tickers.yaml), en orden."""
    cfg = paths.PROJECT_DIR / "config" / "tickers.yaml"
    with cfg.open(encoding="utf-8") as fh:
        return [str(t) for t in yaml.safe_load(fh).get("tickers", [])]


def equal_weights(tickers: list[str]) -> dict[str, float]:
    """Pesos iguales por ticker (suman 1)."""
    w = 1.0 / len(tickers)
    return {t.upper(): w for t in tickers}


def allocate_capital(
    capital: float,
    tickers: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Reparte capital entre los tickers del catálogo.

    Args:
        capital : capital total a repartir.
        tickers : lista de tickers; solo se usa si no se pasan `weights`.
        weights : pesos por ticker (no es necesario que sumen 1: se
            normalizan para que el capital asignado sume exactamente
            `capital`).

    Returns:
        {ticker: capital_asignado}.
    """
    if weights is None:
        weights = equal_weights(tickers if tickers is not None else load_catalog())

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("los pesos deben sumar un valor positivo")
    return {t.upper(): capital * (w / total) for t, w in weights.items()}


def rebalance(
    target_weights: dict[str, float],
    prices: dict[str, float],
    capital: float,
    current_shares: dict[str, float] | None = None,
) -> dict[str, float]:
    """Acciones objetivo por ticker para igualar los pesos.

    Simplificación: se trabaja con fracciones de acción (shares = peso *
    capital / precio); no se redondea a acciones enteras porque la cartera
    es en papel y no hay lotes reales.

    Args:
        target_weights : pesos objetivo por ticker.
        prices : precio actual por ticker.
        capital : capital total de la cartera.
        current_shares : tenencias actuales por ticker. Si se pasan, el
            resultado es el delta a operar (positivo = comprar, negativo =
            vender); si no, son las acciones objetivo a mantener.

    Returns:
        {ticker: acciones} — objetivo o delta según `current_shares`.
    """
    alloc = allocate_capital(capital, weights=target_weights)
    target = {t: alloc[t] / prices[t] for t in alloc}
    if current_shares is None:
        return target
    return {t: target[t] - current_shares.get(t, 0.0) for t in target}
