"""Señales de trading: de probabilidad de subida a decisión.

Convención del repo: p = probabilidad de que el precio suba (clase 1),
la que devuelve predict_model.predict_future como prob_sube.
"""

from __future__ import annotations

from enum import Enum


class Signal(str, Enum):
    """Decisión de trading para un día."""

    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


# Umbral por defecto: se compra solo con probabilidad de subida >= 0.6 y se
# vende solo con probabilidad de bajada >= 0.6 (p <= 0.4). En el medio, HOLD.
DEFAULT_THRESHOLD = 0.6


def signal_from_probability(p: float, threshold: float = DEFAULT_THRESHOLD) -> Signal:
    """Convierte una probabilidad de subida p en señal.

    Args:
        p : probabilidad de subida en [0, 1].
        threshold : umbral en (0.5, 1]. Comprar si p >= threshold, vender si
            p <= 1 - threshold, mantener si no hay señal.

    Returns:
        Signal.BUY / Signal.HOLD / Signal.SELL

    Raises:
        ValueError: si threshold no está en (0.5, 1], porque entonces las
            zonas de compra y venta se solapan y la señal es ambigua.
    """
    if not 0.5 < threshold <= 1.0:
        raise ValueError(f"threshold debe estar en (0.5, 1], no {threshold}")
    if p >= threshold:
        return Signal.BUY
    if p <= 1.0 - threshold:
        return Signal.SELL
    return Signal.HOLD
