"""Gestión de riesgo: position sizing y drawdown.

Sin dependencias del modelo: trabajan sobre capital y series de equity.
"""

from __future__ import annotations

import pandas as pd


def position_size(
    capital: float,
    risk_per_trade: float = 0.02,
    stop_loss_pct: float | None = None,
) -> float:
    """Capital invertido por operación.

    Args:
        capital : capital total de la cuenta.
        risk_per_trade : fracción del capital que se arriesga por operación
            (por defecto 2%, la regla clásica).
        stop_loss_pct : stop-loss en % (0.05 = 5%). Si se indica, el capital
            invertido se escala para que la pérdida potencial (capital *
            risk_per_trade) se alcance justo con el stop. Si es None, se
            invierte una fracción fija risk_per_trade del capital.

    Returns:
        Cantidad de capital a invertir en la operación.
    """
    if stop_loss_pct is None:
        return capital * risk_per_trade
    return capital * risk_per_trade / stop_loss_pct


def max_drawdown(equity: pd.Series) -> float:
    """Drawdown máximo (negativo) de una serie de equity.

    Máxima caída desde un pico anterior: min(equity / cummax(equity) - 1).
    Una serie que solo sube tiene drawdown 0.
    """
    eq = equity if isinstance(equity, pd.Series) else pd.Series(equity)
    running_max = eq.cummax()
    drawdowns = eq / running_max - 1.0
    return float(drawdowns.min())
