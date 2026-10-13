"""Gestión de riesgo y control de exposición."""

from inversion.risk.risk_of_ruin import (
    drawdown_risk_report,
    is_acceptable_risk,
    kelly_fraction_for_target_risk,
    simulate_bankroll_trajectories,
)

__all__ = [
    "simulate_bankroll_trajectories",
    "drawdown_risk_report",
    "is_acceptable_risk",
    "kelly_fraction_for_target_risk",
]
