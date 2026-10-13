"""Simulación Monte Carlo de riesgo de ruina (Gate 16, Sección 14 CRITERIOS_FIABILIDAD).

Antes de apostar dinero real, simular >=10.000 trayectorias de bankroll
y calcular la probabilidad de drawdown >30-50% en horizonte de 1 año.
"""

from __future__ import annotations

import gc
import logging

import numpy as np

logger = logging.getLogger(__name__)

def _log_memory_usage(label: str = "") -> None:
    try:
        import psutil
        proc = psutil.Process()
        mem = proc.memory_info().rss / 1024 / 1024
        logger.debug("Memoria [%s]: %.1f MB", label, mem)
    except ImportError:
        pass


def simulate_bankroll_trajectories(
    initial_bankroll: float = 1000.0,
    win_rate: float = 0.55,
    avg_odds: float = 2.0,
    kelly_fraction: float = 0.1,
    max_stake_pct: float = 0.03,
    n_bets: int = 500,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> dict:
    """Simula trayectorias de bankroll vía Monte Carlo.

    Args:
        initial_bankroll: Capital inicial en euros.
        win_rate: Probabilidad de ganar cada apuesta.
        avg_odds: Cuota media (formato decimal).
        kelly_fraction: Fracción de Kelly a usar (0.1 = 10%).
        max_stake_pct: Fracción máxima del bankroll por apuesta.
        n_bets: Número de apuestas a simular por trayectoria.
        n_simulations: Número de trayectorias independientes.
        seed: Semilla para reproducibilidad.

    Returns:
        dict con trays, bankrolls finales, drawdowns y métricas de riesgo.
    """
    _log_memory_usage("inicio simulate_bankroll_trajectories")
    rng = np.random.default_rng(seed)

    if avg_odds <= 1.0 or win_rate <= 0.0 or initial_bankroll <= 0:
        trajectories = np.zeros((n_simulations, n_bets), dtype=np.float64)
        trajectories[:, 0] = initial_bankroll
        final_bankrolls = np.zeros(n_simulations, dtype=np.float64)
        max_drawdowns = np.ones(n_simulations, dtype=np.float64)
        prob_ruin = 1.0
        prob_drawdown_30pct = 1.0
        prob_drawdown_50pct = 1.0
        median_final = 0.0
        mean_final = 0.0
        percentile_5 = 0.0
        percentile_95 = 0.0
        sharpe_ratio = 0.0
        return {
            "trajectories": trajectories,
            "final_bankrolls": final_bankrolls,
            "max_drawdowns": max_drawdowns,
            "prob_ruin": prob_ruin,
            "prob_drawdown_30pct": prob_drawdown_30pct,
            "prob_drawdown_50pct": prob_drawdown_50pct,
            "median_final": median_final,
            "mean_final": mean_final,
            "percentile_5": percentile_5,
            "percentile_95": percentile_95,
            "sharpe_ratio": sharpe_ratio,
        }

    b = avg_odds - 1.0
    kelly_pct_raw = win_rate - (1.0 - win_rate) / b
    stake_pct = max(0.0, min(kelly_pct_raw * kelly_fraction, max_stake_pct))

    outcomes = rng.random((n_simulations, n_bets)) < win_rate
    trajectories = np.full((n_simulations, n_bets), initial_bankroll, dtype=np.float64)
    bankroll = np.full(n_simulations, initial_bankroll, dtype=np.float64)

    for bet_i in range(n_bets):
        won = outcomes[:, bet_i]
        stake = bankroll * stake_pct
        bankroll[won] += stake[won] * b
        bankroll[~won] -= stake[~won]
        bankroll = np.maximum(bankroll, 0.0)
        trajectories[:, bet_i] = bankroll

    final_bankrolls = trajectories[:, -1].copy()

    peak = np.maximum.accumulate(trajectories, axis=1)
    dd = (peak - trajectories) / np.maximum(peak, 1e-12)
    max_drawdowns = np.max(dd, axis=1)

    prob_ruin = float(np.mean(final_bankrolls <= 0))
    prob_drawdown_30pct = float(np.mean(max_drawdowns > 0.30))
    prob_drawdown_50pct = float(np.mean(max_drawdowns > 0.50))

    median_final = float(np.median(final_bankrolls))
    mean_final = float(np.mean(final_bankrolls))
    percentile_5 = float(np.percentile(final_bankrolls, 5))
    percentile_95 = float(np.percentile(final_bankrolls, 95))

    returns = (final_bankrolls - initial_bankroll) / initial_bankroll
    sigma = float(np.std(returns, ddof=1))
    sharpe_ratio = float(np.mean(returns) / sigma) if sigma > 1e-12 else 0.0

    gc.collect()
    _log_memory_usage("fin simulate_bankroll_trajectories")

    return {
        "trajectories": trajectories,
        "final_bankrolls": final_bankrolls,
        "max_drawdowns": max_drawdowns,
        "prob_ruin": prob_ruin,
        "prob_drawdown_30pct": prob_drawdown_30pct,
        "prob_drawdown_50pct": prob_drawdown_50pct,
        "median_final": median_final,
        "mean_final": mean_final,
        "percentile_5": percentile_5,
        "percentile_95": percentile_95,
        "sharpe_ratio": sharpe_ratio,
    }


def drawdown_risk_report(risk_results: dict) -> str:
    """Genera un reporte formateado del riesgo de ruina."""
    lines = [
        "═" * 60,
        "  REPORTE DE RIESGO DE RUINA (Monte Carlo)",
        "═" * 60,
        "",
        f"  Probabilidad de ruina (bankroll <= 0):      {risk_results['prob_ruin']*100:6.2f}%",
        f"  Probabilidad de drawdown > 30%%:             {risk_results['prob_drawdown_30pct']*100:6.2f}%",
        f"  Probabilidad de drawdown > 50%%:             {risk_results['prob_drawdown_50pct']*100:6.2f}%",
        "",
        f"  Bankroll final — mediana:                   {risk_results['median_final']:8.2f} €",
        f"  Bankroll final — media:                     {risk_results['mean_final']:8.2f} €",
        f"  Bankroll final — P5:                        {risk_results['percentile_5']:8.2f} €",
        f"  Bankroll final — P95:                       {risk_results['percentile_95']:8.2f} €",
        "",
        f"  Sharpe ratio (mean/sigma de retornos):       {risk_results['sharpe_ratio']:6.3f}",
        "",
        "═" * 60,
    ]
    return "\n".join(lines)


def is_acceptable_risk(
    risk_results: dict,
    max_ruin_prob: float = 0.001,
    max_drawdown_30pct: float = 0.10,
) -> bool:
    """Evalúa si el riesgo es aceptable según los umbrales dados.

    Args:
        risk_results: Dict devuelto por simulate_bankroll_trajectories.
        max_ruin_prob: Probabilidad máxima de ruina tolerada (default 0.1%).
        max_drawdown_30pct: Probabilidad máxima de drawdown >30% (default 10%).

    Returns:
        True si ambas métricas están por debajo del umbral.
    """
    return (
        risk_results["prob_ruin"] <= max_ruin_prob
        and risk_results["prob_drawdown_30pct"] <= max_drawdown_30pct
    )


def kelly_fraction_for_target_risk(
    initial_bankroll: float = 1000.0,
    win_rate: float = 0.55,
    avg_odds: float = 2.0,
    target_max_drawdown_30pct: float = 0.05,
    n_simulations: int = 5_000,
) -> float:
    """Busca la fracción Kelly máxima que mantiene el riesgo de drawdown
    por debajo del objetivo mediante búsqueda binaria.

    Args:
        initial_bankroll: Capital inicial.
        win_rate: Probabilidad de ganar.
        avg_odds: Cuota media decimal.
        target_max_drawdown_30pct: Probabilidad máxima tolerada de drawdown >30%.
        n_simulations: Trayectorias por iteración de búsqueda.

    Returns:
        Fracción de Kelly óptima (entre 0.0 y 1.0).
    """
    lo, hi = 0.0, 1.0
    best = 0.0

    for _ in range(15):
        mid = (lo + hi) / 2.0
        result = simulate_bankroll_trajectories(
            initial_bankroll=initial_bankroll,
            win_rate=win_rate,
            avg_odds=avg_odds,
            kelly_fraction=mid,
            n_simulations=n_simulations,
            seed=42,
        )
        if result["prob_drawdown_30pct"] <= target_max_drawdown_30pct:
            best = mid
            lo = mid
        else:
            hi = mid

    return best
