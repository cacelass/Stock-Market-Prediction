"""
Kelly fraccional, control de exposición y Kelly con restricción de riesgo (Boyd).
v3.0 — GROWTH-003: Risk-Constrained Kelly (Busseti, Ryu & Boyd 2016)

CAMBIO PRINCIPAL: KELLY_FRACTION 0.25 → 0.10
══════════════════════════════════════════════════════════════════════════
El modelo tiene Brier=0.2125 vs mercado Brier=0.2011 — no supera al
mercado en predicción global. El edge observado (+17% yield OOS) tiene
IC95 [+10%, +24%] sobre 114 apuestas — demasiado ancho para Quarter Kelly.

Progresión recomendada:
  n_bets_oos < 150:   KELLY_FRACTION=0.10  (actual)
  n_bets_oos 150-300: KELLY_FRACTION=0.12
  n_bets_oos >= 300 y KGR > 0: KELLY_FRACTION=0.15

NOTA: ev_calculator.py usa kelly_pct_conservative() con p_conservadora
= (p_modelo + p_mercado) / 2. Las funciones de este módulo se usan en
walk_forward_backtest() y en apply_kelly_to_bets() para portfolios.

GROWTH-003: kelly_risk_constrained() implementa el marco de Busseti & Boyd
(2016) para optimizar la fracción de Kelly con restricción de riesgo:
  max_f  G(f) − λ · R(f)
donde G(f) = E[log(1 + f·X)] es la tasa de crecimiento logarítmico y
R(f) es una medida de riesgo del drawdown del bankroll. λ controla la
aversion al riesgo: λ → 0 recupera Kelly puro; λ → ∞ tiende a f=0.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KELLY_FRACTION:    float = 0.10   # 10% — post-auditoría (era 0.25)
MAX_STAKE_PCT:     float = 0.03   # cap 3% bankroll por apuesta (era 5%)
MIN_STAKE_PCT:     float = 0.005  # mín 0.5% bankroll
MAX_EXPOSURE_PCT:  float = 0.15   # máx 15% bankroll expuesto (era 20%)


def kelly_stake_pct(
    p_model: float,
    odds_decimal: float,
    fraction: float = KELLY_FRACTION,
    confidence: str = "MEDIA",
) -> float:
    """
    Porcentaje del bankroll según Kelly fraccional.

    Usa p_modelo directa (sin conservadora). Para uso interno en
    walk_forward_backtest() donde la conservadora ya se aplicó antes.
    Retorna 0 si no hay edge matemático (f_full <= 0).

    Ajuste por confianza sobre el Kelly fraccional:
      ALTA: ×1.0  MEDIA: ×0.85  BAJA: ×0.50
    """
    if odds_decimal <= 1.0:
        return 0.0
    b      = odds_decimal - 1.0
    f_full = (p_model * b - (1.0 - p_model)) / b
    if f_full <= 0:
        return 0.0
    conf_mult = {"ALTA": 1.0, "MEDIA": 0.85, "BAJA": 0.50}
    adjusted  = f_full * fraction * conf_mult.get(confidence, 0.85)
    return float(np.clip(adjusted, MIN_STAKE_PCT, MAX_STAKE_PCT))


def kelly_stake_abs(
    p_model: float,
    odds_decimal: float,
    bankroll: float,
    fraction: float = KELLY_FRACTION,
    confidence: str = "MEDIA",
) -> float:
    """Stake absoluto en euros."""
    return kelly_stake_pct(p_model, odds_decimal, fraction, confidence) * bankroll


def apply_kelly_to_bets(
    bets_df: pd.DataFrame,
    bankroll: float,
    fraction: float = KELLY_FRACTION,
    confidence_col: str = "confidence",
) -> pd.DataFrame:
    """
    Aplica Kelly fraccional a un DataFrame de apuestas.

    Usa p_conservative si está disponible (columna 'p_conservative'),
    de lo contrario usa p_model. Controla la exposición total del portfolio.

    Args:
        bets_df:        DataFrame con columnas p_model, odds_market, bet_signal
        bankroll:       bankroll actual en euros
        fraction:       fracción de Kelly (default: KELLY_FRACTION=0.10)
        confidence_col: columna con nivel de confianza (ALTA/MEDIA/BAJA)
    """
    df = bets_df.copy()

    def _compute_kelly(row: pd.Series) -> float:
        if not row.get("bet_signal", False):
            return 0.0
        p_use = row.get("p_conservative", row.get("p_model", 0.0))
        conf  = row.get(confidence_col, "MEDIA")
        return kelly_stake_pct(p_use, row.get("odds_market", 0.0), fraction, conf)

    df["kelly_pct"]   = df.apply(_compute_kelly, axis=1)
    df["kelly_stake"] = df["kelly_pct"] * bankroll

    # Control de exposición total del portfolio
    total   = df["kelly_stake"].sum()
    max_exp = bankroll * MAX_EXPOSURE_PCT
    if total > max_exp:
        logger.warning(
            "Exposición portfolio %.2f€ > límite %.2f€ (%.0f%% bankroll) — escalando.",
            total, max_exp, MAX_EXPOSURE_PCT * 100
        )
        scale = max_exp / total
        df["kelly_pct"]   = df["kelly_pct"]   * scale  # BUG FIX: no duplicar
        df["kelly_stake"] = df["kelly_stake"] * scale  # BUG FIX: reescalar stake

    # ev_per_unit calculado DESPUÉS del posible reescalado
    ev_col = df["ev"] if "ev" in df.columns else pd.Series(0.0, index=df.index)
    df["ev_per_unit"] = df["kelly_stake"] * ev_col
    return df


# ═════════════════════════════════════════════════════════════════════════════
# GROWTH-003 — Risk-Constrained Kelly (Busseti, Ryu & Boyd 2016)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RiskConstrainedResult:
    """Resultado de kelly_risk_constrained()."""
    fraction_opt: float
    """Fracción óptima del bankroll por apuesta."""
    growth: float
    """Tasa de crecimiento logarítmico: mean(log(1 + f·X))."""
    risk: float
    """Medida de riesgo usada (CVaR del drawdown o max drawdown)."""
    max_drawdown: float
    """Máximo drawdown de la trayectoria del bankroll."""
    full_kelly: float
    """Kelly fraccional sin restricción de riesgo (λ=0)."""
    lambda_risk: float
    """Parámetro λ usado."""
    risk_measure_name: str
    """Nombre de la medida de riesgo: 'cvar_drawdown' o 'max_drawdown'."""


def build_betting_returns(bets_df: pd.DataFrame) -> np.ndarray:
    """
    Construye vector de retornos por unidad apostada desde un DataFrame de apuestas.

    Retorno neto por unidad apostada = won * odds - 1.0
    (gana odds-1 si gana, pierde 1 si pierde).

    Args:
        bets_df: DataFrame con columnas 'odds' y 'won' (bool/int/str).

    Returns:
        np.ndarray con los retornos netos por apuesta, en orden de fila.

    Raises:
        ValueError: si falta alguna columna o el DataFrame está vacío.
    """
    if bets_df.empty:
        raise ValueError("bets_df está vacío; no se pueden calcular retornos.")
    for col in ("odds", "won"):
        if col not in bets_df.columns:
            raise ValueError(f"Columna '{col}' requerida no encontrada en bets_df.")
    odds = bets_df["odds"].to_numpy(dtype=float)
    won_raw = bets_df["won"]
    if won_raw.dtype == object or won_raw.dtype.name == "str":
        # Handle string booleans: "True"/"true"/"1" → True, rest → False
        won = won_raw.astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        won = won_raw.to_numpy(dtype=bool)
    if np.any(odds <= 1.0):
        raise ValueError(
            f"Hay cuotas <= 1.0 en el DataFrame (mínimo {odds.min():.2f}). "
            "Las cuotas deben ser decimales > 1.0."
        )
    return won * odds - 1.0


def _growth_rate(f: float, ret: np.ndarray) -> float:
    """G(f) = mean(log(1 + f * ret)) — tasa de crecimiento logarítmico."""
    vals = 1.0 + f * ret
    if np.any(vals <= 0):
        return -1e20
    return float(np.mean(np.log(vals)))


def _bankroll_trajectory(ret: np.ndarray, f: float,
                         initial: float = 1000.0) -> np.ndarray:
    """Simula la trayectoria del bankroll con fracción f constante.

    Cada apuesta: stake = f * bankroll; si gana: +stake*(odds-1); si pierde: -stake.
    Equivalente: bankroll *= (1 + f * ret_i).
    """
    traj = np.empty(len(ret) + 1, dtype=np.float64)
    traj[0] = initial
    for i, r in enumerate(ret):
        traj[i + 1] = traj[i] * (1.0 + f * r)
        if traj[i + 1] <= 0:
            traj[i + 1] = 1e-12
    return traj


def _max_drawdown(traj: np.ndarray) -> float:
    """Máximo drawdown de una trayectoria de bankroll."""
    peak = np.maximum.accumulate(traj)
    dd = (peak - traj) / np.maximum(peak, 1e-12)
    return float(np.max(dd))


def _cvar_drawdown(ret: np.ndarray, f: float, alpha: float = 0.10,
                   initial: float = 1000.0, n_bootstrap: int = 1000,
                   seed: int = 42) -> float:
    """CVaR (Expected Shortfall) del max drawdown por bootstrap.

    Para una secuencia fija de retornos, el drawdown de la trayectoria es
    determinista dado f. Con bootstrap (remuestreo con reemplazo) simulamos
    trayectorias alternativas para estimar la distribución del drawdown y
    calculamos su CVaR al nivel alpha.

    Si n_bootstrap <= 1, retorna el max drawdown determinista (sin incertidumbre).
    """
    if n_bootstrap <= 1:
        return _max_drawdown(_bankroll_trajectory(ret, f, initial))

    rng = np.random.default_rng(seed)
    n = len(ret)
    max_dds = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sampled = ret[idx]
        max_dds[i] = _max_drawdown(_bankroll_trajectory(sampled, f, initial))

    var_threshold = np.percentile(max_dds, (1.0 - alpha) * 100)
    tail = max_dds[max_dds >= var_threshold]
    if len(tail) == 0:
        return float(var_threshold)
    return float(np.mean(tail))


def kelly_risk_constrained(
    bets_df: pd.DataFrame | np.ndarray | None = None,
    returns: np.ndarray | None = None,
    lambda_risk: float = 0.5,
    risk_measure: Literal["cvar_drawdown", "max_drawdown"] = "max_drawdown",
    cvar_alpha: float = 0.10,
    n_bootstrap: int = 500,
    initial_bankroll: float = 1000.0,
    f_bounds: tuple[float, float] = (0.005, 0.15),
    seed: int = 42,
) -> RiskConstrainedResult:
    """
    Kelly con restricción de riesgo (Busseti, Ryu & Boyd 2016).

    Resuelve:  max_f  G(f) − λ · R(f)
    donde G(f) = E[log(1 + f·X)] es la tasa de crecimiento logarítmico
    y R(f) es la medida de riesgo del drawdown del bankroll.

    Normalización de riesgo:
      R_norm(f) = R(f) / R(f_kelly), donde f_kelly es la fracción de
      Kelly clásica de la secuencia observada. Así R_norm(f_kelly) = 1.0
      y R_norm(0) = 0. Esto hace que λ sea adimensional con significado
      claro: λ=0 → Kelly puro; λ=1 → riesgo = crecimiento; λ>1 → riesgo
      domina.

    Referencia: Busseti, Ryu & Boyd (2016), "Risk-Constrained Kelly Gambling".
    Implementación numérica via scipy.optimize.minimize_scalar (bounded).

    Args:
        bets_df:        DataFrame con columnas 'odds' y 'won'. Alternativa a returns.
        returns:        Vector de retornos netos por unidad apostada (won*odds - 1).
        lambda_risk:    Parámetro de aversión al riesgo (λ ≥ 0, default 0.5).
                        λ=0 → Kelly puro; λ=0.5 → equilibrio moderado;
                        λ=1.0 → riesgo igual de importante que crecimiento;
                        λ>1.0 → muy conservador.
        risk_measure:   'max_drawdown' (determinista, rápido) o
                        'cvar_drawdown' (CVaR por bootstrap, como Boyd §4.1).
        cvar_alpha:     Nivel de confianza para CVaR (default 0.10 = 90% CVaR).
        n_bootstrap:    Réplicas bootstrap para CVaR (0 o 1 = exacto).
        initial_bankroll: Bankroll inicial para simulación.
        f_bounds:       Rango de búsqueda para f (min, max). Default (0.5%, 15%).
        seed:           Semilla para reproducibilidad del bootstrap.

    Returns:
        RiskConstrainedResult con fracción óptima y métricas asociadas.

    Raises:
        ValueError: Si no se proporcionan datos o λ < 0.
    """
    from scipy.optimize import minimize_scalar

    if lambda_risk < 0:
        raise ValueError(f"lambda_risk debe ser >= 0, recibido: {lambda_risk}")

    # Construir vector de retornos
    if returns is not None:
        ret = np.asarray(returns, dtype=np.float64)
    elif bets_df is not None:
        ret = build_betting_returns(bets_df)
    else:
        raise ValueError("Se requiere 'bets_df' o 'returns'.")

    if len(ret) == 0:
        raise ValueError("El vector de retornos está vacío.")

    # Kelly clásico de la secuencia observada (λ=0)
    full_kelly_res = minimize_scalar(
        lambda f: -_growth_rate(f, ret),
        bounds=f_bounds,
        method="bounded",
    )
    full_kelly = float(full_kelly_res.x)

    if lambda_risk == 0.0:
        # Sin restricción de riesgo: Kelly puro
        traj = _bankroll_trajectory(ret, full_kelly, initial_bankroll)
        md = _max_drawdown(traj)
        g = _growth_rate(full_kelly, ret)
        return RiskConstrainedResult(
            fraction_opt=full_kelly,
            growth=g,
            risk=0.0,
            max_drawdown=md,
            full_kelly=full_kelly,
            lambda_risk=0.0,
            risk_measure_name=risk_measure,
        )

    # Pre-computar G(f_kelly) y R(f_kelly) para normalizar
    g_kelly = _growth_rate(full_kelly, ret)
    if risk_measure == "max_drawdown":
        r_kelly = _max_drawdown(
            _bankroll_trajectory(ret, full_kelly, initial_bankroll)
        )
    else:
        r_kelly = _cvar_drawdown(
            ret, full_kelly, alpha=cvar_alpha,
            initial=initial_bankroll, n_bootstrap=n_bootstrap, seed=seed,
        )

    def _objective(f: float) -> float:
        """min -(G_norm(f) - λ·R_norm(f)) = λ·R_norm(f) - G_norm(f).

        G_norm(f) = G(f) / G(f_kelly); R_norm(f) = R(f) / R(f_kelly).
        Así ambas medidas están en [0, ~1] y λ es adimensional.
        """
        g = _growth_rate(f, ret)
        if g_kelly < 1e-15:
            g_norm = 0.0
        else:
            g_norm = g / g_kelly

        if risk_measure == "max_drawdown":
            r_f = _max_drawdown(_bankroll_trajectory(ret, f, initial_bankroll))
        else:
            r_f = _cvar_drawdown(
                ret, f, alpha=cvar_alpha,
                initial=initial_bankroll, n_bootstrap=n_bootstrap, seed=seed,
            )
        r_norm = r_f / r_kelly if r_kelly > 1e-12 else 0.0

        return lambda_risk * r_norm - g_norm

    res = minimize_scalar(_objective, bounds=f_bounds, method="bounded")

    f_opt = float(res.x)
    g_opt = _growth_rate(f_opt, ret)

    if risk_measure == "max_drawdown":
        r_opt_raw = _max_drawdown(_bankroll_trajectory(ret, f_opt, initial_bankroll))
    else:
        r_opt_raw = _cvar_drawdown(
            ret, f_opt, alpha=cvar_alpha,
            initial=initial_bankroll, n_bootstrap=n_bootstrap, seed=seed,
        )
    r_opt = r_opt_raw / r_kelly if r_kelly > 1e-12 else 0.0

    traj_opt = _bankroll_trajectory(ret, f_opt, initial_bankroll)
    md_opt = _max_drawdown(traj_opt)

    return RiskConstrainedResult(
        fraction_opt=f_opt,
        growth=g_opt,
        risk=r_opt,
        max_drawdown=md_opt,
        full_kelly=full_kelly,
        lambda_risk=lambda_risk,
        risk_measure_name=risk_measure,
    )
