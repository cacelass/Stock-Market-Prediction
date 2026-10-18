"""Formulas reutilizables de edgebet adaptadas para predicción bursátil.

Fórmulas de edgebet (fútbol):
    1. Poisson para probabilidades de marcador
    2. BTTS: prob(ambos marcan) = 1 - exp(-(xg_home + xg_away) * k)
    3. Over/Under: prob(total > k) = 1 - exp(-total / (k + 1))
    4. Raw probability: línea base + contribuciones de features

Adaptación a stocks:
    1. Poisson para probabilidades de retorno (0%, +1%, +2%, etc.)
    2. Probabilidad de subida >2%: 1 - exp(-prob_up * k)
    3. Probabilidad de retorno > umbral
    4. Probabilidad calibrada con features técnicas
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any

try:
    from scipy.special import gammaln
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def poisson_pmf(k: float, mu: float) -> float:
    """Probabilidad masiva de Poisson: P(X=k) para X~Poisson(mu).
    
    Usada en edgebet para probabilidades de marcador (0-0, 1-0, etc.).
    
    Args:
        k: Valor entero (0, 1, 2, ...)
        mu: Media de la distribución
        
    Returns:
        P(X=k)
    """
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    if HAS_SCIPY:
        return np.exp(-mu + k * np.log(mu) - gammaln(k + 1))
    else:
        # Fallback sin scipy
        return np.exp(-mu) * (mu ** k) / np.math.factorial(int(k))


def poisson_cdf(k: int, mu: float) -> float:
    """Función de distribución acumulada de Poisson: P(X<=k).
    
    Args:
        k: Valor máximo
        mu: Media de la distribución
        
    Returns:
        P(X<=k)
    """
    return sum(poisson_pmf(i, mu) for i in range(k + 1))


def btts_probability(xg_total: float, k: float = 0.8) -> float:
    """Probabilidad de Both Teams To Score (adaptada a stocks).
    
    En edgebet: prob(ambos equipos marcan) = 1 - exp(-(xg_home + xg_away) * k)
    
    Adaptación: prob(subida significativa) = 1 - exp(-signal * k)
    
    Args:
        xg_total: Total de xG (o signal strength en stocks)
        k: Factor de escala (0.8 es el valor de edgebet)
        
    Returns:
        Probabilidad entre 0 y 1
    """
    return float(np.clip(1.0 - np.exp(-xg_total * k), 0.0, 1.0))


def over_under_probability(total: float, threshold: float, k: float = 1.0) -> float:
    """Probabilidad Over/Under (adaptada a stocks).
    
    En edgebet: prob(total > k) = 1 - exp(-total / (k + 1))
    
    Adaptación: prob(return > threshold) = 1 - exp(-signal / (threshold + epsilon))
    
    Args:
        total: Total de goles (o signal strength)
        threshold: Umbral (k en edgebet, umbral de retorno en stocks)
        k: Factor de escala
        
    Returns:
        Probabilidad entre 0 y 1
    """
    epsilon = 0.01  # Evitar división por cero
    return float(np.clip(1.0 - np.exp(-total / (threshold + epsilon) * k), 0.0, 1.0))


def raw_probability(
    base: float,
    features: Dict[str, float],
    weights: Dict[str, float],
    bounds: tuple = (0.05, 0.95)
) -> float:
    """Cálculo de probabilidad raw (línea base + contribuciones).
    
    En edgebet: raw_home = 0.40 + xg_diff * 0.15 + elo_diff * 0.10 + home_adv
    
    Adaptación: prob = base + sum(feature_i * weight_i)
    
    Args:
        base: Probabilidad base (0.40 para home en edgebet)
        features: Dict de features con valores
        weights: Dict de pesos para cada feature
        bounds: Límites (min, max) para la probabilidad final
        
    Returns:
        Probabilidad calibrada entre bounds
    """
    raw = base
    for feat, weight in weights.items():
        raw += features.get(feat, 0.0) * weight
    
    # Clip a bounds
    return float(np.clip(raw, bounds[0], bounds[1]))


def normalize_probabilities(probs: Dict[str, float]) -> Dict[str, float]:
    """Normalizar probabilidades para que sumen 1.
    
    En edgebet: prob_home + prob_draw + prob_away = 1.0
    
    Args:
        probs: Dict de probabilidades raw
        
    Returns:
        Dict con probabilidades normalizadas
    """
    total = sum(probs.values())
    if total <= 0:
        total = 0.001
    return {k: v / total for k, v in probs.items()}


def predict_stock_probability(
    signal: float,
    volatility: float,
    rsi: float,
    base_prob: float = 0.5,
    k_btts: float = 0.8,
    k_ou: float = 1.0
) -> Dict[str, float]:
    """Predicción de probabilidad bursátil usando fórmulas de edgebet.
    
    Combina:
    1. btts_probability para probabilidad de subida significativa
    2. over_under_probability para probabilidad de retorno > umbral
    3. raw_probability para combinar con RSI y volatilidad
    
    Args:
        signal: Signal strength (como xg_total en edgebet)
        volatility: Volatilidad del activo
        rsi: RSI (0-100)
        base_prob: Probabilidad base
        k_btts: Factor para probabilidad de subida
        k_ou: Factor para probabilidad over/under
        
    Returns:
        Dict con probabilidades calibradas
    """
    # Normalizar RSI a 0-1
    rsi_norm = (rsi - 50) / 50.0  # 0 cuando RSI=50, -1 cuando RSI=0, +1 cuando RSI=100
    
    # Probabilidad de subida significativa (BTTS adaptation)
    prob_up = btts_probability(signal * (1 + abs(rsi_norm)), k_btts)
    
    # Probabilidad de retorno > umbral (Over/Under adaptation)
    prob_large_move = over_under_probability(abs(signal), 0.02, k_ou)
    
    # Combinar con RSI y volatilidad (Raw probability adaptation)
    weights = {
        "rsi_contribution": 0.1,
        "volatility_contribution": -0.05,  # Mayor volatilidad = menos certeza
    }
    features = {
        "rsi_contribution": rsi_norm * 0.1,
        "volatility_contribution": volatility * -0.05,
    }
    prob_combined = raw_probability(base_prob, features, weights)
    
    # Promedio ponderado
    final_prob = 0.4 * prob_up + 0.3 * prob_large_move + 0.3 * prob_combined
    
    return {
        "prob_up": float(np.clip(final_prob, 0.05, 0.95)),
        "prob_down": 1.0 - float(np.clip(final_prob, 0.05, 0.95)),
        "prob_large_move": prob_large_move,
    }
