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


def signal_from_probability_sentiment(
    p: float,
    sentiment: float,
    threshold: float = DEFAULT_THRESHOLD,
    sentiment_threshold: float = 0.0,
) -> Signal:
    """Señal híbrida: probabilidad del modelo + score de sentimiento (TRADE-006).

    Exige que el modelo y el sentimiento estén ALINEADOS: se opera solo cuando
    ambos apuntan en la misma dirección. El sentimiento del día t se conoce el
    propio día t (se calcula con la información disponible ese día), así que
    usarlo en la señal del día t no introduce fuga.

    Reglas (simétricas):
        - BUY  si p >= threshold Y sentiment >  sentiment_threshold
        - SELL si p <= 1 - threshold Y sentiment <  -sentiment_threshold
        - HOLD en el resto (probabilidad y sentimiento en desacuerdo, o sin
          señal de ninguno de los dos lados).

    El SELL usa el umbral de sentimiento NEGADO: exige lo contrario exacto del
    BUY (sentimiento negativo más allá de -sentiment_threshold), igual que el
    umbral de probabilidad es simétrico (p <= 1 - threshold).

    Args:
        p : probabilidad de subida en [0, 1] (la que devuelve predict_model).
        sentiment : score de sentimiento del día t, alineado con p.
        threshold : umbral de probabilidad en (0.5, 1], igual que en
            signal_from_probability.
        sentiment_threshold : umbral de sentimiento >= 0. Comprar exige
            sentiment > sentiment_threshold; vender exige sentiment <
            -sentiment_threshold. Con 0.0 (default): BUY si sentiment > 0 y
            SELL si sentiment < 0.

    Returns:
        Signal.BUY / Signal.HOLD / Signal.SELL

    Raises:
        ValueError: si threshold no está en (0.5, 1] o sentiment_threshold < 0.
    """
    if not 0.5 < threshold <= 1.0:
        raise ValueError(f"threshold debe estar en (0.5, 1], no {threshold}")
    if sentiment_threshold < 0.0:
        raise ValueError(f"sentiment_threshold debe ser >= 0, no {sentiment_threshold}")
    if p >= threshold and sentiment > sentiment_threshold:
        return Signal.BUY
    if p <= 1.0 - threshold and sentiment < -sentiment_threshold:
        return Signal.SELL
    return Signal.HOLD
