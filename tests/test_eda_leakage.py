"""
test_eda_leakage.py — Tests del informe de fugas del EDA (DATA-001).

El criterio 3 de DATA-001 exige que el informe de fugas no señale
correlaciones sospechosas con el target sin justificación. Estos tests
verifican que, sobre el dataset real, ninguna feature correlaciona con
el target por encima del umbral que el agente `data` considera fuga.
"""

from pathlib import Path

import pandas as pd

from agents.tools.dataframe_analysis_tool import DataFrameAnalysisTool

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET = REPO_ROOT / "data" / "raw" / "NVDA_ml_ready.csv"


def test_nvda_dataset_has_no_leakage_suspects_with_target():
    """Ninguna feature debe correlacionar >= 0.95 con 'target' (umbral del agente data)."""
    df = pd.read_csv(RAW_DATASET)
    suspects = DataFrameAnalysisTool.leakage_suspects(df, "target")
    assert suspects == []


def test_nvda_max_correlation_with_target_is_low():
    """La correlación más alta con 'target' debe estar muy por debajo del umbral."""
    df = pd.read_csv(RAW_DATASET)
    corr = df.select_dtypes(include="number").corr()["target"].drop("target")
    assert abs(corr).max() < 0.1
