"""test_viabilidad.py — Smoke test del informe de viabilidad out-of-sample (TRADE-003).

El foco no son métricas complejas sino que el script genera VIABILIDAD.md con
las secciones clave y que se puede re-ejecutar (make viabilidad regenera el
informe). TRADE-004 añade costes realistas y calibración de umbral por ticker.
TRADE-006 añade la sección de señal híbrida modelo+sentimiento.
"""

import numpy as np
import pandas as pd
import pytest

from inversion.trading.viabilidad import THRESHOLD_GRID, _calibrate_threshold, main


def _write_sample_ticker(patch_paths, df_with_target) -> None:
    """Deja data/interim/features_AAPL_ml_ready.csv con features + target."""
    out = patch_paths["INTERIM_DATA_DIR"] / "features_AAPL_ml_ready.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_with_target.to_csv(out, index=False)


def _add_sentiment(df) -> pd.DataFrame:
    """Añade las columnas de sentimiento (SENTIMENT_COLS) a un df sintético."""
    out = df.copy()
    out["sentiment_score"] = np.where(out["return"] > 0, 0.5, -0.5)
    out["sentiment_ma5"] = out["sentiment_score"].rolling(5, min_periods=1).mean()
    out["sentiment_vol"] = 0.1
    return out


def test_viabilidad_genera_informe_con_secciones_clave(patch_paths, df_with_target, tmp_path):
    _write_sample_ticker(patch_paths, df_with_target)
    report_dir = tmp_path / "reports" / "backtest"

    rc = main(["--ticker", "AAPL", "--out", str(report_dir)])
    assert rc == 0

    md = report_dir / "VIABILIDAD.md"
    assert md.exists()
    content = md.read_text(encoding="utf-8")
    assert "## Conclusión" in content
    assert "## Resultados por ticker" in content
    assert "## Limitaciones" in content
    assert "CARTERA" in content
    assert "AAPL" in content
    assert "buy&hold" in content
    assert "out-of-sample" in content
    assert "comisión" in content
    assert "slippage" in content
    assert "umbral" in content
    assert "Impacto de los costes" in content


def test_calibrate_threshold_chooses_highest_when_all_sharpe_zero():
    # Precios constantes → Sharpe 0 para cualquier umbral; el desempate elige
    # el umbral más alto de la rejilla (menos operaciones, menos costes).
    prices = pd.Series([100.0] * 10)
    probs = pd.Series([0.9] * 10)
    best = _calibrate_threshold(prices, probs, 10_000.0, 0.001, 0.0005)
    assert best == pytest.approx(THRESHOLD_GRID[-1])


def test_viabilidad_incluye_seccion_hibrida(patch_paths, df_with_target, tmp_path):
    # Con sentimiento en los datos, el informe compara la señal híbrida
    # (modelo + sentimiento, TRADE-006) contra solo-modelo en el MISMO test.
    _write_sample_ticker(patch_paths, _add_sentiment(df_with_target))
    report_dir = tmp_path / "reports" / "backtest"

    rc = main(["--ticker", "AAPL", "--out", str(report_dir)])
    assert rc == 0

    md = report_dir / "VIABILIDAD.md"
    assert md.exists()
    content = md.read_text(encoding="utf-8")
    assert "## Señal híbrida vs solo-modelo" in content
    assert "híbrida" in content
    assert "solo-modelo" in content


def test_viabilidad_se_regenera_al_reejecutar(patch_paths, df_with_target, tmp_path):
    _write_sample_ticker(patch_paths, df_with_target)
    report_dir = tmp_path / "reports" / "backtest"

    assert main(["--ticker", "AAPL", "--out", str(report_dir)]) == 0
    assert (report_dir / "VIABILIDAD.md").exists()

    # Segunda ejecución (lo que hace make viabilidad): regenera, no falla.
    assert main(["--ticker", "AAPL", "--out", str(report_dir)]) == 0
    assert (report_dir / "VIABILIDAD.md").exists()


def test_viabilidad_sin_datos_devuelve_error(patch_paths, tmp_path):
    # No hay ningún CSV de features en data/interim/ → error limpio, rc != 0.
    report_dir = tmp_path / "reports" / "backtest"
    rc = main(["--ticker", "AAPL", "--out", str(report_dir)])
    assert rc != 0
    assert not (report_dir / "VIABILIDAD.md").exists()
