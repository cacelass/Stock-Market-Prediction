"""test_viabilidad.py — Smoke test del informe de viabilidad out-of-sample (TRADE-003).

El foco no son métricas complejas sino que el script genera VIABILIDAD.md con
las secciones clave y que se puede re-ejecutar (make viabilidad regenera el
informe).
"""

from inversion.trading.viabilidad import main


def _write_sample_ticker(patch_paths, df_with_target) -> None:
    """Deja data/interim/features_AAPL_ml_ready.csv con features + target."""
    out = patch_paths["INTERIM_DATA_DIR"] / "features_AAPL_ml_ready.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_with_target.to_csv(out, index=False)


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
