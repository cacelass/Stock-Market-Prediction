"""
test_make_dataset.py — Tests para inversion/data/make_dataset.py
"""

import pandas as pd
import numpy as np
import pytest


def test_load_data_reads_csv(patch_paths):
    """load_data debe leer el CSV de paths.RAW_DATA_FILE y devolver un DataFrame."""
    from inversion.data.make_dataset import load_data

    sample = pd.DataFrame(
        np.random.randn(50, 3),
        columns=["a", "b", "c"],
        index=pd.date_range("2020-01-01", periods=50, freq="B"),
    )
    sample.index.name = "timestamp"
    sample.to_csv(patch_paths["RAW_DATA_FILE"])

    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (50, 3)
    assert df.index.name == "timestamp"


def test_load_data_sorts_index(patch_paths):
    """load_data debe devolver el índice ordenado cronológicamente."""
    from inversion.data.make_dataset import load_data

    idx = pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01"])
    sample = pd.DataFrame({"close": [3, 1, 2]}, index=idx)
    sample.index.name = "timestamp"
    sample.to_csv(patch_paths["RAW_DATA_FILE"])

    df = load_data()
    assert df is not None
    assert df.index.is_monotonic_increasing


def test_load_data_returns_none_on_missing(patch_paths):
    """load_data debe devolver None si el CSV no existe."""
    from inversion.data.make_dataset import load_data

    assert load_data() is None


def test_get_yfinance_history_returns_dataframe():
    """get_yfinance_history debe devolver un DataFrame (vacío si la descarga falla)."""
    from inversion.data.make_dataset import get_yfinance_history

    df = get_yfinance_history("TICKER_INEXISTENTE_XYZ", "2020-01-01", "2020-02-01")
    assert isinstance(df, pd.DataFrame)


def test_process_raw_to_processed_writes_processed_csv(patch_paths):
    """process_raw_to_processed debe leer data/raw/ y escribir el CSV en data/processed/."""
    from inversion.data.make_dataset import process_raw_to_processed

    raw = patch_paths["RAW_DATA_DIR"] / "NVDA_ml_ready.csv"
    sample = pd.DataFrame({"timestamp": ["2020-01-02", "2020-01-01"], "close": [2.0, 1.0], "target": [0, 1]})
    sample.to_csv(raw, index=False)

    written = process_raw_to_processed()

    out = patch_paths["PROCESSED_DATA_DIR"] / "NVDA_ml_ready.csv"
    assert out in written
    assert out.exists()
    df = pd.read_csv(out)
    assert df.shape == sample.shape
    assert list(df.columns) == list(sample.columns)


def test_process_raw_to_processed_sorts_by_timestamp(patch_paths):
    """El CSV procesado debe quedar ordenado cronológicamente por timestamp."""
    from inversion.data.make_dataset import process_raw_to_processed

    raw = patch_paths["RAW_DATA_DIR"] / "NVDA_ml_ready.csv"
    sample = pd.DataFrame({"timestamp": ["2020-03-01", "2020-01-01", "2020-02-01"], "close": [3.0, 1.0, 2.0]})
    sample.to_csv(raw, index=False)

    process_raw_to_processed()

    out = patch_paths["PROCESSED_DATA_DIR"] / "NVDA_ml_ready.csv"
    df = pd.read_csv(out, parse_dates=["timestamp"])
    assert df["timestamp"].is_monotonic_increasing


# ─────────────────────────────────────────────────────────────────────────────
# DATA-002 — Descarga multi-ticker parametrizada
# ─────────────────────────────────────────────────────────────────────────────


def _fake_ohlcv(n: int = 250, start: str = "2020-01-01") -> pd.DataFrame:
    """OHLCV sintético con el mismo formato que get_yfinance_history."""
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=n, freq="B"),
            "open": close + rng.normal(0, 0.5, n),
            "high": close + 2 + np.abs(rng.normal(0, 0.5, n)),
            "low": close - 2 - np.abs(rng.normal(0, 0.5, n)),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, size=n).astype(float),
        }
    )


def test_load_catalog_reads_yaml(tmp_path):
    """load_catalog debe leer tickers, start y end desde el YAML de config."""
    from inversion.data.make_dataset import load_catalog

    cfg = tmp_path / "tickers.yaml"
    cfg.write_text(
        "start: '2000-01-01'\nend: '2020-01-01'\ntickers:\n  - AAPL\n  - MSFT\n",
        encoding="utf-8",
    )

    catalog = load_catalog(cfg)
    assert catalog["tickers"] == ["AAPL", "MSFT"]
    assert catalog["start"] == "2000-01-01"
    assert catalog["end"] == "2020-01-01"


def test_load_catalog_missing_file_raises(tmp_path):
    """load_catalog debe fallar con ValueError si el fichero no existe."""
    from inversion.data.make_dataset import load_catalog

    with pytest.raises(ValueError, match="No se encontró"):
        load_catalog(tmp_path / "no_existe.yaml")


def test_build_ml_ready_schema_matches_nvda():
    """build_ml_ready debe producir el MISMO esquema que NVDA_ml_ready.csv."""
    from inversion.data.make_dataset import ML_READY_COLUMNS, build_ml_ready

    out = build_ml_ready(_fake_ohlcv())

    assert list(out.columns) == ML_READY_COLUMNS
    assert len(ML_READY_COLUMNS) == 20
    assert out["target"].isin([0, 1]).all()
    assert out["timestamp"].is_monotonic_increasing


def test_build_ml_ready_target_drops_no_future_rows():
    """Las últimas TARGET_HORIZON_DAYS filas (sin ventana futura) se descartan."""
    from inversion.data.make_dataset import TARGET_HORIZON_DAYS, build_ml_ready

    n = 260
    df = _fake_ohlcv(n=n)
    out = build_ml_ready(df)

    assert len(out) > 0
    assert out["target"].notna().all()
    # La última fila conservada es la n-6 del raw (la n-5 ya no tiene ventana futura)
    last_kept = pd.to_datetime(df["timestamp"]).iloc[n - TARGET_HORIZON_DAYS - 1]
    assert pd.to_datetime(out["timestamp"]).iloc[-1] == last_kept


def test_build_ml_ready_target_threshold():
    """El target vale 1 cuando el precio sube >2% en 5 días, y 0 si se mantiene."""
    from inversion.data.make_dataset import build_ml_ready

    n = 260
    df = _fake_ohlcv(n=n)

    # Crecimiento exponencial: el retorno a 5 días es ~5.1% > 2% en toda la serie
    df["close"] = 100.0 * (1.01 ** np.arange(n))
    out = build_ml_ready(df)
    assert len(out) > 0
    assert (out["target"] == 1).all()

    # Precio plano: el retorno a 5 días es 0% → target 0
    df["close"] = 100.0
    out_flat = build_ml_ready(df)
    assert len(out_flat) > 0
    assert (out_flat["target"] == 0).all()


def test_download_catalog_writes_per_ticker(monkeypatch, tmp_path, patch_paths):
    """download_catalog debe dejar data/raw/<TICKER>_ml_ready.csv + informe."""
    from inversion.data import make_dataset as md

    cfg = tmp_path / "tickers.yaml"
    cfg.write_text(
        "start: '2020-01-01'\nend: '2021-01-01'\ntickers:\n  - AAPL\n  - MSFT\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(md, "get_yfinance_history", lambda t, s, e: _fake_ohlcv())

    written = md.download_catalog(cfg)

    assert len(written) == 2
    for ticker in ("AAPL", "MSFT"):
        out = patch_paths["RAW_DATA_DIR"] / f"{ticker}_ml_ready.csv"
        assert out.exists()
        df = pd.read_csv(out)
        assert list(df.columns) == md.ML_READY_COLUMNS
        assert df["target"].isin([0, 1]).all()
        # Informe de validación con nulos, rango y tamaño
        report = patch_paths["VALIDATION_DIR"] / f"{ticker}_validation.json"
        assert report.exists()
        assert report.read_text(encoding="utf-8").count("date_min") == 1


def test_schema_consistency_across_tickers():
    """Dos tickers con historias distintas deben compartir el mismo esquema."""
    from inversion.data.make_dataset import build_ml_ready

    a = build_ml_ready(_fake_ohlcv(n=300))
    b = build_ml_ready(_fake_ohlcv(n=120, start="2021-01-01"))

    assert list(a.columns) == list(b.columns)


def test_download_catalog_fails_cleanly_without_network(monkeypatch, tmp_path, patch_paths):
    """Sin red: RuntimeError con mensaje claro y NINGÚN fichero escrito."""
    from inversion.data import make_dataset as md

    cfg = tmp_path / "tickers.yaml"
    cfg.write_text(
        "start: '2020-01-01'\nend: '2021-01-01'\ntickers:\n  - AAPL\n  - MSFT\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(md, "get_yfinance_history", lambda t, s, e: pd.DataFrame())

    with pytest.raises(RuntimeError, match="internet"):
        md.download_catalog(cfg)

    assert list(patch_paths["RAW_DATA_DIR"].glob("*.csv")) == []
    assert list(patch_paths["VALIDATION_DIR"].glob("*.json")) == []
