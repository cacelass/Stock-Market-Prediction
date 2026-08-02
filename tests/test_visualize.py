"""
test_visualize.py — Tests para inversion/visualization/visualize.py
"""

import numpy as np
import pandas as pd
import pytest


from inversion.visualization.visualize import (
    plot_price,
    plot_predictions,
    plot_feature_importance,
    plot_returns_distribution,
)


def test_plot_price_saves_png(sample_df, tmp_path):
    plot_price(sample_df, ticker="NVDA", output_dir=str(tmp_path))
    assert (tmp_path / "NVDA_price_plot.png").exists()


def test_plot_predictions_saves_png(sample_df, tmp_path):
    y_true = np.random.randint(0, 2, size=50)
    y_pred = np.random.randint(0, 2, size=50)
    plot_predictions(sample_df, y_true, y_pred, output_dir=str(tmp_path))
    assert (tmp_path / "predictions_plot.png").exists()


def test_plot_feature_importance_saves_png(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    X = np.random.randn(120, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    plot_feature_importance(rf, feature_names=["f0", "f1", "f2", "f3"], output_dir=str(tmp_path))
    assert (tmp_path / "feature_importance.png").exists()


def test_plot_returns_distribution_saves_png(sample_df, tmp_path):
    from inversion.features.build_features import add_derived_features

    df = add_derived_features(sample_df)
    plot_returns_distribution(df, output_dir=str(tmp_path))
    assert (tmp_path / "returns_distribution.png").exists()
