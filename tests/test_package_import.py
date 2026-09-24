"""Basic tests for the local MLflow extension package."""

from __future__ import annotations

from importlib.metadata import distribution

import dl_mlflow


def test_package_import_exposes_version() -> None:
    """The package root should import successfully and expose a version."""

    assert dl_mlflow.__version__ == "0.0.16"


def test_package_exposes_runtime_entry_point() -> None:
    """Installed MLflow integrations should register without scaffold imports."""
    runtime_points = distribution("deep-learning-mlflow").entry_points
    assert any(
        point.group == "dl_core.runtime_extensions"
        and point.name == "mlflow"
        and point.value == "dl_mlflow"
        for point in runtime_points
    )
