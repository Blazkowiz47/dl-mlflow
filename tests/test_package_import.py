"""Basic tests for the local MLflow extension package."""

from __future__ import annotations

import dl_mlflow


def test_package_import_exposes_version() -> None:
    """The package root should import successfully and expose a version."""

    assert dl_mlflow.__version__ == "0.0.2"
