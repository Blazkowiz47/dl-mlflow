"""Tests for MLflow tracker and metrics source registration."""

from __future__ import annotations

from types import SimpleNamespace

from pytest import MonkeyPatch

import dl_mlflow
from dl_core.core import METRICS_SOURCE_REGISTRY, TRACKER_REGISTRY
from dl_mlflow.callbacks.mlflow import MlflowCallback


class _DummyAccelerator:
    """Small accelerator test double."""

    def is_main_process(self) -> bool:
        """Return that this is the main process."""
        return True


class _DummyTrainer:
    """Small trainer test double."""

    def __init__(self) -> None:
        self.accelerator = _DummyAccelerator()
        self.config = {
            "runtime": {"name": "demo-run"},
            "experiment": {"name": "demo-experiment"},
            "tracking": {
                "backend": "mlflow",
                "uri": "./demo-mlruns",
                "context": "parent-run-123",
                "run_name": "demo-run",
            },
        }
        self.artifact_manager = SimpleNamespace(run_dir=".")


def test_mlflow_tracker_and_metrics_source_are_registered() -> None:
    """Importing dl-mlflow should register tracker and metrics source aliases."""
    assert dl_mlflow.__version__ == "0.0.1"
    assert TRACKER_REGISTRY.is_registered("mlflow")
    assert METRICS_SOURCE_REGISTRY.is_registered("mlflow")


def test_mlflow_callback_uses_tracking_config_for_uri_and_parent(
    monkeypatch: MonkeyPatch,
) -> None:
    """The callback should resolve tracking URI and parent run from config."""
    events: list[tuple[str, str | None]] = []

    def fake_set_tracking_uri(uri: str) -> None:
        events.append(("uri", uri))

    def fake_set_experiment(name: str) -> None:
        events.append(("experiment", name))

    def fake_start_run(
        run_id: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
    ):
        events.append(("start", run_id or run_name))
        return SimpleNamespace(info=SimpleNamespace(run_id="child-run-456"))

    monkeypatch.setattr(
        "dl_mlflow.callbacks.mlflow.mlflow",
        SimpleNamespace(
            set_tracking_uri=fake_set_tracking_uri,
            set_experiment=fake_set_experiment,
            start_run=fake_start_run,
            log_params=lambda *_args, **_kwargs: None,
            log_artifact=lambda *_args, **_kwargs: None,
            end_run=lambda: None,
        ),
    )

    callback = MlflowCallback(tracking_uri="")
    callback.set_trainer(_DummyTrainer())
    callback.on_training_start()

    assert ("uri", "./demo-mlruns") in events
    assert ("experiment", "demo-experiment") in events
    assert ("start", "parent-run-123") in events
    assert ("start", "demo-run") in events
