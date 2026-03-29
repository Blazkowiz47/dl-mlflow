"""Tests for MLflow tracker and metrics source registration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch

import dl_mlflow
from dl_core.core import METRICS_SOURCE_REGISTRY, TRACKER_REGISTRY
from dl_mlflow.trackers.mlflow import MlflowTracker
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
    assert dl_mlflow.__version__ == "0.0.2a12"
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
        parent_run_id: str | None = None,
        nested: bool = False,
    ):
        del parent_run_id
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
    assert ("start", "demo-run") in events


def test_mlflow_callback_prefers_tracking_values_over_callback_defaults(
    monkeypatch: MonkeyPatch,
) -> None:
    """The callback should use injected sweep tracking values as canonical."""
    events: list[tuple[str, str | None]] = []

    def fake_set_tracking_uri(uri: str) -> None:
        events.append(("uri", uri))

    def fake_set_experiment(name: str) -> None:
        events.append(("experiment", name))

    def fake_start_run(
        run_id: str | None = None,
        run_name: str | None = None,
        parent_run_id: str | None = None,
        nested: bool = False,
    ):
        del parent_run_id
        del nested
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

    callback = MlflowCallback(
        experiment_name="stale-experiment",
        run_name="stale-run",
        tracking_uri="",
    )
    callback.set_trainer(_DummyTrainer())
    callback.on_training_start()

    assert ("experiment", "demo-experiment") in events
    assert ("start", "demo-run") in events


def test_mlflow_callback_logs_phase_metrics_with_epoch_steps(
    monkeypatch: MonkeyPatch,
) -> None:
    """The callback should log phase metrics separately with 1-based epoch steps."""
    metric_events: list[tuple[dict[str, float], int]] = []

    monkeypatch.setattr(
        "dl_mlflow.callbacks.mlflow.mlflow",
        SimpleNamespace(
            set_tracking_uri=lambda *_args, **_kwargs: None,
            set_experiment=lambda *_args, **_kwargs: None,
            start_run=lambda **_kwargs: SimpleNamespace(
                info=SimpleNamespace(run_id="child-run-456")
            ),
            log_params=lambda *_args, **_kwargs: None,
            log_artifact=lambda *_args, **_kwargs: None,
            log_metrics=lambda metrics, step: metric_events.append((metrics, step)),
            end_run=lambda: None,
        ),
    )

    callback = MlflowCallback(tracking_uri="")
    callback.set_trainer(_DummyTrainer())
    callback.on_training_start()
    callback.on_test_end(0, {"accuracy": 0.61})
    callback.on_train_end(1, {"loss": 0.5})
    callback.on_validation_end(1, {"accuracy": 0.75})
    callback.on_epoch_end(
        1,
        {
            "train/loss": 0.5,
            "validation/accuracy": 0.75,
            "general/state/global_step": 32.0,
        },
    )

    assert metric_events == [
        ({"test/accuracy": 0.61}, 0),
        ({"train/loss": 0.5}, 1),
        ({"validation/accuracy": 0.75}, 1),
        ({"general/state/global_step": 32.0}, 1),
    ]


def test_mlflow_tracker_setup_sweep_creates_parent_run(
    monkeypatch: MonkeyPatch,
) -> None:
    """The tracker should create one parent MLflow run for the sweep."""
    events: list[tuple[str, str]] = []

    def fake_set_tracking_uri(uri: str) -> None:
        events.append(("uri", uri))

    def fake_set_experiment(name: str) -> None:
        events.append(("experiment", name))

    def fake_start_run(run_name: str | None = None):
        events.append(("start", run_name or ""))
        return SimpleNamespace(info=SimpleNamespace(run_id="parent-run-001"))

    monkeypatch.setattr(
        "dl_mlflow.trackers.mlflow.mlflow",
        SimpleNamespace(
            set_tracking_uri=fake_set_tracking_uri,
            set_experiment=fake_set_experiment,
            start_run=fake_start_run,
            end_run=lambda: events.append(("end", "parent")),
        ),
    )

    tracker = MlflowTracker({"tracking_uri": "./demo-mlruns"})
    tracker_state = tracker.setup_sweep(
        experiment_name="demo-experiment",
        sweep_id="sweep-001",
        sweep_config={"tracking": {}},
        total_runs=2,
    )
    tracker.teardown_sweep()

    assert tracker_state == {
        "tracking_context": "parent-run-001",
        "tracking_uri": "./demo-mlruns",
    }
    assert ("uri", "./demo-mlruns") in events
    assert ("experiment", "demo-experiment") in events
    assert ("start", "demo-experiment-sweep-001") in events
    assert ("end", "parent") in events


def test_mlflow_metrics_source_prefers_remote_metrics(monkeypatch: MonkeyPatch) -> None:
    """The MLflow metrics source should use remote metrics when a run ref exists."""
    source = METRICS_SOURCE_REGISTRY.get("mlflow")

    monkeypatch.setattr(
        "dl_mlflow.metrics_sources.mlflow.mlflow",
        SimpleNamespace(
            tracking=SimpleNamespace(
                MlflowClient=lambda tracking_uri: SimpleNamespace(
                    get_run=lambda run_id: SimpleNamespace(
                        info=SimpleNamespace(status="FINISHED"),
                        data=SimpleNamespace(
                            metrics={"validation/accuracy": 0.97},
                            tags={"mlflow.runName": "demo-run"},
                        ),
                    )
                )
            )
        ),
    )

    run_record = source.collect_run(
        run_index=0,
        run_data={
            "tracking_run_id": "child-run-123",
            "tracking_run_name": "demo-run",
            "tracking_backend": "mlflow",
            "metrics_source_backend": "mlflow",
            "tracking_run_ref": {
                "backend": "mlflow",
                "run_id": "child-run-123",
                "tracking_uri": "./demo-mlruns",
            },
            "status": "running",
            "config_path": str(Path("config.yaml")),
        },
        sweep_data={"tracking_backend": "mlflow"},
    )

    assert run_record["remote_summary_available"] is True
    assert run_record["selection_value"] is None
    assert run_record["final_metrics"]["validation/accuracy"] == 0.97
    assert run_record["status"] == "completed"
