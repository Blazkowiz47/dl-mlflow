"""MLflow tracker implementation for dl-core sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from dl_core.core import BaseTracker, config_field, register_tracker


@register_tracker("mlflow")
class MlflowTracker(BaseTracker):
    """Tracker metadata adapter for local MLflow-backed runs."""

    CONFIG_FIELDS = BaseTracker.CONFIG_FIELDS + [
        config_field(
            "tracking_uri",
            "str",
            "MLflow tracking URI used for the sweep parent run.",
            default="./mlruns",
        ),
        config_field(
            "sweep_name",
            "str | None",
            "Optional MLflow parent run name for the sweep.",
            default=None,
        ),
    ]

    def __init__(self, tracking_config: dict[str, Any] | None = None, **kwargs: Any):
        """Initialize tracker state for MLflow-backed sweeps."""
        super().__init__(tracking_config, **kwargs)
        self.parent_run: Any | None = None

    def get_backend_name(self) -> str:
        """Return the tracker backend name."""
        return "mlflow"

    def setup_sweep(
        self,
        *,
        experiment_name: str,
        sweep_id: str,
        sweep_config: dict[str, Any],
        total_runs: int,
        tracking_context: str | None = None,
        tracking_uri: str | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Create or reuse the MLflow parent run for the sweep."""
        del total_runs

        resolved_tracking_uri = (
            tracking_uri
            or self.tracking_config.get("tracking_uri")
            or self.tracking_config.get("uri")
            or "./mlruns"
        )
        if resume and tracking_context:
            return {
                "tracking_context": tracking_context,
                "tracking_uri": resolved_tracking_uri,
            }

        if tracking_context:
            return {
                "tracking_context": tracking_context,
                "tracking_uri": resolved_tracking_uri,
            }

        sweep_file = sweep_config.get("sweep_file")
        derived_sweep_name = Path(str(sweep_file)).stem if sweep_file else ""
        resolved_experiment_name = (
            self.tracking_config.get("experiment_name") or experiment_name
        )
        sweep_name = (
            self.tracking_config.get("sweep_name")
            or derived_sweep_name
            or f"{resolved_experiment_name}-{sweep_id}"
        )
        mlflow.set_tracking_uri(resolved_tracking_uri)
        mlflow.set_experiment(str(resolved_experiment_name))
        self.parent_run = mlflow.start_run(run_name=str(sweep_name))
        return {
            "tracking_context": self.parent_run.info.run_id,
            "tracking_uri": resolved_tracking_uri,
        }

    def teardown_sweep(self) -> None:
        """Close the MLflow parent sweep run when one was opened."""
        if self.parent_run is None:
            return
        try:
            mlflow.end_run()
        finally:
            self.parent_run = None

    def inject_tracking_config(
        self,
        config: dict[str, Any],
        *,
        run_name: str | None = None,
        tracking_context: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        """Inject MLflow tracking metadata into a run configuration."""
        super().inject_tracking_config(
            config,
            run_name=run_name,
            tracking_context=tracking_context,
            tracking_uri=tracking_uri,
        )
        tracking = config.setdefault("tracking", {})
        if tracking_context:
            tracking["parent_run_id"] = tracking_context
        if tracking_uri:
            tracking["tracking_uri"] = tracking_uri

    def build_run_reference(
        self,
        *,
        result: dict[str, Any] | None = None,
        run_name: str | None = None,
        tracking_context: str | None = None,
        tracking_uri: str | None = None,
    ) -> dict[str, Any] | None:
        """Build an MLflow-specific run reference for sweep tracking."""
        reference = super().build_run_reference(
            result=result,
            run_name=run_name,
            tracking_context=tracking_context,
            tracking_uri=tracking_uri,
        )
        if reference is None:
            return None

        reference["backend"] = "mlflow"
        if tracking_context:
            reference.setdefault("parent_run_id", tracking_context)
        if tracking_uri:
            reference.setdefault("tracking_uri", tracking_uri)
        experiment_name = self.tracking_config.get("experiment_name")
        if isinstance(experiment_name, str) and experiment_name:
            reference.setdefault("experiment_name", experiment_name)
        return reference
