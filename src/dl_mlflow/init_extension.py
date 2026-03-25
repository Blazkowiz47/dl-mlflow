"""MLflow scaffold extension for dl-init-experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from dl_core.init_extensions import InitExtension, ScaffoldContext


def _mlflow_callback_block() -> str:
    """Render the scaffold callback block for local MLflow logging."""

    return """
  mlflow:
    experiment_name: my_experiment
    run_name: my_run
    tracking_uri: ./mlruns
    log_config: true
"""


def _mlflow_tracking_fields() -> str:
    """Render local MLflow-specific additions to the sweep tracking block."""

    return """tracking:
  backend: mlflow
  tracking_uri: ./mlruns
  group: my_experiment
"""


class MlflowInitExtension(InitExtension):
    """Expose local MLflow scaffold wiring when dl-mlflow is installed."""

    name = "mlflow"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register the local MLflow scaffold flag."""

        parser.add_argument(
            "--with-mlflow",
            action="store_true",
            help="Include local MLflow callback wiring and tracking defaults.",
        )

    def is_enabled(
        self,
        args: argparse.Namespace,
        discovered_extensions: dict[str, InitExtension],
    ) -> bool:
        """Enable MLflow wiring when explicitly requested."""

        del discovered_extensions
        return bool(getattr(args, "with_mlflow", False))

    def apply(self, context: ScaffoldContext) -> None:
        """Apply local MLflow-specific scaffold mutations."""

        context.replace_in_file(
            "pyproject.toml",
            '"deep-learning-core"',
            '"deep-learning-core[mlflow]"',
        )
        context.add_dependency("deep-learning-mlflow")
        context.append_bootstrap_import("import dl_mlflow  # noqa: F401")
        context.append_readme_note(
            "Local MLflow support is enabled. Review the `callbacks.mlflow` "
            "block in `configs/base.yaml` before training."
        )
        context.replace_in_file(
            Path("configs") / "base.yaml",
            "  metric_logger:\n    log_frequency: 1\n",
            "  metric_logger:\n    log_frequency: 1\n"
            f"{_mlflow_callback_block()}",
        )
        context.replace_in_file(
            Path("configs") / "base_sweep.yaml",
            "tracking:\n  group: my_experiment\n",
            _mlflow_tracking_fields(),
        )
