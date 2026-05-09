"""MLflow scaffold extension for dl-init."""

from __future__ import annotations

import argparse
from pathlib import Path

from dl_core.init_extensions import InitExtension, ScaffoldContext


def _mlflow_callback_block() -> str:
    """Render the scaffold callback block for local MLflow logging."""

    return """
  mlflow:
    run_name: null
    tracking_uri: ./mlruns
    log_config: true
"""


def _mlflow_tracking_fields() -> str:
    """Render local MLflow-specific additions to the sweep tracking block."""

    return """  backend: mlflow
  tracking_uri: ./mlruns
"""


def _inject_mlflow_tracking_fields(content: str) -> str:
    """Inject MLflow-specific tracking fields into the sweep scaffold."""

    if "tracking:\n" not in content:
        return content

    if "tracking:\n  backend: mlflow\n" in content:
        return content

    return content.replace(
        "tracking:\n",
        f"tracking:\n{_mlflow_tracking_fields()}",
        1,
    )


class MlflowInitExtension(InitExtension):
    """Expose local MLflow scaffold wiring when dl-mlflow is installed."""

    name = "mlflow"

    def display_name(self) -> str:
        """Return the prompt label for MLflow support."""
        return "MLflow"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register the local MLflow scaffold flag."""
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--with-mlflow",
            dest="with_mlflow",
            action="store_true",
            default=None,
            help="Include local MLflow callback wiring and tracking defaults.",
        )
        group.add_argument(
            "--without-mlflow",
            dest="with_mlflow",
            action="store_false",
            default=None,
            help="Exclude local MLflow scaffold wiring even when dl-mlflow is installed.",
        )

    def is_enabled(
        self,
        args: argparse.Namespace,
        discovered_extensions: dict[str, InitExtension],
    ) -> bool:
        """Enable MLflow wiring when explicitly requested."""

        del discovered_extensions
        return self.selection_state(args) is True

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
            context.get_file(Path("configs") / "base_sweep.yaml"),
            _inject_mlflow_tracking_fields(
                context.get_file(Path("configs") / "base_sweep.yaml")
            ),
        )
