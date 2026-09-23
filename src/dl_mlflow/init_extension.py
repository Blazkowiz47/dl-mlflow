"""MLflow scaffold extension for dl-init."""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from dl_core.init_extensions import InitExtension, ScaffoldContext


def _append_gitignore_patterns(
    context: ScaffoldContext,
    *patterns: str,
) -> None:
    """Append MLflow runtime outputs to the generated project gitignore."""
    relative = Path(".gitignore")
    content = context.files.get(relative, "")
    existing_lines = set(content.splitlines())
    missing = [pattern for pattern in patterns if pattern not in existing_lines]
    if not missing:
        return
    prefix = content.rstrip()
    suffix = "\n".join(missing)
    context.set_file(relative, f"{prefix}\n{suffix}\n" if prefix else f"{suffix}\n")


def _mlflow_callback_block() -> str:
    """Render the scaffold callback block for local MLflow logging."""

    return """
  mlflow:
    run_name: null
    tracking_uri: ./mlruns
    log_config: true
"""


def _inject_mlflow_tracking_fields(content: str) -> str:
    """Set the local MLflow sweep backend without silently duplicating it."""
    marker = "tracking:\n"
    if content.count(marker) != 1:
        raise ValueError("Expected one tracking block in configs/base_sweep.yaml")
    tracking = (yaml.safe_load(content) or {}).get("tracking")
    if not isinstance(tracking, dict):
        raise ValueError("Expected a mapping at tracking in configs/base_sweep.yaml")
    backend = tracking.get("backend")
    if backend == "mlflow":
        return content
    if backend is not None:
        raise ValueError("Sweep tracking backend is already configured")
    return content.replace(
        marker, f"{marker}  backend: mlflow\n  tracking_uri: ./mlruns\n", 1
    )


class MlflowInitExtension(InitExtension):
    """Expose local MLflow scaffold wiring when dl-mlflow is installed."""

    name = "mlflow"
    tracking_backend = "mlflow"

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

        context.add_dependency("deep-learning-mlflow")
        _append_gitignore_patterns(context, "mlruns/")
        context.append_bootstrap_import("import dl_mlflow  # noqa: F401")
        context.append_readme_note(
            "Local MLflow support is enabled. Review the `callbacks.mlflow` "
            "block in `configs/base.yaml` before training."
        )
        base_path = Path("configs") / "base.yaml"
        if "  mlflow:\n" not in context.get_file(base_path):
            if "  metric_logger:\n    log_frequency: 1\n" not in context.get_file(base_path):
                raise ValueError("MLflow callback anchor not found in configs/base.yaml")
            context.replace_in_file(
                base_path,
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
