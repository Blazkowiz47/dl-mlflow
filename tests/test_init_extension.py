"""Tests for the local MLflow init extension plugin."""

from __future__ import annotations

from pathlib import Path

from dl_core.init_extensions import ProjectNames, ScaffoldContext

from dl_mlflow.init_extension import MlflowInitExtension


def test_mlflow_init_extension_updates_scaffold_files(tmp_path: Path) -> None:
    """The MLflow init extension should patch the scaffold for local MLflow."""

    context = ScaffoldContext(
        target_dir=tmp_path,
        templates_dir=tmp_path,
        project=ProjectNames(
            project_name="demo",
            project_slug="demo",
            component_name="demo",
            dataset_name="demo",
            dataset_class_name="DemoDataset",
            model_name="resnet_example",
            model_class_name="ResNetExample",
            trainer_name="demo",
            trainer_class_name="DemoTrainer",
        ),
        files={
            Path("pyproject.toml"): (
                "[project]\n"
                "dependencies = [\n"
                '    "deep-learning-core",\n'
                "]\n"
            ),
            Path("README.md"): "# demo\n",
            Path("src") / "bootstrap.py": (
                '"""Project bootstrap hooks for local component loading."""\n'
            ),
            Path("configs") / "base.yaml": (
                "callbacks:\n"
                "  metric_logger:\n"
                "    log_frequency: 1\n"
            ),
            Path("configs") / "base_sweep.yaml": (
                "tracking:\n"
                "  # sweep_name: demo\n"
                "  # Optional override. Defaults to the sweep filename.\n"
                '  run_name_template: "lr_{optimizers.lr}"\n'
                '  description_template: "learning_rate={optimizers.lr}"\n'
            ),
        },
        enabled_extensions={"mlflow"},
    )

    MlflowInitExtension().apply(context)

    assert '"deep-learning-core[mlflow]"' in context.get_file("pyproject.toml")
    assert '"deep-learning-mlflow"' in context.get_file("pyproject.toml")
    assert "import dl_mlflow" in context.get_file(Path("src") / "bootstrap.py")
    assert "backend: mlflow" in context.get_file(
        Path("configs") / "base_sweep.yaml"
    )
    assert "# sweep_name: demo" in context.get_file(
        Path("configs") / "base_sweep.yaml"
    )
    assert "callbacks:" in context.get_file(Path("configs") / "base.yaml")
