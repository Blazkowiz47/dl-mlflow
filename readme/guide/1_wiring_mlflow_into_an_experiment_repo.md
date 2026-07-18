# Wiring MLflow Into An Experiment Repo

1. Install `deep-learning-core[mlflow]` or install `deep-learning-mlflow`
   directly.
2. Run `dl-init --with-mlflow`.
3. Review the generated `callbacks.mlflow` block in `configs/base.yaml`.
4. If needed, override the tracker destination with
   `tracking.experiment_name` in `configs/base_sweep.yaml`.
5. Run local training or sweeps as usual.

This package is intentionally local-only. Azure-backed MLflow remains part of
`dl-azure`.
