# Callback And Scaffold

`dl-mlflow` currently provides two things:

- `callbacks.mlflow`: a local MLflow callback for training runs
- `--with-mlflow`: scaffold wiring for `dl-init`
- a generated `mlruns/` Git ignore entry for local tracking data

The callback writes to a local `tracking_uri` by default and supports
`parent_run_id` for nested sweep runs.
