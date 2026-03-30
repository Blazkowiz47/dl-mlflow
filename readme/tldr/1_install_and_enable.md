# Install And Enable

```bash
uv add "dl-core[mlflow]" dl-mlflow
uv run dl-init --root-dir . --with-mlflow
```

The generated config enables a local MLflow callback with `./mlruns` as the
default tracking directory.
