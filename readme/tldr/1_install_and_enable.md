# Install And Enable

```bash
uv add "deep-learning-core[mlflow]"
uv run dl-init --root-dir . --with-mlflow
```

The generated config enables a local MLflow callback with `./mlruns` as the
default tracking directory and ignores that directory in Git.
