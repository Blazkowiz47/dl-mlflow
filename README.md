# dl-mlflow

Local MLflow integration layer for `dl-core`.

`dl-mlflow` adds:
- an MLflow callback for local training runs
- `dl-init-experiment --with-mlflow` scaffold wiring
- the `dl-core[mlflow]` install path

## Install

```bash
uv add "dl-core[mlflow]" dl-mlflow
```

## Quick Start

```bash
uv run dl-init-experiment --root-dir . --with-mlflow
uv run dl-run --config configs/base.yaml
```

The scaffold points MLflow at a local `./mlruns` directory by default.

## Docs

- [Overview](./readme/README.md)
- [TLDR: Install And Enable](./readme/tldr/1_install_and_enable.md)
- [Guide: Wiring MLflow](./readme/guide/1_wiring_mlflow_into_an_experiment_repo.md)
- [Technical: Callback And Scaffold](./readme/technical/1_callback_and_scaffold.md)
