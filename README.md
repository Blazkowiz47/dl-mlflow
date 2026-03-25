# dl-mlflow

Local MLflow integration layer for `dl-core`.

`dl-mlflow` adds local MLflow tracking on top of `dl-core` without Azure
dependencies. It is the public MLflow variant behind `dl-core[mlflow]`.

## Install

Install from PyPI:

```bash
pip install "dl-core[mlflow]"
```

Install the package directly:

```bash
pip install dl-mlflow
```

Install in a `uv` project:

```bash
uv add "dl-core[mlflow]" dl-mlflow
```

## Quick Start

```bash
uv run dl-init-experiment --root-dir . --with-mlflow
uv run dl-run --config configs/base.yaml
```

The scaffold points MLflow at a local `./mlruns` directory by default.

## What You Get

- the `mlflow` callback for local training runs
- `dl-init-experiment --with-mlflow` scaffold support
- local `./mlruns` tracking defaults for generated experiment repositories

Azure-backed MLflow wiring remains part of `dl-azure`.

## Companion Packages

- [`dl-core`](https://github.com/Blazkowiz47/dl-core)
- [`dl-azure`](https://github.com/Blazkowiz47/dl-azure)
- [`dl-wandb`](https://github.com/Blazkowiz47/dl-wandb)

## Docs

- [Documentation Index](https://github.com/Blazkowiz47/dl-mlflow/tree/main/readme)
- [GitHub Repository](https://github.com/Blazkowiz47/dl-mlflow)
