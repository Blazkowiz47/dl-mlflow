# deep-learning-mlflow

Local MLflow integration layer for `deep-learning-core`.

`deep-learning-mlflow` adds local MLflow tracking on top of
`deep-learning-core` without Azure dependencies. It is the public MLflow
variant behind `deep-learning-core[mlflow]`.

## Install

The package is now available on PyPI under the `deep-learning-mlflow` name.
TestPyPI remains available for validation flows.

PyPI install target:

```bash
pip install "deep-learning-core[mlflow]"
```

Install the package directly:

```bash
pip install deep-learning-mlflow
```

Install in a `uv` project:

```bash
uv add "deep-learning-core[mlflow]" deep-learning-mlflow
```

## Quick Start

```bash
uv init
uv add deep-learning-mlflow
uv run dl-init-experiment --root-dir . --with-mlflow
uv run dl-run --config configs/base.yaml
uv run dl-sweep experiments/lr_sweep.yaml
```

The scaffold points MLflow at a local `./mlruns` directory by default.
Tracker experiment naming defaults to the repository root name unless
`tracking.experiment_name` overrides it.

Concrete local tracking flow:

```bash
uv run dl-init-experiment --root-dir . --with-mlflow
uv run dl-run --config configs/base.yaml
uv run dl-analyze --sweep experiments/lr_sweep.yaml
```

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

## License

MIT. See [LICENSE](LICENSE).
