# dl-mlflow Repository Guidelines

## Scope

- Work from the `dl-mlflow/` package root unless a narrower directory is clearly better.
- `dl-mlflow` is the MLflow integration layer on top of `dl-core`.
- Keep reusable, vendor-neutral framework behavior in `dl-core`.
- Keep MLflow-specific callbacks, tracking, metrics-source wiring, and scaffold wiring in `dl-mlflow`.

## Structure

- `src/dl_mlflow/` contains the package code.
- `src/dl_mlflow/init_extension.py` owns MLflow-specific experiment scaffold changes.
- `tests/` contains package-level pytest coverage.
- `readme/` and `README.md` contain user-facing package documentation.
- `.github/workflows/` contains CI and publish workflows.
- `dist/` is build output, not source.

## Commands

- Prefer `rg` and `rg --files` for search.
- Prefer non-destructive commands by default.
- Prefer `uv` for Python execution, validation, lockfile refresh, and builds.
- Prefer targeted validation for changed files before broader test runs.
- Typical validation commands:
  `uv run --extra dev pytest`
  `uv run python -m compileall src/dl_mlflow`
  `uv build --no-sources`

## Development Rules

- Keep edits minimal and task-scoped.
- Use type hints and concise docstrings for public modules and APIs.
- Match existing style: 4-space indentation, `snake_case`, `PascalCase`, short module docstrings, and f-strings.
- Prefer `apply_patch` for focused edits.
- Do not overwrite unrelated user changes in a dirty tree.
- Whenever package behavior, public APIs, CLI behavior, scaffold output, dependencies, or versions change, review `README.md` and the relevant `readme/` docs and keep them consistent with the code. If no documentation update is needed, state why.

## Package Rules

- Do not move MLflow-specific logic into `dl-core`.
- Keep scaffold logic in `src/dl_mlflow/init_extension.py` aligned with the generated experiment-repo experience.
- When scaffold behavior changes, update the matching tests in `tests/test_init_extension.py`.
- Preserve compatibility with the generated `AGENTS.md` and config conventions produced by `dl-init`.
- Keep default local MLflow behavior explicit in scaffolded config instead of relying on hidden defaults.

## Execution Policy

- Never run training jobs, sweeps, worker processes, or publish workflows unless explicitly requested.
- Do not push commits or trigger GitHub Actions unless the user asked for that outcome.
- If validation was skipped, state that clearly.

## Versioning And Release

- When bumping the package version, update `pyproject.toml` and `src/dl_mlflow/__init__.py`.
- Refresh `uv.lock` instead of editing it by hand when package metadata changes.
- Use concise release commits such as `release: bump dl-mlflow to 0.1.8`.
- Use `gh run list`, `gh run view`, and `gh run watch` when GitHub Actions work is requested.
- Treat `publish` as TestPyPI by default unless the user explicitly asks for a real PyPI release.

## Git And PR Rules

- Stage and commit only task-scoped files.
- Use clear prefix-based commit messages such as `fix: ...`, `docs: ...`, `test: ...`, or `release: ...`.
- Keep pull requests focused and include validation evidence plus any scaffold or release impact.

## Agent Behavior

- Before substantial tool use, restate the goal and give a short plan.
- For multi-step tasks, provide concise progress updates.
- If uncertain, gather evidence with tools instead of guessing.
