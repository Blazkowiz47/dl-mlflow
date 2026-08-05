# deep-learning-mlflow Release History

The main README shows only the latest release. This page preserves the
release-by-release changes.

## 0.0.14

- the supported core range includes the architecture-free
  `deep-learning-core==0.1.0` trainer and registry boundary
- MLflow callbacks, trackers, metric sources, and scaffold behavior remain
  unchanged

## 0.0.13

- MLflow implements the public RL callback hooks `on_episode_end()`,
  `on_update_end()`, and `on_evaluation_end()`
- custom callbacks can subclass the integration without relying on private
  underscore methods
- the core compatibility floor moved to `deep-learning-core>=0.0.34,<0.1`

## 0.0.12

- RL episodes, algorithm updates, and aggregate evaluations are logged
  alongside supervised epoch metrics
- evaluation episodes remain separate from training-episode series
- top-level tracking URIs take precedence over callback fallbacks
- generated repositories depend directly on `deep-learning-mlflow` and ignore
  local `mlruns/` data
- the core compatibility floor moved to `deep-learning-core>=0.0.26,<0.1`

## 0.0.11

- the core compatibility floor moved to `deep-learning-core>=0.0.25,<0.1`
- local MLflow callback, scaffold defaults, and run-artifact upload behavior
  remained available without Azure dependencies

Structured release notes begin with 0.0.11. Earlier package history remains
available through the repository's Git history.
