"""MLflow metrics source backed by local normalized artifact files."""

from __future__ import annotations

from dl_core.core import register_metrics_source
from dl_core.metrics_sources.local import LocalMetricsSource


@register_metrics_source("mlflow")
class MlflowMetricsSource(LocalMetricsSource):
    """Reuse local normalized artifact files for MLflow-backed sweep analysis."""
