import pytest
import os
import json
import numpy as np
from datetime import datetime, timedelta
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge
from evaluation.Online.online_eval import OnlineEvaluator, get_evaluator


@pytest.fixture
def tmp_evaluator(tmp_path, monkeypatch):
    """Create isolated OnlineEvaluator for tests without changing source code."""
    log_path = tmp_path / "logs" / "online_metrics.json"

    # Patch Prometheus constructors to use a fresh local registry
    local_registry = CollectorRegistry()
    monkeypatch.setattr(
        "evaluation.Online.online_eval.Counter",
        lambda *a, **kw: Counter(*a, registry=local_registry, **kw),
    )
    monkeypatch.setattr(
        "evaluation.Online.online_eval.Histogram",
        lambda *a, **kw: Histogram(*a, registry=local_registry, **kw),
    )
    monkeypatch.setattr(
        "evaluation.Online.online_eval.Gauge",
        lambda *a, **kw: Gauge(*a, registry=local_registry, **kw),
    )

    return OnlineEvaluator(log_path=str(log_path))


def test_log_recommendations_and_save(tmp_evaluator):
    tmp_evaluator.log_recommendation("u1", ["m1", "m2"], 0.5)
    tmp_evaluator.log_recommendation("u2", ["m3"], 0.4)

    assert len(tmp_evaluator.metrics["recommendations"]) == 2
    assert os.path.exists(tmp_evaluator.log_path)
    assert all(isinstance(rt, float) for rt in tmp_evaluator.metrics["response_times"])


def test_log_interactions_and_hit_rate(tmp_evaluator):
    tmp_evaluator.log_recommendation("u1", ["m1", "m2"], 0.5)
    tmp_evaluator.log_interaction("u1", "m1", "click")
    assert tmp_evaluator.metrics["user_interactions"][0]["action_type"] == "click"

    tmp_evaluator.log_interaction("u2", "m9", "watch")
    assert len(tmp_evaluator.metrics["user_interactions"]) == 2


def test_log_error_and_error_rate(tmp_evaluator):
    tmp_evaluator.log_recommendation("u1", ["m1"], 0.5)
    tmp_evaluator.log_error("ValueError", "Invalid input")

    assert len(tmp_evaluator.metrics["errors"]) == 1
    assert "ValueError" in tmp_evaluator.metrics["errors"][0]["type"]
    assert tmp_evaluator.error_rate._value.get() > 0


def test_model_deployment_and_quality(tmp_evaluator):
    tmp_evaluator.log_model_deployment("v1.0", "XGBoost")
    assert tmp_evaluator.metrics["model_versions"][0]["version"] == "v1.0"

    tmp_evaluator.log_recommendation_quality(
        user_id="u1", recommended_items=["m1", "m2"], selected_item="m1", rating_score=4.0
    )
    assert tmp_evaluator.metrics["recommendation_quality"][0]["rating"] == 4.0


def test_compute_online_metrics(tmp_evaluator):
    now = datetime.now()
    past_time = (now - timedelta(hours=2)).isoformat()

    tmp_evaluator.metrics["recommendations"] = [
        {"timestamp": now.isoformat(), "user_id": "u1", "items": ["m1", "m2"]},
        {"timestamp": past_time, "user_id": "u2", "items": ["m3"]},
    ]
    tmp_evaluator.metrics["user_interactions"] = [
        {"timestamp": now.isoformat(), "user_id": "u1", "item_id": "m2", "action_type": "click"},
        {"timestamp": past_time, "user_id": "u2", "item_id": "m9", "action_type": "watch"},
    ]
    tmp_evaluator.metrics["response_times"] = [0.5, 0.7]
    tmp_evaluator.metrics["errors"] = []

    metrics = tmp_evaluator.compute_online_metrics(last_hours=1)
    assert isinstance(metrics, dict)
    assert "num_recommendations" in metrics
    assert metrics["num_recommendations"] >= 1
    assert 0.0 <= metrics["hit_rate"] <= 1.0
    assert metrics["user_coverage"] == 1  # only u1 is recent


def test_get_evaluator_singleton(tmp_path):
    log_path = tmp_path / "singleton.json"
    eval1 = get_evaluator()
    eval2 = get_evaluator()
    assert eval1 is eval2
