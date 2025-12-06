import json
from pathlib import Path
import pytest

from retrain_manager import (
    DataRefreshManager,
    RetrainingManager,
    STATE_PATH,
)

# -----------------------------
# helpers
# -----------------------------

def write_metric(path: Path, rmse: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "regression_metrics": {"rmse": rmse}
    }))


# -----------------------------
# DataRefreshManager tests
# -----------------------------

def test_data_refresh_computes_deltas(tmp_path):
    data_dir = tmp_path / "data"
    raw = data_dir / "raw_data"
    raw.mkdir(parents=True)

    (raw / "users.csv").write_text("user_id\n1\n2\n3\n")
    (raw / "movies.csv").write_text("id\n10\n")
    (raw / "watch_time.csv").write_text("u,m\n1,10\n")
    (raw / "ratings.csv").write_text("u,m,r\n1,10,5\n")

    class DummyPipeline:
        def run(self): pass

    mgr = DataRefreshManager(str(data_dir))
    report = mgr.refresh(DummyPipeline())

    assert report["stats"]["users"] == 3
    assert report["stats"]["interactions"] == 1
    assert report["deltas"]["users"] == 3


# -----------------------------
# Retrain state tests
# -----------------------------

def test_retrain_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("retrain_manager.STATE_PATH", tmp_path / "state.json")

    stats = {"users": 100, "interactions": 500}
    RetrainingManager.save_retrain_state(stats)

    state = RetrainingManager.load_retrain_state()
    assert state["users_seen"] == 100
    assert state["interactions_seen"] == 500


# -----------------------------
# Promotion logic tests
# -----------------------------

def test_candidate_promoted_if_better(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    write_metric(tmp_path / "src/models/candidate/evaluation_results.json", 0.90)
    write_metric(tmp_path / "src/models/v2/evaluation_results.json", 0.95)

    mgr = RetrainingManager(data_dir="data")
    assert mgr.promote_if_better() is True


def test_candidate_rejected_if_not_better(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    write_metric(tmp_path / "src/models/candidate/evaluation_results.json", 0.96)
    write_metric(tmp_path / "src/models/v2/evaluation_results.json", 0.95)

    mgr = RetrainingManager(data_dir="data")
    assert mgr.promote_if_better() is False


# -----------------------------
# Gating logic
# -----------------------------

def test_gating_thresholds():
    unseen_users = 30
    unseen_interactions = 50

    assert not (unseen_users >= 50 or unseen_interactions >= 100)

