import subprocess
import pytest
from retrain_manager import RetrainingManager


def test_run_invokes_subprocesses(monkeypatch):
    calls = []

    def fake_run(cmd, *_, **__):
        calls.append(cmd)
        return None

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(RetrainingManager, "promote_if_better", lambda self: False)
    monkeypatch.setattr(RetrainingManager, "register_candidate_provenance", lambda self: None)

    mgr = RetrainingManager(data_dir="data")
    mgr.run()

    joined = [" ".join(c) for c in calls]
    assert any("retrain_cf" in s for s in joined)
    assert any("train_xgboost" in s for s in joined)


def test_exit_code_10_on_promotion(monkeypatch):
    from retrainer import main

    monkeypatch.setattr("retrainer.RetrainingManager.run", lambda _: True)
    monkeypatch.setattr("retrainer.DataPipeline.run", lambda *_: None)

    with pytest.raises(SystemExit) as e:
        main()

    assert e.value.code == 10
