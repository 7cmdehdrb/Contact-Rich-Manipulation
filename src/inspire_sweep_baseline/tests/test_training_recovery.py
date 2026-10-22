"""CPU/process tests for checkpoint rollback, isolation, and deliberate stops."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def recovery(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("training_recovery")


@pytest.fixture
def coordinator(recovery):
    return importlib.import_module("train_resilient")


def save_checkpoint(directory, iteration, std=1.0):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"model_{iteration}.pt"
    torch.save({"iter": iteration, "actor_state_dict": {"distribution.std_param": torch.tensor([std])},
                "critic_state_dict": {"weight": torch.ones(1)}, "optimizer_state_dict": {}, "infos": None}, path)
    return path


def arguments(tmp_path, **overrides):
    values = dict(task="test-task", max_iterations=4000, rollback_iterations=500,
                  max_restarts=10, job_dir=tmp_path / "job", resume_checkpoint=None)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_rollback_greatest_saved_boundary_and_corrupt_candidate(recovery, tmp_path):
    for iteration in (0, 2000, 2450, 2500, 3000):
        save_checkpoint(tmp_path, iteration)
    lineage = [{"directory": str(tmp_path), "eligible_through": None}]
    selected, details = recovery.select_rollback(lineage)
    assert selected.iteration == 2500 and details["rollback_target"] == 2500
    (tmp_path / "model_2500.pt").write_bytes(b"interrupted save")
    selected, details = recovery.select_rollback(lineage)
    assert selected.iteration == 2450
    assert "Unreadable checkpoint" in details["rejected"][0]


def test_early_failure_uses_zero_and_never_unrelated_runs(recovery, tmp_path):
    owned, unrelated = tmp_path / "owned", tmp_path / "unrelated"
    zero = save_checkpoint(owned, 0)
    save_checkpoint(owned, 250)
    save_checkpoint(unrelated, 10000)
    selected, details = recovery.select_rollback([{"directory": str(owned)}])
    assert selected.path == zero and details["latest_iteration"] == 250
    zero.unlink()
    selected, details = recovery.select_rollback([{"directory": str(owned)}])
    assert selected is None and details["fallback"] == "fresh_iteration_zero_no_valid_checkpoint"


def test_corrupt_bootstrap_restarts_fresh_but_later_missing_rollback_stops(recovery, tmp_path):
    zero = save_checkpoint(tmp_path, 0)
    zero.write_bytes(b"interrupted first save")
    selected, details = recovery.select_rollback([{"directory": str(tmp_path)}])
    assert selected is None and details["rollback_target"] == 0 and details["rejected"]
    save_checkpoint(tmp_path, 1000)
    with pytest.raises(ValueError, match="No valid checkpoint"):
        recovery.select_rollback([{"directory": str(tmp_path)}])


@pytest.mark.parametrize("std,expected", [(float("nan"), "Non-finite"), (float("inf"), "Non-finite"),
                                        (-.01, "Nonpositive Gaussian"), (0., "Nonpositive Gaussian")])
def test_bad_model_state_rejected(recovery, tmp_path, std, expected):
    path = save_checkpoint(tmp_path, 500, std=std)
    with pytest.raises(ValueError, match=expected):
        recovery.inspect_checkpoint(path)


def test_filename_and_iteration_must_agree(recovery, tmp_path):
    path = save_checkpoint(tmp_path, 0)
    renamed = path.with_name("model_500.pt")
    path.rename(renamed)
    with pytest.raises(ValueError, match="filename/iter disagree"):
        recovery.inspect_checkpoint(renamed)


@pytest.mark.parametrize("log_std", [-1000., 1000.])
def test_log_std_underflow_and_overflow_rejected(recovery, tmp_path, log_std):
    path = save_checkpoint(tmp_path, 0)
    data = torch.load(path, weights_only=True)
    data["actor_state_dict"] = {"distribution.log_std_param": torch.tensor([log_std])}
    torch.save(data, path)
    with pytest.raises(ValueError, match="Invalid exponentiated Gaussian"):
        recovery.inspect_checkpoint(path)


def test_old_future_branch_excluded_without_deleting(recovery, tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    for iteration in (0, 2000, 2100, 2500, 3000):
        save_checkpoint(old, iteration)
    untouched = hashlib.sha256((old / "model_3000.pt").read_bytes()).hexdigest()
    lineage = [{"directory": str(old)}]
    recovery.truncate_lineage(lineage, 2500)
    for iteration in (2500, 2600):
        save_checkpoint(new, iteration)
    lineage.append({"directory": str(new)})
    selected, details = recovery.select_rollback(lineage)
    assert details["latest_iteration"] == 2600 and selected.iteration == 2100
    assert hashlib.sha256((old / "model_3000.pt").read_bytes()).hexdigest() == untouched


def test_restore_resets_after_loaded_curriculum_and_absolute_budget(recovery):
    calls = []

    class Environment:
        @property
        def unwrapped(self):
            return self

        def set_learning_iteration(self, iteration):
            calls.append(("curriculum", iteration))
            self.learning_iteration = iteration

        def reset(self):
            calls.append(("reset", self.learning_iteration))

    class Runner:
        env = Environment()
        current_learning_iteration = 0

        def load(self, checkpoint):
            calls.append(("load", checkpoint))
            self.current_learning_iteration = 2500

    assert recovery.restore_for_training(Runner(), "model_2500.pt") == 2500
    assert calls == [("load", "model_2500.pt"), ("curriculum", 2500), ("reset", 2500)]
    assert recovery.remaining_iterations(90000, 2500) == 87500
    assert list(range(2500, 2500 + recovery.remaining_iterations(2503, 2500))) == [2500, 2501, 2502]


def test_bootstrap_save_preserves_real_logger_initialization_lifecycle(recovery, tmp_path):
    """Use installed RSL-RL classes: Logger.writer does not exist before learn."""
    from types import SimpleNamespace
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.utils.logger import Logger

    def algorithm_state():
        return {"actor_state_dict": {"distribution.std_param": torch.ones(1)},
                "critic_state_dict": {"weight": torch.ones(1)}, "optimizer_state_dict": {}}

    logger = Logger(log_dir=str(tmp_path), cfg={"algorithm": {"rnd_cfg": None}, "logger": "tensorboard"},
                    env_cfg={}, num_envs=1, is_distributed=False, gpu_world_size=1,
                    gpu_global_rank=0, device="cpu")
    logger.git_status_repos = []
    runner = object.__new__(OnPolicyRunner)
    runner.logger = logger
    runner.alg = SimpleNamespace(save=algorithm_state)
    runner.current_learning_iteration = 2500
    assert not hasattr(logger, "writer")
    # Reproduce the real pre-learn failure, then verify the production fix.
    with pytest.raises(AttributeError, match="writer"):
        runner.save(str(tmp_path / "premature.pt"))
    bootstrap = tmp_path / "model_2500.pt"
    recovery.save_bootstrap_checkpoint(runner, bootstrap)
    assert not hasattr(logger, "writer"), "Bootstrap must not initialize a duplicate writer"
    assert recovery.inspect_checkpoint(bootstrap).iteration == 2500
    assert not list(tmp_path.glob("*.tmp"))
    logger.init_logging_writer()
    try:
        writer = logger.writer
        regular = tmp_path / "regular.pt"
        runner.save(str(regular))
        assert logger.writer is writer
        initial_state = torch.load(bootstrap, weights_only=True)
        regular_state = torch.load(regular, weights_only=True)
        assert initial_state.keys() == regular_state.keys()
        assert initial_state["iter"] == regular_state["iter"] == 2500
        assert initial_state["infos"] is regular_state["infos"] is None
        torch.testing.assert_close(initial_state["actor_state_dict"]["distribution.std_param"],
                                   regular_state["actor_state_dict"]["distribution.std_param"])
    finally:
        logger.writer.close()


@pytest.fixture
def fake_worker(tmp_path):
    path = tmp_path / "worker.py"
    path.write_text('''import argparse, json, os, time
from pathlib import Path
import torch
p=argparse.ArgumentParser()
p.add_argument("--task")
p.add_argument("--max_iterations",type=int)
p.add_argument("--resilient-attempt-dir",type=Path)
p.add_argument("--resilient-target-iterations",type=int)
p.add_argument("--resilient-checkpoint",type=Path)
p.add_argument("--fake-mode",default="crashes")
a=p.parse_args()
d=a.resilient_attempt_dir
n=int(d.name.rsplit("_",1)[1])
restored=torch.load(a.resilient_checkpoint,weights_only=True)["iter"] if a.resilient_checkpoint else 0
assert a.max_iterations == a.resilient_target_iterations
(d/"worker.json").write_text(json.dumps({"pid":os.getpid(),"restored":restored,"remaining":a.max_iterations-restored}))
if a.fake_mode == "wait":
    print("READY",flush=True)
    time.sleep(60)
def save(i):
    path=d/f"model_{i}.pt"
    torch.save({"iter":i,"actor_state_dict":{"std":torch.ones(1)},"critic_state_dict":{"weight":torch.ones(1)},"optimizer_state_dict":{},"infos":None},path)
    return path
if a.fake_mode not in ("success", "completed-crash") and n == 0:
    for i in (0,2000,2100,2500,3000):save(i)
    raise RuntimeError("normal expects all elements of std >= 0.0")
if a.fake_mode not in ("success", "completed-crash") and n == 1:
    assert restored == 2500
    for i in (2500,2600):save(i)
    raise SystemExit(0) # incomplete normal exit must still trigger recovery
if a.fake_mode not in ("success", "completed-crash"):assert restored == 2100
checkpoint=save(a.max_iterations-1)
(d/"training_status.json").write_text(json.dumps({"status":"completed","target_iterations":a.max_iterations,"checkpoint":str(checkpoint)}))
if a.fake_mode == "completed-crash":raise RuntimeError("Kit cleanup failed after final checkpoint")
''')
    return path


def test_real_coordinator_restarts_crash_and_early_zero_exit(coordinator, tmp_path, fake_worker):
    args = arguments(tmp_path)
    assert coordinator.supervise(args, [], worker_script=fake_worker) == 0
    result = json.loads((args.job_dir / "job.json").read_text())
    assert result["status"] == "completed"
    assert [entry["restored_iteration"] for entry in result["attempts"]] == [0, 2500, 2100]
    workers = [json.loads((Path(entry["directory"]) / "worker.json").read_text()) for entry in result["attempts"]]
    assert len({worker["pid"] for worker in workers}) == 3
    assert [worker["remaining"] for worker in workers] == [4000, 1500, 1900]
    assert (args.job_dir / "attempt_000/model_3000.pt").exists()
    assert (args.job_dir / "attempt_002/model_3999.pt").exists()
    assert "normal expects" in (args.job_dir / "attempt_000/console.log").read_text()


def test_retry_limit_zero_and_explicit_initial_resume(coordinator, tmp_path, fake_worker):
    args = arguments(tmp_path, max_restarts=0)
    assert coordinator.supervise(args, [], worker_script=fake_worker) == 1
    result = json.loads((args.job_dir / "job.json").read_text())
    assert result["status"] == "retry_limit_reached" and len(result["attempts"]) == 1
    initial = save_checkpoint(tmp_path / "existing", 2500)
    before = initial.read_bytes()
    args = arguments(tmp_path, job_dir=tmp_path / "resumed", resume_checkpoint=initial)
    assert coordinator.supervise(args, ["--fake-mode", "success"], worker_script=fake_worker) == 0
    result = json.loads((args.job_dir / "job.json").read_text())
    assert result["attempts"][0]["restored_iteration"] == 2500
    assert result["lineage"][0]["eligible_through"] == 2500
    assert initial.read_bytes() == before


def test_completed_training_does_not_rollback_after_worker_cleanup_crash(coordinator, tmp_path, fake_worker):
    args = arguments(tmp_path)
    assert coordinator.supervise(args, ["--fake-mode", "completed-crash"], worker_script=fake_worker) == 0
    result = json.loads((args.job_dir / "job.json").read_text())
    assert result["status"] == "completed" and len(result["attempts"]) == 1
    assert result["attempts"][0]["returncode"] == 1
    assert "worker exited with 1" in result["worker_exit_warning"]
    assert not (args.job_dir / "attempt_001").exists()


def test_ctrl_c_stops_real_coordinator_without_respawning(tmp_path, fake_worker):
    job = tmp_path / "interrupted"
    source = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from pathlib import Path; from argparse import Namespace; "
        "from train_resilient import supervise; "
        "a=Namespace(task='test',max_iterations=4000,rollback_iterations=500,max_restarts=10,"
        "job_dir=Path(sys.argv[2]),resume_checkpoint=None); "
        "raise SystemExit(supervise(a,['--fake-mode','wait'],worker_script=sys.argv[3]))"
    )
    with (tmp_path / "supervisor.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-u", "-c", source, str(SCRIPTS), str(job), str(fake_worker)],
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 20
            while not (job / "attempt_000/worker.json").exists():
                assert process.poll() is None
                assert time.monotonic() < deadline
                time.sleep(.05)
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=15) == 130
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    result = json.loads((job / "job.json").read_text())
    assert result["status"] == "stopped" and len(result["attempts"]) == 1
    assert not (job / "attempt_001").exists()


def test_coordinator_rejects_unscoped_resume_arguments(coordinator, tmp_path):
    with pytest.raises(ValueError, match="explicit initial checkpoint"):
        coordinator.supervise(arguments(tmp_path), ["--load_run", ".*"])
    assert not (tmp_path / "job").exists()


def test_output_failure_reaps_worker(coordinator, tmp_path, monkeypatch):
    child_pids = []

    def broken_print(value, **kwargs):
        child_pids.append(int(value))
        raise BrokenPipeError("terminal consumer disconnected")

    monkeypatch.setattr("builtins.print", broken_print)
    with pytest.raises(BrokenPipeError):
        coordinator.run_worker([sys.executable, "-u", "-c",
                                "import os,time; print(os.getpid(),flush=True); time.sleep(60)"],
                               tmp_path / "child.log")
    assert len(child_pids) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(child_pids[0], 0)
