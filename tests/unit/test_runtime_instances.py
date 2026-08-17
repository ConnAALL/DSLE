from __future__ import annotations

import json
import multiprocessing
import subprocess
from pathlib import Path

import pytest

from dsle.exceptions import ConfigurationError, RuntimeUnavailableError
from dsle.runtime import instances as runtime_instances


def _hold_instance_launch_lock(state_dir, entered, release) -> None:
    with runtime_instances._instance_launch_lock(Path(state_dir)):
        entered.set()
        release.wait(timeout=5.0)


def write_config(tmp_path: Path, count: int = 2) -> Path:
    generated = runtime_instances.generate_instances(
        count,
        wineprefix_root=tmp_path / "wineprefixes",
        display_start=90,
        vnc_port_start=5901,
    )
    config = tmp_path / "config" / "instances.json"
    runtime_instances.write_instance_config(config, generated)
    return config


def test_generate_write_load_and_resolve_isolated_instances(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    loaded = runtime_instances.load_instances(config)
    assert tuple(loaded) == ("dsr-1", "dsr-2")
    assert loaded["dsr-1"].display == ":90"
    assert loaded["dsr-2"].display == ":91"
    assert loaded["dsr-1"].vnc_port == 5901
    assert loaded["dsr-2"].vnc_port == 5902
    assert loaded["dsr-1"].wineprefix != loaded["dsr-2"].wineprefix
    assert loaded["dsr-1"].xdg_runtime_dir != loaded["dsr-2"].xdg_runtime_dir
    assert runtime_instances.resolve_instance("dsr-2", config) == loaded["dsr-2"]


def test_instance_config_is_group_readable_for_host_tools(tmp_path: Path) -> None:
    config = write_config(tmp_path, count=1)

    assert config.stat().st_mode & 0o777 == 0o640


def test_instance_config_refuses_overwrite_and_duplicate_resources(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    generated = runtime_instances.generate_instances(1, wineprefix_root=tmp_path / "other")
    with pytest.raises(FileExistsError):
        runtime_instances.write_instance_config(config, generated)

    document = json.loads(config.read_text(encoding="utf-8"))
    document["instances"]["dsr-2"]["display_num"] = 90
    document["instances"]["dsr-2"]["display"] = ":90"
    config.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="display_num values must be unique"):
        runtime_instances.load_instances(config)


def test_instance_config_requires_unique_runtime_and_explicit_save_directories(
    tmp_path: Path,
) -> None:
    config = write_config(tmp_path)
    document = json.loads(config.read_text(encoding="utf-8"))
    document["instances"]["dsr-2"]["xdg_runtime_dir"] = document["instances"]["dsr-1"][
        "xdg_runtime_dir"
    ]
    config.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="xdg_runtime_dir values must be unique"):
        runtime_instances.load_instances(config)

    document["instances"]["dsr-2"]["xdg_runtime_dir"] = str(tmp_path / "runtime-2")
    shared_save = str(tmp_path / "shared-save")
    document["instances"]["dsr-1"]["save_dir"] = shared_save
    document["instances"]["dsr-2"]["save_dir"] = shared_save
    config.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="save_dir values must be unique"):
        runtime_instances.load_instances(config)


def test_status_is_honest_before_any_process_is_started(tmp_path: Path) -> None:
    config = write_config(tmp_path, count=1)
    assert runtime_instances.instance_status(config_path=config) == {
        "dsr-1": {"name": "dsr-1", "status": "not_started", "running": False}
    }


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="the live-game runtime targets Linux and tests its process lock with fork",
)
def test_instance_launch_lock_serializes_independent_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    first_entered = context.Event()
    first_release = context.Event()
    second_entered = context.Event()
    second_release = context.Event()
    first = context.Process(
        target=_hold_instance_launch_lock,
        args=(str(tmp_path), first_entered, first_release),
    )
    second = context.Process(
        target=_hold_instance_launch_lock,
        args=(str(tmp_path), second_entered, second_release),
    )
    try:
        first.start()
        assert first_entered.wait(timeout=2.0)
        second.start()
        assert not second_entered.wait(timeout=0.2)
        first_release.set()
        assert second_entered.wait(timeout=2.0)
        second_release.set()
        first.join(timeout=2.0)
        second.join(timeout=2.0)
        assert first.exitcode == 0
        assert second.exitcode == 0
    finally:
        first_release.set()
        second_release.set()
        if first.pid is not None:
            first.join(timeout=6.0)
        if second.pid is not None:
            second.join(timeout=6.0)


def test_start_rejects_mode_mismatch_before_launching_any_process(
    tmp_path: Path, monkeypatch
) -> None:
    config = write_config(tmp_path, count=1)
    metadata = tmp_path / "run" / "dsr-1.json"
    metadata.parent.mkdir()
    metadata.write_text(
        json.dumps(
            {
                "name": "dsr-1",
                "status": "ready",
                "mode": "headless",
                "supervisor_pid": 4321,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_instances, "_is_instance_supervisor", lambda *_args: True)
    monkeypatch.setattr(
        runtime_instances.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a mode mismatch must be rejected before launching")
        ),
    )

    with pytest.raises(ConfigurationError, match="already running in mode 'headless'"):
        runtime_instances.start_instances(
            config_path=config,
            names=("dsr-1",),
            mode="headless-vnc",
        )


def test_stale_metadata_never_triggers_child_process_signals(tmp_path: Path, monkeypatch) -> None:
    config = write_config(tmp_path, count=1)
    metadata = tmp_path / "run" / "dsr-1.json"
    metadata.parent.mkdir()
    metadata.write_text(
        json.dumps(
            {
                "name": "dsr-1",
                "status": "ready",
                "supervisor_pid": 1001,
                "wine_pid": 2002,
                "vnc_pid": 3003,
                "xorg_pid": 4004,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(runtime_instances, "_is_instance_supervisor", lambda *_args: False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("stale child process metadata must never be signalled")

    monkeypatch.setattr(runtime_instances, "_terminate_process_group", forbidden)
    result = runtime_instances.stop_instances(config_path=config)
    assert result["dsr-1"]["status"] == "stopped"
    assert result["dsr-1"]["running"] is False


def test_restart_invalidates_stale_failure_before_supervisor_launch(
    tmp_path: Path, monkeypatch
) -> None:
    config = write_config(tmp_path, count=1)
    metadata = tmp_path / "run" / "dsr-1.json"
    metadata.parent.mkdir()
    metadata.write_text(
        json.dumps({"name": "dsr-1", "status": "failed", "error": "old failure"}),
        encoding="utf-8",
    )

    class Process:
        pid = 4321
        returncode = None

        def __init__(self, *_args, **_kwargs):
            preflight = json.loads(metadata.read_text(encoding="utf-8"))
            assert preflight["status"] == "starting"
            assert "old failure" not in preflight
            metadata.write_text(
                json.dumps(
                    {
                        "name": "dsr-1",
                        "status": "ready",
                        "supervisor_pid": self.pid,
                    }
                ),
                encoding="utf-8",
            )

        def poll(self):
            return None

    monkeypatch.setattr(runtime_instances.subprocess, "Popen", Process)
    monkeypatch.setattr(
        runtime_instances,
        "_is_instance_supervisor",
        lambda pid, name, selected_config: (
            pid == 4321 and name == "dsr-1" and Path(selected_config) == config
        ),
    )

    result = runtime_instances.start_instances(
        config_path=config,
        names=("dsr-1",),
        mode="headless",
    )

    assert result["dsr-1"]["status"] == "ready"
    assert result["dsr-1"]["running"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 0},
        {"count": 1, "display_start": -1},
        {"count": 2, "vnc_port_start": 65535},
    ],
)
def test_generation_rejects_invalid_ranges(kwargs) -> None:
    with pytest.raises(ValueError):
        runtime_instances.generate_instances(**kwargs)


def test_instance_names_are_canonical_and_cannot_escape_runtime_paths(tmp_path: Path) -> None:
    config = write_config(tmp_path, count=1)
    document = json.loads(config.read_text(encoding="utf-8"))
    document["instances"]["../../outside"] = document["instances"].pop("dsr-1")
    config.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="dsr-1 through dsr-30"):
        runtime_instances.load_instances(config)
    with pytest.raises(ConfigurationError, match="dsr-1 through dsr-30"):
        runtime_instances._metadata_path(tmp_path, "/tmp/outside")
    with pytest.raises(ValueError, match="name_prefix"):
        runtime_instances.generate_instances(1, name_prefix="../")


def test_supervisor_identity_includes_the_exact_instance_config(tmp_path: Path) -> None:
    first = (tmp_path / "first" / "instances.json").resolve()
    second = (tmp_path / "second" / "instances.json").resolve()
    arguments = (
        "/usr/bin/python3",
        "-m",
        "dsle.runtime.instances",
        "--supervise",
        "--config",
        str(first),
        "--name",
        "dsr-1",
        "--mode",
        "headless",
    )

    assert runtime_instances._supervisor_arguments_match(arguments, "dsr-1", first)
    assert not runtime_instances._supervisor_arguments_match(arguments, "dsr-1", second)
    assert not runtime_instances._supervisor_arguments_match(arguments, "dsr-2", first)


def test_start_timeout_terminates_and_rolls_back_the_new_supervisor(
    tmp_path: Path, monkeypatch
) -> None:
    config = write_config(tmp_path, count=1)
    terminated: list[tuple[int, float]] = []
    rollback_calls: list[tuple[str, ...]] = []

    class Process:
        pid = 4321
        returncode = None

        def __init__(self, *_args, **_kwargs):
            pass

        def poll(self):
            return None

    monkeypatch.setattr(runtime_instances, "INSTANCE_READINESS_TIMEOUT_S", 0.0)
    monkeypatch.setattr(runtime_instances.subprocess, "Popen", Process)
    monkeypatch.setattr(
        runtime_instances,
        "_terminate_child",
        lambda process, timeout_s: terminated.append((process.pid, timeout_s)),
    )
    monkeypatch.setattr(runtime_instances, "_is_instance_supervisor", lambda *_args: False)

    def stop_pool(*, config_path, names):
        assert Path(config_path) == config
        rollback_calls.append(tuple(names))
        return {name: {"name": name, "status": "stopped", "running": False} for name in names}

    monkeypatch.setattr(runtime_instances, "stop_instances", stop_pool)

    result = runtime_instances.start_instances(config_path=config, names=("dsr-1",))

    assert terminated == [(4321, 3.0)]
    assert rollback_calls == [("dsr-1",)]
    assert result["dsr-1"]["status"] == "failed"
    assert result["dsr-1"]["running"] is False
    assert "did not report ready" in result["dsr-1"]["error"]


def test_wine_audio_is_forced_to_pulse_for_reusable_prefixes(monkeypatch) -> None:
    calls = []

    def fake_run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(runtime_instances.subprocess, "run", fake_run)
    monkeypatch.setattr(runtime_instances, "_command", lambda name: f"/usr/bin/{name}")

    runtime_instances._configure_wine_audio(
        {"WINEPREFIX": "/state/wineprefixes/dsr-1", "DISPLAY": ":90"}
    )

    arguments, kwargs = calls[0]
    assert arguments == [
        "/usr/bin/wine",
        "reg",
        "add",
        r"HKCU\Software\Wine\Drivers",
        "/v",
        "Audio",
        "/t",
        "REG_SZ",
        "/d",
        "pulse",
        "/f",
    ]
    assert kwargs["env"]["WINEPREFIX"] == "/state/wineprefixes/dsr-1"
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True


def test_reusable_runtime_tree_is_reclaimed_without_following_symlinks(
    tmp_path: Path, monkeypatch
) -> None:
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    drive = prefix / "drive_c"
    drive.mkdir()
    registry = prefix / "user.reg"
    registry.write_text("registry", encoding="ascii")
    link = prefix / "dosdevice"
    link.symlink_to(drive, target_is_directory=True)
    calls = []

    monkeypatch.setattr(runtime_instances.os, "geteuid", lambda: 123)
    monkeypatch.setattr(runtime_instances.os, "getegid", lambda: 456)
    monkeypatch.setattr(
        runtime_instances.os,
        "chown",
        lambda path, uid, gid, **kwargs: calls.append((Path(path), uid, gid, kwargs)),
    )

    runtime_instances._claim_runtime_tree(prefix)

    claimed = {path for path, _uid, _gid, _kwargs in calls}
    assert {prefix, drive, registry, link} <= claimed
    assert all(uid == 123 and gid == 456 for _path, uid, gid, _kwargs in calls)
    assert all(kwargs == {"follow_symlinks": False} for _path, _uid, _gid, kwargs in calls)


def test_reusable_runtime_tree_rejects_a_symbolic_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "prefix"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeUnavailableError, match="real directory"):
        runtime_instances._claim_runtime_tree(link)
