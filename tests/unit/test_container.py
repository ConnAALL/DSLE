from __future__ import annotations

import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest

import dsle.container as container_module
from dsle.container import ContainerSession, ManagedDarkSoulsEnv, validate_game_directory
from dsle.exceptions import ConfigurationError, RuntimeUnavailableError
from dsle.runtime.version import REQUIRED_GAME_DIRECTORIES

IMAGE_ID = "sha256:" + "1" * 64
CONTAINER_ID = "2" * 64


@pytest.fixture(autouse=True)
def known_test_build(monkeypatch) -> None:
    monkeypatch.setattr(container_module, "verify_game_executable", lambda *_args, **_kwargs: None)


def game_directory(tmp_path: Path) -> Path:
    game = tmp_path / "legal-game-install"
    game.mkdir()
    (game / "DarkSoulsRemastered.exe").write_bytes(b"MZ\x00test")
    for name in REQUIRED_GAME_DIRECTORIES:
        (game / name).mkdir()
    return game.resolve()


def completed(command, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


@pytest.mark.parametrize("value", [0, -1, True, False, "1", float("nan"), float("inf")])
@pytest.mark.parametrize("option", ["startup_timeout", "stop_timeout", "request_timeout"])
def test_container_session_rejects_invalid_timeouts(option: str, value: object) -> None:
    kwargs = {option: value}
    with pytest.raises(ValueError, match="finite positive"):
        ContainerSession(
            image="dsle-runtime:0.1.0",
            game_dir="/not-inspected",
            output_dir="/not-inspected",
            docker_binary="docker",
            **kwargs,
        )


def test_game_directory_requires_real_nonempty_executable(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ConfigurationError, match="does not exist"):
        validate_game_directory(missing)

    game = tmp_path / "game"
    game.mkdir()
    executable = game / "DarkSoulsRemastered.exe"
    executable.touch()
    with pytest.raises(ConfigurationError, match="non-empty"):
        validate_game_directory(game)


@pytest.mark.parametrize("name", ["", 7, "-leading-option"])
def test_invalid_explicit_container_name_has_no_output_side_effect(tmp_path: Path, name) -> None:
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Container name"):
        ContainerSession(
            image="dsle-runtime:0.1.0",
            game_dir=tmp_path / "not-inspected",
            output_dir=output,
            name=name,
            docker_binary="docker",
        )
    assert not output.exists()


@pytest.mark.parametrize("ports", [None, 5901, "5901", b"5901"])
def test_vnc_ports_reject_non_sequences_before_filesystem_access(tmp_path: Path, ports) -> None:
    with pytest.raises(ValueError, match="vnc_ports"):
        ContainerSession(
            image="dsle-runtime:0.1.0",
            game_dir=tmp_path / "not-inspected",
            output_dir=tmp_path / "output",
            docker_binary="docker",
            vnc_ports=ports,
        )


def test_game_directory_requires_assets_and_forces_verified_v104(
    tmp_path: Path, monkeypatch
) -> None:
    game = game_directory(tmp_path)
    calls = []
    monkeypatch.setattr(
        container_module,
        "verify_game_executable",
        lambda executable, **kwargs: calls.append((executable, kwargs)),
    )
    assert validate_game_directory(game) == game
    assert calls == [(game / "DarkSoulsRemastered.exe", {"allow_unverified": False})]

    (game / "script").rmdir()
    with pytest.raises(ConfigurationError, match="script"):
        validate_game_directory(game)


def test_prebuilt_session_uses_exact_readonly_game_mount_and_owned_id(
    tmp_path: Path, monkeypatch
) -> None:
    game = game_directory(tmp_path)
    state = tmp_path / "state"
    calls: list[list[str]] = []
    session_holder: list[ContainerSession] = []
    stopped = False

    def fake_run(command, **_kwargs):
        nonlocal stopped
        calls.append(command)
        arguments = command[1:]
        if arguments[:2] == ["image", "inspect"]:
            return completed(command, stdout=IMAGE_ID + "\n")
        if arguments[0] == "run":
            return completed(command, stdout=CONTAINER_ID + "\n")
        if arguments[:2] == ["container", "inspect"]:
            if stopped:
                return completed(command, returncode=1, stderr="No such container")
            session = session_holder[0]
            identity = f"{CONTAINER_ID}|/{session.name}|{session.session_id}\n"
            return completed(command, stdout=identity)
        if arguments[:2] == ["container", "ls"]:
            return completed(command, stdout="" if stopped else CONTAINER_ID + "\n")
        if arguments[:2] == ["container", "exec"]:
            return completed(command)
        if arguments[:2] == ["container", "stop"]:
            stopped = True
            return completed(command, stdout=CONTAINER_ID + "\n")
        raise AssertionError(arguments)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(container_module, "ping", lambda *_args, **_kwargs: {"status": "ready"})
    session = ContainerSession(
        image="dsle-runtime:0.1.0",
        game_dir=game,
        output_dir=state,
        name="dsle-owned-test",
        docker_binary="docker",
    )
    session_holder.append(session)
    session.socket_path.touch()
    session.start()
    persistent_marker = session.output_dir / "results" / "metrics.jsonl"
    persistent_marker.write_text("episode data\n", encoding="utf-8")

    assert stat.S_IMODE(session.token_path.stat().st_mode) == 0o600
    run_call = next(command for command in calls if command[1] == "run")
    arguments = run_call[1:]
    assert "--pull=never" in arguments
    assert "--rm" in arguments
    assert arguments[arguments.index("--gpus") :][:2] == ["--gpus", "all"]
    assert arguments[arguments.index("--runtime") :][:2] == ["--runtime", "nvidia"]
    assert arguments[arguments.index("--cap-add") :][:2] == ["--cap-add", "SYS_PTRACE"]
    assert "seccomp=unconfined" in arguments
    assert "apparmor=unconfined" in arguments
    assert IMAGE_ID in arguments
    assert "dsle-runtime:0.1.0" not in arguments

    mounts = [arguments[index + 1] for index, value in enumerate(arguments) if value == "--mount"]
    assert mounts == [
        f"type=bind,source={game},target=/opt/dsle/game,readonly",
        f"type=bind,source={state.resolve()},target=/var/lib/dsle",
        f"type=bind,source={session._transport_dir},target=/var/lib/dsle/rpc",
    ]
    assert "readonly" not in mounts[1]
    assert "readonly" not in mounts[2]
    assert not any(command[1:2] in (["build"], ["pull"]) for command in calls)
    health_call = next(command for command in calls if command[1:3] == ["container", "exec"])
    assert health_call[-5:] == [
        CONTAINER_ID,
        "dsle-doctor",
        "--container",
        "--require-game",
        "--quiet",
    ]

    # The fake readiness marker is a regular file, unlike the real server's
    # Unix socket; remove only this test-created marker before lifecycle cleanup.
    session.socket_path.unlink()
    session.close()
    stop_call = next(command for command in calls if command[1:3] == ["container", "stop"])
    assert stop_call[-1] == CONTAINER_ID
    assert not session.token_path.exists()
    assert persistent_marker.read_text(encoding="utf-8") == "episode data\n"
    assert not session._transport_dir.exists()


def test_default_output_directory_is_unique_named_and_persistent_root(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(container_module, "user_data_path", lambda _name: tmp_path)
    first = container_module.default_output_directory("Asylum Demon")
    second = container_module.default_output_directory("Asylum Demon")
    assert first.parent == tmp_path / "runs"
    assert "asylum-demon" in first.name
    assert first != second


def test_managed_game_directory_defaults_to_existing_cwd_game(tmp_path: Path, monkeypatch) -> None:
    local_game = tmp_path / "game"
    local_game.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DSLE_GAME_DIR", raising=False)

    assert container_module._select_game_directory(None) == local_game

    canonical_game = tmp_path / "Dark.Souls.Remastered.v1.04"
    canonical_game.mkdir()
    assert container_module._select_game_directory(None) == canonical_game
    assert container_module._select_game_directory("/explicit/game") == "/explicit/game"

    monkeypatch.setenv("DSLE_GAME_DIR", "/configured/game")
    assert container_module._select_game_directory(None) == "/configured/game"


def test_missing_cwd_game_preserves_required_game_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DSLE_GAME_DIR", raising=False)

    with pytest.raises(ConfigurationError, match="game_dir is required"):
        container_module.make_managed("asylum_demon")


def test_long_default_output_path_does_not_contain_rpc_socket(tmp_path: Path, monkeypatch) -> None:
    long_root = tmp_path / ("deep-output-directory-" * 4)
    monkeypatch.setattr(container_module, "user_data_path", lambda _name: long_root)
    output = container_module.default_output_directory("crossbreed_priscilla")
    assert len(os.fsencode(output / "run" / ("rpc-" + "a" * 16 + ".sock"))) >= 104

    session = ContainerSession(
        image="dsle-runtime:0.1.0",
        game_dir=game_directory(tmp_path),
        output_dir=output,
        docker_binary="docker",
    )
    transport_dir = session._transport_dir
    assert session.output_dir == output.resolve()
    assert len(os.fsencode(session.socket_path)) < 104
    assert transport_dir.parent == Path(tempfile.gettempdir()).resolve()
    session.close()
    assert not transport_dir.exists()


def test_readiness_cannot_run_before_a_token_is_created(tmp_path: Path) -> None:
    session = ContainerSession(
        image="dsle-runtime:0.1.0",
        game_dir=game_directory(tmp_path),
        output_dir=tmp_path / "output",
        docker_binary="docker",
    )
    try:
        with pytest.raises(RuntimeError, match="call start"):
            session._wait_until_ready()
    finally:
        session.close()


def test_output_directory_requires_marker_before_reusing_existing_data(tmp_path: Path) -> None:
    output = tmp_path / "existing-output"
    output.mkdir()
    important = output / "important.txt"
    important.write_text("keep me", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="non-empty uninitialized"):
        ContainerSession(
            image="dsle-runtime:0.1.0",
            game_dir=game_directory(tmp_path),
            output_dir=output,
            docker_binary="docker",
        )
    assert important.read_text(encoding="utf-8") == "keep me"


def test_output_directory_rejects_reserved_symlink(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / ".dsle-output").write_text("dsle-output\n", encoding="ascii")
    victim = tmp_path / "victim"
    victim.mkdir()
    (output / "logs").symlink_to(victim, target_is_directory=True)
    with pytest.raises(ConfigurationError, match="Reserved DSLE output path"):
        ContainerSession(
            image="dsle-runtime:0.1.0",
            game_dir=game_directory(tmp_path),
            output_dir=output,
            docker_binary="docker",
        )
    assert list(victim.iterdir()) == []


def test_run_timeout_recovers_cidfile_and_removes_owned_container(
    tmp_path: Path, monkeypatch
) -> None:
    game = game_directory(tmp_path)
    calls: list[list[str]] = []
    session_holder: list[ContainerSession] = []
    stopped = False
    run_timeout = None

    def fake_run(command, **kwargs):
        nonlocal run_timeout, stopped
        calls.append(command)
        arguments = command[1:]
        if arguments[:2] == ["image", "inspect"]:
            return completed(command, stdout=IMAGE_ID + "\n")
        if arguments[0] == "run":
            run_timeout = kwargs["timeout"]
            cidfile = Path(arguments[arguments.index("--cidfile") + 1])
            cidfile.write_text(CONTAINER_ID + "\n", encoding="ascii")
            raise subprocess.TimeoutExpired(command, timeout=run_timeout)
        if arguments[:2] == ["container", "inspect"]:
            session = session_holder[0]
            return completed(
                command,
                stdout=f"{CONTAINER_ID}|/{session.name}|{session.session_id}\n",
            )
        if arguments[:2] == ["container", "stop"]:
            stopped = True
            return completed(command, stdout=CONTAINER_ID + "\n")
        if arguments[:2] == ["container", "ls"]:
            return completed(command, stdout="" if stopped else CONTAINER_ID + "\n")
        raise AssertionError(arguments)

    monkeypatch.setattr(subprocess, "run", fake_run)
    session = ContainerSession(
        image="dsle-runtime:0.1.0",
        game_dir=game,
        output_dir=tmp_path / "output",
        docker_binary="docker",
        startup_timeout=73,
    )
    session_holder.append(session)
    transport_dir = session._transport_dir
    with pytest.raises(RuntimeUnavailableError, match="failed to execute"):
        session.start()
    assert stopped
    assert run_timeout == 73
    assert not transport_dir.exists()
    assert session not in container_module._ACTIVE_SESSIONS


def test_malformed_pending_identity_retains_container_uncertainty(
    tmp_path: Path, monkeypatch
) -> None:
    session = ContainerSession(
        image="dsle-runtime:0.1.0",
        game_dir=game_directory(tmp_path),
        output_dir=tmp_path / "output",
        docker_binary="docker",
    )
    session._container_may_exist = True
    monotonic_values = iter((0.0, 3.0))
    monkeypatch.setattr(container_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        session,
        "_run",
        lambda *_args, **_kwargs: completed([], stdout="partial-identity\n"),
    )

    with pytest.raises(RuntimeUnavailableError, match="malformed pending-container identity"):
        session._recover_container_id()

    assert session._container_may_exist
    session._container_may_exist = False
    session.close()


def test_vnc_mode_uses_collision_free_loopback_publication(tmp_path: Path) -> None:
    game = game_directory(tmp_path)
    session = ContainerSession(
        image="dsle-runtime:0.1.0",
        game_dir=game,
        output_dir=tmp_path / "state",
        docker_binary="docker",
        vnc_ports=(5901, 5907),
    )
    arguments = session._run_arguments(IMAGE_ID)
    published = [
        arguments[index + 1] for index, value in enumerate(arguments) if value == "--publish"
    ]
    assert published == [
        "127.0.0.1::5901/tcp",
        "127.0.0.1::5907/tcp",
    ]


def test_vnc_mode_can_reserve_stable_loopback_host_ports(tmp_path: Path) -> None:
    session = ContainerSession(
        image="dsle-runtime:0.1.0",
        game_dir=game_directory(tmp_path),
        output_dir=tmp_path / "state",
        docker_binary="docker",
        vnc_ports=(5901, 5907),
        vnc_host_ports={5901: 6901, 5907: 6907},
    )
    arguments = session._run_arguments(IMAGE_ID)
    published = [
        arguments[index + 1] for index, value in enumerate(arguments) if value == "--publish"
    ]
    assert published == [
        "127.0.0.1:6901:5901/tcp",
        "127.0.0.1:6907:5907/tcp",
    ]


@pytest.mark.parametrize("value", [(-1,), (5931,), (True,), (5901, 5901), "5901"])
def test_vnc_ports_are_bounded_and_unique(tmp_path: Path, value: object) -> None:
    with pytest.raises(ValueError, match="vnc_ports"):
        ContainerSession(
            image="dsle-runtime:0.1.0",
            game_dir=game_directory(tmp_path),
            output_dir=tmp_path / "state",
            docker_binary="docker",
            vnc_ports=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "value",
    [
        {5903: 6903},
        {5901: 0},
        {5901: 65536},
        {5901: True},
        {5901: 6901, 5902: 6901},
        [(5901, 6901)],
    ],
)
def test_vnc_host_ports_must_match_selected_ports(tmp_path: Path, value: object) -> None:
    with pytest.raises(ValueError, match="vnc_host_ports"):
        ContainerSession(
            image="dsle-runtime:0.1.0",
            game_dir=game_directory(tmp_path),
            output_dir=tmp_path / "state",
            docker_binary="docker",
            vnc_ports=(5901, 5902),
            vnc_host_ports=value,  # type: ignore[arg-type]
        )


def test_managed_gui_mode_requires_manual_container_workflow() -> None:
    with pytest.raises(ValueError, match="cannot expose a host GUI"):
        container_module.make_managed(
            "asylum_demon",
            game_dir="/not-inspected-before-mode-validation",
            instance_mode="gui",
        )


def test_managed_mode_cannot_attach_inside_a_fresh_container() -> None:
    with pytest.raises(ValueError, match="runtime='external'"):
        container_module.make_managed(
            "asylum_demon",
            game_dir="/not-inspected-before-lifecycle-validation",
            start_instance=False,
        )


def test_name_collision_failure_never_stops_an_unowned_container(
    tmp_path: Path, monkeypatch
) -> None:
    game = game_directory(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        arguments = command[1:]
        if arguments[:2] == ["image", "inspect"]:
            return completed(command, stdout=IMAGE_ID + "\n")
        if arguments[0] == "run":
            return completed(command, returncode=125, stderr="name is already in use")
        if arguments[:2] == ["container", "inspect"]:
            return completed(
                command,
                stdout=f"{'3' * 64}|/already-owned-by-someone-else|foreign-session\n",
            )
        raise AssertionError(arguments)

    monkeypatch.setattr(subprocess, "run", fake_run)
    session = ContainerSession(
        image="dsle-runtime:0.1.0",
        game_dir=game,
        output_dir=tmp_path / "state",
        name="already-owned-by-someone-else",
        docker_binary="docker",
    )
    with pytest.raises(RuntimeUnavailableError, match="already in use"):
        session.start()
    assert not any(command[1:3] == ["container", "stop"] for command in calls)


def test_identity_mismatch_refuses_to_stop_container(tmp_path: Path, monkeypatch) -> None:
    game = game_directory(tmp_path)
    calls: list[list[str]] = []
    inspect_count = 0
    session_holder: list[ContainerSession] = []

    def fake_run(command, **_kwargs):
        nonlocal inspect_count
        calls.append(command)
        arguments = command[1:]
        if arguments[:2] == ["image", "inspect"]:
            return completed(command, stdout=IMAGE_ID + "\n")
        if arguments[0] == "run":
            return completed(command, stdout=CONTAINER_ID + "\n")
        if arguments[:2] == ["container", "inspect"]:
            inspect_count += 1
            session = session_holder[0]
            label = session.session_id if inspect_count == 1 else "different-owner"
            return completed(command, stdout=f"{CONTAINER_ID}|/{session.name}|{label}\n")
        if arguments[:2] == ["container", "stop"]:
            raise AssertionError("identity-mismatched container must not be stopped")
        raise AssertionError(arguments)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(ContainerSession, "_wait_until_ready", lambda self: None)
    session = ContainerSession(
        image="dsle-runtime:0.1.0",
        game_dir=game,
        output_dir=tmp_path / "state",
        docker_binary="docker",
    )
    session_holder.append(session)
    session.start()
    with pytest.raises(RuntimeUnavailableError, match="mismatched identity"):
        session.close()
    assert not any(command[1:3] == ["container", "stop"] for command in calls)
    assert session.token_path.exists(), (
        "credentials must remain while an unowned process may use them"
    )


def test_stop_uses_verified_remove_fallback_without_touching_image(monkeypatch) -> None:
    session = object.__new__(ContainerSession)
    session._container_id = CONTAINER_ID
    session._container_may_exist = False
    session.name = "dsle-removal-fallback"
    session.session_id = "a" * 32
    session.stop_timeout = 0.001
    removed = False
    calls: list[list[str]] = []

    def fake_run(arguments, **_kwargs):
        nonlocal removed
        calls.append(arguments)
        if arguments[:2] == ["container", "inspect"]:
            return completed(
                arguments,
                stdout=f"{CONTAINER_ID}|/{session.name}|{session.session_id}\n",
            )
        if arguments[:2] == ["container", "stop"]:
            return completed(arguments, stdout=CONTAINER_ID + "\n")
        if arguments[:2] == ["container", "ls"]:
            return completed(arguments, stdout="" if removed else CONTAINER_ID + "\n")
        if arguments[:2] == ["container", "rm"]:
            removed = True
            return completed(arguments, stdout=CONTAINER_ID + "\n")
        raise AssertionError(arguments)

    monkeypatch.setattr(session, "_run", fake_run)
    session._stop_owned_container()
    assert removed and session.container_id is None
    assert not any(arguments[:2] == ["image", "rm"] for arguments in calls)


def test_stop_tolerates_container_disappearing_after_identity_check(monkeypatch) -> None:
    session = object.__new__(ContainerSession)
    session._container_id = CONTAINER_ID
    session._container_may_exist = False
    session.name = "dsle-disappeared-during-stop"
    session.session_id = "c" * 32
    session.stop_timeout = 1.0
    existence_checks: list[str] = []

    monkeypatch.setattr(
        session,
        "_inspect_identity",
        lambda: (CONTAINER_ID, session.name, session.session_id),
    )
    monkeypatch.setattr(
        session,
        "_run",
        lambda arguments, **_kwargs: completed(
            arguments,
            returncode=1,
            stderr="Error: No such container",
        ),
    )

    def absent(container_id: str) -> bool:
        existence_checks.append(container_id)
        return False

    monkeypatch.setattr(session, "_container_exists_by_id", absent)

    session._stop_owned_container()

    assert existence_checks == [CONTAINER_ID]
    assert session.container_id is None
    assert session._container_may_exist is False


def test_daemon_inspect_error_never_claims_container_was_deleted(monkeypatch) -> None:
    session = object.__new__(ContainerSession)
    session._container_id = CONTAINER_ID
    session._container_may_exist = False
    session.name = "dsle-daemon-error"
    session.session_id = "b" * 32
    session.stop_timeout = 1.0

    def fake_run(arguments, **_kwargs):
        if arguments[:2] == ["container", "inspect"]:
            return completed(arguments, returncode=1, stderr="daemon temporarily unavailable")
        if arguments[:2] == ["container", "ls"]:
            return completed(arguments, returncode=1, stderr="daemon temporarily unavailable")
        raise AssertionError(arguments)

    monkeypatch.setattr(session, "_run", fake_run)
    with pytest.raises(RuntimeUnavailableError, match="could not verify"):
        session._stop_owned_container()
    assert session.container_id == CONTAINER_ID


def test_close_succeeds_when_owned_container_is_proven_absent(tmp_path: Path, monkeypatch) -> None:
    session = ContainerSession(
        image="dsle-runtime:0.1.0",
        game_dir=game_directory(tmp_path),
        output_dir=tmp_path / "state",
        docker_binary="docker",
    )
    session._container_id = CONTAINER_ID
    session._token = "a" * 64
    session._started_instances = True
    monkeypatch.setattr(session, "_inspect_identity", lambda: None)

    session.close()

    assert session.container_id is None
    assert session._closed
    assert session not in container_module._ACTIVE_SESSIONS
    assert not session._transport_dir.exists()


def test_state_mount_cannot_overlap_game_installation(tmp_path: Path) -> None:
    game = game_directory(tmp_path)
    with pytest.raises(ConfigurationError, match="inside the read-only game"):
        ContainerSession(
            image="dsle-runtime:0.1.0",
            game_dir=game,
            output_dir=game / "state",
            docker_binary="docker",
        )


def test_token_is_a_regular_file_not_a_symlink(tmp_path: Path, monkeypatch) -> None:
    game = game_directory(tmp_path)
    session = ContainerSession(
        image="dsle-runtime:0.1.0",
        game_dir=game,
        output_dir=tmp_path / "state",
        docker_binary="docker",
    )
    session.token_path.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "attacker-token"
    target.write_text("unchanged", encoding="utf-8")
    session.token_path.symlink_to(target)
    monkeypatch.setattr(session, "_inspect_local_image", lambda: IMAGE_ID)
    with pytest.raises(RuntimeUnavailableError, match="Could not create private RPC token"):
        session.start()
    assert target.read_text(encoding="utf-8") == "unchanged"
    assert not session.token_path.exists()
    assert not session._transport_dir.exists()
    assert session not in container_module._ACTIVE_SESSIONS


def test_managed_environment_close_owns_its_session() -> None:
    class Environment(gym.Env[np.ndarray, int]):
        metadata: ClassVar[dict[str, Any]] = {}

        def __init__(self):
            self.action_space = gym.spaces.Discrete(1)
            self.observation_space = gym.spaces.Box(0, 1, (1,), np.uint8)

    class Session:
        def __init__(self):
            self.close_calls = 0
            self.output_dir = Path("/tmp/dsle-test-output")

        def close(self):
            self.close_calls += 1

    session = Session()
    environment = ManagedDarkSoulsEnv(Environment(), session)  # type: ignore[arg-type]
    environment.close()
    environment.close()
    assert session.close_calls == 1


def test_managed_environment_delegates_passive_observe_and_menu_return() -> None:
    class Environment(gym.Env[np.ndarray, int]):
        metadata: ClassVar[dict[str, Any]] = {}

        def __init__(self):
            self.action_space = gym.spaces.Discrete(1)
            self.observation_space = gym.spaces.Box(0, 1, (1,), np.uint8)
            self.menu_timeouts: list[float] = []

        def observe(self):
            return np.asarray([1], dtype=np.uint8), {"player_dead": False}

        def return_to_menu(self, *, timeout_s):
            self.menu_timeouts.append(timeout_s)

    class Session:
        def __init__(self):
            self.output_dir = Path("/tmp/dsle-test-output")
            self.vnc_ports = {5901: 49152}

        def close(self):
            pass

    remote = Environment()
    environment = ManagedDarkSoulsEnv(remote, Session())  # type: ignore[arg-type]

    observation, info = environment.observe()
    environment.return_to_menu(timeout_s=8.0)

    np.testing.assert_array_equal(observation, np.asarray([1], dtype=np.uint8))
    assert info == {"player_dead": False}
    assert remote.menu_timeouts == [8.0]


def test_managed_environment_close_can_retry_a_failed_session_close() -> None:
    class Environment(gym.Env[np.ndarray, int]):
        metadata: ClassVar[dict[str, Any]] = {}

        def __init__(self):
            self.action_space = gym.spaces.Discrete(1)
            self.observation_space = gym.spaces.Box(0, 1, (1,), np.uint8)

    class Session:
        def __init__(self):
            self.close_calls = 0
            self.output_dir = Path("/tmp/dsle-test-output")

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeUnavailableError("temporary identity mismatch")

    session = Session()
    environment = ManagedDarkSoulsEnv(Environment(), session)  # type: ignore[arg-type]
    with pytest.raises(RuntimeUnavailableError, match="identity mismatch"):
        environment.close()
    environment.close()
    environment.close()
    assert session.close_calls == 2


def test_shared_session_starts_a_numbered_instance_pool(monkeypatch) -> None:
    session = object.__new__(ContainerSession)
    session._owner_pid = os.getpid()
    session._container_id = CONTAINER_ID
    session._token = "a" * 64
    session.socket_path = Path("/tmp/dsle-test.sock")
    session.request_timeout = 180.0
    monkeypatch.setattr(session, "_require_owned_container", lambda: None)
    calls: list[tuple[str, dict[str, object]]] = []

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def request(self, operation, payload):
            calls.append((operation, payload))
            return (
                {
                    "statuses": {
                        f"dsr-{index}": {
                            "name": f"dsr-{index}",
                            "status": "ready",
                            "running": True,
                        }
                        for index in range(1, 4)
                    }
                },
                None,
            )

        def close(self):
            pass

    monkeypatch.setattr(container_module, "RPCClient", Client)
    statuses = session.start_instances(3, mode="headless-vnc")
    assert tuple(statuses) == ("dsr-1", "dsr-2", "dsr-3")
    assert calls == [("start_instances", {"count": 3, "mode": "headless-vnc"})]
    with pytest.raises(ValueError, match=r"\[1, 30\]"):
        session.start_instances(31)


def test_forked_child_never_closes_the_parent_container_session(monkeypatch) -> None:
    parent = object.__new__(ContainerSession)
    parent._owner_pid = 1001
    parent_close_calls = []
    monkeypatch.setattr(parent, "close", lambda: parent_close_calls.append(True))

    child_owned = object.__new__(ContainerSession)
    child_owned._owner_pid = 2002
    child_close_calls = []
    monkeypatch.setattr(child_owned, "close", lambda: child_close_calls.append(True))

    active = {parent, child_owned}
    monkeypatch.setattr(container_module, "_ACTIVE_SESSIONS", active)
    monkeypatch.setattr(container_module.os, "getpid", lambda: 2002)

    container_module._close_active_sessions()

    assert parent_close_calls == []
    assert child_close_calls == [True]
    assert parent not in active


def test_explicit_close_from_a_forked_child_is_a_safe_noop(monkeypatch) -> None:
    session = object.__new__(ContainerSession)
    session._owner_pid = 1001
    active = {session}
    monkeypatch.setattr(container_module, "_ACTIVE_SESSIONS", active)
    monkeypatch.setattr(container_module.os, "getpid", lambda: 2002)

    session.close()

    assert session not in active


def test_make_managed_can_return_owned_vector_and_persistent_output(
    tmp_path: Path, monkeypatch
) -> None:
    sessions = []

    class Environment(gym.Env[np.ndarray, int]):
        metadata: ClassVar[dict[str, Any]] = {}

        def __init__(self, instance: str):
            self.instance = instance
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Box(0, 1, (1,), np.uint8)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, dtype=np.uint8), {"instance": self.instance}

        def step(self, action):
            return np.zeros(1, dtype=np.uint8), float(action), False, False, {}

    class Session:
        def __init__(self, *, output_dir, **_kwargs):
            self.output_dir = Path(output_dir)
            self.container_id = CONTAINER_ID
            self.started_pool = None
            self.environments = []
            sessions.append(self)

        def start(self):
            return self

        def start_instances(self, count, *, mode):
            self.started_pool = (count, mode)

        def make(self, _boss, *, instance, **_kwargs):
            environment = Environment(instance)
            self.environments.append(environment)
            return environment

        def close(self):
            for environment in self.environments:
                environment.close()
            self.container_id = None

    monkeypatch.setattr(container_module, "ContainerSession", Session)
    output = tmp_path / "experiment-output"
    environment = container_module.make_managed(
        "asylum_demon",
        game_dir="/games/dsr",
        output_dir=output,
        num_instances=3,
    )
    assert environment.num_envs == 3
    assert environment.output_dir == output
    assert sessions[0].started_pool == (3, "headless")
    _observation, info = environment.reset(seed=1)
    assert tuple(info["instance"]) == ("dsr-1", "dsr-2", "dsr-3")
    environment.close()
    assert sessions[0].container_id is None


def test_make_managed_selects_unique_default_output_when_omitted(
    tmp_path: Path, monkeypatch
) -> None:
    expected_output = tmp_path / "generated-output"
    sessions = []

    class Session:
        def __init__(self, *, output_dir, **_kwargs):
            self.output_dir = Path(output_dir)
            self.container_id = CONTAINER_ID
            sessions.append(self)

        def start(self):
            return self

        def make(self, _boss, **_kwargs):
            return object()

        def close(self):
            self.container_id = None

    monkeypatch.setattr(container_module, "default_output_directory", lambda _boss: expected_output)
    monkeypatch.setattr(container_module, "ContainerSession", Session)
    monkeypatch.setattr(
        container_module,
        "ManagedDarkSoulsEnv",
        lambda environment, session: (environment, session),
    )

    _environment, session = container_module.make_managed(
        "asylum_demon",
        game_dir="/games/dsr",
    )

    assert session is sessions[0]
    assert session.output_dir == expected_output
