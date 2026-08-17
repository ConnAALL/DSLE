from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

import dsle
from dsle.container import ContainerSession
from dsle.runtime.version import verify_game_executable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = PROJECT_ROOT / "docker" / "compose.yaml"
RUN_DOCKER = os.environ.get("DSLE_RUN_DOCKER_TESTS") == "1"
RUN_DOCKER_LIVE = os.environ.get("DSLE_RUN_DOCKER_LIVE_TESTS") == "1"


def _run(
    command: list[str],
    *,
    environment: dict[str, str],
    timeout: float,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _compose(project: str, *arguments: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        project,
        "--file",
        str(COMPOSE_FILE),
        *arguments,
    ]


@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.skipif(not RUN_DOCKER, reason="set DSLE_RUN_DOCKER_TESTS=1")
def test_build_and_start_authenticated_nvidia_runtime(tmp_path: Path) -> None:
    """Build and validate the publishable image without launching a game instance."""

    game_value = os.environ.get("DSLE_GAME_DIR")
    assert game_value, "DSLE_GAME_DIR must point to a legally obtained game installation"
    game_dir = Path(game_value).expanduser().resolve(strict=True)
    executable = game_dir / "DarkSoulsRemastered.exe"
    game_build = verify_game_executable(executable)
    assert game_build.verified

    state_dir = tmp_path / "state"
    run_dir = state_dir / "run"
    run_dir.mkdir(parents=True)
    (state_dir / ".dsle-output").write_text("dsle-output\n", encoding="ascii")
    (state_dir / ".container-session.lock").touch(mode=0o600)
    token_file = run_dir / "dsle.token"
    token_file.write_text(secrets.token_urlsafe(48) + "\n", encoding="ascii")
    token_file.chmod(0o600)

    project = f"dsle-test-{os.getpid()}"
    container_name = f"{project}-runtime"
    image = os.environ.get("DSLE_DOCKER_TEST_IMAGE", "dsle-runtime:0.1.0-test")
    environment = os.environ.copy()
    environment.update(
        {
            "DSLE_CONTAINER_NAME": container_name,
            "DSLE_GAME_DIR": str(game_dir),
            "DSLE_IMAGE": image,
            "DSLE_IMAGE_TARGET": "runtime",
            "DSLE_STATE_DIR": str(state_dir),
        }
    )

    try:
        _run(
            _compose(project, "build", "runtime"),
            environment=environment,
            timeout=3600,
        )
        image_game_check = _run(
            [
                "docker",
                "run",
                "--rm",
                "--pull=never",
                "--network=none",
                "--entrypoint",
                "python3",
                image,
                "-c",
                (
                    "from pathlib import Path; "
                    "game=Path('/opt/dsle/game'); "
                    "assert not game.exists() or "
                    "(game.is_dir() and next(game.iterdir(), None) is None), "
                    "'final image contains files under /opt/dsle/game'; "
                    "print('game-installation-absent')"
                ),
            ],
            environment=environment,
            timeout=60,
        )
        assert image_game_check.stdout.strip() == "game-installation-absent"
        _run(
            _compose(project, "up", "--detach", "--no-build", "runtime"),
            environment=environment,
            timeout=120,
        )

        deadline = time.monotonic() + 180
        health = "starting"
        while time.monotonic() < deadline:
            inspected = _run(
                ["docker", "inspect", "--format", "{{.State.Health.Status}}", container_name],
                environment=environment,
                timeout=15,
            )
            health = inspected.stdout.strip()
            if health == "healthy":
                break
            if health == "unhealthy":
                break
            time.sleep(2)
        if health != "healthy":
            logs = _run(
                _compose(project, "logs", "--no-color", "runtime"),
                environment=environment,
                timeout=30,
                check=False,
            )
            pytest.fail(f"runtime health was {health!r}\n{logs.stdout}\n{logs.stderr}")

        details = json.loads(
            _run(
                ["docker", "inspect", container_name],
                environment=environment,
                timeout=15,
            ).stdout
        )[0]
        game_mount = next(
            mount for mount in details["Mounts"] if mount["Destination"] == "/opt/dsle/game"
        )
        assert game_mount["Type"] == "bind"
        assert Path(game_mount["Source"]).resolve() == game_dir
        assert game_mount["RW"] is False
        assert details["HostConfig"]["Runtime"] == "nvidia"
        capabilities = {
            capability.removeprefix("CAP_") for capability in details["HostConfig"]["CapAdd"]
        }
        assert "SYS_PTRACE" in capabilities
        security = set(details["HostConfig"]["SecurityOpt"])
        assert {"seccomp=unconfined", "apparmor=unconfined"} <= security
        labels = details["Config"]["Labels"]
        assert labels["org.opencontainers.image.version"] == "0.1.0"
        assert labels["io.dsle.image.flavor"] == "core"
        for runtime_directory in (
            state_dir / "pulse",
            state_dir / "xdg",
            state_dir / "xdg/container",
        ):
            metadata = runtime_directory.stat()
            assert metadata.st_uid == os.getuid()
            assert metadata.st_gid == os.getgid()

        _run(
            _compose(project, "exec", "--no-tty", "runtime", "dsle-health"),
            environment=environment,
            timeout=60,
        )
        _run(
            _compose(
                project,
                "exec",
                "--no-tty",
                "runtime",
                "dsle-instances",
                "configure",
                "--count",
                "1",
                "--resolution",
                "800x600",
                "--force",
            ),
            environment=environment,
            timeout=30,
        )
        listing = _run(
            _compose(project, "exec", "--no-tty", "runtime", "dsle-instances", "list"),
            environment=environment,
            timeout=30,
        )
        assert "dsr-1" in listing.stdout
        assert "5901" in listing.stdout

        mounted_fingerprint = _run(
            _compose(
                project,
                "exec",
                "--no-tty",
                "runtime",
                "sha256sum",
                "/opt/dsle/game/DarkSoulsRemastered.exe",
            ),
            environment=environment,
            timeout=120,
        ).stdout.split()[0]
        assert mounted_fingerprint == game_build.sha256

        artifact_counts = json.loads(
            _run(
                _compose(
                    project,
                    "exec",
                    "--no-tty",
                    "runtime",
                    "python3",
                    "-c",
                    (
                        "import json,pathlib; "
                        "root=pathlib.Path('/opt/dsle'); "
                        "s=list((root/'assets/scenarios').glob('*.sl2')); "
                        "t=list((root/'assets/templates').rglob('*.png')); "
                        "e=sorted(p.name for p in (root/'examples').glob('*.py')); "
                        "print(json.dumps({'saves':len(s),'save_sizes':sorted({p.stat().st_size for p in s}),'templates':len(t),'examples':e}))"
                    ),
                ),
                environment=environment,
                timeout=30,
            ).stdout
        )
        assert artifact_counts == {
            "saves": 43,
            "save_sizes": [4_326_608],
            "templates": 47,
            "examples": [
                "common.py",
                "dqn.py",
                "expert_agent.py",
                "ppo.py",
                "random_agent.py",
                "scope.py",
                "scope_parallel.py",
            ],
        }
    finally:
        _run(
            _compose(project, "down", "--timeout", "45"),
            environment=environment,
            timeout=90,
            check=False,
        )
    for runtime_directory in (
        state_dir / "pulse",
        state_dir / "xdg",
        state_dir / "xdg/container",
    ):
        metadata = runtime_directory.stat()
        assert metadata.st_uid == os.getuid()
        assert metadata.st_gid == os.getgid()


@pytest.mark.docker
@pytest.mark.game
@pytest.mark.boss
@pytest.mark.slow
@pytest.mark.skipif(
    not RUN_DOCKER_LIVE,
    reason="set DSLE_RUN_DOCKER_LIVE_TESTS=1 and DSLE_DOCKER_TEST_IMAGE",
)
def test_managed_prebuilt_container_resets_and_steps_asylum_demon(tmp_path: Path) -> None:
    """Release gate: manage a prebuilt image and execute one real boss transition."""

    image = os.environ.get("DSLE_DOCKER_TEST_IMAGE")
    game_value = os.environ.get("DSLE_GAME_DIR")
    assert image, "DSLE_DOCKER_TEST_IMAGE must name an existing prebuilt image"
    assert game_value, "DSLE_GAME_DIR must point to a legally obtained game installation"
    game_dir = Path(game_value).expanduser().resolve(strict=True)
    executable = game_dir / "DarkSoulsRemastered.exe"
    game_build = verify_game_executable(executable)
    assert game_build.verified
    host_fingerprint = game_build.sha256
    prebuilt_image_id = _run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        environment=os.environ.copy(),
        timeout=15,
    ).stdout.strip()

    managed_container_id: str | None = None
    output_dir = tmp_path / "s"
    with ContainerSession(
        image=image,
        game_dir=game_dir,
        # A short child keeps Docker's temporary test metadata readable; RPC
        # transport itself is decoupled from this persistent output path.
        output_dir=output_dir,
        startup_timeout=180,
        request_timeout=240,
    ) as session:
        assert session.container_id is not None
        managed_container_id = session.container_id
        inspected = json.loads(
            _run(
                ["docker", "inspect", session.container_id],
                environment=os.environ.copy(),
                timeout=15,
            ).stdout
        )[0]
        game_mount = next(
            mount for mount in inspected["Mounts"] if mount["Destination"] == "/opt/dsle/game"
        )
        assert game_mount["Type"] == "bind"
        assert Path(game_mount["Source"]).resolve() == game_dir
        assert game_mount["RW"] is False
        results_dir = output_dir / "results"
        assert results_dir.stat().st_uid == os.getuid()
        assert results_dir.stat().st_gid == os.getgid()
        assert results_dir.stat().st_mode & 0o070 == 0o070

        mounted_fingerprint = _run(
            [
                "docker",
                "container",
                "exec",
                session.container_id,
                "sha256sum",
                "/opt/dsle/game/DarkSoulsRemastered.exe",
            ],
            environment=os.environ.copy(),
            timeout=120,
        ).stdout.split()[0]
        assert mounted_fingerprint == host_fingerprint

        _run(
            [
                "docker",
                "container",
                "exec",
                session.container_id,
                "python3",
                "-c",
                (
                    "from pathlib import Path; "
                    "Path('/var/lib/dsle/results/container-proof.txt').write_text('container data\\n')"
                ),
            ],
            environment=os.environ.copy(),
            timeout=30,
        )

        status = session.start_instance("dsr-1", mode="headless")
        assert status["status"] == "ready"
        assert status["running"] is True

        environment = session.make(
            "asylum_demon",
            instance="dsr-1",
            start_instance=False,
            max_steps=1,
        )
        try:
            observation, info = environment.reset()
            assert environment.observation_space.contains(observation)
            assert info["player_state_valid"] and info["boss_state_valid"]

            observation, reward, terminated, truncated, info = environment.step(0)
            assert environment.observation_space.contains(observation)
            assert isinstance(reward, float)
            assert not terminated
            assert truncated
            assert info["player_state_valid"] and info["boss_state_valid"]
        finally:
            environment.close()
    assert managed_container_id is not None
    removed = _run(
        ["docker", "container", "inspect", managed_container_id],
        environment=os.environ.copy(),
        timeout=15,
        check=False,
    )
    assert removed.returncode != 0, "closing the environment must remove its owned container"
    retained_image_id = _run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        environment=os.environ.copy(),
        timeout=15,
    ).stdout.strip()
    assert retained_image_id == prebuilt_image_id, "closing must never remove the prebuilt image"
    assert output_dir.is_dir(), "closing the environment must retain its host output directory"
    proof = output_dir / "results" / "container-proof.txt"
    assert proof.read_text(encoding="utf-8") == "container data\n"
    assert proof.stat().st_uid == os.getuid() and proof.stat().st_gid == os.getgid()
    with proof.open("a", encoding="utf-8") as stream:
        stream.write("host data\n")
    assert proof.read_text(encoding="utf-8") == "container data\nhost data\n"


@pytest.mark.docker
@pytest.mark.game
@pytest.mark.boss
@pytest.mark.slow
@pytest.mark.skipif(
    not RUN_DOCKER_LIVE,
    reason="set DSLE_RUN_DOCKER_LIVE_TESTS=1 and DSLE_DOCKER_TEST_IMAGE",
)
def test_public_api_runs_two_instances_and_removes_its_container(tmp_path: Path) -> None:
    """Release gate for the public N-instance lifecycle and persistent output."""

    image = os.environ.get("DSLE_DOCKER_TEST_IMAGE")
    game_value = os.environ.get("DSLE_GAME_DIR")
    assert image, "DSLE_DOCKER_TEST_IMAGE must name an existing prebuilt image"
    assert game_value, "DSLE_GAME_DIR must point to a legally obtained game installation"
    game_dir = Path(game_value).expanduser().resolve(strict=True)
    output_dir = tmp_path / "v"
    environment = dsle.make(
        "asylum_demon",
        runtime="managed",
        game_dir=game_dir,
        image=image,
        output_dir=output_dir,
        num_instances=2,
        max_steps=1,
    )
    container_id = environment._container_session.container_id
    assert container_id is not None
    try:
        observations, infos = environment.reset(seed=11)
        assert observations.shape[0] == 2
        assert np.all(infos["player_state_valid"])
        assert np.all(infos["boss_state_valid"])
        for observation in observations:
            assert environment.single_observation_space.contains(observation)

        observations, rewards, terminated, truncated, infos = environment.step(
            np.zeros(2, dtype=np.int64)
        )
        assert observations.shape[0] == 2
        assert np.all(np.isfinite(rewards))
        assert not np.any(terminated)
        assert np.all(truncated)
        assert np.all(infos["player_state_valid"])
        assert np.all(infos["boss_state_valid"])
    finally:
        environment.close()

    removed = _run(
        ["docker", "container", "inspect", container_id],
        environment=os.environ.copy(),
        timeout=15,
        check=False,
    )
    assert removed.returncode != 0
    assert output_dir.is_dir()
