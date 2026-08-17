from __future__ import annotations

import base64
import os
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_GAME_DIRECTORIES = ("chr", "event", "map", "mtd", "param", "script")


def _make_game_layout(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "DarkSoulsRemastered.exe").write_bytes(b"MZ")
    for name in REQUIRED_GAME_DIRECTORIES:
        (root / name).mkdir()
    return root


def _write_fake_docker(path: Path) -> Path:
    path.mkdir()
    executable = path / "docker"
    executable.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "${DSLE_FAKE_DOCKER_LOG}"
arguments="$*"
if [[ "${arguments}" == "info" || "${arguments}" == "compose version" ]]; then
  exit 0
fi
if [[ "${arguments}" == *"run --rm --pull=never --entrypoint python3"* ]]; then
  if [[ "${DSLE_FAKE_FAIL_STAGE}" == "dependencies" ]]; then
    printf '%s\n' "ModuleNotFoundError: No module named rich" >&2
    exit 74
  fi
  exit 0
fi
if [[ "${arguments}" == *" up --detach runtime" ]]; then
  [[ "${DSLE_FAKE_FAIL_STAGE}" != "up" ]] || exit 71
  exit 0
fi
if [[ "${arguments}" == *" exec --no-tty runtime dsle-health" ]]; then
  exit 0
fi
if [[ "${arguments}" == *" exec --no-tty runtime python3 -c "* && "${arguments}" == *"dsle.__file__"* ]]; then
  printf '%s\n' /workspace/dsle/src/dsle/__init__.py
  exit 0
fi
if [[ "${arguments}" == *" dsle-instances configure "* ]]; then
  [[ "${DSLE_FAKE_FAIL_STAGE}" != "configure" ]] || exit 72
  exit 0
fi
if [[ "${arguments}" == *" dsle-instances start "* ]]; then
  exit 0
fi
if [[ "${arguments}" == *" down --timeout 45" ]]; then
  exit 0
fi
exit 0
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _write_fake_tigervnc(path: Path) -> Path:
    executable = path / "xtigervncviewer"
    executable.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" > "${DSLE_FAKE_VNC_LOG}"
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def test_container_contract_keeps_game_external_and_results_host_owned() -> None:
    dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    entrypoint = (PROJECT_ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    ignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "COPY game" not in dockerfile
    assert "DarkSoulsRemastered.exe" not in dockerfile
    assert "sha256sum --check --strict --quiet assets/checksums.sha256" in dockerfile
    assert "pip==26.2.1" in dockerfile
    assert "pip==25.1.1" not in dockerfile
    assert "COPY docker/container_requirements.txt " in dockerfile
    assert "--requirement /opt/dsle/container_requirements.txt" in dockerfile
    assert "COPY LICENSE /tmp/dsle-payload/opt/dsle/LICENSE" in dockerfile
    assert 'org.opencontainers.image.licenses="GPL-3.0-only"' in dockerfile
    ignore_rules = set(ignore.splitlines())
    assert {
        "**",
        "!LICENSE",
        "!src/**",
        "!assets/checksums.sha256",
        "!assets/scenarios/*.sl2",
        "!assets/templates",
        "!docker/container_requirements.txt",
        "!docker/entrypoint.sh",
        "!docker/xorg/nvidia.conf.template",
    } <= ignore_rules
    assert "**/DarkSoulsRemastered.exe" in ignore_rules
    assert not (PROJECT_ROOT / "docker" / "Dockerfile.dockerignore").exists()
    assert 'chown "${HOST_UID}:${HOST_GID}"' in entrypoint
    assert '"${reserved_directories[@]}"' in entrypoint
    assert 'chmod 2770 "${durable_directories[@]}"' in entrypoint
    assert 'PULSE_RUNTIME_DIR="/tmp/dsle-pulse-runtime"' in entrypoint
    assert 'XDG_RUNTIME_DIR="${PULSE_RUNTIME_DIR}"' in entrypoint
    assert "pulseaudio -n" in entrypoint
    assert "flock --nonblock 9" in entrypoint


def test_research_dependencies_are_cached_before_application_payload() -> None:
    dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")

    stable_base = dockerfile.index("FROM ubuntu:${UBUNTU_VERSION} AS runtime-base")
    research_dependencies = dockerfile.index("FROM runtime-base AS research-dependencies")
    torch_install = dockerfile.index("torch==2.7.1", research_dependencies)
    application_payload = dockerfile.index("FROM runtime-base AS application-payload")
    source_copy = dockerfile.index("COPY src /tmp/dsle-payload/opt/dsle/src", application_payload)
    runtime_target = dockerfile.index("FROM runtime-base AS runtime")
    research_target = dockerfile.index("FROM research-dependencies AS research")

    assert stable_base < research_dependencies < torch_install < application_payload
    assert application_payload < source_copy < runtime_target < research_target
    assert "FROM runtime AS research" not in dockerfile
    assert dockerfile.count("COPY --from=application-payload /tmp/dsle-payload/ /") == 2
    wine_template_copies = [
        line
        for line in dockerfile.splitlines()
        if line.startswith("COPY --from=") and "wine-template" in line
    ]
    assert wine_template_copies
    assert all(line.startswith("COPY --from=dxvk ") for line in wine_template_copies)
    assert not any(
        stage in line
        for line in wine_template_copies
        for stage in ("runtime-base", "research-dependencies", "application-payload")
    )


def test_core_and_research_targets_share_one_verified_application_payload() -> None:
    dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")

    for instruction in (
        "COPY LICENSE /tmp/dsle-payload/opt/dsle/LICENSE",
        "COPY src /tmp/dsle-payload/opt/dsle/src",
        "COPY assets /tmp/dsle-payload/opt/dsle/assets",
        "COPY examples /tmp/dsle-payload/opt/dsle/examples",
        "COPY docker/xorg/nvidia.conf.template "
        "/tmp/dsle-payload/opt/dsle/xorg/nvidia.conf.template",
        "COPY --chmod=0755 docker/bin/dsle-doctor /tmp/dsle-payload/usr/local/bin/dsle-doctor",
        "COPY --chmod=0755 docker/bin/dsle-health /tmp/dsle-payload/usr/local/bin/dsle-health",
        "COPY --chmod=0755 docker/bin/dsle-instances "
        "/tmp/dsle-payload/usr/local/bin/dsle-instances",
        "COPY --chmod=0755 docker/bin/dsle-server /tmp/dsle-payload/usr/local/bin/dsle-server",
        "COPY --chmod=0755 docker/entrypoint.sh /tmp/dsle-payload/usr/local/bin/dsle-entrypoint",
    ):
        assert dockerfile.count(instruction) == 1

    payload_stage = dockerfile.index("FROM runtime-base AS application-payload")
    assert dockerfile.index(
        "sha256sum --check --strict --quiet assets/checksums.sha256", payload_stage
    )
    payload_copy = "COPY --from=application-payload /tmp/dsle-payload/ /"
    assert dockerfile.count(payload_copy) == 2
    for metadata in (
        'org.opencontainers.image.version="0.1.0"',
        'org.opencontainers.image.licenses="GPL-3.0-only"',
        "WORKDIR /opt/dsle",
        'VOLUME ["/var/lib/dsle"]',
        'ENTRYPOINT ["/usr/local/bin/dsle-entrypoint"]',
        'CMD ["dsle-server"]',
    ):
        assert dockerfile.count(metadata) == 1
        assert dockerfile.index(metadata) < dockerfile.index(
            "FROM runtime-base AS research-dependencies"
        )
    assert dockerfile.count('io.dsle.image.flavor="core"') == 1
    assert dockerfile.count('io.dsle.image.flavor="research"') == 1


def test_lifecycle_script_initializes_a_fresh_persistent_state_dir(tmp_path: Path) -> None:
    state_dir = tmp_path / "new-state"
    environment = os.environ.copy()
    environment["DSLE_STATE_DIR"] = str(state_dir)
    environment["DSLE_CONTROL_DIR"] = str(tmp_path / "control")
    result = subprocess.run(
        ["bash", "-c", "source scripts/_container_common.sh; prepare_state_dir"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (state_dir / ".dsle-output").read_text(encoding="ascii") == "dsle-output\n"
    lock = state_dir / ".container-session.lock"
    assert lock.is_file() and lock.stat().st_size == 0
    assert lock.stat().st_mode & 0o077 == 0


def test_lifecycle_script_refuses_a_nonempty_unmarked_state_dir(tmp_path: Path) -> None:
    state_dir = tmp_path / "not-dsle"
    state_dir.mkdir()
    important = state_dir / "important.txt"
    important.write_text("unchanged", encoding="utf-8")
    environment = os.environ.copy()
    environment["DSLE_STATE_DIR"] = str(state_dir)
    environment["DSLE_CONTROL_DIR"] = str(tmp_path / "control")
    result = subprocess.run(
        ["bash", "-c", "source scripts/_container_common.sh; prepare_state_dir"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "non-empty uninitialized" in result.stderr
    assert important.read_text(encoding="utf-8") == "unchanged"


def test_lifecycle_metadata_stays_outside_state_and_never_evaluates_values(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    control_dir = tmp_path / "control"
    executed = tmp_path / "command-substitution-ran"
    literal_image = f"$(touch {executed})"
    environment = os.environ.copy()
    environment.update(
        {
            "DSLE_CONTROL_DIR": str(control_dir),
            "DSLE_IMAGE": literal_image,
            "DSLE_IMAGE_TARGET": "runtime",
            "DSLE_STATE_DIR": str(state_dir),
        }
    )
    subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/_container_common.sh; prepare_state_dir; persist_image_env",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    metadata = control_dir / "image.meta"
    assert metadata.is_file()
    assert metadata.stat().st_mode & 0o777 == 0o600
    assert metadata.read_text(encoding="ascii").startswith("DSLE-METADATA-V1\n")
    assert "$(" not in metadata.read_text(encoding="ascii")
    assert not (state_dir / "image.meta").exists()
    assert not (state_dir / "image.env").exists()
    assert not executed.exists()

    loaded_environment = environment.copy()
    loaded_environment.pop("DSLE_IMAGE")
    loaded_environment.pop("DSLE_IMAGE_TARGET")
    loaded = subprocess.run(
        [
            "bash",
            "-c",
            (
                "source scripts/_container_common.sh; "
                "unset DSLE_IMAGE DSLE_IMAGE_TARGET; "
                "load_image_env; "
                'printf \'%s\\n%s\\n\' "${DSLE_IMAGE}" "${DSLE_IMAGE_TARGET}"'
            ),
        ],
        cwd=PROJECT_ROOT,
        env=loaded_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert loaded.stdout.splitlines() == [literal_image, "runtime"]
    assert not executed.exists()


def test_script_lifecycle_output_is_default_on_and_environment_switchable() -> None:
    command = (
        "source scripts/_container_common.sh; "
        'checking "Checking dependencies"; debug "Container id: abc123"; ready "Game running"'
    )
    visible = subprocess.run(
        ["bash", "-c", command],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert visible.stdout == ""
    assert visible.stderr.splitlines() == [
        "[CHECK] Checking dependencies",
        "[DEBUG] Container id: abc123",
        "[READY] Game running",
    ]

    quiet_environment = os.environ.copy()
    quiet_environment["DSLE_VERBOSE"] = "0"
    quiet = subprocess.run(
        ["bash", "-c", command],
        cwd=PROJECT_ROOT,
        env=quiet_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert quiet.stdout == ""
    assert quiet.stderr == ""


def test_lifecycle_lock_rejects_a_concurrent_compose_mutation(tmp_path: Path) -> None:
    common = PROJECT_ROOT / "scripts" / "_container_common.sh"
    control = tmp_path / "control"
    command = (
        'source "$1"; DSLE_CONTROL_DIR="$2"; export DSLE_CONTROL_DIR; '
        "acquire_lifecycle_lock; printf 'ready\\n'; read -r _"
    )
    holder = subprocess.Popen(
        ["bash", "-c", command, "bash", str(common), str(control)],
        cwd=PROJECT_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline() == "ready\n"

    contender = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; DSLE_CONTROL_DIR="$2"; export DSLE_CONTROL_DIR; acquire_lifecycle_lock',
            "bash",
            str(common),
            str(control),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert contender.returncode != 0
    assert "another DSLE lifecycle command is already running" in contender.stderr

    _stdout, holder_stderr = holder.communicate("\n", timeout=3.0)
    assert holder.returncode == 0, holder_stderr


def test_lifecycle_metadata_rejects_unknown_keys_without_executing_them(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    control_dir.mkdir(mode=0o700)
    executed = tmp_path / "unknown-key-ran"

    def encoded(value: str) -> str:
        return base64.b64encode(value.encode("utf-8")).decode("ascii")

    metadata = control_dir / "image.meta"
    metadata.write_text(
        "\n".join(
            (
                "DSLE-METADATA-V1",
                f"DSLE_IMAGE={encoded('dsle-runtime:0.1.0')}",
                f"DSLE_IMAGE_TARGET={encoded('runtime')}",
                f"DSLE_UNKNOWN={encoded(f'$(touch {executed})')}",
                "",
            )
        ),
        encoding="ascii",
    )
    metadata.chmod(0o600)
    environment = os.environ.copy()
    environment["DSLE_CONTROL_DIR"] = str(control_dir)
    result = subprocess.run(
        ["bash", "-c", "source scripts/_container_common.sh; load_image_env"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unknown metadata key" in result.stderr
    assert not executed.exists()


def test_lifecycle_metadata_rejects_command_syntax_instead_of_sourcing_it(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    control_dir.mkdir(mode=0o700)
    executed = tmp_path / "raw-command-ran"
    metadata = control_dir / "image.meta"
    metadata.write_text(
        "\n".join(
            (
                "DSLE-METADATA-V1",
                f"DSLE_IMAGE=$(touch {executed})",
                "DSLE_IMAGE_TARGET=cnVudGltZQ==",
                "",
            )
        ),
        encoding="ascii",
    )
    metadata.chmod(0o600)
    environment = os.environ.copy()
    environment["DSLE_CONTROL_DIR"] = str(control_dir)
    result = subprocess.run(
        ["bash", "-c", "source scripts/_container_common.sh; load_image_env"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid base64" in result.stderr
    assert not executed.exists()


def test_lifecycle_metadata_writer_refuses_a_symlink_without_touching_its_target(
    tmp_path: Path,
) -> None:
    control_dir = tmp_path / "control"
    control_dir.mkdir(mode=0o700)
    victim = tmp_path / "important-host-file"
    victim.write_text("unchanged\n", encoding="utf-8")
    metadata = control_dir / "image.meta"
    metadata.symlink_to(victim)
    environment = os.environ.copy()
    environment.update(
        {
            "DSLE_CONTROL_DIR": str(control_dir),
            "DSLE_IMAGE": "dsle-runtime:0.1.0",
            "DSLE_IMAGE_TARGET": "runtime",
        }
    )
    result = subprocess.run(
        ["bash", "-c", "source scripts/_container_common.sh; persist_image_env"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "refusing symbolic-link metadata destination" in result.stderr
    assert metadata.is_symlink()
    assert victim.read_text(encoding="utf-8") == "unchanged\n"
    assert not tuple(control_dir.glob(".image.meta.tmp.*"))


def test_deployment_metadata_restores_the_recorded_custom_state_directory(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    state_dir = tmp_path / "persistent-output"
    game_dir = tmp_path / "legal-game"
    game_dir.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "DSLE_CONTROL_DIR": str(control_dir),
            "DSLE_GAME_DIR": str(game_dir),
            "DSLE_RESOLUTION": "800x600",
            "DSLE_STATE_DIR": str(state_dir),
            "DSLE_VNC_HOST_END": "5930",
            "DSLE_VNC_HOST_START": "5901",
        }
    )
    subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/_container_common.sh; prepare_state_dir; persist_deployment_env",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    restored_environment = os.environ.copy()
    restored_environment["DSLE_CONTROL_DIR"] = str(control_dir)
    restored = subprocess.run(
        [
            "bash",
            "-c",
            (
                "source scripts/_container_common.sh; "
                "unset DSLE_GAME_DIR DSLE_STATE_DIR; "
                "load_deployment_env; "
                'printf \'%s\\n%s\\n\' "${DSLE_STATE_DIR}" "${DSLE_GAME_DIR}"'
            ),
        ],
        cwd=PROJECT_ROOT,
        env=restored_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.splitlines() == [str(state_dir.resolve()), str(game_dir)]
    assert (control_dir / "deployment.meta").is_file()
    assert not (state_dir / "deployment.meta").exists()
    assert not (state_dir / "deployment.env").exists()


def test_manual_game_layout_uses_the_public_required_directory_set(tmp_path: Path) -> None:
    complete = _make_game_layout(tmp_path / "complete-game")
    accepted = subprocess.run(
        [
            "bash",
            "-c",
            'source scripts/_container_common.sh; validate_game_dir "$1"',
            "bash",
            str(complete),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stderr

    incomplete = tmp_path / "incomplete-game"
    incomplete.mkdir()
    (incomplete / "DarkSoulsRemastered.exe").write_bytes(b"MZ")
    for name in REQUIRED_GAME_DIRECTORIES[:-1]:
        (incomplete / name).mkdir()
    rejected = subprocess.run(
        [
            "bash",
            "-c",
            'source scripts/_container_common.sh; validate_game_dir "$1"',
            "bash",
            str(incomplete),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert f"game asset directory is missing: {incomplete}/script" in rejected.stderr


def test_manual_scripts_default_to_existing_cwd_game_and_preserve_missing_error(
    tmp_path: Path,
) -> None:
    common = PROJECT_ROOT / "scripts" / "_container_common.sh"
    local_game = _make_game_layout(tmp_path / "game")
    selected = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; select_game_dir ""',
            "bash",
            str(common),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert selected.stdout.strip() == str(local_game)
    assert f"Using default game directory: {local_game}" in selected.stderr

    empty_directory = tmp_path / "without-game"
    empty_directory.mkdir()
    missing = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; validate_game_dir "$(select_game_dir "")"',
            "bash",
            str(common),
        ],
        cwd=empty_directory,
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing.returncode != 0
    assert "provide --game-dir PATH or set DSLE_GAME_DIR" in missing.stderr


@pytest.mark.parametrize("relationship", ["state-inside-game", "state-contains-game"])
def test_start_rejects_game_state_overlap_before_docker_or_state_initialization(
    tmp_path: Path,
    relationship: str,
) -> None:
    game_dir = _make_game_layout(tmp_path / "game")
    state_dir = game_dir / "state" if relationship == "state-inside-game" else tmp_path
    fake_bin = tmp_path / "fake-bin"
    _write_fake_docker(fake_bin)
    docker_log = tmp_path / "docker.log"
    environment = os.environ.copy()
    environment.update(
        {
            "DSLE_CONTROL_DIR": str(tmp_path / "control"),
            "DSLE_FAKE_DOCKER_LOG": str(docker_log),
            "DSLE_FAKE_FAIL_STAGE": "none",
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )

    result = subprocess.run(
        [
            "/usr/bin/bash",
            "scripts/start.sh",
            "--game-dir",
            str(game_dir),
            "--state-dir",
            str(state_dir),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "DSLE_STATE_DIR cannot" in result.stderr
    assert not docker_log.exists(), "overlap must fail before invoking Docker or Compose"
    assert not (state_dir / ".dsle-output").exists()


@pytest.mark.parametrize(("failure_stage", "expected_status"), [("up", 71), ("configure", 72)])
def test_start_failure_removes_partial_container_but_preserves_image_and_state(
    tmp_path: Path,
    failure_stage: str,
    expected_status: int,
) -> None:
    game_dir = _make_game_layout(tmp_path / "game")
    state_dir = tmp_path / "state"
    fake_bin = tmp_path / "fake-bin"
    _write_fake_docker(fake_bin)
    docker_log = tmp_path / "docker.log"
    environment = os.environ.copy()
    environment.update(
        {
            "DSLE_CONTROL_DIR": str(tmp_path / "control"),
            "DSLE_FAKE_DOCKER_LOG": str(docker_log),
            "DSLE_FAKE_FAIL_STAGE": failure_stage,
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )

    result = subprocess.run(
        [
            "/usr/bin/bash",
            "scripts/start.sh",
            "--game-dir",
            str(game_dir),
            "--state-dir",
            str(state_dir),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == expected_status
    commands = docker_log.read_text(encoding="utf-8").splitlines()
    assert sum(command.endswith(" down --timeout 45") for command in commands) == 1
    assert commands[-1].endswith(" down --timeout 45")
    assert not any(" image rm" in command or " volume rm" in command for command in commands)
    assert (state_dir / ".dsle-output").read_text(encoding="ascii") == "dsle-output\n"
    assert (state_dir / "run" / "dsle.token").is_file()
    assert (
        "[STOP] Startup failed; stopping and removing the partial runtime container"
        in result.stderr
    )


def test_successful_container_only_start_does_not_run_failure_cleanup(tmp_path: Path) -> None:
    game_dir = _make_game_layout(tmp_path / "game")
    fake_bin = tmp_path / "fake-bin"
    _write_fake_docker(fake_bin)
    docker_log = tmp_path / "docker.log"
    environment = os.environ.copy()
    environment.update(
        {
            "DSLE_CONTROL_DIR": str(tmp_path / "control"),
            "DSLE_FAKE_DOCKER_LOG": str(docker_log),
            "DSLE_FAKE_FAIL_STAGE": "none",
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )

    result = subprocess.run(
        [
            "/usr/bin/bash",
            "scripts/start.sh",
            "--game-dir",
            str(game_dir),
            "--state-dir",
            str(tmp_path / "state"),
            "--container-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    commands = docker_log.read_text(encoding="utf-8").splitlines()
    assert any(command.endswith(" up --detach runtime") for command in commands)
    assert not any(command.endswith(" down --timeout 45") for command in commands)


def test_boss_vnc_sweep_launcher_is_executable_and_documents_lifecycle() -> None:
    launcher = PROJECT_ROOT / "scripts" / "boss-vnc-sweep.sh"
    assert launcher.is_file()
    assert os.access(launcher, os.X_OK)

    syntax = subprocess.run(
        ["/usr/bin/bash", "-n", str(launcher)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    help_result = subprocess.run(
        [str(launcher), "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "all 22 configured standard (non-boosted) boss saves" in help_result.stdout
    assert "same dsr-1 display" in help_result.stdout
    assert "container and VNC listener are stopped and removed" in help_result.stdout
    assert "Stable loopback host port used for the full run" in help_result.stdout

    payload = launcher.read_text(encoding="utf-8")
    assert "export DSLE_BOSS_SWEEP_VNC=1" in payload
    assert 'export DSLE_BOSS_SWEEP_VNC_HOST_PORT="${vnc_host_port}"' in payload
    assert 'exec "${SCRIPT_DIR}/boss-screenshot-sweep.sh" "${forwarded[@]}"' in payload


def test_recorded_boss_showcase_launcher_is_executable_and_documents_lifecycle() -> None:
    launcher = PROJECT_ROOT / "scripts" / "record-boss-showcase.sh"
    example = PROJECT_ROOT / "examples" / "boss_showcase.py"
    assert launcher.is_file() and os.access(launcher, os.X_OK)
    assert example.is_file() and os.access(example, os.X_OK)

    syntax = subprocess.run(
        ["/usr/bin/bash", "-n", str(launcher)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    environment = os.environ.copy()
    environment["DSLE_PYTHON"] = sys.executable
    help_result = subprocess.run(
        [str(launcher), "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "every supported regular boss" in help_result.stdout
    assert "--fight-seconds" in help_result.stdout
    assert "--menu-seconds" in help_result.stdout
    assert "--fps" in help_result.stdout

    source = example.read_text(encoding="utf-8")
    assert "vnc_ports=(VNC_CONTAINER_PORT,)" in source
    assert 'mode="headless-vnc"' in source
    assert 'difficulty="standard"' in source
    assert "environment.observe()" in source
    assert "environment.return_to_menu" in source


def test_single_instance_development_workflow_uses_live_read_only_repository_mount() -> None:
    launcher = PROJECT_ROOT / "scripts" / "dev.sh"
    override = (PROJECT_ROOT / "docker" / "compose.dev.yaml").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")

    assert launcher.is_file() and os.access(launcher, os.X_OK)
    syntax = subprocess.run(
        ["/usr/bin/bash", "-n", str(launcher)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr
    help_result = subprocess.run(
        [str(launcher), "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "scripts/dev.sh boss BOSS" in help_result.stdout
    assert "scripts/dev.sh menu" in help_result.stdout
    assert "Persistent Wine prefix, saves, logs, and results" in help_result.stdout
    assert "--no-vnc-viewer" in help_result.stdout

    assert "PYTHONPATH: /workspace/dsle/src" in override
    assert "DSLE_ASSET_DIR: /workspace/dsle/assets" in override
    assert "target: /workspace/dsle" in override
    assert "target: /workspace/dsle/src" not in override
    assert override.count("read_only: true") == 1
    assert "${DSLE_DEV_SOURCE_DIR:?" in override
    assert "${DSLE_VNC_PORT_BINDING:-127.0.0.1:5901-5930:5901-5930}" in compose

    payload = launcher.read_text(encoding="utf-8")
    assert "--count 1" in payload
    assert "--instances dsr-1" in payload
    assert 'DSLE_VNC_PORT_BINDING="127.0.0.1:${vnc_port}:5901"' in payload
    assert "python3 -m dsle.cli.dev" in payload
    assert "find_tigervnc_viewer" in payload
    assert 'nohup "${viewer}" "127.0.0.1::${vnc_port}"' in payload
    assert "compose down --timeout 45" in payload


def test_development_start_configures_exactly_one_live_source_instance(tmp_path: Path) -> None:
    game_dir = _make_game_layout(tmp_path / "game")
    fake_bin = tmp_path / "fake-bin"
    _write_fake_docker(fake_bin)
    _write_fake_tigervnc(fake_bin)
    docker_log = tmp_path / "docker.log"
    vnc_log = tmp_path / "vnc.log"
    environment = os.environ.copy()
    environment.update(
        {
            "DSLE_DEV_CONTROL_DIR": str(tmp_path / "control"),
            "DSLE_FAKE_DOCKER_LOG": str(docker_log),
            "DSLE_FAKE_FAIL_STAGE": "none",
            "DSLE_FAKE_VNC_LOG": str(vnc_log),
            "DISPLAY": ":99",
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )

    result = subprocess.run(
        [
            "/usr/bin/bash",
            "scripts/dev.sh",
            "start",
            "--game-dir",
            str(game_dir),
            "--state-dir",
            str(tmp_path / "state"),
            "--vnc-port",
            "6901",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    commands = docker_log.read_text(encoding="utf-8").splitlines()
    assert any(
        "--file " + str(PROJECT_ROOT / "docker" / "compose.dev.yaml") in line for line in commands
    )
    assert any("dsle-instances configure --count 1" in line for line in commands)
    assert any(
        "dsle-instances start --instances dsr-1 --mode headless-vnc" in line for line in commands
    )
    assert any("python3 -m dsle.cli.dev menu" in line for line in commands)
    dependency_check = next(
        index
        for index, line in enumerate(commands)
        if "run --rm --pull=never --entrypoint python3" in line
    )
    container_start = next(index for index, line in enumerate(commands) if "up --detach" in line)
    assert dependency_check < container_start
    assert "vncviewer 127.0.0.1::6901" in result.stderr
    assert "no boss selected; navigating dsr-1 to the main menu" in result.stderr
    for _attempt in range(50):
        if vnc_log.is_file():
            break
        time.sleep(0.01)
    assert vnc_log.read_text(encoding="utf-8").strip() == "127.0.0.1::6901"
    assert "opened TigerVNC Viewer" in result.stderr


def test_development_start_explains_stale_prebuilt_image(tmp_path: Path) -> None:
    game_dir = _make_game_layout(tmp_path / "game")
    fake_bin = tmp_path / "fake-bin"
    _write_fake_docker(fake_bin)
    docker_log = tmp_path / "docker.log"
    environment = os.environ.copy()
    environment.update(
        {
            "DSLE_DEV_CONTROL_DIR": str(tmp_path / "control"),
            "DSLE_FAKE_DOCKER_LOG": str(docker_log),
            "DSLE_FAKE_FAIL_STAGE": "dependencies",
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )

    result = subprocess.run(
        [
            "/usr/bin/bash",
            "scripts/dev.sh",
            "start",
            "--game-dir",
            str(game_dir),
            "--state-dir",
            str(tmp_path / "state"),
            "--no-vnc-viewer",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "missing current dependencies" in result.stderr
    assert "rerun this command with --build" in result.stderr
    assert "No module named rich" in result.stderr
    commands = docker_log.read_text(encoding="utf-8").splitlines()
    assert any("run --rm --pull=never --entrypoint python3" in line for line in commands)
    assert not any("up --detach runtime" in line for line in commands)
    assert not any("down --timeout 45" in line for line in commands)


def test_source_distribution_contains_the_complete_public_source_project(tmp_path: Path) -> None:
    output = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    archives = tuple(output.glob("dsle-0.1.0.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0], mode="r:gz") as archive:
        names = {
            member.name.split("/", 1)[1]
            for member in archive.getmembers()
            if "/" in member.name and member.isfile()
        }

    expected_files = {"LICENSE", "MANIFEST.in", "README.md", "pyproject.toml"}
    for directory in ("src/dsle", "assets", "docker", "docs", "examples", "scripts", "tests"):
        expected_files.update(
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in (PROJECT_ROOT / directory).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    assert expected_files <= names
    assert not any("DarkSoulsRemastered.exe" in name for name in names)
    assert not any(
        part in {".dsle-control", ".dsle-state", ".git", ".pytest_cache", ".ruff_cache"}
        for name in names
        for part in Path(name).parts
    )


def test_host_wheel_contains_api_and_configs_but_no_runtime_assets(tmp_path: Path) -> None:
    """The wheel is the host control plane; the OCI image owns live assets."""

    output = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheels = tuple(output.glob("dsle-0.1.0-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        license_name = next(name for name in names if name.endswith(".dist-info/licenses/LICENSE"))
        metadata = archive.read(metadata_name).decode("utf-8")
        packaged_license = archive.read(license_name)
    assert "dsle/_version.py" in names
    assert "dsle/py.typed" in names
    assert "dsle/config/defaults.yaml" in names
    assert (
        sum(name.startswith("dsle/config/bosses/") and name.endswith(".yaml") for name in names)
        == 22
    )
    assert not any(name.endswith(".sl2") for name in names)
    assert not any("DarkSoulsRemastered.exe" in name for name in names)
    assert not any(
        name.startswith(("assets/", "docker/", "docs/", "examples/", "scripts/", "tests/"))
        for name in names
    )
    assert "License-Expression: GPL-3.0-only\n" in metadata
    assert "License-File: LICENSE\n" in metadata
    assert "Classifier: License ::" not in metadata
    assert packaged_license == (PROJECT_ROOT / "LICENSE").read_bytes()

    target = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--target",
            str(target),
            str(wheels[0]),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    environment = os.environ.copy()
    # The isolated wheel smoke is a separate installed artifact, not a child
    # coverage worker. pytest-cov exports auto-start variables that would make
    # it emit incompatible statement-only data beside this branch run.
    for variable in (
        "COVERAGE_PROCESS_START",
        "COV_CORE_SOURCE",
        "COV_CORE_CONFIG",
        "COV_CORE_DATAFILE",
    ):
        environment.pop(variable, None)
    environment["PYTHONPATH"] = str(target)
    smoke = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import dsle; "
                "assert dsle.__version__ == '0.1.0'; "
                "assert len(dsle.list_bosses()) == 22; "
                "assert dsle.load_boss('asylum_demon').boss_id == 'asylum_demon'"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert smoke.returncode == 0
