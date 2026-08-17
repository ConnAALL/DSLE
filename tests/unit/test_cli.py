from __future__ import annotations

import json
from pathlib import Path

import pytest

from dsle.cli import doctor
from dsle.cli import instances as instances_cli
from dsle.runtime.version import REQUIRED_GAME_DIRECTORIES


def make_game_layout(root: Path, *, missing: frozenset[str] = frozenset()) -> Path:
    root.mkdir()
    (root / "DarkSoulsRemastered.exe").write_bytes(b"MZ")
    for name in REQUIRED_GAME_DIRECTORIES:
        if name not in missing:
            (root / name).mkdir()
    return root


def test_doctor_game_checks_distinguish_required_missing_and_unknown_build(
    tmp_path: Path, monkeypatch
) -> None:
    missing = doctor._game_checks(None, required=True)
    assert missing[0].status == "fail"
    optional = doctor._game_checks(None, required=False)
    assert optional[0].status == "warn"

    game = make_game_layout(tmp_path / "game")
    monkeypatch.setenv("DSLE_ALLOW_UNVERIFIED_GAME", "1")
    checks = doctor._game_checks(game, required=True)
    assert not any(check.status == "fail" for check in checks)
    assert next(check for check in checks if check.name == "game build").status == "warn"

    incomplete_game = make_game_layout(tmp_path / "incomplete-game", missing=frozenset({"script"}))
    incomplete_checks = doctor._game_checks(incomplete_game, required=True)
    asset_check = next(check for check in incomplete_checks if check.name == "game assets")
    assert asset_check.status == "fail"
    assert asset_check.detail == "missing: script"


def test_doctor_json_exit_code_reflects_failed_preflight(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        doctor,
        "_host_checks",
        lambda: [doctor.Check("Docker", "fail", "daemon unavailable")],
    )
    code = doctor.main(["--host", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["ok"] is False
    assert any(check["name"] == "Docker" for check in payload["checks"])


def test_doctor_defaults_to_existing_cwd_game(tmp_path: Path, monkeypatch) -> None:
    game = make_game_layout(tmp_path / "game")
    selected: list[tuple[Path | None, bool]] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DSLE_GAME_DIR", raising=False)
    monkeypatch.setattr(doctor, "_resource_checks", lambda _path: [])
    monkeypatch.setattr(doctor, "_host_checks", lambda: [])
    monkeypatch.setattr(
        doctor,
        "_game_checks",
        lambda game_dir, required: selected.append((game_dir, required)) or [],
    )

    assert doctor.main(["--host", "--quiet"]) == 0
    assert selected == [(game, False)]


def test_container_doctor_fails_a_confirmed_writable_game_mount(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor,
        "_command_check",
        lambda name, _command: doctor.Check(name, "pass", "ok"),
    )
    monkeypatch.setattr(doctor, "_mount_is_read_only", lambda _path: False)
    monkeypatch.setattr(doctor, "_linux_security_checks", lambda: [])
    checks = doctor._container_checks()
    mount = next(check for check in checks if check.name == "game mount mode")
    assert mount.status == "fail"
    assert "writable" in mount.detail


def test_container_doctor_requires_a_live_pulseaudio_socket(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor,
        "_command_check",
        lambda name, _command: doctor.Check(name, "pass", "ok"),
    )
    monkeypatch.setattr(doctor, "_mount_is_read_only", lambda _path: True)
    monkeypatch.setattr(doctor, "_linux_security_checks", lambda: [])
    monkeypatch.setenv("PULSE_SERVER", "unix:/definitely/missing/dsle-pulse.sock")

    checks = doctor._container_checks()

    pulse = next(check for check in checks if check.name == "PulseAudio null sink")
    assert pulse.status == "fail"
    assert "cannot connect" in pulse.detail


def test_instances_cli_configure_persists_resolution_and_lists(tmp_path: Path, capsys) -> None:
    config = tmp_path / "config" / "instances.json"
    state = tmp_path / "state"
    assert (
        instances_cli.main(
            [
                "configure",
                "--config",
                str(config),
                "--state-dir",
                str(state),
                "--count",
                "2",
                "--resolution",
                "1280x720",
            ]
        )
        == 0
    )
    document = json.loads(config.read_text(encoding="utf-8"))
    assert {entry["resolution"] for entry in document["instances"].values()} == {"1280x720"}

    assert instances_cli.main(["list", "--config", str(config), "--json"]) == 0
    listed = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert [row["name"] for row in listed] == ["dsr-1", "dsr-2"]
    assert {row["resolution"] for row in listed} == {"1280x720"}

    assert instances_cli.main(["list", "--config", str(config)]) == 0
    assert "1280x720" in capsys.readouterr().out


def test_instances_cli_start_returns_failure_when_any_instance_is_not_ready(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = tmp_path / "config" / "instances.json"
    assert (
        instances_cli.main(
            ["configure", "--config", str(config), "--state-dir", str(tmp_path), "--count", "1"]
        )
        == 0
    )
    capsys.readouterr()
    original = instances_cli._runtime_function

    def runtime_function(name: str):
        if name == "start_instances":
            return lambda **_kwargs: {
                "dsr-1": {
                    "name": "dsr-1",
                    "status": "failed",
                    "running": False,
                    "error": "Xorg failed",
                }
            }
        return original(name)

    monkeypatch.setattr(instances_cli, "_runtime_function", runtime_function)
    assert instances_cli.main(["start", "--config", str(config), "--all"]) == 1
    captured = capsys.readouterr()
    assert "Xorg failed" in captured.out
    assert "dsr-1=failed" in captured.err


def test_instances_cli_rejects_bad_resolution_without_writing(tmp_path: Path) -> None:
    config = tmp_path / "instances.json"
    try:
        instances_cli.main(["configure", "--config", str(config), "--resolution", "200x100"])
    except SystemExit as exc:
        assert exc.code == 2
    assert not config.exists()


@pytest.mark.parametrize("command", ["start", "status", "stop"])
def test_instances_cli_requires_an_explicit_selection(command: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        instances_cli.build_parser().parse_args([command])
    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("builder", "program"),
    [(doctor.build_parser, "dsle-doctor"), (instances_cli.build_parser, "dsle-instances")],
)
def test_public_clis_expose_stable_program_names_and_version(builder, program, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        builder().parse_args(["--version"])
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"{program} 0.1.0"

    with pytest.raises(SystemExit) as exc_info:
        builder().parse_args(["--help"])
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: {program} ")
