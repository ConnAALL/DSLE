from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src" / "dsle"


def python_sources(root: Path):
    yield from sorted(root.rglob("*.py"))


def base_name(node: ast.expr) -> str:
    while isinstance(node, ast.Subscript):
        node = node.value
    return ast.unparse(node)


def test_environment_package_has_no_baseline_dependencies() -> None:
    forbidden = {"cma", "scipy", "torch"}
    found: list[tuple[Path, str]] = []
    for path in python_sources(SOURCE_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".", 1)[0]}
            else:
                continue
            for name in names & forbidden:
                found.append((path.relative_to(PROJECT_ROOT), name))
    assert not found, f"Learning dependencies leaked into src/dsle: {found}"


def test_there_is_one_authoritative_environment_and_one_transport_proxy() -> None:
    concrete: list[tuple[Path, str]] = []
    for path in python_sources(SOURCE_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {base_name(base) for base in node.bases}
            if bases & {"gym.Env", "gymnasium.Env"}:
                concrete.append((path.relative_to(PROJECT_ROOT), node.name))
    assert concrete == [
        (Path("src/dsle/env.py"), "DarkSoulsEnv"),
        (Path("src/dsle/remote.py"), "RemoteDarkSoulsEnv"),
    ]

    # The proxy must not duplicate reward, reset, or terminal semantics. Those
    # live exclusively in DarkSoulsEnv inside the runtime process.
    remote_source = (SOURCE_ROOT / "remote.py").read_text(encoding="utf-8")
    assert "transition_reward" not in remote_source
    assert "BossConfig" not in remote_source


def test_docker_is_nvidia_only_and_cannot_copy_the_game() -> None:
    dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")
    build_inputs = [
        line.strip().casefold()
        for line in dockerfile.splitlines()
        if line.lstrip().upper().startswith(("COPY ", "ADD "))
    ]
    assert not any("darksouls" in line or "game" in line for line in build_inputs)
    assert "runtime: nvidia" in compose
    assert "gpus: all" in compose
    assert "NVIDIA_DRIVER_CAPABILITIES" in compose
    assert "read_only: true" in compose
    assert "igpu" not in dockerfile.casefold()
    assert "igpu" not in compose.casefold()


def test_every_learning_baseline_lives_under_examples() -> None:
    expected = {
        "random_agent.py",
        "expert_agent.py",
        "ppo.py",
        "dqn.py",
        "scope.py",
        "scope_parallel.py",
    }
    assert expected <= {path.name for path in (PROJECT_ROOT / "examples").glob("*.py")}
