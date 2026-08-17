from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from dsle.config import ConfigRepository
from dsle.exceptions import ConfigurationError
from dsle.suites import DSLE_5, UNCONFIGURED_DSR_BOSSES

PACKAGE_CONFIG = Path(__file__).resolve().parents[2] / "src" / "dsle" / "config"


def one_boss_config(tmp_path: Path) -> Path:
    root = tmp_path / "config"
    (root / "bosses").mkdir(parents=True)
    shutil.copy2(PACKAGE_CONFIG / "defaults.yaml", root / "defaults.yaml")
    shutil.copy2(
        PACKAGE_CONFIG / "bosses" / "asylum_demon.yaml",
        root / "bosses" / "asylum_demon.yaml",
    )
    return root


def mutate_boss(root: Path, mutation) -> None:
    path = root / "bosses" / "asylum_demon.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutation(document)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def test_registry_contains_every_supported_base_game_boss() -> None:
    repository = ConfigRepository()
    assert len(repository.list_bosses()) == 22
    assert len(repository.list_bosses(include_experimental=False)) == 22
    assert len(repository.list_bosses(include_experimental=True)) == 22
    assert set(DSLE_5) <= set(repository.list_bosses(include_experimental=False))
    assert "centipede_demon" in repository.list_bosses(include_experimental=False)
    assert "bed_of_chaos" in repository.list_bosses(include_experimental=False)
    assert UNCONFIGURED_DSR_BOSSES == ()


def test_every_boss_has_standard_and_boosted_saves() -> None:
    repository = ConfigRepository()
    for boss_id in repository.list_bosses(include_experimental=False):
        boss = repository.get(boss_id)
        assert boss.source.name == f"{boss_id}.yaml"
        assert boss.save_state("standard") == f"{boss_id}.sl2"
        assert boss.save_state("boosted") == f"{boss_id}_boosted.sl2"
        assert boss.victory.templates
        assert boss.setup
        assert boss.cleanup
        assert boss.metadata.description
        assert boss.metadata.tags == ()


def test_alias_resolution_is_case_insensitive() -> None:
    repository = ConfigRepository()
    assert repository.get("Asylum Demon").boss_id == "asylum_demon"
    assert repository.get("O&S").boss_id == "ornstein_and_smough"
    assert repository.get("Gravelord Nito").boss_id == "nito"


def test_unknown_boss_suggests_the_closest_canonical_id() -> None:
    repository = ConfigRepository()

    with pytest.raises(KeyError) as captured:
        repository.get("strya_dmeon")

    assert "Did you mean 'stray_demon'?" in str(captured.value)


def test_special_boss_operations_preserve_ground_truth() -> None:
    repository = ConfigRepository()
    four_kings = repository.get("four_kings")
    assert [operation.op for operation in four_kings.setup].count("teleport_player") == 2
    assert any(
        operation.op == "hold_key" and operation.params.get("key") == "w"
        for operation in four_kings.setup
    )

    stray = repository.get("stray_demon")
    assert all(operation.op != "wait_until_ready" for operation in stray.setup)

    priscilla = repository.get("crossbreed_priscilla")
    assert [operation.op for operation in priscilla.setup[-7:]] == [
        "teleport_player",
        "sleep",
        "repeat_actions",
        "sleep",
        "teleport_player",
        "sleep",
        "set_flag",
    ]
    assert dict(priscilla.setup[-7].params) == {
        "x": -23.03441047668457,
        "y": 698.2172241210938,
        "z": 60.07139587402344,
    }
    assert dict(priscilla.setup[-6].params) == {"seconds": 1.0}
    assert dict(priscilla.setup[-5].params) == {
        "actions": ("light_attack",),
        "duration_s": 0.05,
        "hold_s": 0.05,
        "interval_s": 0.0,
    }
    assert dict(priscilla.setup[-4].params) == {"seconds": 1.0}
    assert dict(priscilla.setup[-3].params) == {
        "x": -23.04547882080078,
        "y": 710.9319458007812,
        "z": 60.551387786865234,
    }
    assert dict(priscilla.setup[-2].params) == {"seconds": 0.1}
    assert dict(priscilla.setup[-1].params) == {
        "flag": "no_damage",
        "enable": False,
    }

    gwyn = repository.get("gwyn_lord_of_cinder")
    assert all(
        operation.op not in {"teleport_player", "repeat_actions"} for operation in gwyn.setup
    )
    assert [operation.op for operation in gwyn.setup] == [
        "ensure_menu",
        "load_save",
        "wait_until_ready",
        "set_flag",
        "tap_key",
        "sleep",
        "set_flag",
        "sleep",
        "set_flag",
    ]
    protection = [
        (index, operation.params["enable"])
        for index, operation in enumerate(gwyn.setup)
        if operation.op == "set_flag" and operation.params.get("flag") == "no_damage"
    ]
    assert [enabled for _index, enabled in protection] == [True, False]
    wait_index = next(
        index for index, operation in enumerate(gwyn.setup) if operation.op == "wait_until_ready"
    )
    interact_index = next(
        index
        for index, operation in enumerate(gwyn.setup)
        if operation.op == "tap_key" and operation.params.get("key") == "e"
    )
    assert gwyn.setup[wait_index].params["interact_after_ready"] is False
    # Protection starts at the fog boundary, before entering Gwyn's arena, and
    # normal damage is restored as soon as the short entry sequence finishes.
    assert protection == [(wait_index + 1, True), (len(gwyn.setup) - 1, False)]
    assert interact_index == wait_index + 2

    nito = repository.get("nito")
    assert [operation.op for operation in nito.setup] == [
        "ensure_menu",
        "load_save",
        "wait_until_ready",
        "sleep",
        "set_flag",
        "set_flag",
        "set_flag",
        "sleep",
        "teleport_player",
        "sleep",
        "tap_key",
        "sleep",
        "set_flag",
        "sleep",
    ]
    teleports = [operation.params for operation in nito.setup if operation.op == "teleport_player"]
    assert [(item["x"], item["y"], item["z"]) for item in teleports] == [
        (-138.84, -37.63, -265.102),
    ]
    readiness = [operation for operation in nito.setup if operation.op == "wait_until_ready"]
    assert len(readiness) == 1
    assert readiness[0].params == {}
    assert nito.readiness.mode == "traverse_light"
    assert nito.readiness.interact_after_ready is True
    damage_protection = [
        operation.params["enable"]
        for operation in nito.setup
        if operation.op == "set_flag" and operation.params["flag"] == "no_damage"
    ]
    assert damage_protection == [False, True, False]
    assert nito.setup[-4].params == {"key": "esc", "hold_s": 0.1}
    assert nito.metadata.expected_hp == 4317

    centipede = repository.get("centipede")
    assert centipede.availability == "supported"
    assert centipede.save_state("standard") == "centipede_demon.sl2"
    assert centipede.save_state("boosted") == "centipede_demon_boosted.sl2"
    assert centipede.memory.hp_chains == ((8, 1840, 200, 1000),)
    assert centipede.memory.defeated_offsets == (0, 15475)
    assert centipede.memory.defeated_bit == 2
    assert centipede.metadata.expected_hp == 3432
    assert centipede.victory.templates[0] == "centipede_demon_post"
    assert all(operation.op != "teleport_player" for operation in centipede.setup)
    assert [operation.op for operation in centipede.setup] == [
        "ensure_menu",
        "load_save",
        "wait_until_ready",
        "sleep",
        "set_flag",
        "set_flag",
        "sleep",
        "tap_key",
        "sleep",
    ]
    escape = centipede.setup[-2]
    assert escape.params == {"key": "esc", "hold_s": 0.1}

    bed = repository.get("bed")
    assert bed.availability == "supported"
    assert bed.save_state("standard") == "bed_of_chaos.sl2"
    assert bed.save_state("boosted") == "bed_of_chaos_boosted.sl2"
    assert bed.memory.hp_chains == ()
    assert bed.memory.hp_mode == "defeated_flag"
    assert bed.memory.defeated_offsets == (0, 2)
    assert bed.memory.defeated_bit == 5
    assert bed.metadata.expected_hp == 1
    assert bed.victory.templates[0] == "bed_of_chaos_post"
    assert [operation.op for operation in bed.setup] == [
        "ensure_menu",
        "load_save",
        "wait_until_ready",
        "sleep",
        "set_flag",
        "set_flag",
        "set_flag",
        "sleep",
        "teleport_player",
        "sleep",
        "set_flag",
        "sleep",
    ]
    assert dict(bed.setup[-4].params) == {
        "x": 522.002,
        "y": 394.87,
        "z": -440.13,
    }
    bed_damage_protection = [
        operation.params["enable"]
        for operation in bed.setup
        if operation.op == "set_flag" and operation.params["flag"] == "no_damage"
    ]
    assert bed_damage_protection == [False, True, False]


def test_victory_cleanup_reserves_time_and_recovers_when_no_template_appears() -> None:
    defaults = ConfigRepository().defaults
    for preset_name in ("default", "retreat_before_title"):
        wait = next(
            operation
            for operation in defaults.cleanup_presets[preset_name]
            if operation.op == "wait_for_victory"
        )
        assert wait.params["timeout_s"] == 50.0
        assert wait.params["continue_on_timeout"] is True


def test_only_known_unreliable_flags_allow_hp_zero_victory() -> None:
    repository = ConfigRepository()
    hp_zero = {
        boss_id
        for boss_id in repository.list_bosses()
        if repository.get(boss_id).victory.hp_zero_is_victory
    }
    assert hp_zero == {
        "ceaseless_discharge",
        "demon_firesage",
        "moonlight_butterfly",
        "stray_demon",
    }


def test_unknown_difficulty_has_an_actionable_error() -> None:
    boss = ConfigRepository().get("asylum_demon")
    with pytest.raises(ValueError, match=r"standard, boosted|boosted, standard"):
        boss.save_state("nightmare")


def test_config_repository_requires_at_least_one_boss(tmp_path: Path) -> None:
    root = tmp_path / "config"
    (root / "bosses").mkdir(parents=True)
    shutil.copy2(PACKAGE_CONFIG / "defaults.yaml", root / "defaults.yaml")
    with pytest.raises(ConfigurationError, match="No boss YAML"):
        ConfigRepository(root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update(schema_version=2), "schema_version"),
        (lambda raw: raw.update(id="wrong_name"), "match its filename"),
        (lambda raw: raw.update(save_states={"boosted": "x.sl2"}), "standard"),
        (lambda raw: raw["memory"].update(hp_chains=[]), "HP chain"),
        (
            lambda raw: raw["memory"].update(hp_mode="defeated_flag"),
            "requires memory.hp_chains to be empty",
        ),
        (lambda raw: raw["memory"].update(hp_mode="largest"), "hp_mode"),
        (lambda raw: raw["memory"]["defeated_flag"].update(bit=9), "defeated_flag"),
        (lambda raw: raw["victory"].update(templates=[]), "templates"),
        (lambda raw: raw.update(availability="missing"), "availability"),
        (lambda raw: raw.update(typo_field=True), "unknown key"),
        (
            lambda raw: raw["victory"].update(hp_zero_is_victory="false"),
            "must be a boolean",
        ),
        (lambda raw: raw.update(reward={"win_bonus": float("nan")}), "finite"),
        (
            lambda raw: raw.update(save_states={"standard": "../outside.sl2"}),
            "relative .sl2 filename",
        ),
        (
            lambda raw: raw["save_states"].update(standard="bad save.sl2"),
            "relative .sl2 filename",
        ),
        (
            lambda raw: raw["save_states"].update(standard=".hidden.sl2"),
            "relative .sl2 filename",
        ),
        (
            lambda raw: raw["save_states"].update(standard="ümlaut.sl2"),
            "relative .sl2 filename",
        ),
        (
            lambda raw: raw["save_states"].update(standard=r"back\\slash.sl2"),
            "relative .sl2 filename",
        ),
        (
            lambda raw: raw.update(setup={"replace": [{"op": "launch_missiles"}]}),
            "unsupported operation",
        ),
        (
            lambda raw: raw.update(
                setup={"replace": [{"op": "sleep", "seconds": 1, "secondz": 1}]}
            ),
            "unknown key",
        ),
        (
            lambda raw: raw.update(
                setup={
                    "replace": [
                        {
                            "op": "repeat_actions",
                            "actions": [{"name": "not_an_action"}],
                            "duration_s": 1,
                        }
                    ]
                }
            ),
            "unknown action",
        ),
        (
            lambda raw: raw.update(
                setup={
                    "replace": [
                        {
                            "op": "walk_until_template",
                            "template": "talk",
                            "continue_on_timeout": "true",
                        }
                    ]
                }
            ),
            "must be a boolean",
        ),
    ],
)
def test_invalid_boss_documents_fail_before_backend_allocation(
    tmp_path: Path, mutation, message: str
) -> None:
    root = one_boss_config(tmp_path)
    mutate_boss(root, mutation)
    with pytest.raises(ConfigurationError, match=message):
        ConfigRepository(root)


def test_experimental_filter_is_explicit(tmp_path: Path) -> None:
    root = one_boss_config(tmp_path)
    mutate_boss(root, lambda raw: raw.update(availability="experimental"))
    repository = ConfigRepository(root)
    assert repository.list_bosses() == ()
    assert repository.list_bosses(include_experimental=False) == ()
    assert repository.list_bosses(include_experimental=True) == ("asylum_demon",)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update(extra_section={}), "unknown key"),
        (
            lambda raw: raw["environment"].update(auto_lock_on="false"),
            "must be a boolean",
        ),
        (
            lambda raw: raw["observation"].update(template_threshold=float("inf")),
            "finite",
        ),
        (
            lambda raw: raw["cleanup_presets"]["default"][0].update(enable="true"),
            "must be a boolean",
        ),
    ],
)
def test_invalid_defaults_fail_strictly_before_runtime(
    tmp_path: Path, mutation, message: str
) -> None:
    root = one_boss_config(tmp_path)
    path = root / "defaults.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutation(document)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        ConfigRepository(root)


def test_yaml_loader_rejects_symlinks_and_oversized_documents(tmp_path: Path) -> None:
    outside = tmp_path / "outside-defaults.yaml"
    outside.write_text("schema_version: 1\n", encoding="utf-8")
    linked_root = tmp_path / "linked-config"
    (linked_root / "bosses").mkdir(parents=True)
    (linked_root / "defaults.yaml").symlink_to(outside)
    with pytest.raises(ConfigurationError, match="Could not load"):
        ConfigRepository(linked_root)

    large_root = tmp_path / "large-config"
    (large_root / "bosses").mkdir(parents=True)
    (large_root / "defaults.yaml").write_bytes(b"#" * ((1 << 20) + 1))
    with pytest.raises(ConfigurationError, match="exceeds"):
        ConfigRepository(large_root)
