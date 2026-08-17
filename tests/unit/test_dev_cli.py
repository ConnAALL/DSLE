from __future__ import annotations

import json
from types import SimpleNamespace

from dsle.cli import dev
from tests.fakes import state


def test_boss_command_monitors_death_cleans_up_and_returns_to_menu(monkeypatch, capsys) -> None:
    calls: list[object] = []

    class Backend:
        def reset(self, boss, save_state):
            calls.append(("reset", boss.boss_id, save_state))
            return SimpleNamespace(state=state(boss_hp=1176))

        def observe(self, boss):
            calls.append(("observe", boss.boss_id))
            return SimpleNamespace(state=state(player_hp=0, player_dead=True, boss_hp=900))

        def finish(self, boss, outcome):
            calls.append(("finish", boss.boss_id, outcome.reason, outcome.win))

        def return_to_menu(self, *, timeout_s):
            calls.append(("menu", timeout_s))

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setenv("DSLE_CONTAINER", "1")
    monkeypatch.setattr(
        dev,
        "_new_backend",
        lambda instance, **_kwargs: calls.append(instance) or Backend(),
    )

    assert dev.main(["boss", "capra_demon", "--difficulty", "boosted"]) == 0
    captured = capsys.readouterr()
    assert "fight through VNC" in captured.err
    assert "[COMBAT] dsr-1 sample=0" in captured.err
    assert "player_location={'x': 1.25, 'y': 2.5, 'z': -3.75}" in captured.err
    assert "boss_defeated=no" in captured.err
    assert '"result": "player_dead"' in captured.out
    assert '"status": "complete"' in captured.out
    assert calls == [
        "dsr-1",
        ("reset", "capra_demon", "capra_demon_boosted.sl2"),
        ("observe", "capra_demon"),
        ("finish", "capra_demon", "player_dead", False),
        ("menu", 60.0),
        "close",
    ]


def test_manual_boss_monitor_preserves_simultaneous_death_victory_precedence() -> None:
    boss = dev.ConfigRepository().get("asylum_demon")

    class Backend:
        @staticmethod
        def observe(_boss):
            return SimpleNamespace(
                state=state(player_hp=0, player_dead=True, boss_hp=0, boss_defeated=True)
            )

    outcome, _final = dev._wait_for_outcome(Backend(), boss, poll_s=0.1)

    assert outcome.reason == "boss_defeated"
    assert outcome.win is True


def test_menu_command_attaches_returns_and_closes(monkeypatch, capsys) -> None:
    calls: list[object] = []

    class Backend:
        def return_to_menu(self, *, timeout_s: float) -> None:
            calls.append(("menu", timeout_s))

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setenv("DSLE_CONTAINER", "1")
    monkeypatch.setattr(
        dev,
        "_new_backend",
        lambda instance, **_kwargs: calls.append(instance) or Backend(),
    )

    assert dev.main(["menu", "--timeout", "12.5"]) == 0
    assert calls == ["dsr-1", ("menu", 12.5), "close"]
    assert capsys.readouterr().out == "dsr-1 is at the title menu\n"


def test_live_commands_reject_accidental_host_execution(monkeypatch, capsys) -> None:
    monkeypatch.delenv("DSLE_CONTAINER", raising=False)

    assert dev.main(["boss", "asylum_demon"]) == 2
    assert "must run inside the DSLE runtime container" in capsys.readouterr().err


def test_boss_command_suggests_a_close_boss_id(monkeypatch, capsys) -> None:
    monkeypatch.setenv("DSLE_CONTAINER", "1")

    assert dev.main(["boss", "strya_dmeon"]) == 2
    assert "Did you mean 'stray_demon'?" in capsys.readouterr().err


def test_list_reads_the_current_config_tree(capsys) -> None:
    assert dev.main(["list", "--json"]) == 0
    bosses = json.loads(capsys.readouterr().out)
    assert len(bosses) == 22
    assert "asylum_demon" in bosses
