"""Interactive commands for a live, single-instance DSLE development container."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence

from dsle._version import __version__
from dsle.config import ConfigRepository
from dsle.exceptions import DSLEError
from dsle.models import BossConfig, EpisodeOutcome, RuntimeObservation
from dsle.progress import LifecycleLogger
from dsle.runtime.live import LiveGameBackend


def _require_container() -> None:
    if os.environ.get("DSLE_CONTAINER") != "1":
        raise RuntimeError("dsle-dev must run inside the DSLE runtime container")


def _verbose_from_environment() -> bool:
    return os.environ.get("DSLE_VERBOSE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _new_backend(instance: str, *, lifecycle: LifecycleLogger) -> LiveGameBackend:
    return LiveGameBackend(instance, lifecycle=lifecycle)


def _report_combat(
    lifecycle: LifecycleLogger,
    instance: str,
    sample: int,
    observation: RuntimeObservation,
) -> None:
    state = observation.state
    player = (
        f"{state.player_hp}/{state.player_hp_max}"
        if state.player_state_valid
        and state.player_hp is not None
        and state.player_hp_max is not None
        else "unavailable"
    )
    boss_hps = state.boss_hps or (state.boss_hp,)
    boss_hp_maxes = state.boss_hp_maxes or (state.boss_hp_max,)
    boss_health = " ".join(
        f"boss{index + 1}_hp="
        + (
            f"{hp}/{boss_hp_maxes[index]}"
            if state.boss_state_valid
            and hp is not None
            and index < len(boss_hp_maxes)
            and boss_hp_maxes[index] is not None
            else str(hp)
            if state.boss_state_valid and hp is not None
            else "unavailable"
        )
        for index, hp in enumerate(boss_hps)
    )
    lifecycle.event(
        "COMBAT",
        f"{instance} sample={sample} player_hp={player} "
        f"player_location={state.player_location} {boss_health} "
        f"boss_defeated={'yes' if state.boss_defeated else 'no'} "
        f"player_dead={'yes' if state.player_dead else 'no'}",
    )


def _wait_for_outcome(
    backend: LiveGameBackend,
    boss: BossConfig,
    *,
    poll_s: float,
    status_interval_s: float = 1.0,
    lifecycle: LifecycleLogger | None = None,
) -> tuple[EpisodeOutcome, RuntimeObservation]:
    """Observe a manual fight until the boss or player is confirmed dead."""

    if poll_s <= 0:
        raise ValueError("poll_s must be positive")
    if status_interval_s <= 0:
        raise ValueError("status_interval_s must be positive")
    samples = 0
    victory_streak = 0
    next_status = 0.0
    selected_lifecycle = lifecycle or LifecycleLogger(False)
    runtime_instance = getattr(backend, "instance", None)
    instance_name = getattr(runtime_instance, "name", "dsr-1")
    while True:
        current = backend.observe(boss)
        samples += 1
        state = current.state
        victory_signal = state.boss_state_valid and state.boss_defeated
        victory_streak = victory_streak + 1 if victory_signal else 0
        now = time.monotonic()
        terminal_signal = victory_signal or state.player_dead
        if now >= next_status or terminal_signal:
            _report_combat(selected_lifecycle, instance_name, samples, current)
            next_status = now + status_interval_s
        if (
            samples >= boss.limits.minimum_victory_step
            and victory_streak >= boss.limits.victory_confirmations
        ):
            return EpisodeOutcome(True, True, False, "boss_defeated", samples), current
        # Preserve the environment's simultaneous-death behavior: a confirmed
        # boss defeat wins before player death is considered.
        if state.player_dead:
            return EpisodeOutcome(False, True, False, "player_dead", samples), current
        time.sleep(poll_s)


def _boss(args: argparse.Namespace) -> int:
    _require_container()
    boss = ConfigRepository().get(args.boss)
    lifecycle = LifecycleLogger(args.verbose)
    backend = _new_backend(args.instance, lifecycle=lifecycle)
    try:
        initial = backend.reset(boss, boss.save_state(args.difficulty))
        lifecycle.ready(
            f"{boss.boss_id} is ready on {args.instance}; fight through VNC. "
            "Monitoring player and boss health..."
        )
        _report_combat(lifecycle, args.instance, 0, initial)
        try:
            outcome, final = _wait_for_outcome(
                backend,
                boss,
                poll_s=args.poll_interval,
                status_interval_s=args.status_interval,
                lifecycle=lifecycle,
            )
        except KeyboardInterrupt:
            backend.return_to_menu(timeout_s=args.menu_timeout)
            print(f"{boss.boss_id} interrupted; {args.instance} is back at the title menu")
            return 0

        try:
            backend.finish(boss, outcome)
        finally:
            # Boss-specific cleanup reaches the title flow; verify and finish
            # navigation so both wins and deaths end at Continue/New Game.
            backend.return_to_menu(timeout_s=args.menu_timeout)

        state = final.state
        print(
            json.dumps(
                {
                    "boss": boss.boss_id,
                    "boss_hp": state.boss_hp,
                    "boss_hps": list(state.boss_hps or (state.boss_hp,)),
                    "difficulty": args.difficulty,
                    "instance": args.instance,
                    "player_hp": state.player_hp,
                    "result": outcome.reason,
                    "samples": outcome.steps,
                    "status": "complete",
                    "win": outcome.win,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        backend.close()


def _menu(args: argparse.Namespace) -> int:
    _require_container()
    backend = _new_backend(args.instance, lifecycle=LifecycleLogger(args.verbose))
    try:
        backend.return_to_menu(timeout_s=args.timeout)
    finally:
        backend.close()
    print(f"{args.instance} is at the title menu")
    return 0


def _list(args: argparse.Namespace) -> int:
    repository = ConfigRepository()
    bosses = repository.list_bosses(include_experimental=args.include_experimental)
    if args.json_output:
        print(json.dumps(list(bosses), indent=2))
    else:
        for boss in bosses:
            print(boss)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsle-dev",
        description="Load boss encounters in an already running DSLE development instance.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--quiet",
        action="store_false",
        dest="verbose",
        default=_verbose_from_environment(),
        help="suppress setup and combat telemetry",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    boss = subparsers.add_parser(
        "boss",
        help="run a manual boss fight and return to menu on death or victory",
    )
    boss.add_argument("boss", help="boss id or configured alias")
    boss.add_argument(
        "--difficulty",
        choices=("standard", "boosted"),
        default="standard",
    )
    boss.add_argument("--instance", default="dsr-1")
    boss.add_argument("--poll-interval", type=float, default=0.1, help=argparse.SUPPRESS)
    boss.add_argument("--status-interval", type=float, default=1.0)
    boss.add_argument("--menu-timeout", type=float, default=60.0, help=argparse.SUPPRESS)
    boss.set_defaults(handler=_boss)

    menu = subparsers.add_parser("menu", help="return the running game to its title menu")
    menu.add_argument("--instance", default="dsr-1")
    menu.add_argument("--timeout", type=float, default=30.0)
    menu.set_defaults(handler=_menu)

    listing = subparsers.add_parser("list", help="list boss ids from the live source mount")
    listing.add_argument("--all", action="store_true", dest="include_experimental")
    listing.add_argument("--json", action="store_true", dest="json_output")
    listing.set_defaults(handler=_list)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (DSLEError, KeyError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"dsle-dev: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
