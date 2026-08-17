#!/usr/bin/env python3
"""The paper's small reactive expert-system reference baseline."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from pathlib import Path

try:  # Support both ``python examples/expert_agent.py`` and module imports.
    from .common import default_output_dir, prepare_output_file, run_episodes
except ImportError:  # pragma: no cover - exercised by direct script execution
    from common import default_output_dir, prepare_output_file, run_episodes

import dsle
from dsle import Action


class ExpertPolicy:
    """Heal below a threshold; otherwise alternate forward and light attack."""

    def __init__(
        self,
        heal_threshold: float = 0.4,
        heal_cooldown_s: float = 1.6,
        max_heals: int = 5,
        *,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.heal_threshold = float(heal_threshold)
        self.heal_cooldown_s = float(heal_cooldown_s)
        self.max_heals = int(max_heals)
        if not math.isfinite(self.heal_threshold) or not 0.0 < self.heal_threshold < 1.0:
            raise ValueError("heal_threshold must be finite and in (0, 1)")
        if not math.isfinite(self.heal_cooldown_s) or self.heal_cooldown_s < 0.0:
            raise ValueError("heal_cooldown_s must be finite and non-negative")
        if self.max_heals < 0:
            raise ValueError("max_heals cannot be negative")
        self._sleep = sleeper
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._heals_attempted = 0
        self._heals_committed = 0
        self._pending_heal = False
        self._cooldown_waited = False
        self._hp_at_dispatch = 0
        self._max_hp_seen = 1

    def act(self, _observation, info: dict[str, object]) -> int:
        hp = info.get("player_hp")
        hp_max = info.get("player_hp_max")
        if isinstance(hp_max, int) and hp_max > self._max_hp_seen:
            self._max_hp_seen = hp_max
        hp_fraction = (hp / self._max_hp_seen) if isinstance(hp, int) else 1.0

        # The transition returned immediately after HEAL can already contain
        # the restored HP. Confirm it before entering the fallback wait path;
        # delaying this check by another combat transition lets incoming damage
        # hide a successful flask use.
        if self._pending_heal and isinstance(hp, int) and hp > self._hp_at_dispatch:
            self._heals_committed += 1
            self._pending_heal = False
            self._cooldown_waited = False

        # DSLE intentionally has no no-op in its 14 policy actions. Wait in
        # wall-clock time for the drink animation, then send one normal combat
        # action so the next Gym transition refreshes HP before deciding whether
        # the flask actually landed.
        if self._pending_heal and not self._cooldown_waited:
            self._sleep(self.heal_cooldown_s)
            self._cooldown_waited = True
            action = Action.MOVE_FORWARD if self._step % 2 == 0 else Action.LIGHT_ATTACK
        else:
            if self._pending_heal:
                self._pending_heal = False
                self._cooldown_waited = False

            if hp_fraction < self.heal_threshold and self._heals_committed < self.max_heals:
                self._heals_attempted += 1
                self._hp_at_dispatch = hp if isinstance(hp, int) else 0
                self._pending_heal = True
                self._cooldown_waited = False
                action = Action.HEAL
            else:
                action = Action.MOVE_FORWARD if self._step % 2 == 0 else Action.LIGHT_ATTACK
        self._step += 1
        return int(action)

    @property
    def heals_attempted(self) -> int:
        return self._heals_attempted

    @property
    def heals_committed(self) -> int:
        return self._heals_committed


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid values before allocating a live game environment."""

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")


def run(args: argparse.Namespace):
    validate_args(args)
    policy = ExpertPolicy(
        args.heal_threshold,
        args.heal_cooldown_seconds,
        args.max_heals,
    )
    output = prepare_output_file(args.output)
    env = dsle.make(
        args.boss,
        instance=args.instance,
        difficulty=args.difficulty,
        max_steps=args.max_steps,
    )
    return run_episodes(
        env,
        policy,
        args.episodes,
        output,
        metadata={
            "algorithm": "expert",
            "phase": "evaluate",
            "boss": args.boss,
            "instance": args.instance,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boss", default="asylum_demon", choices=dsle.list_bosses())
    parser.add_argument("--instance", default="dsr-1")
    parser.add_argument("--difficulty", default="standard", choices=("standard", "boosted"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=7200)
    parser.add_argument("--heal-threshold", type=float, default=0.4)
    parser.add_argument(
        "--heal-cooldown-seconds",
        type=float,
        default=1.6,
        help="Wall-clock wait after sending heal (the action space has no no-op)",
    )
    parser.add_argument("--max-heals", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_dir() / "expert.jsonl",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
