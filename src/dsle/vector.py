"""Gymnasium-native construction of isolated DSLE instance pools."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv

from dsle.exceptions import ConfigurationError
from dsle.runtime.instances import validate_instance_name


def make_vector_env(
    boss: str,
    num_envs: int,
    *,
    instances: Sequence[str] | None = None,
    asynchronous: bool = True,
    **env_kwargs: Any,
) -> VectorEnv:
    """Map each Gymnasium environment to a distinct live-game instance."""

    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or not 1 <= num_envs <= 30:
        raise ValueError("num_envs must be an integer in [1, 30]")
    if not isinstance(asynchronous, bool):
        raise ValueError("asynchronous must be a boolean")
    if isinstance(instances, (str, bytes)):
        raise ValueError("instances must be a sequence of unique instance names")
    if instances is None:
        names = tuple(f"dsr-{index + 1}" for index in range(num_envs))
    else:
        try:
            names = tuple(instances)
        except TypeError as exc:
            raise ValueError("instances must be a sequence of unique instance names") from exc
    try:
        names = tuple(validate_instance_name(name) for name in names)
    except ConfigurationError as exc:
        raise ValueError("instances must contain only dsr-1 through dsr-30") from exc
    if len(names) != num_envs or len(set(names)) != num_envs:
        raise ValueError("instances must contain exactly num_envs unique names")
    if "instance" in env_kwargs:
        raise ValueError("Pass instance names through instances, not env_kwargs")
    if env_kwargs.get("backend") is not None:
        raise ValueError("Vector environments cannot share one backend object; use backend_factory")

    def factory(instance: str):
        def build():
            from dsle.env import DarkSoulsEnv

            return DarkSoulsEnv(boss, instance=instance, **env_kwargs)

        return build

    env_functions = [factory(name) for name in names]
    vector_type = AsyncVectorEnv if asynchronous else SyncVectorEnv
    return vector_type(env_functions)


__all__ = ["make_vector_env"]
