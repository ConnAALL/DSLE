# Python API guide

DSLE exposes a Gymnasium-compatible environment for Dark Souls: Remastered
boss encounters. The normal host-side API starts an owned NVIDIA container,
starts one or more isolated game processes, and connects Gymnasium objects to
the authoritative environments inside that container.

The commercial game is never part of the Python package or container image. A
legally obtained installation is attached as a read-only bind mount whenever a
managed container starts.

See [host setup and permissions](SETUP.md) before using a live environment.

## Public API at a glance

The package root exports the stable, commonly used interface:

```python
import dsle

dsle.make
dsle.DarkSoulsEnv
dsle.Action
dsle.ACTION_SPECS
dsle.action_names
dsle.RewardConfig
dsle.list_bosses
dsle.load_boss
dsle.__version__
```

Container lifecycle helpers, wrappers, vector helpers, suites, and exceptions
are available from their respective modules:

```python
from dsle.container import ContainerSession, default_output_directory
from dsle.exceptions import DSLEError
from dsle.suites import resolve_suite
from dsle.vector import make_vector_env
from dsle.wrappers import FrameStack, ResizeObservation
```

For ordinary applications, prefer `dsle.make(...)`. It selects the correct
host/container transport and ensures that owned resources are cleaned up.

## First environment

The prebuilt `dsle-runtime:0.1.0` image must exist before this code runs. The
setup guide explains how to build it.

```python
import dsle

env = dsle.make(
    "asylum_demon",
    game_dir="/games/Dark.Souls.Remastered.v1.04",
    output_dir="runs/asylum-random",
    verbose=True,
)

try:
    observation, info = env.reset(seed=42)

    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    print(info["win"], info["terminated_reason"], info["truncated_reason"])
finally:
    env.close()
```

Closing a managed environment stops its game processes and removes its owned
container. The prebuilt image, game installation, and host output directory
remain. Always use `try/finally` or the managed context manager:

```python
with dsle.make("capra_demon", game_dir="/games/dsr") as env:
    observation, info = env.reset()
    # Train or evaluate here.
```

## Runtime selection

`dsle.make` supports four runtime selectors.

| Runtime | Use case | Container ownership | Instance ownership |
|---|---|---:|---:|
| `auto` | Recommended default | Managed on a host; none inside DSLE's container | Follows selected runtime |
| `managed` | Explicit host-managed execution | Starts and removes one container | Starts and stops requested instances |
| `external` | Attach to a separately started DSLE server | None | Optional through `start_instance` |
| `local` | Code already running inside the supported container | None | None |

`auto` selects `local` when `DSLE_CONTAINER=1` or when a test backend is
injected. Otherwise it selects `managed`.

### Managed runtime

Managed mode is the normal public interface:

```python
env = dsle.make(
    "nito",
    runtime="managed",
    game_dir="/games/dsr",
    image="dsle-runtime:0.1.0",
    output_dir="runs/nito-001",
)
```

It uses only an existing local image. Environment construction never builds or
pulls an image implicitly. The created container is ephemeral and runs with
`--rm`; experiment output is a separate writable host bind mount.

If `game_dir` is omitted, managed mode checks, in order:

1. `DSLE_GAME_DIR`.
2. `Dark.Souls.Remastered.v1.04/` in the current working directory.
3. `game/` in the current working directory.

If `output_dir` is omitted, a unique directory is created beneath the user's
platform data directory. The resolved directory is exposed as `env.output_dir`.

### External runtime

Use external mode when the runtime container is managed separately, for
example with `scripts/start.sh`:

```bash
./scripts/start.sh --game-dir /games/dsr --instances 1
```

Then connect through the authenticated Unix socket in that deployment's state
directory:

```python
import dsle

env = dsle.make(
    "iron_golem",
    runtime="external",
    socket_path=".dsle-state/run/dsle.sock",
    token_file=".dsle-state/run/dsle.token",
    start_instance=False,
    instance="dsr-1",
)

try:
    observation, info = env.reset()
finally:
    env.close()  # Closes this proxy, not the external container.
```

Set `start_instance=True` only when the external server is running but the
selected game instance has not been started. External environments never stop
or remove the external container when closed.

### Local runtime

Local mode constructs `DarkSoulsEnv` in the current process. It is intended
for code executing inside the DSLE runtime after the instance controller has
already started the named game process:

```python
env = dsle.make(
    "taurus_demon",
    runtime="local",
    start_instance=False,
    instance="dsr-1",
)
```

Local mode does not accept host container options such as `game_dir`,
`output_dir`, `image`, `socket_path`, or `token_file`. It also does not own the
game-process lifecycle.

## `dsle.make` reference

```python
dsle.make(
    boss,
    *,
    runtime="auto",
    game_dir=None,
    output_dir=None,
    image=None,
    container_name=None,
    num_instances=1,
    socket_path=None,
    token_file=None,
    start_instance=None,
    instance_mode="headless",
    verbose=True,
    **environment_options,
)
```

### Runtime parameters

| Parameter | Meaning |
|---|---|
| `boss` | Boss ID or configured alias. Use `dsle.list_bosses()` to discover IDs. |
| `runtime` | One of `auto`, `managed`, `external`, or `local`. |
| `game_dir` | Host game installation for managed mode. It is mounted read-only. |
| `output_dir` | Persistent host state/results directory for managed mode. |
| `image` | Existing local runtime image; defaults to `DSLE_IMAGE` or `dsle-runtime:0.1.0`. |
| `container_name` | Optional explicit name for the owned managed container. |
| `num_instances` | Number of isolated games, from 1 through 30. Values above 1 return a Gymnasium vector environment. |
| `socket_path` | External runtime server's Unix socket. |
| `token_file` | Private external runtime authentication-token file. |
| `start_instance` | Whether the selected runtime should start the game process. The default follows runtime ownership. |
| `instance_mode` | `headless`, `headless-vnc`, or, for supported external workflows, `gui`. Managed mode accepts only the first two. |
| `verbose` | Enables Rich lifecycle, setup, cleanup, write, and combat telemetry. Defaults to `True`. |

### Environment parameters

Extra keyword arguments configure the authoritative `DarkSoulsEnv`:

| Parameter | Default | Meaning |
|---|---:|---|
| `instance` | `dsr-1` | Isolated game process to control. Multi-instance construction assigns `dsr-1` through `dsr-N` automatically. |
| `difficulty` | `standard` | Initial save profile. Current boss configs provide `standard` and `boosted`. |
| `obs_mode` | `grayscale` | `grayscale`, `rgb`, or `state_only`. |
| `action_ms` | `250` | Milliseconds to hold each policy action; zero is accepted. |
| `action_repeat` | `1` | Number of real action/observation transitions performed for one API call. |
| `max_steps` | boss default | Maximum real transitions before a `max_steps` truncation. |
| `auto_lock_on` | `True` | Repairs lock-on when the runtime can confirm that it was lost. |
| `lock_on_interval` | `1` | Number of real transitions between lock-on checks. |
| `reward` | boss default | A `dsle.RewardConfig` override. |
| `render_mode` | `None` | Set to `rgb_array` to enable `render()`. |

`config_dir`, `asset_dir`, `instance_config`, `backend`, and `backend_factory`
exist for local development and testing. Managed and external RPC modes reject
host paths because those paths are not automatically mounted into the runtime.

External RPC construction also accepts `connect_timeout` and `request_timeout`
as environment options. Both are finite positive seconds.

## Boss discovery and configuration

List supported boss IDs:

```python
import dsle

for boss_id in dsle.list_bosses():
    print(boss_id)
```

The current supported IDs are:

```text
asylum_demon
bed_of_chaos
bell_gargoyles
capra_demon
ceaseless_discharge
centipede_demon
chaos_witch_quelaag
crossbreed_priscilla
dark_sun_gwyndolin
demon_firesage
four_kings
gaping_dragon
great_grey_wolf_sif
gwyn_lord_of_cinder
iron_golem
moonlight_butterfly
nito
ornstein_and_smough
pinwheel
seath_the_scaleless
stray_demon
taurus_demon
```

`load_boss` returns the immutable validated configuration object:

```python
config = dsle.load_boss("ornstein_and_smough")
print(config.display_name)
print(dict(config.save_states))
print(len(config.memory.hp_chains))
print(config.setup)
```

Unknown names include a nearest-match suggestion when one is available.

Boss YAML files define save profiles, readiness, setup operations, cleanup
operations, memory chains, victory signals, reward weights, and limits. They do
not contain agent or learning-algorithm code.

## Gymnasium lifecycle

### `reset`

```python
observation, info = env.reset(seed=42)
```

`reset` loads the boss save, executes the YAML setup sequence, verifies valid
player and boss state, optionally repairs lock-on, and returns control at the
configured fight-ready point.

Change difficulty between episodes without rebuilding the environment:

```python
observation, info = env.reset(options={"difficulty": "boosted"})
```

No other reset option is currently accepted. Gymnasium seeds affect action
space sampling and agent-side randomness; they cannot seed the external game's
AI, animation, or timing randomness. Equal seeds and actions therefore do not
guarantee identical trajectories.

### `step`

```python
observation, reward, terminated, truncated, info = env.step(action)
```

`action` must be an integer in the discrete action space. A step performs the
input, captures the matching game state and frame, computes reward, checks
terminal conditions, and runs boss cleanup after a completed episode.

| Condition | `terminated` | `truncated` | Reason |
|---|---:|---:|---|
| Verified boss victory | `True` | `False` | `boss_defeated` |
| Player death | `True` | `False` | `player_dead` |
| Step limit | `False` | `True` | `max_steps` |
| Capture, memory, or runtime failure | `False` | `True` | `runtime_error` |

Call `reset()` before the first `step()` and again after every terminated or
truncated episode.

### `close`

`close()` is idempotent. For managed environments it closes all proxies,
requests graceful instance shutdown, stops and removes the owned container,
and restores host ownership of persistent output. It does not remove the image
or output directory.

An `atexit` safety handler attempts cleanup if a managed session is still open,
but applications should not rely on interpreter shutdown for normal lifecycle
management.

## Observation spaces

DSLE uses channel-first visual observations.

| Mode | Shape | Type | Contents |
|---|---|---|---|
| `grayscale` | `(1, 600, 800)` | `numpy.uint8` | One grayscale game frame. |
| `rgb` | `(3, 600, 800)` | `numpy.uint8` | RGB channels in CHW order. |
| `state_only` | `(2,)` | `numpy.float32` | `[player_hp_fraction, boss_hp_fraction]`, clipped to `[0, 1]`. |

The configured frame size comes from `config/defaults.yaml`. Diagnostic state
is returned in `info` for all observation modes.

## Action space

Every boss uses the same `gymnasium.spaces.Discrete(14)` interface.

| Value | Enum | Name | Input |
|---:|---|---|---|
| 0 | `MOVE_FORWARD` | `move_forward` | W |
| 1 | `MOVE_LEFT` | `move_left` | A |
| 2 | `MOVE_BACKWARD` | `move_backward` | S |
| 3 | `MOVE_RIGHT` | `move_right` | D |
| 4 | `LIGHT_ATTACK` | `light_attack` | Left mouse |
| 5 | `STRONG_ATTACK` | `strong_attack` | Left Shift + left mouse |
| 6 | `HEAL` | `heal` | R |
| 7 | `BACKSTEP` | `backstep` | Space |
| 8 | `ROLL_FORWARD` | `roll_forward` | W + Space |
| 9 | `ROLL_LEFT` | `roll_left` | A + Space |
| 10 | `ROLL_BACKWARD` | `roll_backward` | S + Space |
| 11 | `ROLL_RIGHT` | `roll_right` | D + Space |
| 12 | `LEFT_CLICK` | `left_click` | Left mouse |
| 13 | `RIGHT_CLICK` | `right_click` | Right mouse |

Use named actions to avoid numeric literals:

```python
from dsle import Action

observation, reward, terminated, truncated, info = env.step(Action.ROLL_LEFT)
```

Programmatic discovery is also available:

```python
print(dsle.action_names())
print(dsle.ACTION_SPECS[dsle.Action.STRONG_ATTACK])
```

Lock-on and menu interaction inputs are orchestration operations, not policy
actions, so they are intentionally absent from the 14-action space.

## The `info` dictionary

`reset`, `step`, `observe`, and successful health writes return a consistent
diagnostic mapping.

| Key | Meaning |
|---|---|
| `boss` | Canonical boss ID. |
| `instance` | Instance name such as `dsr-1`. |
| `difficulty` | Active save profile. |
| `step` | Real transition count in the current episode. |
| `player_hp`, `player_hp_max` | Current and maximum player health. |
| `player_location` | `{"x": ..., "y": ..., "z": ...}` diagnostic coordinates. |
| `boss_hp`, `boss_hp_max` | Aggregate/primary boss health and maximum. |
| `boss_hps`, `boss_hp_maxes` | One entry for every configured health location. |
| `death_count` | Current player death counter when readable. |
| `locked_on` | Lock-on state when readable. |
| `player_state_valid`, `boss_state_valid` | Whether the corresponding memory read is valid. |
| `boss_damage_dealt` | Non-negative boss HP delta used for this reward. |
| `player_damage_taken` | Non-negative player HP delta used for this reward. |
| `boss_defeated`, `player_dead` | Raw verified outcome signals. |
| `win` | True only when this transition terminates as `boss_defeated`. |
| `terminated_reason`, `truncated_reason` | Episode outcome strings or `None`. |
| `elapsed_seconds` | Monotonic wall time since reset completed. |

On a runtime-error truncation, `info["runtime_error"]` contains the exception
type and message. If post-episode cleanup fails, `info["cleanup_error"]` is
present without changing the completed transition.

For encounters with multiple health locations, use `boss_hps` rather than
assuming one bar. Ornstein and Smough, for example, exposes both `boss1_hp` and
`boss2_hp` through this list and verbose telemetry.

## Reward configuration

The transition reward is the sum of:

```text
boss_damage_weight * (nonnegative boss damage / boss maximum)
+ player_damage_weight * (nonnegative player damage / player maximum)
+ step_penalty
+ one applicable terminal or truncation reward
```

Missing or invalid memory reads are never interpreted as damage. Action repeat
accumulates each real transition's damage and step penalty, then applies the
terminal component once.

Override weights with `RewardConfig`:

```python
reward = dsle.RewardConfig(
    boss_damage=2.0,
    player_damage=-0.5,
    step_penalty=-0.002,
    win_bonus=50.0,
    death_penalty=-20.0,
    timeout_penalty=-2.0,
    runtime_error_penalty=-20.0,
)

env = dsle.make("gaping_dragon", game_dir="/games/dsr", reward=reward)
```

Every weight must be a finite number.

## Passive observation and manual control

`observe()` refreshes the frame and state without injecting a policy action or
incrementing the Gymnasium step counter:

```python
observation, info = env.observe()
```

It requires an active episode and is intended for viewers, recorders, and
human-driven diagnostics. Learning agents should use `step()` for transitions.

`return_to_menu()` ends manual control and waits for the title menu to be
visually verified:

```python
env.return_to_menu(timeout_s=60.0)
```

This is useful when one long-lived game instance is reused across several boss
demonstrations.

## Health writers

Health writers are explicit diagnostic/intervention APIs. They require a
successful `reset()` and an active episode, validate the requested value,
perform an immediate memory readback, and return refreshed `info`.

```python
observation, info = env.reset()

info = env.set_player_hp(info["player_hp_max"])
info = env.set_boss_hp(1, 100)
```

Boss numbers are one-based and match `boss_hps`/`boss_hp_maxes` positions. For
a multi-health encounter:

```python
info = env.set_boss_hp(2, info["boss_hp_maxes"][1])
```

Zero every configured boss health location atomically:

```python
info = env.set_boss_hp(-1, 0)
```

`-1` is valid only with health zero. The operation first verifies that every
configured health location is readable; if validation fails, none are written.
Setting HP to zero changes health only and does not forge the independent
boss-defeated event flag.

Bed of Chaos derives binary health from its defeated flag: 1 before defeat and
0 afterward. That value is read-only, so `set_boss_hp` is unavailable for that
encounter.

## Rendering and episode recording

Enable `render()` when creating the environment:

```python
env = dsle.make(
    "moonlight_butterfly",
    game_dir="/games/dsr",
    render_mode="rgb_array",
)

try:
    observation, info = env.reset()
    frame = env.render()  # HWC RGB uint8, or None before reset.
finally:
    env.close()
```

`RecordBossEpisode` writes compressed RGB frame chunks and a JSON episode
summary without requiring a video codec:

```python
from dsle.wrappers import RecordBossEpisode

base = dsle.make(
    "asylum_demon",
    game_dir="/games/dsr",
    render_mode="rgb_array",
)
env = RecordBossEpisode(base, "runs/frames", every=1, chunk_frames=16)
```

Each recorded episode produces `episode-NNNNNN.json` and one or more
`episode-NNNNNN-frames-NNNNNN.npz` chunks. Calling `close()` during an active
episode finalizes it with status `interrupted`.

## Multiple instances

Set `num_instances` from 2 through 30 to start an isolated pool in one managed
container:

```python
import dsle

envs = dsle.make(
    "asylum_demon",
    game_dir="/games/dsr",
    output_dir="runs/asylum-vector",
    num_instances=4,
)

try:
    observations, infos = envs.reset(seed=42)

    for _rollout_step in range(128):
        actions = envs.action_space.sample()
        observations, rewards, terminated, truncated, infos = envs.step(actions)
finally:
    envs.close()
```

The returned object is a synchronous Gymnasium vector environment. The example
uses a fixed rollout length because sub-environments can finish at different
times and follow Gymnasium's vector autoreset behavior. Instances
are deterministically named `dsr-1` through `dsr-N`. Each uses its own X11
display, Wine prefix, save directory, runtime directory, and optional VNC port.
They share the read-only game installation and immutable container layers.

Algorithms that need strict
per-candidate episode boundaries should track the vector terminal arrays or
use one proxy per instance as demonstrated by `examples/scope_parallel.py`.

### VNC-enabled managed pools

```python
envs = dsle.make(
    "asylum_demon",
    game_dir="/games/dsr",
    num_instances=4,
    instance_mode="headless-vnc",
)

print(envs.vnc_ports)
# Mapping from container ports 5901..5904 to dynamically assigned host ports.
```

Published VNC endpoints bind to `127.0.0.1` only. Pass the mappings to
`dsle-viewer --endpoint ...` to view a managed pool.

## Advanced container sessions

`ContainerSession` is useful when one container and one game process must be
reused across several different boss environment objects:

```python
from dsle.container import ContainerSession

session = ContainerSession(
    image="dsle-runtime:0.1.0",
    game_dir="/games/dsr",
    output_dir="runs/sequential-bosses",
    vnc_ports=(5901,),
)

try:
    session.start()
    session.start_instance("dsr-1", mode="headless-vnc")

    for boss in ("asylum_demon", "taurus_demon", "capra_demon"):
        env = session.make(
            boss,
            instance="dsr-1",
            start_instance=False,
            obs_mode="state_only",
        )
        try:
            observation, info = env.reset()
            # Run or passively monitor this fight.
            env.return_to_menu(timeout_s=60.0)
        finally:
            env.close()
finally:
    session.close()
```

One output directory cannot be shared by two live sessions. The container
entrypoint holds an exclusive lock to prevent concurrent Wine prefixes, save
slots, sockets, and metadata from corrupting each other.

## Preprocessing wrappers

DSLE includes small algorithm-independent Gymnasium wrappers:

```python
from dsle.wrappers import (
    ClipReward,
    FrameStack,
    ResizeObservation,
    ScaleReward,
    make_atari_style,
)
```

| Wrapper | Purpose |
|---|---|
| `ResizeObservation(env, (84, 84))` | Resize a CHW visual observation. Uses OpenCV when installed and a nearest-neighbor core fallback otherwise. |
| `FrameStack(env, count=4)` | Concatenate recent CHW observations on the channel axis. |
| `ClipReward(env)` | Clamp reward to `[-1, 1]`. |
| `ScaleReward(env, scale)` | Multiply reward by a finite constant. |
| `make_atari_style(boss, **kwargs)` | Create grayscale, 84×84, four-frame-stacked, clipped-reward single environment. |

Example:

```python
from dsle.wrappers import FrameStack, ResizeObservation

base = dsle.make("asylum_demon", game_dir="/games/dsr")
env = FrameStack(ResizeObservation(base, (84, 84)), count=4)
```

These image wrappers require a channel-first visual mode, not `state_only`.

## Benchmark suites

Named subsets live in `dsle.suites`:

```python
from dsle.suites import resolve_suite

print(resolve_suite("dsle5"))
print(resolve_suite("melee"))
print(resolve_suite("ranged"))
print(resolve_suite("multi"))
print(resolve_suite("full"))
```

Suites are boss ID collections only. They do not construct environments or
contain policies.

## Verbose lifecycle output

Verbose output is enabled by default. Messages are written to stderr and
include stages such as:

- `[CHECK]` dependency and input validation.
- `[START]` container, game, reset, and operation startup.
- `[READY]` successful readiness milestones.
- `[SETUP]` individual declarative setup operations.
- `[COMBAT]` player health/location, every boss health, and outcome flags.
- `[WRITE]` explicit health interventions.
- `[CLEANUP]` individual declarative cleanup operations.
- `[DONE]`, `[STOP]`, and `[ERROR]` lifecycle outcomes.

Disable non-error telemetry per environment:

```python
env = dsle.make("asylum_demon", game_dir="/games/dsr", verbose=False)
```

Shell scripts also accept `--quiet`, and `DSLE_VERBOSE=0` is available for
automation.

## Exceptions

Catch `DSLEError` when an application wants one boundary for DSLE-specific
failures:

```python
from dsle.exceptions import DSLEError

try:
    env = dsle.make("asylum_demon", game_dir="/games/dsr")
except DSLEError as exc:
    print(f"DSLE could not start: {exc}")
```

| Exception | Meaning |
|---|---|
| `ConfigurationError` | Invalid YAML, path, output layout, or runtime option. |
| `AssetError` | Required save or screenshot-template asset is unavailable. |
| `RuntimeUnavailableError` | Docker, GPU, image, game build, process, or live backend is unavailable. |
| `RuntimeCommunicationError` | Capture, memory, input, or RPC communication failed. |
| `ResetError` | A boss could not reach a validated episode start. |

Ordinary `ValueError`, `TypeError`, and `RuntimeError` are used for immediate
API contract violations such as invalid arguments, stepping before reset, or
using an already closed environment.

## Baseline examples

Learning algorithms intentionally live outside the environment package. The
`examples/` directory contains SCOPE, parallel SCOPE, PPO, DQN, random, and
expert consumers of this API. See [the examples guide](../examples/README.md)
for commands and dependency groups.
