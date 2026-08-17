# Dark Souls Learning Environment

**A Gymnasium-compatible research platform for learning agents in Dark Souls:
Remastered boss encounters.**

Current release: **v0.1.0**.

[Paper](https://arxiv.org/abs/2608.09902v1) ·
[Python API](docs/API.md) ·
[Setup guide](docs/SETUP.md) ·
[Examples](examples/README.md) ·
[License](LICENSE)

The Dark Souls Learning Environment (DSLE) presents all 22 base-game boss
encounters as reproducible learning tasks. Each environment step executes a
real action in a running copy of the game and returns a visual or diagnostic
observation through the familiar Gymnasium API.

DSLE separates agent code from game orchestration. Boss behavior is described
with validated YAML configurations, while an NVIDIA container provides Wine,
DXVK, isolated game processes, save restoration, visual capture, and
process-memory instrumentation.

The environment and benchmark are introduced in
[**DSLE: A Learning Environment for Dark Souls Boss Encounters**](https://arxiv.org/abs/2608.09902v1).
If you use DSLE in research, please cite the paper using the entry in
[Citing DSLE](#citing-dsle).

> [!IMPORTANT]
> Dark Souls: Remastered is not included. Users must supply their own legally
> obtained, supported installation. DSLE attaches that directory as a
> read-only runtime mount and never copies it into the container image.

## Features

- All 22 base-game boss encounters, each with standard and boosted scenario
  saves.
- Gymnasium-compatible `reset()` and `step()` behavior.
- Grayscale, RGB, and compact health-state observations.
- A stable 14-action space covering movement, attacks, dodges, and healing.
- Declarative YAML setup, readiness, memory, victory, and cleanup rules.
- Automatic managed-container startup and cleanup from the Python API.
- One to 30 isolated game instances in a shared NVIDIA container.
- Persistent host-side outputs that survive container removal.
- Passive VNC viewing for one instance or a sampling-controlled CCTV grid.
- SCOPE, PPO, DQN, random, and expert baselines kept outside the environment
  package.
- Unit, contract, container, live-game, screenshot-sweep, and recording tests.

## Quick start

### Requirements

DSLE currently supports Linux on x86-64 with:

- Python 3.10 or newer;
- an NVIDIA GPU and driver with Vulkan support;
- Docker Engine with Docker Compose;
- NVIDIA Container Toolkit; and
- the supported Dark Souls: Remastered installation.

One instance requires at least 8 GiB of system RAM. Allow roughly 3 GiB for
each additional live instance and at least 30 GiB of free disk space for the
core image and runtime state. See the
[setup, dependency, and permission guide](docs/SETUP.md) before the first run.

### Install DSLE

From an extracted source directory or Git checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
./scripts/build.sh --core
```

The build creates the local `dsle-runtime:0.1.0` image. Normal API calls use
that prebuilt image and do not build or pull images implicitly.

Place the game directory at either `Dark.Souls.Remastered.v1.04/` or
`game/` in the current directory, set `DSLE_GAME_DIR`, or pass
`game_dir=...` explicitly.

### Use the Gymnasium API

```python
import dsle

env = dsle.make(
    "asylum_demon",
    game_dir="/games/Dark.Souls.Remastered.v1.04",
)

try:
    observation, info = env.reset()

    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
finally:
    env.close()
```

By default, `dsle.make()` starts and owns an ephemeral container. Calling
`close()` stops the game, removes that container, and retains the prebuilt
image and persistent output directory. Always close environments, preferably
with `try/finally` or a context manager.

## Runtime modes

DSLE supports three execution arrangements through one constructor:

| Runtime | Intended use | Container ownership |
|---|---|---|
| `managed` | Ordinary host-side Python programs | DSLE starts and removes it |
| `external` | Programs attaching to an existing DSLE deployment | Caller owns it |
| `local` | Code already running inside the runtime container | Caller owns it |

`runtime="auto"` is the default. It selects managed mode on a normal host and
local mode inside the DSLE container. The full constructor and mode-specific
options are documented in the [Python API guide](docs/API.md).

Lifecycle and combat telemetry are enabled by default. Messages report
dependency checks, container and instance startup, setup operations, player
location, player health, every configured boss health value, victory, death,
and cleanup. Pass `verbose=False` to silence informational output without
hiding errors.

## Parallel environments

Set `num_instances` to create isolated games named `dsr-1` through
`dsr-N`. They share one container and game installation but have separate
displays, Wine prefixes, saves, and runtime state.

```python
import dsle

envs = dsle.make(
    "asylum_demon",
    game_dir="/games/Dark.Souls.Remastered.v1.04",
    num_instances=4,
)

try:
    observations, infos = envs.reset()
    actions = envs.action_space.sample()
    observations, rewards, terminated, truncated, infos = envs.step(actions)
finally:
    envs.close()
```

The [parallel SCOPE example](examples/scope_parallel.py) shows how to distribute
a population across a managed instance pool while preserving candidate order
and training artifacts.

## Manual container workflow

The repository scripts expose the same runtime for interactive work and
in-container training:

```bash
./scripts/start.sh --game-dir /games/Dark.Souls.Remastered.v1.04 --instances 4
./scripts/status.sh
./scripts/exec.sh python3 examples/random_agent.py --boss asylum_demon
./scripts/stop.sh
```

`start.sh` launches VNC-enabled instances. `stop.sh` removes the container
and network while preserving the image, Wine prefixes, logs, recordings, and
results beneath the selected state directory.

## Interactive boss development

The development workflow mounts the repository read-only into one container,
so Python, YAML, save, and template changes are available to each new command
without rebuilding the image:

```bash
./scripts/dev.sh start --game-dir /games/Dark.Souls.Remastered.v1.04
./scripts/dev.sh boss asylum_demon
./scripts/dev.sh boss capra_demon --difficulty boosted
./scripts/dev.sh menu
./scripts/dev.sh stop
```

Starting without `--boss` leaves the game at the main menu. If TigerVNC
Viewer is installed, DSLE opens the loopback VNC session automatically; use
`--no-vnc-viewer` when no window is wanted.

## Multi-instance CCTV viewer

The optional CCTV viewer passively samples up to 30 VNC feeds in one window.
It never sends keyboard, mouse, focus, reset, or environment-control commands.

```bash
sudo apt install python3-tk
python -m pip install '.[viewer]'
./scripts/start.sh --instances 4
./scripts/view-instances.sh --period 1.0
```

Each tile is labeled with its instance name. Sampling period, displayed
instances, grid columns, and explicit host-port mappings can be configured from
the command line.

## Agent examples

Learning algorithms are consumers of DSLE rather than part of the environment
package. The [examples directory](examples/README.md) contains:

- a uniform random policy;
- a reactive expert policy;
- PPO and DQN visual baselines;
- serial SCOPE with CMA-ES;
- parallel SCOPE over API-managed instances; and
- a single-instance boss-showcase recorder.

Install their optional dependencies with:

```bash
python -m pip install '.[examples]'
./scripts/build.sh --research
```

The research image adds CUDA PyTorch, SciPy, and CMA-ES while retaining the same
external, read-only game mount.

## Validation and recordings

Run host preflight checks before allocating a game process:

```bash
dsle-doctor --host --game-dir /games/Dark.Souls.Remastered.v1.04 --require-game
```

Load every regular boss and save a lossless screenshot 10 seconds after each
fight begins:

```bash
./scripts/boss-screenshot-sweep.sh --game-dir /games/Dark.Souls.Remastered.v1.04
```

Record all supported bosses as one continuous 30 FPS MP4, with a verified menu
pause between encounters:

```bash
DSLE_PYTHON=.venv/bin/python ./scripts/record-boss-showcase.sh --game-dir /games/Dark.Souls.Remastered.v1.04 --output-dir runs/boss-showcase
```

Both workflows preserve their output and remove only the containers they own.
The [setup guide](docs/SETUP.md#testing) documents the complete unit,
distribution, Docker, live-game, and boss-sweep test layers.

## Documentation

- [Python API](docs/API.md) — environment construction, observations, actions,
  rewards, diagnostics, health controls, wrappers, configuration, vector
  execution, and exception behavior.
- [Setup and permissions](docs/SETUP.md) — host packages, Docker, NVIDIA,
  game and output paths, required security options, validation, and
  troubleshooting.
- [Examples](examples/README.md) — baseline commands, paper defaults,
  checkpoints, output files, and parallel training.
- [Bundled assets](assets/README.md) — included scenario/template material,
  excluded commercial files, checksums, and publication review.

## Citing DSLE

If DSLE contributes to an experiment, benchmark, publication, or derived
research artifact, please cite:

```bibtex
@misc{gezgin2026dsle,
  title         = {DSLE: A Learning Environment for Dark Souls Boss Encounters},
  author        = {Gezgin, Derin and O'Connor, Jim and Goodwin, Tanner and Parker, Gary B.},
  year          = {2026},
  eprint        = {2608.09902},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2608.09902}
}
```

## License and game assets

DSLE is licensed under the
[GNU General Public License version 3 only](LICENSE). The license applies to
DSLE's code and does not grant redistribution rights for Dark Souls:
Remastered or other third-party material.

The game executable and installation are never distributed by DSLE. Review
the [bundled asset notice](assets/README.md) for the separate provenance and
publication requirements covering included scenario saves and screenshot
templates.
