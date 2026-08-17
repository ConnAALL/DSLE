# Host setup, dependencies, and permissions

This guide prepares a Linux workstation to build the DSLE NVIDIA runtime and
run Dark Souls: Remastered through the Python API or repository scripts.

DSLE does not distribute the game. The user's legally obtained game directory
is attached to each runtime container as a read-only bind mount and is never
copied into the image.

## Supported deployment

| Component | Requirement |
|---|---|
| Host operating system | 64-bit Linux on x86-64. Ubuntu is the documented path. |
| Python | 3.10 or newer. |
| GPU | NVIDIA GPU with Vulkan support. Integrated-GPU execution is not included. |
| GPU driver | A working proprietary NVIDIA host driver; `nvidia-smi` must succeed. |
| Container engine | Docker Engine with an accessible daemon. |
| Compose | Docker Compose v2, invoked as `docker compose`. |
| GPU container support | NVIDIA Container Toolkit, registered as Docker runtime `nvidia`. |
| Game | The supported Dark Souls: Remastered benchmark executable and required asset directories. |

The runtime image supplies Wine, DXVK, Xorg, PulseAudio, VNC, Vulkan userspace
libraries, and live Python dependencies. Do not install those components on the
host merely to satisfy DSLE. The host needs the NVIDIA driver, Docker stack,
Python tooling, and any optional desktop viewers.

Rootless Docker is not part of the tested release configuration. The supported
path is the system Docker daemon with the calling user allowed to access its
Unix socket.

## Resource planning

The exact requirement depends on resolution, boss, policy, recording, and the
number of live processes. Use these as conservative starting points:

| Resource | One instance | Additional instances |
|---|---:|---:|
| System RAM | At least 8 GiB total | Budget roughly 3 GiB each |
| Shared memory | 2 GiB for the container | Shared by all processes in that container |
| Free disk | At least 30 GiB for the core build and runtime state | Wine prefixes, logs, checkpoints, and videos add to this |
| VNC ports | One only when VNC mode is enabled | One per VNC-enabled instance |

Thirty instances are supported by the orchestration API, but that maximum is
not a promise that every workstation can run 30 games. Choose a count that fits
the available RAM, GPU memory, CPU, and disk bandwidth.

## 1. Prepare the Ubuntu host

Install the small host-side base set:

```bash
sudo apt update
sudo apt install -y \
  bash \
  ca-certificates \
  coreutils \
  curl \
  findutils \
  python3 \
  python3-pip \
  python3-venv \
  util-linux
```

These packages provide Python and the standard shell tools used by the
repository launchers, including `flock`, `readlink`, `stat`, and `find`.

`git` is needed only for a Git checkout. A downloaded ZIP also works after it
is extracted into a normal folder; DSLE does not require a `.git` directory.
If an archive did not retain executable bits, either invoke a launcher with
`bash scripts/start.sh ...` or restore the launcher permissions:

```bash
chmod +x scripts/*.sh
```

## 2. Install and authorize Docker

Install Docker Engine from Docker's official repository, including
`docker-buildx-plugin` and `docker-compose-plugin`. The maintained Ubuntu
instructions are in the [Docker Engine installation guide](https://docs.docker.com/engine/install/ubuntu/).

After the official repository is configured, the relevant package set is:

```bash
sudo apt install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

sudo systemctl enable --now docker
```

Verify the client, daemon, Buildx, and Compose plugin:

```bash
docker --version
docker info
docker buildx version
docker compose version
```

### Docker socket permission

DSLE runs Docker commands as the user who starts the Python process or shell
script. That user must be able to access the Docker daemon without inserting
`sudo` in front of every internal command.

The common system-Docker configuration is:

```bash
sudo usermod -aG docker "$USER"
```

Log out and back in after changing group membership, then verify:

```bash
docker info
```

The `docker` group grants root-equivalent control of the Docker daemon. Only
trusted users should receive this membership. See Docker's
[Linux post-installation guidance](https://docs.docker.com/engine/install/linux-postinstall/)
for the security warning and supported procedure.

Do not run the DSLE Python API itself with `sudo`. Doing so creates root-owned
state and result files and prevents the ownership protections from matching the
normal user.

## 3. Install the NVIDIA driver and container runtime

Install a host NVIDIA driver appropriate for the GPU and distribution. Verify
the driver before debugging Docker:

```bash
nvidia-smi
```

Then install NVIDIA Container Toolkit using NVIDIA's maintained
[installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
After the packages are installed, register the runtime with Docker and restart
the daemon:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify that Docker advertises a runtime named `nvidia`:

```bash
docker info --format '{{json .Runtimes}}'
```

DSLE explicitly requests both `--gpus all` and `--runtime nvidia`. Merely having
`nvidia-smi` on the host is insufficient if the runtime is not registered with
Docker.

## 4. Obtain and install the Python package

Enter the extracted repository directory and create a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The core dependency set is intentionally small:

- Gymnasium for the environment interface.
- NumPy for observations and state vectors.
- platformdirs for default persistent-output paths.
- PyYAML for declarative boss configuration.
- Rich for lifecycle and combat telemetry.

The managed host API needs only the core set. Wine-facing dependencies are
already pinned in `docker/container_requirements.txt` and installed in the
runtime image.

### Optional dependency groups

Install only the features needed on the host:

```bash
# Host image/video tools and live-runtime adapters
python -m pip install -e '.[runtime]'

# SCOPE, PPO, and DQN examples
python -m pip install -e '.[examples]'

# Tests, type checks, linting, and package builds
python -m pip install -e '.[runtime,test]'

# Packaging/release validation
python -m pip install -e '.[release]'
```

For the multi-instance CCTV viewer, install both its Python extra and the
operating-system Tk binding:

```bash
sudo apt install -y python3-tk
python -m pip install -e '.[viewer]'
```

TigerVNC is optional and independent of the CCTV GUI:

```bash
sudo apt install -y tigervnc-viewer
```

When TigerVNC is installed and a graphical desktop is available,
`scripts/dev.sh start` opens its loopback VNC endpoint automatically. Use
`--no-vnc-viewer` to suppress that behavior.

## 5. Supply the game directory

The path must be the complete game installation root containing:

```text
DarkSoulsRemastered.exe
chr/
event/
map/
mtd/
param/
script/
```

The executable is fingerprinted before managed execution because DSLE's memory
reads and writes target the supported benchmark build. A different executable
can have incompatible pointer layouts even if the game appears to launch.

There are three normal ways to select the directory:

1. Pass `game_dir="/absolute/path/to/game"` to `dsle.make`.
2. Set `DSLE_GAME_DIR=/absolute/path/to/game`.
3. Put the full installation in `Dark.Souls.Remastered.v1.04/` or `game/` in
   the current working directory.

The two conventional repository-local names are excluded from Git and from the
Docker build context. Keeping the game inside the checkout is therefore
supported, but it remains a runtime bind mount rather than image content.

### Game-directory permissions

The calling user must be able to traverse the directory and read the executable
and assets. The Docker daemon must also be allowed to bind-mount that path.
The mount inside the container is read-only, so DSLE cannot update or replace
the supplied game files.

Avoid placing the game in an encrypted, network, removable, or sandboxed path
that the Docker daemon cannot access. A path containing a comma is rejected
because Docker's structured bind-mount syntax treats commas as separators.

Do not place the writable DSLE output/state directory inside the game folder,
and do not place the game folder inside the output/state directory. Both
layouts are rejected to preserve the read-only boundary.

## 6. Build the runtime image

Build the core NVIDIA image once:

```bash
./scripts/build.sh --core
```

This produces `dsle-runtime:0.1.0`. The build context is an allowlist containing
application source, declared assets, and Docker runtime files. The game
directory and `DarkSoulsRemastered.exe` are excluded even if they are stored in
the repository.

The build downloads Ubuntu packages, Wine, Winetricks, and DXVK, so the Docker
daemon needs network access during the initial build. Normal API construction
does not build, pull, or update the image.

The larger research image adds CUDA PyTorch, SciPy, and CMA-ES for the supplied
training examples:

```bash
./scripts/build.sh --research
```

It is tagged `dsle-runtime:0.1.0-research`.

## 7. Run preflight checks

Run the host doctor with the game required:

```bash
dsle-doctor --host \
  --game-dir /absolute/path/to/game \
  --require-game
```

It checks:

- Linux and Python compatibility.
- x86-64 architecture.
- available RAM and disk.
- game executable, fingerprint, and core asset directories.
- Docker CLI, daemon, and Compose.
- host NVIDIA driver and GPU visibility.
- Docker's registered NVIDIA runtime.

Machine-readable output is available for provisioning and CI:

```bash
dsle-doctor --host --game-dir /absolute/path/to/game --require-game --json
```

The managed API runs a second in-container doctor during startup. That check
verifies Wine, NVIDIA visibility, Vulkan, Xorg, x11vnc, writable state,
PulseAudio, read-only game mount status, `SYS_PTRACE`, seccomp, and AppArmor.

## 8. Smoke-test the container and game

First validate the image, GPU, mounts, server, and runtime dependencies without
launching the game:

```bash
./scripts/start.sh \
  --game-dir /absolute/path/to/game \
  --container-only

./scripts/status.sh
./scripts/stop.sh
```

Then run one short real environment:

```python
import dsle

env = dsle.make(
    "asylum_demon",
    game_dir="/absolute/path/to/game",
    output_dir="runs/smoke-test",
    max_steps=1,
)

try:
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(0)
finally:
    env.close()
```

The container should be absent after `close()`, while `runs/smoke-test/` and the
prebuilt image remain.

## Required permissions and security options

DSLE intentionally does not request a fully privileged container. It does need
specific permissions for GPU rendering and process-memory instrumentation.

| Permission or option | Why it is needed | Scope |
|---|---|---|
| Docker daemon access | Create, inspect, execute in, stop, and remove owned containers | Host user; Docker access is root-equivalent |
| `--gpus all` and runtime `nvidia` | Vulkan/DXVK rendering and optional learning workloads | NVIDIA devices exposed to the container |
| `NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display` | Graphics, driver diagnostics, compute, and display support | Container only |
| `CAP_SYS_PTRACE` | Read and write the Wine game process's memory | Container process namespace |
| `seccomp=unconfined` | Permit process-memory syscalls used by instrumentation | Owned DSLE container |
| `apparmor=unconfined` | Prevent the default profile from blocking memory instrumentation | Owned DSLE container |
| Read-only game bind | Share one game installation across isolated processes without mutating it | `/opt/dsle/game` |
| Writable state bind | Persist Wine prefixes, saves, logs, recordings, and results | `/var/lib/dsle` |
| 2 GiB shared memory | Support Xorg, capture, Wine, and multiple game processes | Shared container `/dev/shm` |
| Loopback VNC publication | Optional local observation of headless instances | `127.0.0.1` only |

The ptrace capability and relaxed security profiles apply inside the owned
container; they do not give the game a writable game mount. Nevertheless,
users should review these options before running DSLE on a shared or
multi-tenant host.

If an organizational Docker policy forbids `SYS_PTRACE`, unconfined seccomp,
or unconfined AppArmor, the current live memory backend will not work. Failing
silently is not supported: `dsle-doctor` reports the missing requirement.

## Output and state permissions

Managed API runs create this host-side layout:

```text
OUTPUT_DIR/
├── .container-session.lock
├── .dsle-output
├── config/
├── dxvk-cache/
├── logs/
├── pulse/
├── recordings/
├── results/
├── rpc/
├── run/
├── wineprefixes/
└── xdg/
```

The calling user needs write and directory-traverse permission on the selected
output directory and its parent. The directory must be either:

- New or empty, so DSLE can initialize it; or
- A previously initialized DSLE directory with a valid `.dsle-output` marker.

DSLE refuses an arbitrary non-empty directory. Reserved control paths must not
be symbolic links, must be owned by the calling user, and must have the expected
private permissions. These checks prevent a container from overwriting
unrelated host data through a misleading output path.

The host UID and GID are passed into the container. Durable directories are
returned to that ownership and are group-writable, so logs, checkpoints, and
recordings remain accessible after the root-running container exits. Private
RPC/token/runtime directories retain restrictive permissions.

Only one live container may use an output/state directory at a time. Choose a
different output path for each concurrent run. Omitting `output_dir` is the
safest default because it creates a unique path automatically.

The manual Compose workflow uses `.dsle-state/` and `.dsle-control/` by
default. The development workflow uses `.dsle-dev-state/` and
`.dsle-dev-control/`. Stopping those workflows removes containers and networks
but intentionally keeps the state directories.

## Networking and VNC

The normal environment uses a private authenticated Unix socket, not a TCP API
port. The bearer token is stored in a private, non-symlink regular file and
must not be readable by group or other users.

VNC is disabled in ordinary `headless` API mode. In `headless-vnc` mode, every
published VNC port binds to `127.0.0.1`. Local viewing therefore needs no public
firewall rule:

```bash
vncviewer 127.0.0.1::5901
```

The manual scripts reserve host ports 5901 through 5930. API-managed containers
normally request dynamic loopback host ports and expose the mappings through
`env.vnc_ports`.

Do not make the unauthenticated VNC service publicly reachable. For viewing a
remote research machine, keep VNC on loopback and use an authenticated SSH
tunnel controlled by that machine's administrator.

## Environment variables

| Variable | Purpose |
|---|---|
| `DSLE_GAME_DIR` | Default host game directory when `game_dir` is omitted. |
| `DSLE_IMAGE` | Default existing runtime image. |
| `DSLE_VERBOSE` | Shell-script verbosity; accepts `1/0`, `true/false`, `yes/no`, or `on/off`. |
| `DSLE_INSTANCE_COUNT` | Default instance count for `scripts/start.sh`. |
| `DSLE_RESOLUTION` | Default manual-workflow resolution, for example `800x600`. |
| `DSLE_STATE_DIR` | Manual workflow's persistent host state directory. |
| `DSLE_CONTROL_DIR` | Manual workflow's private host lifecycle metadata directory. |
| `NVIDIA_VISIBLE_DEVICES` | GPU selection for the Compose workflow; defaults to `all`. |
| `NO_COLOR` | Disable ANSI color while retaining lifecycle text. |

Prefer explicit Python arguments in library code and environment variables in
deployment wrappers.

## Development setup

For live editing of Python, YAML, saves, and templates, use the development
workflow:

```bash
./scripts/dev.sh start --game-dir /absolute/path/to/game
./scripts/dev.sh list
./scripts/dev.sh boss asylum_demon
./scripts/dev.sh menu
./scripts/dev.sh stop
```

The repository is mounted read-only at `/workspace/dsle`. Each `boss`, `menu`,
or `list` command starts a fresh Python process, so most host source edits are
visible without rebuilding the image or recreating the container. Changes to
the already-running process supervisor still require a stop/start cycle.

## Test dependency levels

Install the runtime and test extras first:

```bash
python -m pip install -e '.[runtime,test]'
```

Run hardware-independent unit and contract tests:

```bash
python -m pytest -m 'not docker and not game'
```

Build and validate the NVIDIA container without starting a game process:

```bash
DSLE_GAME_DIR=/absolute/path/to/game \
DSLE_RUN_DOCKER_TESTS=1 \
python -m pytest tests/integration/test_docker_runtime.py -m docker
```

Run a real one-boss reset/step test from a supported runtime selection:

```bash
DSLE_GAME_DIR=/absolute/path/to/game \
DSLE_RUN_LIVE_TESTS=1 \
DSLE_TEST_RUNTIME=managed \
python -m pytest tests/integration/test_live_boss.py::test_asylum_demon_reset_and_short_episode
```

The full regular-save screenshot sweep starts every advertised boss, waits ten
seconds after each fight begins, and writes a PNG plus JSON metadata:

```bash
./scripts/boss-screenshot-sweep.sh \
  --game-dir /absolute/path/to/game \
  --python .venv/bin/python
```

These hardware tests allocate real containers and game processes. Run them on
an otherwise idle workstation and allow their managed cleanup to complete.

## Troubleshooting

### `the Docker daemon is unavailable to this user`

Run `docker info`. If it succeeds only with `sudo`, fix Docker socket access and
log in again so group membership is refreshed. Continue running DSLE as the
regular host user.

### `runtime named 'nvidia' was not found`

Install NVIDIA Container Toolkit, run
`sudo nvidia-ctk runtime configure --runtime=docker`, restart Docker, and check
`docker info --format '{{json .Runtimes}}'`.

### Host `nvidia-smi` works but container startup fails

Run `dsle-doctor --host --require-game --game-dir PATH`. A working host driver
does not prove that Docker has NVIDIA runtime support. Also confirm that the
user is communicating with the expected Docker context and daemon.

### `Unsupported DarkSoulsRemastered.exe fingerprint`

The supplied executable does not match the memory layout used by the runtime.
Use the supported benchmark build. An executable that launches successfully is
not necessarily safe for memory reads or health writes.

### `Refusing non-empty uninitialized output directory`

Select a new or empty output directory, or omit `output_dir` and let DSLE create
a unique one. Do not add `.dsle-output` manually to an unrelated directory.

### `output directory is already used by another container`

Another live session holds the directory's lifecycle lock. Give concurrent
runs different output directories. If a workstation lost power, first confirm
with `docker ps` that no container is still using the path, then start a new run
with a new output directory.

### Permission errors in results

Confirm that the output directory is owned by the user running DSLE and that
the API was not started with `sudo`. Use a fresh user-owned output directory to
separate the new run from state created by a different UID.

### VNC viewer does not open

The game can still run headlessly. Install `tigervnc-viewer`, make sure a host
graphical display is available, or connect manually to the printed loopback
endpoint. `--no-vnc-viewer` prevents automatic launch in development mode.

### CCTV viewer reports missing dependencies

Install both dependency layers:

```bash
sudo apt install -y python3-tk
python -m pip install -e '.[viewer]'
```

### Image not found

Managed construction never pulls or builds automatically. Run
`./scripts/build.sh --core`, or pass the exact name of another locally built
compatible image with `image=...`.
