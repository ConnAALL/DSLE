#!/usr/bin/env bash
set -Eeuo pipefail

# Manage one long-lived development container and one VNC-enabled DSR process.
# The local repository is bind-mounted and reloaded by each boss/menu command,
# so testing YAML, Python, saves, and templates needs no image rebuild.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
export COMPOSE_PROJECT_NAME="${DSLE_DEV_PROJECT_NAME:-dsle-dev}"
export DSLE_CONTROL_DIR="${DSLE_DEV_CONTROL_DIR:-${PROJECT_ROOT}/.dsle-dev-control}"
source "${SCRIPT_DIR}/_container_common.sh"

DEFAULT_STATE_DIR="${PROJECT_ROOT}/.dsle-dev-state"
DEV_COMPOSE_FILE="${PROJECT_ROOT}/docker/compose.dev.yaml"
export DSLE_CONTAINER_NAME="${DSLE_DEV_CONTAINER_NAME:-dsle-dev-runtime}"
export DSLE_DEV_SOURCE_DIR="${PROJECT_ROOT}"

# Layer the development source mounts over the ordinary NVIDIA runtime service.
compose() {
  docker compose \
    --project-name "${COMPOSE_PROJECT_NAME}" \
    --file "${COMPOSE_FILE}" \
    --file "${DEV_COMPOSE_FILE}" \
    "$@"
}

usage() {
  cat <<'EOF'
Usage:
  scripts/dev.sh start [--game-dir PATH] [options]
  scripts/dev.sh boss BOSS [--difficulty standard|boosted]
  scripts/dev.sh menu
  scripts/dev.sh list [--all]
  scripts/dev.sh status
  scripts/dev.sh shell
  scripts/dev.sh stop

Global options:
  --quiet            Suppress default lifecycle messages (errors still print).

Start options:
  --game-dir PATH    Game installation (default: a recognized local game folder).
  --state-dir PATH   Persistent Wine prefix, saves, logs, and results.
  --image IMAGE      Existing image (default: dsle-runtime:0.1.0).
  --resolution WxH   Game/VNC resolution (default: 800x600).
  --vnc-port PORT    First loopback VNC port (default: 5901).
  --boss BOSS        Load this boss after startup (default: stay at main menu).
  --difficulty MODE  standard or boosted (default: standard).
  --no-vnc-viewer    Do not automatically open an installed TigerVNC Viewer.
  --build             Build the selected image before startup.

Examples:
  scripts/dev.sh start --game-dir /games/Dark.Souls.Remastered.v1.04
  scripts/dev.sh boss asylum_demon
  scripts/dev.sh boss capra_demon --difficulty boosted
  scripts/dev.sh menu
  scripts/dev.sh stop

Connect to the running game with: vncviewer 127.0.0.1::5901
The boss command stays attached until the boss or player dies, then returns to the menu.
EOF
}

load_dev_deployment() {
  load_deployment_env
  export DSLE_CONTAINER_NAME DSLE_DEV_SOURCE_DIR
}

require_dev_runtime() {
  local container_id
  container_id="$(compose ps --quiet runtime)"
  [[ -n "${container_id}" ]] || die "development container is not running; use scripts/dev.sh start"
}

run_live_dev_command() {
  compose exec --no-tty --env "DSLE_VERBOSE=${DSLE_VERBOSE}" runtime \
    python3 -m dsle.cli.dev "$@"
}

# Find the common executable names provided by TigerVNC packages.
find_tigervnc_viewer() {
  local candidate
  for candidate in xtigervncviewer tigervncviewer vncviewer; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done
  return 1
}

# Open the loopback VNC endpoint without tying the viewer to this shell.
open_tigervnc_viewer() {
  local vnc_port="$1"
  local viewer
  if [[ -z "${DISPLAY:-}" ]]; then
    note "TigerVNC auto-open skipped because the host has no graphical DISPLAY"
    return
  fi
  if ! viewer="$(find_tigervnc_viewer)"; then
    note "TigerVNC Viewer was not found; connect manually if desired"
    return
  fi
  nohup "${viewer}" "127.0.0.1::${vnc_port}" >/dev/null 2>&1 &
  note "opened TigerVNC Viewer (${viewer}) on 127.0.0.1::${vnc_port}"
}

start_development_runtime() {
  local game_dir="${DSLE_GAME_DIR:-}"
  local resolution="${DSLE_RESOLUTION:-800x600}"
  local vnc_port="${DSLE_DEV_VNC_PORT:-5901}"
  local boss=""
  local difficulty="standard"
  local build_first=0
  local open_vnc_viewer=1

  while (($#)); do
    case "$1" in
      --game-dir)
        (($# >= 2)) || die "--game-dir requires a path"
        game_dir="$2"
        shift 2
        ;;
      --state-dir)
        (($# >= 2)) || die "--state-dir requires a path"
        DSLE_STATE_DIR="$2"
        shift 2
        ;;
      --image)
        (($# >= 2)) || die "--image requires a reference"
        DSLE_IMAGE="$2"
        shift 2
        ;;
      --resolution)
        (($# >= 2)) || die "--resolution requires WxH"
        resolution="$2"
        shift 2
        ;;
      --vnc-port)
        (($# >= 2)) || die "--vnc-port requires a port"
        vnc_port="$2"
        shift 2
        ;;
      --boss)
        (($# >= 2)) || die "--boss requires a boss id"
        boss="$2"
        shift 2
        ;;
      --difficulty)
        (($# >= 2)) || die "--difficulty requires standard or boosted"
        difficulty="$2"
        shift 2
        ;;
      --build)
        build_first=1
        shift
        ;;
      --no-vnc-viewer)
        open_vnc_viewer=0
        shift
        ;;
      --quiet)
        DSLE_VERBOSE=0
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown start option: $1"
        ;;
    esac
  done

  acquire_lifecycle_lock
  [[ "${resolution}" =~ ^[0-9]{3,4}x[0-9]{3,4}$ ]] || \
    die "--resolution must look like 800x600"
  [[ "${difficulty}" == "standard" || "${difficulty}" == "boosted" ]] || \
    die "--difficulty must be standard or boosted"
  [[ "${vnc_port}" =~ ^[1-9][0-9]{0,4}$ ]] || die "--vnc-port must be an integer"
  ((vnc_port <= 65506)) || \
    die "--vnc-port must be at most 65506 because Compose reserves a 30-port range"

  game_dir="$(select_game_dir "${game_dir}")"
  validate_game_dir "${game_dir}"
  validate_game_state_separation \
    "${DSLE_GAME_DIR}" \
    "${DSLE_STATE_DIR:-${DEFAULT_STATE_DIR}}"
  require_docker
  prepare_state_dir
  prepare_runtime_token
  load_image_env
  DSLE_RESOLUTION="${resolution}"
  DSLE_VNC_HOST_START="${vnc_port}"
  DSLE_VNC_HOST_END="$((vnc_port + 29))"
  DSLE_VNC_PORT_BINDING="127.0.0.1:${vnc_port}:5901"
  DSLE_HOST_UID="$(id -u)"
  DSLE_HOST_GID="$(id -g)"
  export \
    DSLE_HOST_GID \
    DSLE_HOST_UID \
    DSLE_IMAGE \
    DSLE_RESOLUTION \
    DSLE_VNC_HOST_END \
    DSLE_VNC_HOST_START \
    DSLE_VNC_PORT_BINDING
  persist_deployment_env

  if ((build_first)); then
    starting "Building the NVIDIA runtime image"
    compose build runtime
    ready "NVIDIA runtime image built"
  fi

  # Development mode imports the live-mounted checkout rather than the Python
  # package baked into the image. Detect an old image missing new core
  # dependencies before starting a container that would immediately exit.
  checking "Checking that the prebuilt image supports the live source tree"
  local dependency_error=""
  local health_output=""
  if ! dependency_error="$(
    docker run \
      --rm \
      --pull=never \
      --entrypoint python3 \
      "${DSLE_IMAGE:-dsle-runtime:0.1.0}" \
      -c 'import rich' \
      2>&1
  )"; then
    die "prebuilt image ${DSLE_IMAGE:-dsle-runtime:0.1.0} is missing current dependencies; rerun this command with --build (${dependency_error})"
  fi
  ready "Prebuilt image dependencies match the live source tree"

  starting "Starting one development container with live source and read-only game mounts"
  local cleanup_runtime_on_failure=1
  cleanup_failed_start() {
    local exit_status=$?
    trap - EXIT
    if ((cleanup_runtime_on_failure == 1 && exit_status != 0)); then
      note "development startup failed; removing the partial container/network"
      compose down --timeout 45 || true
    fi
    exit "${exit_status}"
  }
  trap cleanup_failed_start EXIT
  compose up --detach runtime
  ready "Development container is running; waiting for runtime health checks"

  local runtime_ready=0
  local last_health_error=""
  for _attempt in {1..12}; do
    if health_output="$(compose exec --no-tty runtime dsle-health 2>&1)"; then
      runtime_ready=1
      break
    fi
    last_health_error="${health_output}"
    [[ "${health_output}" != *'is not running'* ]] || break
    sleep 2
  done
  if ((runtime_ready == 0)); then
    [[ -z "${last_health_error}" ]] || \
      debug "Last development health error: ${last_health_error}"
    compose logs --no-color --tail 100 runtime || true
  fi
  ((runtime_ready == 1)) || die "development runtime did not become healthy"
  ready "Development runtime and dependencies are healthy"

  # Prove that imports and runtime assets resolve to the bind-mounted checkout.
  local mounted_source
  mounted_source="$(compose exec --no-tty runtime python3 -c \
    'import dsle; print(dsle.__file__)')"
  [[ "${mounted_source}" == /workspace/dsle/src/dsle/* ]] || \
    die "container did not import the live source mount: ${mounted_source}"

  starting "Configuring and launching dsr-1 with VNC enabled"
  compose exec --no-tty runtime dsle-instances configure \
    --count 1 \
    --resolution "${resolution}" \
    --force
  compose exec --no-tty runtime dsle-instances start \
    --instances dsr-1 \
    --mode headless-vnc

  ready "Game running: dsr-1"
  note "VNC: vncviewer 127.0.0.1::${vnc_port}"
  if ((open_vnc_viewer)); then
    open_tigervnc_viewer "${vnc_port}"
  fi

  if [[ -n "${boss}" ]]; then
    run_live_dev_command boss "${boss}" --difficulty "${difficulty}"
  else
    note "no boss selected; navigating dsr-1 to the main menu"
    run_live_dev_command menu
  fi

  ready "Development instance is at the main menu"
  note "load a boss with: scripts/dev.sh boss asylum_demon"
  note "local repository edits are used by the next boss/menu command"
  cleanup_runtime_on_failure=0
  trap - EXIT
}

# --quiet is accepted before the subcommand, matching conventional CLI usage.
if [[ "${1:-}" == "--quiet" ]]; then
  DSLE_VERBOSE=0
  shift
fi
command="${1:-help}"
if (($#)); then
  shift
fi

case "${command}" in
  start)
    start_development_runtime "$@"
    ;;
  boss)
    (($# >= 1)) || die "boss requires a boss id"
    boss="$1"
    shift
    difficulty="standard"
    while (($#)); do
      case "$1" in
        --difficulty)
          (($# >= 2)) || die "--difficulty requires standard or boosted"
          difficulty="$2"
          shift 2
          ;;
        --quiet)
          DSLE_VERBOSE=0
          shift
          ;;
        *)
          die "unknown boss option: $1"
          ;;
      esac
    done
    [[ "${difficulty}" == "standard" || "${difficulty}" == "boosted" ]] || \
      die "--difficulty must be standard or boosted"
    load_dev_deployment
    require_docker
    require_dev_runtime
    run_live_dev_command boss "${boss}" --difficulty "${difficulty}"
    ;;
  menu)
    (($# == 0)) || die "menu does not accept arguments"
    load_dev_deployment
    require_docker
    require_dev_runtime
    run_live_dev_command menu
    ;;
  list)
    (($# <= 1)) || die "list accepts only --all"
    if (($# == 1)); then
      [[ "$1" == "--all" ]] || die "list accepts only --all"
    fi
    load_dev_deployment
    require_docker
    require_dev_runtime
    run_live_dev_command list "$@"
    ;;
  status)
    (($# == 0)) || die "status does not accept arguments"
    load_dev_deployment
    require_docker
    compose ps
    compose exec --no-tty runtime dsle-instances status --instances dsr-1
    note "live repository: ${PROJECT_ROOT}"
    note "VNC: vncviewer 127.0.0.1::${DSLE_VNC_HOST_START}"
    ;;
  shell)
    (($# == 0)) || die "shell does not accept arguments"
    load_dev_deployment
    require_docker
    require_dev_runtime
    compose exec runtime bash
    ;;
  stop)
    (($# == 0)) || die "stop does not accept arguments"
    acquire_lifecycle_lock
    load_dev_deployment
    require_docker
    stopping "Stopping the development game instance"
    compose exec --no-tty runtime dsle-instances stop --instances dsr-1 || true
    compose down --timeout 45
    done_event "Container removed; image and persistent state remain in ${DSLE_STATE_DIR}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    die "unknown command: ${command}; use scripts/dev.sh --help"
    ;;
esac
