#!/usr/bin/env bash
set -Eeuo pipefail

# Start the manual NVIDIA runtime workflow: validate the external game, prepare
# persistent state, launch one container, and start the requested game instances.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_container_common.sh"

# Defaults can be overridden through flags or their matching environment values.
game_dir="${DSLE_GAME_DIR:-}"
instance_count="${DSLE_INSTANCE_COUNT:-1}"
resolution="${DSLE_RESOLUTION:-800x600}"
container_only=0
build_first=0

usage() {
  cat <<'EOF'
Usage: scripts/start.sh [--game-dir PATH] [options]

Options:
  --game-dir PATH    Game installation (default: a recognized local game folder).
  --instances N       Start N isolated game instances (default: 1, maximum: 30).
  --resolution WxH    Wine/X11 desktop size for every instance (default: 800x600).
  --state-dir PATH    Persistent prefixes, saves, logs, and recordings.
  --build             Build the runtime image before starting it.
  --container-only    Start and validate the container without launching the game.
  --quiet             Suppress default lifecycle messages (errors still print).
  -h, --help          Show this help.
EOF
}

# Parse lifecycle options before creating state or contacting Docker.
while (($#)); do
  case "$1" in
    --game-dir)
      (($# >= 2)) || die "--game-dir requires a path"
      game_dir="$2"
      shift 2
      ;;
    --instances)
      (($# >= 2)) || die "--instances requires a number"
      instance_count="$2"
      shift 2
      ;;
    --resolution)
      (($# >= 2)) || die "--resolution requires WxH"
      resolution="$2"
      shift 2
      ;;
    --state-dir)
      (($# >= 2)) || die "--state-dir requires a path"
      DSLE_STATE_DIR="$2"
      shift 2
      ;;
    --build)
      build_first=1
      shift
      ;;
    --container-only)
      container_only=1
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
      die "unknown option: $1"
      ;;
  esac
done

# Validate scalar inputs, the read-only game source, Docker access, and the
# persistent output/control layout before launching anything.
acquire_lifecycle_lock
[[ "${instance_count}" =~ ^[0-9]+$ ]] || die "--instances must be an integer"
((instance_count >= 1 && instance_count <= 30)) || die "--instances must be between 1 and 30"
[[ "${resolution}" =~ ^[0-9]{3,4}x[0-9]{3,4}$ ]] || die "--resolution must look like 800x600"

game_dir="$(select_game_dir "${game_dir}")"
validate_game_dir "${game_dir}"
validate_game_state_separation \
  "${DSLE_GAME_DIR}" \
  "${DSLE_STATE_DIR:-${DEFAULT_STATE_DIR}}"
require_docker
prepare_state_dir
prepare_runtime_token
load_image_env
DSLE_VNC_HOST_START="${DSLE_VNC_HOST_START:-5901}"
DSLE_VNC_HOST_END="${DSLE_VNC_HOST_END:-5930}"
DSLE_RESOLUTION="${resolution}"
DSLE_VNC_PORT_BINDING="127.0.0.1:${DSLE_VNC_HOST_START}-${DSLE_VNC_HOST_END}:5901-5930"
DSLE_HOST_UID="$(id -u)"
DSLE_HOST_GID="$(id -g)"
export \
  DSLE_HOST_GID \
  DSLE_HOST_UID \
  DSLE_RESOLUTION \
  DSLE_VNC_HOST_START \
  DSLE_VNC_HOST_END \
  DSLE_VNC_PORT_BINDING
persist_deployment_env

# Building is explicit; normal startup requires an existing local image.
if ((build_first)); then
  starting "Building the NVIDIA runtime image"
  compose build runtime
  ready "NVIDIA runtime image built"
fi

# Any failure before successful startup removes the partial container/network
# while retaining the image and initialized host state directory.
starting "Starting the runtime container with a read-only game mount"
cleanup_runtime_on_failure=1
cleanup_failed_start() {
  local exit_status=$?
  trap - EXIT
  if ((cleanup_runtime_on_failure == 1 && exit_status != 0)); then
    stopping "Startup failed; stopping and removing the partial runtime container"
    if ! compose down --timeout 45; then
      note "automatic container cleanup failed; run scripts/stop.sh"
    fi
  fi
  exit "${exit_status}"
}
trap cleanup_failed_start EXIT
compose up --detach runtime
ready "Container is running; waiting for runtime health checks"

# Container creation is not readiness: require the full doctor and authenticated
# runtime-server health check before configuring game processes.
runtime_ready=0
last_health_error=""
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
  [[ -z "${last_health_error}" ]] || debug "Last runtime health error: ${last_health_error}"
  compose logs --no-color --tail 100 runtime || true
fi
((runtime_ready == 1)) || die "runtime server did not become healthy within the startup window"
ready "Container runtime and dependencies are healthy"

# Container-only mode supports image, GPU, mount, and server diagnostics without
# paying the cost of launching Dark Souls.
if ((container_only)); then
  done_event "Container is ready; no game instances were requested"
  cleanup_runtime_on_failure=0
  exit 0
fi

# One container hosts N isolated displays, Wine prefixes, saves, and VNC ports.
starting "Configuring ${instance_count} isolated instance(s)"
compose exec --no-tty runtime dsle-instances configure \
  --count "${instance_count}" \
  --resolution "${resolution}" \
  --force

starting "Launching ${instance_count} game instance(s) with VNC enabled"
compose exec --no-tty runtime dsle-instances start --all --mode headless-vnc
ready "All ${instance_count} game instance(s) are running"
done_event "Startup complete; use scripts/status.sh for process and VNC details"
cleanup_runtime_on_failure=0
