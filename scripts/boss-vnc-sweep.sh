#!/usr/bin/env bash
set -Eeuo pipefail

# Add a stable loopback VNC endpoint to the standard screenshot sweep. All boss
# loading and cleanup remains delegated to boss-screenshot-sweep.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

usage() {
  cat <<'EOF'
Usage: scripts/boss-vnc-sweep.sh [--game-dir PATH] [options]

Loads all 22 configured standard (non-boosted) boss saves serially in one
owned NVIDIA container. The same dsr-1 display remains available through one
loopback-only VNC endpoint while each fight is shown for 10 seconds. The
TigerVNC connection command is printed after startup.

The underlying screenshot sweep also writes one PNG and JSON record per boss.
The container and VNC listener are stopped and removed at exit; the selected
output directory and prebuilt image remain.

Options:
  --game-dir PATH    Game installation (default: a recognized local game folder).
  --image IMAGE      Existing prebuilt image (default: dsle-runtime:0.1.0).
  --output-dir PATH  New/empty persistent run directory (unique default if omitted).
  --python PATH      Python with DSLE runtime/test dependencies (default: python3).
  --vnc-port PORT    Stable loopback host port used for the full run (default: 5901).
  --quiet            Suppress default lifecycle messages (test progress still prints).
  -h, --help         Show this help.
EOF
}

# Consume the VNC-only option and forward every ordinary sweep option unchanged.
vnc_host_port="${DSLE_BOSS_SWEEP_VNC_HOST_PORT:-5901}"
forwarded=()
while (($#)); do
  case "$1" in
    --vnc-port)
      (($# >= 2)) || { echo "DSLE: --vnc-port requires a port" >&2; exit 2; }
      vnc_host_port="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      forwarded+=("$1")
      shift
      ;;
  esac
done

# Reject invalid ports before the underlying script starts a container.
if [[ ! "${vnc_host_port}" =~ ^[1-9][0-9]{0,4}$ ]] || ((vnc_host_port > 65535)); then
  echo "DSLE: --vnc-port must be an integer in [1, 65535]" >&2
  exit 2
fi

# The integration fixture publishes this fixed host port and reuses it if a
# failed game runtime must be replaced during the sweep.
export DSLE_BOSS_SWEEP_VNC=1
export DSLE_BOSS_SWEEP_VNC_HOST_PORT="${vnc_host_port}"
exec "${SCRIPT_DIR}/boss-screenshot-sweep.sh" "${forwarded[@]}"
