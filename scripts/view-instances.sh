#!/usr/bin/env bash
set -Eeuo pipefail

# Open a passive CCTV-style grid for every VNC-enabled DSLE instance. The
# viewer reads frames only: it never focuses a game window or sends input.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
source "${SCRIPT_DIR}/_container_common.sh"

usage() {
  cat <<'EOF'
Usage: scripts/view-instances.sh [--dev] [viewer options]

Open every running DSLE instance in one passive CCTV-style grid.

Options:
  --dev              Use the development deployment from .dsle-dev-control.
  --period SECONDS   Sample each instance at this interval (default: 1.0).
  --instances NAMES  Show a comma-separated subset such as dsr-1,dsr-4.
  --columns N        Set grid columns instead of automatic layout.
  -h, --help         Show the complete viewer help.

The ordinary deployment is discovered from scripts/start.sh metadata. For
custom/dynamically mapped containers, run `dsle-viewer --help` and supply one
--endpoint NAME=HOST:PORT argument per feed.
EOF
}

use_dev=0
viewer_args=()
while (($#)); do
  case "$1" in
    --dev)
      use_dev=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      viewer_args+=("$1")
      shift
      ;;
  esac
done

if ((use_dev)); then
  export DSLE_CONTROL_DIR="${DSLE_DEV_CONTROL_DIR:-${PROJECT_ROOT}/.dsle-dev-control}"
fi

python_bin="${DSLE_PYTHON:-python3}"
require_command "${python_bin}"
load_deployment_env
instance_config="${DSLE_STATE_DIR}/config/instances.json"
[[ -f "${instance_config}" ]] || \
  die "instance configuration was not found: ${instance_config}; start the game instances first"

if ! "${python_bin}" -c 'import tkinter; from PIL import Image, ImageTk' >/dev/null 2>&1; then
  die "viewer dependencies are missing; install with python -m pip install '.[viewer]' and sudo apt install python3-tk"
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${python_bin}" -m dsle.cli.viewer \
  --config "${instance_config}" \
  --base-port "${DSLE_VNC_HOST_START}" \
  "${viewer_args[@]}"
