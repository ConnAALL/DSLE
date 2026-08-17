#!/usr/bin/env bash
set -Eeuo pipefail

# Run the host-side single-instance showcase. The Python program owns one
# temporary container, records its read-only VNC feed, and preserves artifacts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
source "${SCRIPT_DIR}/_container_common.sh"

python_bin="${DSLE_PYTHON:-python3}"
require_command "${python_bin}"

# Use the live source checkout while leaving dependency selection to the host
# environment (install the runtime extra for OpenCV video encoding).
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${python_bin}" "${PROJECT_ROOT}/examples/boss_showcase.py" "$@"
