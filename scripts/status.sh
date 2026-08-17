#!/usr/bin/env bash
set -Eeuo pipefail

# Show the manual Compose container state followed by every configured game
# instance's process, display, and VNC status.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_container_common.sh"

if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != "--quiet" ]]; }; then
  die "usage: scripts/status.sh [--quiet]"
fi

# Resolve the exact deployment previously recorded by start.sh.
require_docker
load_deployment_env
checking "Reading container and game-instance status"
compose ps

# Container status remains useful even if the in-container controller is down.
if ! compose exec --no-tty runtime dsle-instances status --all; then
  note "live instance status is unavailable; the container status above is still authoritative"
fi
