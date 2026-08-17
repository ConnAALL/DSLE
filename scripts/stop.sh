#!/usr/bin/env bash
set -Eeuo pipefail

# Gracefully stop every manually managed game instance, then remove the Compose
# container/network while preserving the image and host-side runtime state.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_container_common.sh"

if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != "--quiet" ]]; }; then
  die "usage: scripts/stop.sh [--quiet]"
fi

# Resolve the exact deployment previously recorded by start.sh.
acquire_lifecycle_lock
require_docker
load_deployment_env

# Prefer instance-aware cleanup; Compose teardown remains the fallback.
stopping "Requesting a graceful stop for game instances"
if ! compose exec --no-tty runtime dsle-instances stop --all; then
  note "the instance controller was unavailable; stopping the container instead"
fi

# The persistent state bind is intentionally not deleted.
compose down --timeout 45
done_event "Container removed; persistent state remains in ${DSLE_STATE_DIR}"
