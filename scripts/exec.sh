#!/usr/bin/env bash
set -Eeuo pipefail

# Execute a command inside the manually managed DSLE runtime container. With no
# arguments, open Bash; noninteractive callers automatically disable TTY use.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_container_common.sh"

# Reload the exact image, game, state, and port metadata recorded by start.sh.
require_docker
load_deployment_env

# An interactive shell is the convenient default for manual diagnostics.
if (($# == 0)); then
  set -- bash
fi

# Compose must not allocate a TTY when input or output is redirected.
exec_options=()
if [[ ! -t 0 || ! -t 1 ]]; then
  exec_options+=(--no-tty)
fi
compose exec "${exec_options[@]}" runtime "$@"
