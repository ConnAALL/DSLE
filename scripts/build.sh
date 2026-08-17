#!/usr/bin/env bash
set -Eeuo pipefail

# Build the NVIDIA runtime image from the repository's allowlisted Docker
# context. Select either the minimal core image or research-dependency image.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_container_common.sh"

# Validate Docker and initialize the persistent state/control directories used
# by the matching manual start/status/stop workflow.
require_docker
prepare_state_dir

# Recognize DSLE's flavor switches and pass remaining options to Compose build.
flavor="core"
compose_args=()
while (($#)); do
  case "$1" in
    --research)
      flavor="research"
      shift
      ;;
    --core)
      flavor="core"
      shift
      ;;
    --quiet)
      DSLE_VERBOSE=0
      shift
      ;;
    --)
      shift
      compose_args+=("$@")
      break
      ;;
    *)
      compose_args+=("$1")
      shift
      ;;
  esac
done

# Resolve and persist the target/tag so later start commands use the image that
# this invocation built.
if [[ "${flavor}" == "research" ]]; then
  DSLE_IMAGE_TARGET="research"
  DSLE_IMAGE="${DSLE_IMAGE:-dsle-runtime:0.1.0-research}"
else
  DSLE_IMAGE_TARGET="runtime"
  DSLE_IMAGE="${DSLE_IMAGE:-dsle-runtime:0.1.0}"
fi
export DSLE_IMAGE DSLE_IMAGE_TARGET
persist_image_env

# The root .dockerignore keeps the commercial game outside this build context.
starting "Building the NVIDIA ${flavor} image (the game is not part of the build context)"
compose build "${compose_args[@]}" runtime
done_event "NVIDIA ${flavor} image is ready: ${DSLE_IMAGE}"
