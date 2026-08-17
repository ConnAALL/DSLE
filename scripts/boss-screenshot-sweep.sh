#!/usr/bin/env bash
set -Eeuo pipefail

# Run the release-grade serial sweep for all configured standard boss saves.
# The Python test owns one temporary container and preserves screenshots/JSON.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_container_common.sh"

# Command-line defaults may also be supplied through environment variables.
game_dir="${DSLE_GAME_DIR:-}"
image="${DSLE_DOCKER_TEST_IMAGE:-dsle-runtime:0.1.0}"
output_dir="${DSLE_BOSS_SWEEP_OUTPUT_DIR:-}"
python_bin="${DSLE_PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage: scripts/boss-screenshot-sweep.sh [--game-dir PATH] [options]

Loads every supported regular boss save in one owned NVIDIA container and
writes a lossless screenshot 10 seconds after each fight starts, plus JSON
timing metadata for each encounter. A pre-screenshot failure is retried once
after replacing the game runtime.

Options:
  --game-dir PATH    Game installation (default: a recognized local game folder).
  --image IMAGE      Existing prebuilt image (default: dsle-runtime:0.1.0).
  --output-dir PATH  New/empty persistent run directory (unique default if omitted).
  --python PATH      Python with DSLE runtime/test dependencies (default: python3).
  --quiet            Suppress default lifecycle messages (test progress still prints).
  -h, --help         Show this help.
EOF
}

# Parse only sweep-specific options; unknown arguments fail before Docker runs.
while (($#)); do
  case "$1" in
    --game-dir)
      (($# >= 2)) || die "--game-dir requires a path"
      game_dir="$2"
      shift 2
      ;;
    --image)
      (($# >= 2)) || die "--image requires a reference"
      image="$2"
      shift 2
      ;;
    --output-dir)
      (($# >= 2)) || die "--output-dir requires a path"
      output_dir="$2"
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires an executable"
      python_bin="$2"
      shift 2
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

# Fail early when the host, game attachment, Python test runner, or prebuilt
# runtime image is unavailable.
require_docker
require_command "${python_bin}"
game_dir="$(select_game_dir "${game_dir}")"
validate_game_dir "${game_dir}"
docker image inspect "${image}" >/dev/null 2>&1 || \
  die "prebuilt image was not found locally: ${image}"

# Opt into the otherwise skipped live integration module and pass its selected
# image/output paths without embedding the game in a Docker build context.
export DSLE_DOCKER_TEST_IMAGE="${image}"
export DSLE_RUN_BOSS_SCREENSHOT_SWEEP=1
if [[ -n "${output_dir}" ]]; then
  export DSLE_BOSS_SWEEP_OUTPUT_DIR="${output_dir}"
fi

# Pytest performs serial reset, observation validation, delayed capture,
# cleanup, retry, artifact manifesting, and owned-container teardown.
starting "Running the serial regular-save screenshot sweep from prebuilt image ${image}"
cd "${PROJECT_ROOT}"
"${python_bin}" -m pytest \
  tests/integration/test_boss_screenshot_sweep.py \
  --verbose \
  --show-capture=all
