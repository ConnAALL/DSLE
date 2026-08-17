#!/usr/bin/env bash
set -Eeuo pipefail

# Shared helpers for the manual Docker/Compose scripts. This file is sourced,
# not executed directly; it centralizes safe paths, metadata, and validation.

# Resolve repository-relative defaults once so callers work from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
COMPOSE_FILE="${PROJECT_ROOT}/docker/compose.yaml"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-dsle}"
DEFAULT_STATE_DIR="${PROJECT_ROOT}/.dsle-state"
DEFAULT_CONTROL_DIR="${PROJECT_ROOT}/.dsle-control"
METADATA_HEADER="DSLE-METADATA-V1"

# Lifecycle output is intentionally on by default. Every public script also
# accepts --quiet; DSLE_VERBOSE=0 is useful for automation and CI wrappers.
DSLE_VERBOSE="${DSLE_VERBOSE:-1}"
for _dsle_argument in "$@"; do
  if [[ "${_dsle_argument}" == "--quiet" ]]; then
    DSLE_VERBOSE=0
  fi
done
case "${DSLE_VERBOSE,,}" in
  1|true|yes|on) DSLE_VERBOSE=1 ;;
  0|false|no|off) DSLE_VERBOSE=0 ;;
  *)
    printf '[ERROR] DSLE_VERBOSE must be 1/0, true/false, yes/no, or on/off\n' >&2
    exit 2
    ;;
esac
export DSLE_VERBOSE

# Consistent user-facing output and basic command/path validation.
die() {
  _dsle_log ERROR 31 "$*"
  exit 1
}

note() {
  _dsle_log INFO 34 "$*"
}

debug() {
  _dsle_log DEBUG 36 "$*"
}

checking() {
  _dsle_log CHECK 36 "$*"
}

ready() {
  _dsle_log READY 32 "$*"
}

starting() {
  _dsle_log START 35 "$*"
}

stopping() {
  _dsle_log STOP 33 "$*"
}

done_event() {
  _dsle_log DONE 32 "$*"
}

_dsle_log() {
  local label="$1"
  local color="$2"
  shift 2
  if [[ "${label}" != "ERROR" && "${DSLE_VERBOSE}" != "1" ]]; then
    return
  fi
  if [[ -t 2 && -z "${NO_COLOR:-}" ]]; then
    printf '\033[1;%sm[%s]\033[0m %s\n' "${color}" "${label}" "$*" >&2
  else
    printf '[%s] %s\n' "${label}" "$*" >&2
  fi
}

require_command() {
  checking "Checking host command: $1"
  command -v "$1" >/dev/null 2>&1 || die "required command '$1' was not found"
  ready "Host command available: $1"
}

absolute_dir() {
  local path="$1"
  [[ -d "${path}" ]] || die "directory does not exist: ${path}"
  (cd "${path}" && pwd -P)
}

# When no explicit value was supplied, check the conventional local folder
# names in priority order without recursively guessing through the filesystem.
select_game_dir() {
  local requested="$1"
  if [[ -n "${requested}" ]]; then
    printf '%s\n' "${requested}"
    return
  fi
  local name candidate
  for name in Dark.Souls.Remastered.v1.04 game; do
    candidate="${PWD}/${name}"
    if [[ -d "${candidate}" ]]; then
      debug "Using default game directory: ${candidate}"
      printf '%s\n' "${candidate}"
      return
    fi
  done
}

# Prepare host-only lifecycle metadata and persistent container state. Marker,
# lock, ownership, and link checks prevent accidental use of arbitrary paths.
prepare_control_dir() {
  local requested="${DSLE_CONTROL_DIR:-${DEFAULT_CONTROL_DIR}}"
  [[ ! -L "${requested}" ]] || die "refusing symbolic-link control directory: ${requested}"
  mkdir -p -- "${requested}"
  DSLE_CONTROL_DIR="$(absolute_dir "${requested}")"
  [[ "$(stat -c '%u' -- "${DSLE_CONTROL_DIR}")" == "$(id -u)" ]] || \
    die "DSLE control directory is not owned by this user: ${DSLE_CONTROL_DIR}"
  chmod 0700 -- "${DSLE_CONTROL_DIR}"
  export DSLE_CONTROL_DIR
  if [[ "${_DSLE_LOGGED_CONTROL_DIR:-}" != "${DSLE_CONTROL_DIR}" ]]; then
    debug "Control directory: ${DSLE_CONTROL_DIR}"
    _DSLE_LOGGED_CONTROL_DIR="${DSLE_CONTROL_DIR}"
  fi
}

# Serialize lifecycle mutations for one Compose project. This prevents two
# simultaneous start/stop commands from deleting each other's network or
# partially created container.
acquire_lifecycle_lock() {
  prepare_control_dir
  [[ -z "${_DSLE_LIFECYCLE_LOCK_FD:-}" ]] || return
  command -v flock >/dev/null 2>&1 || die "required command 'flock' was not found"
  local lock_file="${DSLE_CONTROL_DIR}/lifecycle.lock"
  [[ ! -L "${lock_file}" ]] || die "refusing symbolic-link lifecycle lock: ${lock_file}"
  if [[ ! -e "${lock_file}" ]]; then
    (
      umask 077
      set -o noclobber
      : > "${lock_file}"
    ) 2>/dev/null || true
  fi
  [[ -f "${lock_file}" ]] || die "invalid lifecycle lock: ${lock_file}"
  local owner links
  IFS=: read -r owner links < <(stat -c '%u:%h' -- "${lock_file}")
  [[ "${owner}" == "$(id -u)" && "${links}" == "1" ]] || \
    die "lifecycle lock must be user-owned with one hard link: ${lock_file}"
  chmod 0600 -- "${lock_file}"
  exec {_DSLE_LIFECYCLE_LOCK_FD}<>"${lock_file}"
  flock --nonblock "${_DSLE_LIFECYCLE_LOCK_FD}" || \
    die "another DSLE lifecycle command is already running for ${COMPOSE_PROJECT_NAME}"
  debug "Lifecycle lock acquired for ${COMPOSE_PROJECT_NAME}"
}

prepare_state_dir() {
  prepare_control_dir
  local requested="${DSLE_STATE_DIR:-${DEFAULT_STATE_DIR}}"
  if [[ -n "${DSLE_GAME_DIR:-}" ]]; then
    validate_game_state_separation "${DSLE_GAME_DIR}" "${requested}"
  fi
  mkdir -p -- "${requested}"
  DSLE_STATE_DIR="$(absolute_dir "${requested}")"
  if [[ "${DSLE_CONTROL_DIR}" == "${DSLE_STATE_DIR}" || \
        "${DSLE_CONTROL_DIR}" == "${DSLE_STATE_DIR}/"* ]]; then
    die "DSLE_CONTROL_DIR must stay outside the container-mounted DSLE_STATE_DIR"
  fi
  local marker="${DSLE_STATE_DIR}/.dsle-output"
  local lock_file="${DSLE_STATE_DIR}/.container-session.lock"
  if [[ -L "${marker}" || -L "${lock_file}" ]]; then
    die "refusing symbolic-link DSLE control file in ${DSLE_STATE_DIR}"
  fi
  if [[ ! -e "${marker}" ]]; then
    local first_entry
    first_entry="$(find "${DSLE_STATE_DIR}" -mindepth 1 -maxdepth 1 -print -quit)"
    [[ -z "${first_entry}" ]] || \
      die "refusing non-empty uninitialized state directory: ${DSLE_STATE_DIR}"
    (
      umask 077
      set -o noclobber
      printf 'dsle-output\n' > "${marker}"
      : > "${lock_file}"
    ) || die "could not initialize DSLE state directory: ${DSLE_STATE_DIR}"
  fi
  [[ -f "${marker}" && "$(<"${marker}")" == "dsle-output" ]] || \
    die "invalid DSLE state marker: ${marker}"
  [[ -f "${lock_file}" ]] || die "invalid DSLE session lock: ${lock_file}"
  chmod 0600 "${marker}" "${lock_file}"
  export DSLE_STATE_DIR
  ready "Persistent state directory: ${DSLE_STATE_DIR}"
}

# Create the private token used to authenticate host-to-container RPC requests.
prepare_runtime_token() {
  local run_dir="${DSLE_STATE_DIR}/run"
  local token_file="${run_dir}/dsle.token"
  mkdir -p "${run_dir}"
  chmod 0700 "${run_dir}"
  if [[ -L "${token_file}" ]]; then
    die "refusing symbolic-link runtime token: ${token_file}"
  fi
  if [[ ! -e "${token_file}" ]]; then
    local generated_token
    generated_token="$(head -c 32 /dev/urandom | base64 | tr -d '\n')"
    (
      umask 077
      set -o noclobber
      printf '%s\n' "${generated_token}" > "${token_file}"
    ) 2>/dev/null || true
  fi
  [[ -f "${token_file}" && -s "${token_file}" ]] || \
    die "runtime token is missing or invalid: ${token_file}"
  chmod 0600 "${token_file}"
}

deployment_env_path() {
  echo "${DSLE_CONTROL_DIR:-${DEFAULT_CONTROL_DIR}}/deployment.meta"
}

image_env_path() {
  echo "${DSLE_CONTROL_DIR:-${DEFAULT_CONTROL_DIR}}/image.meta"
}

# Metadata values are base64 encoded and written atomically rather than sourced
# as shell code. Readers enforce ownership, permissions, keys, and canonical form.
_encode_metadata_value() {
  local value="$1"
  [[ "${value}" != *$'\n'* ]] || die "DSLE metadata values cannot contain newlines"
  printf '%s' "${value}" | base64 | tr -d '\n'
}

_decode_metadata_value() {
  local encoded="$1"
  local description="$2"
  local destination_name="$3"
  local -n destination="${destination_name}"
  [[ "${encoded}" =~ ^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$ ]] || \
    die "invalid base64 value for ${description}"
  local decoded
  if ! decoded="$(printf '%s' "${encoded}" | base64 --decode 2>/dev/null)"; then
    die "invalid base64 value for ${description}"
  fi
  [[ "${decoded}" != *$'\n'* ]] || die "decoded ${description} contains a newline"
  local canonical
  canonical="$(_encode_metadata_value "${decoded}")"
  [[ "${canonical}" == "${encoded}" ]] || die "non-canonical value for ${description}"
  destination="${decoded}"
}

_validate_metadata_file() {
  local path="$1"
  [[ ! -L "${path}" && -f "${path}" ]] || \
    die "metadata must be a non-symlink regular file: ${path}"
  local owner links mode
  IFS=: read -r owner links mode < <(stat -c '%u:%h:%a' -- "${path}")
  [[ "${owner}" == "$(id -u)" ]] || die "metadata is not owned by this user: ${path}"
  [[ "${links}" == "1" ]] || die "metadata must have exactly one hard link: ${path}"
  [[ "${mode}" == "600" ]] || die "metadata must have mode 0600: ${path}"
}

_parse_metadata_file() {
  local path="$1"
  local destination_name="$2"
  shift 2
  local -a allowed_keys=("$@")
  local -n destination="${destination_name}"
  destination=()
  _validate_metadata_file "${path}"

  local line key encoded value allowed_key is_allowed
  local line_number=0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    ((line_number += 1))
    if ((line_number == 1)); then
      [[ "${line}" == "${METADATA_HEADER}" ]] || die "invalid metadata header: ${path}"
      continue
    fi
    [[ "${line}" == *=* ]] || die "invalid metadata record in ${path}"
    key="${line%%=*}"
    encoded="${line#*=}"
    is_allowed=0
    for allowed_key in "${allowed_keys[@]}"; do
      if [[ "${key}" == "${allowed_key}" ]]; then
        is_allowed=1
        break
      fi
    done
    ((is_allowed == 1)) || die "unknown metadata key ${key@Q} in ${path}"
    [[ ! -v 'destination[$key]' ]] || die "duplicate metadata key ${key@Q} in ${path}"
    value=""
    _decode_metadata_value "${encoded}" "${key} in ${path}" value
    [[ -n "${value}" ]] || die "metadata value ${key@Q} cannot be empty"
    destination["${key}"]="${value}"
  done < "${path}"

  ((line_number > 0)) || die "metadata file is empty: ${path}"
  ((${#destination[@]} == ${#allowed_keys[@]})) || \
    die "metadata file does not contain its exact required key set: ${path}"
}

_write_metadata_file() {
  local destination="$1"
  shift
  (($# > 0 && $# % 2 == 0)) || die "internal error: metadata requires key/value pairs"
  prepare_control_dir
  [[ ! -L "${destination}" ]] || die "refusing symbolic-link metadata destination: ${destination}"
  [[ ! -e "${destination}" || -f "${destination}" ]] || \
    die "metadata destination is not a regular file: ${destination}"

  local temporary
  temporary="$(mktemp "${DSLE_CONTROL_DIR}/.$(basename "${destination}").tmp.XXXXXX")"
  if ! (
    umask 077
    {
      printf '%s\n' "${METADATA_HEADER}"
      local key value encoded
      while (($#)); do
        key="$1"
        value="$2"
        shift 2
        [[ "${key}" =~ ^DSLE_[A-Z0-9_]+$ ]] || exit 65
        encoded="$(_encode_metadata_value "${value}")"
        printf '%s=%s\n' "${key}" "${encoded}"
      done
    } > "${temporary}"
    chmod 0600 -- "${temporary}"
    mv -fT -- "${temporary}" "${destination}"
  ); then
    if [[ -e "${temporary}" || -L "${temporary}" ]]; then
      unlink -- "${temporary}"
    fi
    die "could not atomically write DSLE metadata: ${destination}"
  fi
}

# Load or persist the selected image and the active manual deployment so later
# status, exec, and stop commands address the same paths and Compose project.
load_image_env() {
  prepare_control_dir
  local env_path
  env_path="$(image_env_path)"
  if [[ -e "${env_path}" || -L "${env_path}" ]]; then
    local -A metadata=()
    _parse_metadata_file \
      "${env_path}" metadata \
      DSLE_IMAGE DSLE_IMAGE_TARGET
    DSLE_IMAGE="${DSLE_IMAGE:-${metadata[DSLE_IMAGE]}}"
    DSLE_IMAGE_TARGET="${DSLE_IMAGE_TARGET:-${metadata[DSLE_IMAGE_TARGET]}}"
    export DSLE_IMAGE DSLE_IMAGE_TARGET
  fi
}

persist_image_env() {
  prepare_control_dir
  local env_path
  env_path="$(image_env_path)"
  _write_metadata_file \
    "${env_path}" \
    DSLE_IMAGE "${DSLE_IMAGE:-dsle-runtime:0.1.0}" \
    DSLE_IMAGE_TARGET "${DSLE_IMAGE_TARGET:-runtime}"
}

load_deployment_env() {
  prepare_control_dir
  local env_path
  env_path="$(deployment_env_path)"
  [[ -e "${env_path}" || -L "${env_path}" ]] || \
    die "manual deployment metadata does not exist: ${env_path}"
  local -A metadata=()
  _parse_metadata_file \
    "${env_path}" metadata \
    DSLE_GAME_DIR DSLE_STATE_DIR DSLE_RESOLUTION \
    DSLE_VNC_HOST_START DSLE_VNC_HOST_END
  DSLE_GAME_DIR="${metadata[DSLE_GAME_DIR]}"
  DSLE_STATE_DIR="${metadata[DSLE_STATE_DIR]}"
  DSLE_RESOLUTION="${metadata[DSLE_RESOLUTION]}"
  DSLE_VNC_HOST_START="${metadata[DSLE_VNC_HOST_START]}"
  DSLE_VNC_HOST_END="${metadata[DSLE_VNC_HOST_END]}"
  export \
    DSLE_GAME_DIR \
    DSLE_RESOLUTION \
    DSLE_STATE_DIR \
    DSLE_VNC_HOST_END \
    DSLE_VNC_HOST_START
  prepare_state_dir
  load_image_env
}

persist_deployment_env() {
  prepare_control_dir
  # Starting from an explicitly selected prebuilt image must also survive into
  # later status/exec/stop invocations, even when build.sh was not used first.
  persist_image_env
  local env_path
  env_path="$(deployment_env_path)"
  _write_metadata_file \
    "${env_path}" \
    DSLE_GAME_DIR "${DSLE_GAME_DIR}" \
    DSLE_STATE_DIR "${DSLE_STATE_DIR}" \
    DSLE_RESOLUTION "${DSLE_RESOLUTION:-800x600}" \
    DSLE_VNC_HOST_START "${DSLE_VNC_HOST_START:-5901}" \
    DSLE_VNC_HOST_END "${DSLE_VNC_HOST_END:-5930}"
}

# Validate the externally supplied game layout and keep writable state separate
# from the read-only commercial game directory.
validate_game_dir() {
  local candidate="$1"
  checking "Checking the attached Dark Souls: Remastered installation"
  [[ -n "${candidate}" ]] || die "provide --game-dir PATH or set DSLE_GAME_DIR"
  candidate="$(absolute_dir "${candidate}")"
  [[ -f "${candidate}/DarkSoulsRemastered.exe" ]] || \
    die "DarkSoulsRemastered.exe was not found in ${candidate}"
  local asset
  for asset in chr event map mtd param script; do
    [[ -d "${candidate}/${asset}" ]] || die "game asset directory is missing: ${candidate}/${asset}"
  done
  DSLE_GAME_DIR="${candidate}"
  export DSLE_GAME_DIR
  ready "Game installation verified: ${DSLE_GAME_DIR}"
}

_path_is_same_or_descendant() {
  local candidate="$1"
  local ancestor="$2"
  [[ "${candidate}" == "${ancestor}" ]] || \
    [[ "${ancestor}" == "/" && "${candidate}" == /* ]] || \
    [[ "${candidate}" == "${ancestor%/}/"* ]]
}

validate_game_state_separation() {
  local game_dir="$1"
  local state_dir="$2"
  local game_real state_real
  game_real="$(readlink -m -- "${game_dir}")"
  state_real="$(readlink -m -- "${state_dir}")"

  if _path_is_same_or_descendant "${state_real}" "${game_real}"; then
    die "DSLE_STATE_DIR cannot be inside the read-only game installation"
  fi
  if _path_is_same_or_descendant "${game_real}" "${state_real}"; then
    die "DSLE_STATE_DIR cannot contain the game installation"
  fi
}

# Keep all manual lifecycle commands on one explicit Compose file/project name.
compose() {
  docker compose \
    --project-name "${COMPOSE_PROJECT_NAME}" \
    --file "${COMPOSE_FILE}" \
    "$@"
}

# Manual scripts require an accessible Docker daemon and Compose v2 plugin.
require_docker() {
  require_command docker
  checking "Checking access to the Docker daemon"
  docker info >/dev/null 2>&1 || die "the Docker daemon is unavailable to this user"
  ready "Docker daemon is available"
  checking "Checking Docker Compose v2"
  docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is unavailable"
  ready "Docker Compose v2 is available"
}
