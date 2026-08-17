#!/usr/bin/env bash
set -Eeuo pipefail

# Resolve the attached game, persistent state, RPC token, and host-ownership
# settings supplied by Compose or the API-managed container launcher.
GAME_DIR="${DSLE_GAME_DIR:-/opt/dsle/game}"
STATE_DIR="${DSLE_STATE_DIR:-/var/lib/dsle}"
PULSE_DIR="${PULSE_DIR:-${STATE_DIR}/pulse}"
PULSE_SOCKET="${PULSE_SOCKET:-${PULSE_DIR}/native}"
TOKEN_FILE="${DSLE_RUNTIME_TOKEN_FILE:-${STATE_DIR}/run/dsle.token}"
OUTPUT_DIR="${DSLE_OUTPUT_DIR:-${STATE_DIR}/results}"
SESSION_LOCK="${STATE_DIR}/.container-session.lock"
OUTPUT_MARKER="${STATE_DIR}/.dsle-output"
HOST_UID="${DSLE_HOST_UID:-}"
HOST_GID="${DSLE_HOST_GID:-}"

# Host ownership is optional, but UID and GID must be supplied together as
# numeric values when persistent files should belong to the calling user.
if [[ -n "${HOST_UID}" || -n "${HOST_GID}" ]]; then
  if [[ ! "${HOST_UID}" =~ ^[0-9]+$ || ! "${HOST_GID}" =~ ^[0-9]+$ ]]; then
    echo "DSLE runtime error: DSLE_HOST_UID and DSLE_HOST_GID must both be numeric." >&2
    exit 65
  fi
fi

# Canonicalize persistent paths and require every writable runtime location to
# remain inside the single host-mounted DSLE state directory.
state_real="$(readlink -m "${STATE_DIR}")"
output_real="$(readlink -m "${OUTPUT_DIR}")"
if [[ "${output_real}" != "${state_real}/"* ]]; then
  echo "DSLE runtime error: DSLE_OUTPUT_DIR must be inside DSLE_STATE_DIR." >&2
  exit 65
fi
STATE_DIR="${state_real}"
OUTPUT_DIR="${output_real}"
PULSE_DIR="${STATE_DIR}/pulse"
PULSE_SOCKET="${PULSE_DIR}/native"
XDG_RUNTIME_DIR="$(readlink -m "${XDG_RUNTIME_DIR:-${STATE_DIR}/xdg/container}")"
if [[ "${XDG_RUNTIME_DIR}" != "${STATE_DIR}/"* ]]; then
  echo "DSLE runtime error: XDG_RUNTIME_DIR must be inside DSLE_STATE_DIR." >&2
  exit 65
fi
SESSION_LOCK="${STATE_DIR}/.container-session.lock"
OUTPUT_MARKER="${STATE_DIR}/.dsle-output"

# Accept only state directories initialized by the host manager, and reject
# link-based substitutions for the ownership marker or lifecycle lock.
if [[ -L "${OUTPUT_MARKER}" || ! -f "${OUTPUT_MARKER}" ]] || \
   [[ "$(<"${OUTPUT_MARKER}")" != "dsle-output" ]]; then
  echo "DSLE runtime error: ${STATE_DIR} is not an initialized DSLE output directory." >&2
  exit 65
fi
if [[ -L "${SESSION_LOCK}" || ! -f "${SESSION_LOCK}" ]]; then
  echo "DSLE runtime error: unsafe or missing session lock: ${SESSION_LOCK}" >&2
  exit 65
fi

# The commercial game must arrive through the external read-only mount; it is
# deliberately absent from the image itself.
if [[ ! -f "${GAME_DIR}/DarkSoulsRemastered.exe" ]]; then
  echo "DSLE runtime error: ${GAME_DIR}/DarkSoulsRemastered.exe is missing." >&2
  echo "Mount a legally obtained Dark Souls: Remastered installation at ${GAME_DIR}." >&2
  exit 64
fi

# Validate and create the known state layout before Wine or the RPC server can
# write prefixes, logs, sockets, recordings, or experiment results.
reserved_directories=(
  "${STATE_DIR}/config"
  "${STATE_DIR}/dxvk-cache"
  "${STATE_DIR}/logs"
  "${STATE_DIR}/pulse"
  "${STATE_DIR}/recordings"
  "${STATE_DIR}/results"
  "${STATE_DIR}/rpc"
  "${STATE_DIR}/run"
  "${STATE_DIR}/wineprefixes"
  "${STATE_DIR}/xdg"
  "${XDG_RUNTIME_DIR}"
  "${OUTPUT_DIR}"
)
for directory in "${reserved_directories[@]}"; do
  if [[ -L "${directory}" || ( -e "${directory}" && ! -d "${directory}" ) ]]; then
    echo "DSLE runtime error: reserved output path is unsafe: ${directory}" >&2
    exit 65
  fi
done
mkdir -p \
  "${reserved_directories[@]}" \
  "${XDG_RUNTIME_DIR:-${STATE_DIR}/xdg/container}"
chmod 700 \
  "${PULSE_DIR}" \
  "${STATE_DIR}/rpc" \
  "${STATE_DIR}/run" \
  "${STATE_DIR}/xdg" \
  "${XDG_RUNTIME_DIR:-${STATE_DIR}/xdg/container}"

# API-created output directories belong to the calling host user. Keep all
# durable result locations host-writable even though Wine and the runtime run
# as root inside the container. The setgid bit carries that group onto newly
# created child directories, while umask keeps new files group-writable.
if [[ -n "${HOST_UID}" ]]; then
  durable_directories=(
    "${STATE_DIR}/config"
    "${STATE_DIR}/dxvk-cache"
    "${STATE_DIR}/logs"
    "${STATE_DIR}/recordings"
    "${STATE_DIR}/wineprefixes"
    "${OUTPUT_DIR}"
  )
  # Also assign private runtime directories to the host user. They retain
  # mode 0700; this merely ensures an abruptly stopped manual container does
  # not leave root-owned empty directories in the persistent output bind.
  chown "${HOST_UID}:${HOST_GID}" \
    "${reserved_directories[@]}" \
    "${XDG_RUNTIME_DIR:-${STATE_DIR}/xdg/container}"
  chmod 2770 "${durable_directories[@]}"
  umask 0002
fi

# The persistent output bind is an exclusive live-runtime resource. Holding
# this advisory lock for the entrypoint lifetime prevents two API/Compose
# containers from sharing Wine prefixes, save slots, sockets, or metadata.
exec 9<>"${SESSION_LOCK}"
if ! flock --nonblock 9; then
  echo "DSLE runtime error: output directory is already used by another container: ${STATE_DIR}" >&2
  exit 73
fi

# Require a private, host-created authentication token before exposing the
# runtime RPC server, even though its transport is a local Unix socket.
if [[ -L "${TOKEN_FILE}" || ! -f "${TOKEN_FILE}" || ! -s "${TOKEN_FILE}" ]]; then
  echo "DSLE runtime error: a non-empty regular token file is required at ${TOKEN_FILE}." >&2
  echo "Use the DSLE host manager or scripts/start.sh to create it securely." >&2
  exit 65
fi
token_mode="$(stat -c '%a' "${TOKEN_FILE}")"
if [[ ! "${token_mode}" =~ ^[0-7]{3,4}$ ]] || (( (8#${token_mode} & 077) != 0 )); then
  echo "DSLE runtime error: ${TOKEN_FILE} must not be accessible by group or other users." >&2
  exit 65
fi

export PULSE_SERVER="unix:${PULSE_SOCKET}"
# PulseAudio runs as root in the runtime, whereas persistent output is assigned
# to the calling host UID. Give Pulse its own ephemeral root-owned XDG runtime
# directory while keeping only the native socket in the persistent bind.
PULSE_RUNTIME_DIR="/tmp/dsle-pulse-runtime"
mkdir -p "${PULSE_RUNTIME_DIR}"
chmod 700 "${PULSE_RUNTIME_DIR}"
if [[ -L "${PULSE_SOCKET}" || -S "${PULSE_SOCKET}" ]]; then
  unlink "${PULSE_SOCKET}"
elif [[ -e "${PULSE_SOCKET}" ]]; then
  echo "DSLE runtime error: refusing non-socket PulseAudio path: ${PULSE_SOCKET}" >&2
  exit 65
fi
if ! XDG_RUNTIME_DIR="${PULSE_RUNTIME_DIR}" \
  pulseaudio -n --daemonize=yes --exit-idle-time=-1 --log-target=stderr \
  --load="module-native-protocol-unix socket=${PULSE_SOCKET} auth-anonymous=1" \
  --load="module-null-sink sink_name=dsle_null sink_properties=device.description=DSLE_Null_Sink" \
  --load="module-always-sink"; then
  echo "DSLE runtime warning: PulseAudio did not start; continuing without game audio." >&2
fi

# Replace the shell with the requested server or diagnostic command so signals
# and the final exit status propagate directly through the container runtime.
exec "$@"
