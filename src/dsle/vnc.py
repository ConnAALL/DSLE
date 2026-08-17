"""Small read-only VNC client used by the multi-instance CCTV viewer."""

from __future__ import annotations

import socket
import struct
import threading
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dsle.exceptions import ConfigurationError, RuntimeCommunicationError
from dsle.runtime.instances import load_instances

_RFB_VERSION_LENGTH = 12
_RAW_ENCODING = 0
_ZLIB_ENCODING = 6
_DESKTOP_SIZE_ENCODING = -223
_MAX_SERVER_NAME_BYTES = 4096
_MAX_RECTANGLES = 1024
_MAX_FRAME_WIDTH = 4096
_MAX_FRAME_HEIGHT = 2160


class VNCProtocolError(RuntimeCommunicationError):
    """The VNC peer returned unsupported or malformed RFB data."""


class _Socket(Protocol):
    def recv(self, size: int) -> bytes: ...

    def sendall(self, data: bytes) -> None: ...

    def settimeout(self, timeout: float) -> None: ...

    def close(self) -> None: ...


SocketFactory = Callable[[tuple[str, int], float], _Socket]


@dataclass(frozen=True)
class VNCEndpoint:
    """One named loopback VNC feed."""

    name: str
    host: str
    port: int


@dataclass(frozen=True)
class VNCFrame:
    """One framebuffer encoded as four-byte BGRX pixels."""

    width: int
    height: int
    bgrx: bytes
    server_name: str


def _default_socket_factory(address: tuple[str, int], timeout: float) -> _Socket:
    return socket.create_connection(address, timeout=timeout)


def _instance_order(name: str) -> tuple[int, str]:
    prefix, separator, suffix = name.rpartition("-")
    if separator and prefix == "dsr" and suffix.isdigit():
        return int(suffix), name
    return 1_000_000, name


def build_vnc_endpoints(
    config: str | Path,
    *,
    host: str = "127.0.0.1",
    host_port_start: int = 5901,
    selected_names: Sequence[str] | None = None,
) -> tuple[VNCEndpoint, ...]:
    """Map configured container VNC ports onto one contiguous host range."""

    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be non-empty")
    if (
        isinstance(host_port_start, bool)
        or not isinstance(host_port_start, int)
        or not 1 <= host_port_start <= 65535
    ):
        raise ValueError("host_port_start must be an integer in [1, 65535]")
    instances = load_instances(config)
    if not instances:
        raise ConfigurationError("The instance configuration contains no instances")
    ordered_names = tuple(sorted(instances, key=_instance_order))
    if selected_names is None:
        names = ordered_names
    else:
        names = tuple(selected_names)
        if not names:
            raise ConfigurationError("at least one viewer instance must be selected")
        if len(set(names)) != len(names):
            raise ConfigurationError("viewer instance names must be unique")
        unknown = [name for name in names if name not in instances]
        if unknown:
            raise ConfigurationError(
                f"unknown viewer instance(s): {', '.join(unknown)}; "
                f"available: {', '.join(ordered_names)}"
            )
    container_port_start = min(instance.vnc_port for instance in instances.values())
    endpoints: list[VNCEndpoint] = []
    for name in names:
        port = host_port_start + instances[name].vnc_port - container_port_start
        if not 1 <= port <= 65535:
            raise ConfigurationError(f"host VNC port for {name} falls outside [1, 65535]: {port}")
        endpoints.append(VNCEndpoint(name=name, host=host.strip(), port=port))
    return tuple(endpoints)


class ReadOnlyVNCClient:
    """Request raw framebuffer updates without ever sending input events."""

    def __init__(
        self,
        endpoint: VNCEndpoint,
        *,
        timeout: float = 3.0,
        socket_factory: SocketFactory | None = None,
    ):
        if timeout <= 0:
            raise ValueError("VNC timeout must be positive")
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self._socket_factory = socket_factory or _default_socket_factory
        self._socket: _Socket | None = None
        self._width = 0
        self._height = 0
        self._server_name = endpoint.name
        self._framebuffer = bytearray()
        self._zlib = zlib.decompressobj()
        self._lock = threading.Lock()

    @staticmethod
    def _read_exact(connection: _Socket, size: int) -> bytes:
        if size < 0:
            raise VNCProtocolError("negative VNC payload size")
        payload = bytearray()
        while len(payload) < size:
            chunk = connection.recv(size - len(payload))
            if not chunk:
                raise VNCProtocolError("VNC connection closed during a frame")
            payload.extend(chunk)
        return bytes(payload)

    @classmethod
    def _read_reason(cls, connection: _Socket) -> str:
        length = struct.unpack(">I", cls._read_exact(connection, 4))[0]
        if length > _MAX_SERVER_NAME_BYTES:
            raise VNCProtocolError("VNC failure reason is unreasonably large")
        return cls._read_exact(connection, length).decode("utf-8", errors="replace")

    @staticmethod
    def _validate_dimensions(width: int, height: int) -> None:
        if not 1 <= width <= _MAX_FRAME_WIDTH or not 1 <= height <= _MAX_FRAME_HEIGHT:
            raise VNCProtocolError(f"unsupported VNC framebuffer size: {width}x{height}")

    def _negotiate_security(self, connection: _Socket, minor: int) -> None:
        if minor == 3:
            security_type = struct.unpack(">I", self._read_exact(connection, 4))[0]
            if security_type == 0:
                raise VNCProtocolError(
                    f"VNC server rejected the connection: {self._read_reason(connection)}"
                )
            if security_type != 1:
                raise VNCProtocolError(
                    "VNC server requires authentication; only local no-auth feeds are supported"
                )
            return

        count = self._read_exact(connection, 1)[0]
        if count == 0:
            raise VNCProtocolError(
                f"VNC server rejected the connection: {self._read_reason(connection)}"
            )
        security_types = self._read_exact(connection, count)
        if 1 not in security_types:
            raise VNCProtocolError(
                "VNC server requires authentication; only local no-auth feeds are supported"
            )
        connection.sendall(b"\x01")
        result = struct.unpack(">I", self._read_exact(connection, 4))[0]
        if result != 0:
            reason = self._read_reason(connection) if minor >= 8 else "security negotiation failed"
            raise VNCProtocolError(f"VNC security negotiation failed: {reason}")

    def _connect(self) -> _Socket:
        connection = self._socket_factory((self.endpoint.host, self.endpoint.port), self.timeout)
        try:
            connection.settimeout(self.timeout)
            version = self._read_exact(connection, _RFB_VERSION_LENGTH)
            if not version.startswith(b"RFB 003.") or not version.endswith(b"\n"):
                raise VNCProtocolError(f"invalid RFB version banner: {version!r}")
            try:
                server_minor = int(version[8:11])
            except ValueError as exc:
                raise VNCProtocolError(f"invalid RFB version banner: {version!r}") from exc
            minor = 8 if server_minor >= 8 else 7 if server_minor >= 7 else 3
            connection.sendall(f"RFB 003.00{minor}\n".encode("ascii"))
            self._negotiate_security(connection, minor)

            # Shared mode is essential: the CCTV viewer never takes ownership
            # away from TigerVNC or another passive observer.
            connection.sendall(b"\x01")
            server_init = self._read_exact(connection, 24)
            width, height = struct.unpack(">HH", server_init[:4])
            self._validate_dimensions(width, height)
            name_length = struct.unpack(">I", server_init[20:24])[0]
            if name_length > _MAX_SERVER_NAME_BYTES:
                raise VNCProtocolError("VNC desktop name is unreasonably large")
            server_name = self._read_exact(connection, name_length).decode(
                "utf-8", errors="replace"
            )

            # Ask for 32-bit little-endian true color. Raw rectangles then
            # arrive as B, G, R, unused bytes and need no codec dependency.
            pixel_format = struct.pack(">BBBBHHHBBB3x", 32, 24, 0, 1, 255, 255, 255, 16, 8, 0)
            connection.sendall(b"\x00\x00\x00\x00" + pixel_format)
            # Prefer zlib to keep 30-feed sampling practical, with raw as a
            # universally supported fallback.
            connection.sendall(struct.pack(">BBHii", 2, 0, 2, _ZLIB_ENCODING, _RAW_ENCODING))
            self._width = width
            self._height = height
            self._server_name = server_name or self.endpoint.name
            self._framebuffer = bytearray(width * height * 4)
            self._zlib = zlib.decompressobj()
            self._socket = connection
            return connection
        except Exception:
            connection.close()
            raise

    def _handle_update(self, connection: _Socket) -> None:
        _padding, rectangle_count = struct.unpack(">BH", self._read_exact(connection, 3))
        if rectangle_count > _MAX_RECTANGLES:
            raise VNCProtocolError(f"VNC update contains too many rectangles: {rectangle_count}")
        for _index in range(rectangle_count):
            x, y, width, height, encoding = struct.unpack(
                ">HHHHi", self._read_exact(connection, 12)
            )
            if encoding == _DESKTOP_SIZE_ENCODING:
                self._validate_dimensions(width, height)
                self._width = width
                self._height = height
                self._framebuffer = bytearray(width * height * 4)
                continue
            if encoding not in {_RAW_ENCODING, _ZLIB_ENCODING}:
                raise VNCProtocolError(f"VNC server returned unsupported encoding {encoding}")
            if width < 1 or height < 1 or x + width > self._width or y + height > self._height:
                raise VNCProtocolError(
                    f"VNC rectangle falls outside framebuffer: {x},{y} {width}x{height}"
                )
            expected_bytes = width * height * 4
            if encoding == _RAW_ENCODING:
                rectangle = self._read_exact(connection, expected_bytes)
            else:
                compressed_length = struct.unpack(">I", self._read_exact(connection, 4))[0]
                if compressed_length > max(expected_bytes * 2, 1 << 20):
                    raise VNCProtocolError(
                        f"compressed VNC rectangle is unreasonably large: {compressed_length}"
                    )
                try:
                    rectangle = self._zlib.decompress(
                        self._read_exact(connection, compressed_length)
                    )
                except zlib.error as exc:
                    raise VNCProtocolError(f"invalid compressed VNC rectangle: {exc}") from exc
                if len(rectangle) != expected_bytes:
                    raise VNCProtocolError(
                        f"compressed VNC rectangle decoded to {len(rectangle)} bytes; "
                        f"expected {expected_bytes}"
                    )
            row_bytes = width * 4
            for row in range(height):
                source = row * row_bytes
                destination = ((y + row) * self._width + x) * 4
                self._framebuffer[destination : destination + row_bytes] = rectangle[
                    source : source + row_bytes
                ]

    def _receive_frame(self, connection: _Socket) -> VNCFrame:
        while True:
            message_type = self._read_exact(connection, 1)[0]
            if message_type == 0:
                self._handle_update(connection)
                return VNCFrame(
                    width=self._width,
                    height=self._height,
                    bgrx=bytes(self._framebuffer),
                    server_name=self._server_name,
                )
            if message_type == 2:  # Bell; it has no payload.
                continue
            if message_type == 3:  # ServerCutText; discard clipboard text.
                length = struct.unpack(">I", self._read_exact(connection, 7)[3:])[0]
                if length > _MAX_SERVER_NAME_BYTES:
                    raise VNCProtocolError("VNC clipboard payload is unreasonably large")
                self._read_exact(connection, length)
                continue
            raise VNCProtocolError(f"unsupported VNC server message type {message_type}")

    def capture(self) -> VNCFrame:
        """Fetch one complete framebuffer, reconnecting after earlier failures."""

        with self._lock:
            try:
                connection = self._socket or self._connect()
                connection.sendall(
                    struct.pack(
                        ">BBHHHH",
                        3,
                        0,
                        0,
                        0,
                        self._width,
                        self._height,
                    )
                )
                return self._receive_frame(connection)
            except Exception:
                self.close()
                raise

    def close(self) -> None:
        """Close the passive VNC connection; the game is never affected."""

        connection = self._socket
        self._socket = None
        if connection is not None:
            connection.close()

    def __enter__(self) -> ReadOnlyVNCClient:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()
