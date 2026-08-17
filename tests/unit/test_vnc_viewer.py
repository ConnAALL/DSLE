from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from dsle.cli.viewer import _resolve_endpoints, build_parser, main
from dsle.runtime.instances import generate_instances, write_instance_config
from dsle.vnc import ReadOnlyVNCClient, VNCEndpoint, build_vnc_endpoints


class FakeSocket:
    def __init__(self, responses: bytes):
        self.responses = bytearray(responses)
        self.sent: list[bytes] = []
        self.timeout: float | None = None
        self.closed = False

    def recv(self, size: int) -> bytes:
        selected = min(size, 3, len(self.responses))
        result = bytes(self.responses[:selected])
        del self.responses[:selected]
        return result

    def sendall(self, data: bytes) -> None:
        self.sent.append(bytes(data))

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def close(self) -> None:
        self.closed = True


def rfb_session(rectangle: bytes, *, encoding: int) -> bytes:
    name = b"DSLE test"
    server_init = struct.pack(">HH", 2, 1) + (b"\x00" * 16) + struct.pack(">I", len(name)) + name
    update = b"\x00" + struct.pack(">BH", 0, 1)
    update += struct.pack(">HHHHi", 0, 0, 2, 1, encoding)
    if encoding == 6:
        compressor = zlib.compressobj()
        encoded = compressor.compress(rectangle) + compressor.flush(zlib.Z_SYNC_FLUSH)
        update += struct.pack(">I", len(encoded)) + encoded
    else:
        update += rectangle
    return b"RFB 003.008\n" + b"\x01\x01" + struct.pack(">I", 0) + server_init + update


@pytest.mark.parametrize("encoding", [0, 6])
def test_read_only_vnc_client_decodes_raw_and_compressed_frames(encoding: int) -> None:
    pixels = bytes([30, 20, 10, 0, 3, 2, 1, 0])
    connection = FakeSocket(rfb_session(pixels, encoding=encoding))
    client = ReadOnlyVNCClient(
        VNCEndpoint("dsr-1", "127.0.0.1", 5901),
        socket_factory=lambda _address, _timeout: connection,
    )

    frame = client.capture()

    assert (frame.width, frame.height, frame.server_name) == (2, 1, "DSLE test")
    assert frame.bgrx == pixels
    assert connection.sent[0] == b"RFB 003.008\n"
    assert connection.sent[1:3] == [b"\x01", b"\x01"]
    # After setup, the client only sends SetPixelFormat, SetEncodings, and
    # FramebufferUpdateRequest. RFB keyboard/pointer message types are 4 and 5.
    assert [message[0] for message in connection.sent[3:]] == [0, 2, 3]
    assert struct.unpack(">H", connection.sent[4][2:4])[0] == 2
    assert connection.sent[-1][1] == 0  # non-incremental full snapshot


def test_configured_endpoints_map_to_contiguous_host_ports(tmp_path: Path) -> None:
    instances = generate_instances(
        3,
        wineprefix_root=tmp_path / "wineprefixes",
        vnc_port_start=5901,
    )
    config = write_instance_config(tmp_path / "instances.json", instances)

    assert build_vnc_endpoints(config, host_port_start=6901) == (
        VNCEndpoint("dsr-1", "127.0.0.1", 6901),
        VNCEndpoint("dsr-2", "127.0.0.1", 6902),
        VNCEndpoint("dsr-3", "127.0.0.1", 6903),
    )
    assert build_vnc_endpoints(
        config,
        host="localhost",
        host_port_start=6901,
        selected_names=("dsr-3", "dsr-1"),
    ) == (
        VNCEndpoint("dsr-3", "localhost", 6903),
        VNCEndpoint("dsr-1", "localhost", 6901),
    )


def test_explicit_endpoints_support_dynamic_managed_ports() -> None:
    args = build_parser().parse_args(
        [
            "--endpoint",
            "dsr-1=127.0.0.1:32781",
            "--endpoint",
            "dsr-2=127.0.0.1:41102",
        ]
    )
    assert _resolve_endpoints(args) == (
        VNCEndpoint("dsr-1", "127.0.0.1", 32781),
        VNCEndpoint("dsr-2", "127.0.0.1", 41102),
    )


def test_viewer_cli_rejects_missing_or_ambiguous_endpoint_sources(capsys) -> None:
    assert main([]) == 2
    assert "provide --config" in capsys.readouterr().err

    assert main(["--config", "unused.json", "--endpoint", "dsr-1=localhost:5901"]) == 2
    assert "cannot be combined" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "nan", "61"])
def test_viewer_cli_bounds_the_sampling_period(value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--period", value])
