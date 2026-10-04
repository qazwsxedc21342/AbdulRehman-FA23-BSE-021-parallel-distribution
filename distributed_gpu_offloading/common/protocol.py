"""
CSC-334 : Parallel and Distributed Computing
Lab 04 : Custom Distributed Task Offloading & Remote GPU Rendering System

common/protocol.py
------------------
Wire protocol shared by the client and the server daemon.

Framing layout (every message on the socket):

    +-----------------+--------------+-------------------+
    | length (4 bytes) | type (1 byte)| payload (N bytes) |
    | uint32 big-endian| uint8        | N = length        |
    +-----------------+--------------+-------------------+

* ``length`` is the length of the *payload* only (not counting the 5-byte header).
* ``type`` identifies the message (see :class:`MsgType`).
* Control messages carry a UTF-8 encoded JSON object as payload.
* Bulk data (file chunks) carries raw bytes as payload.

This gives us deterministic framing (no delimiters, no ambiguity), cheap
error detection (bad magic/length -> protocol error) and the ability to
mix JSON control traffic with binary bulk transfers on one socket.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import socket
import struct
import time
from typing import Any, Optional, Tuple

# --------------------------------------------------------------------------- #
# Protocol constants
# --------------------------------------------------------------------------- #

PROTOCOL_VERSION = 1
HEADER = struct.Struct(">IB")          # payload length (uint32) + message type (uint8)
HEADER_SIZE = HEADER.size              # 5 bytes
MAX_PAYLOAD = 16 * 1024 * 1024         # hard cap: 16 MiB per frame (guards memory)
DEFAULT_CHUNK_SIZE = 1024 * 1024       # 1 MiB bulk chunks
HASH_BLOCK = 1024 * 1024


class ProtocolError(Exception):
    """Raised when the peer violates the framing/JSON contract."""


class MsgType(enum.IntEnum):
    """Message identifiers used on the wire."""

    # --- handshake / discovery (Task 1) ---------------------------------- #
    HELLO = 1              # C -> S  initial handshake
    HELLO_ACK = 2          # S -> C  handshake acceptance + worker info
    PING = 3               # C -> S  latency probe
    PONG = 4               # S -> C  latency reply

    # --- job submission & input transfer --------------------------------- #
    JOB_SUBMIT = 10        # C -> S  job description (metadata only)
    JOB_ACCEPTED = 11      # S -> C  job id assigned / queued
    JOB_REJECTED = 12      # S -> C  validation failure
    FILE_BEGIN = 13        # C -> S  transfer start (size, sha256)
    FILE_CHUNK = 14        # C -> S  raw bytes
    FILE_END = 15          # C -> S  transfer finished (sha256)
    FILE_ACK = 16          # S -> C  checksum / size validation result

    # --- execution feedback (Task 4) ------------------------------------- #
    PROGRESS = 20          # S -> C  asynchronous percentage stream
    LOG = 21               # S -> C  structured log line
    HEARTBEAT = 22         # S -> C  keep-alive while queued/running

    # --- job lifecycle ---------------------------------------------------- #
    JOB_DONE = 30          # S -> C  execution finished successfully
    JOB_FAILED = 31        # S -> C  execution failed
    JOB_CANCEL = 32        # C -> S  request cancellation

    # --- output download -------------------------------------------------- #
    OUTPUT_REQUEST = 40    # C -> S  ask for the rendered artefact
    OUTPUT_BEGIN = 41      # S -> C  output metadata (size, sha256)
    OUTPUT_CHUNK = 42      # S -> C  raw bytes
    OUTPUT_END = 43        # S -> C  transfer finished
    OUTPUT_ACK = 44        # C -> S  checksum validation result

    # --- misc ------------------------------------------------------------- #
    ERROR = 90             # either direction, fatal protocol/runtime error


class MessageTypeError(ProtocolError):
    pass


# --------------------------------------------------------------------------- #
# Send / receive primitives
# --------------------------------------------------------------------------- #

def send_frame(sock: socket.socket, msg_type: int, payload: bytes = b"") -> None:
    """Send one framed message. Raises on socket errors / oversize payloads."""
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(f"payload too large: {len(payload)} > {MAX_PAYLOAD}")
    sock.sendall(HEADER.pack(len(payload), int(msg_type)) + payload)


def send_json(sock: socket.socket, msg_type: int, obj: Any) -> None:
    """Send a control message (JSON payload)."""
    send_frame(sock, msg_type, json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes or raise ConnectionError/TimeoutError."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("peer closed the connection while receiving")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket) -> Tuple[int, bytes]:
    """Receive one framed message, returning ``(msg_type, payload)``."""
    header = recv_exact(sock, HEADER_SIZE)
    length, msg_type = HEADER.unpack(header)
    if length > MAX_PAYLOAD:
        raise ProtocolError(f"peer announced illegal frame length {length}")
    payload = recv_exact(sock, length) if length else b""
    return int(msg_type), payload


def recv_json(sock: socket.socket, expect: Optional[int] = None) -> Tuple[int, dict]:
    """Receive a control message and decode its JSON payload."""
    msg_type, payload = recv_frame(sock)
    if expect is not None and msg_type != expect:
        raise MessageTypeError(f"expected {MsgType(expect).name}, got {MsgType(msg_type).name if msg_type in MsgType._value2member_map_ else msg_type}")
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"malformed JSON payload: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError("control payload must be a JSON object")
    return msg_type, obj


def try_recv_json(sock: socket.socket) -> Tuple[int, dict]:
    """Non-strict variant: decode whatever control message arrives."""
    return recv_json(sock, expect=None)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def sha256_file(path: str) -> str:
    """Streaming SHA-256 of a file (constant memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now() -> float:
    return time.time()


def safe_filename(name: str) -> str:
    """Strip directories and dangerous characters from an inbound filename."""
    name = os.path.basename(name.replace("\\", "/"))
    name = "".join(ch for ch in name if ch.isalnum() or ch in "._- ()[]")
    return name[:180] or "input.bin"


def set_socket_options(sock: socket.socket, *, tcp_nodelay: bool = True,
                       keepalive: bool = True) -> None:
    if tcp_nodelay:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
    if keepalive:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            pass


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"
