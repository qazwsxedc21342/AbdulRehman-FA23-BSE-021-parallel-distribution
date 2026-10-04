"""
tests/test_protocol.py
----------------------
Unit tests for the shared wire protocol (framing, hashing, sanitisation).

Run from the project root::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import socket
import struct
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.protocol import (  # noqa: E402
    HEADER,
    HEADER_SIZE,
    MAX_PAYLOAD,
    MsgType,
    ProtocolError,
    human_bytes,
    recv_frame,
    safe_filename,
    send_frame,
    send_json,
    sha256_bytes,
    sha256_file,
)


def socketpair():
    return socket.socketpair()


class FramingTests(unittest.TestCase):
    def test_roundtrip_binary_frame(self):
        a, b = socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        payload = os.urandom(4096)
        send_frame(a, MsgType.FILE_CHUNK, payload)
        mtype, received = recv_frame(b)
        self.assertEqual(mtype, MsgType.FILE_CHUNK)
        self.assertEqual(received, payload)

    def test_roundtrip_json_frame(self):
        a, b = socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        send_json(a, MsgType.HELLO, {"protocol_version": 1, "nonce": "abc"})
        mtype, payload = recv_frame(b)
        self.assertEqual(mtype, MsgType.HELLO)
        self.assertIn(b'"nonce":"abc"', payload)

    def test_empty_payload(self):
        a, b = socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        send_frame(a, MsgType.HEARTBEAT, b"")
        mtype, payload = recv_frame(b)
        self.assertEqual((mtype, payload), (MsgType.HEARTBEAT, b""))

    def test_header_layout(self):
        self.assertEqual(HEADER_SIZE, 5)
        self.assertEqual(HEADER.pack(7, MsgType.PING), struct.pack(">IB", 7, 3))

    def test_oversize_payload_rejected(self):
        a, b = socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        with self.assertRaises(ProtocolError):
            send_frame(a, MsgType.FILE_CHUNK, b"\0" * (MAX_PAYLOAD + 1))

    def test_partial_frame_raises_connection_error(self):
        a, b = socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        # Claim 100 bytes then close: receiver must not hang or return junk.
        a.sendall(HEADER.pack(100, MsgType.FILE_CHUNK))
        a.close()
        with self.assertRaises((ConnectionError, OSError)):
            recv_frame(b)

    def test_closed_socket_raises(self):
        a, b = socketpair()
        self.addCleanup(b.close)
        a.close()
        with self.assertRaises((ConnectionError, OSError)):
            recv_frame(b)


class HashTests(unittest.TestCase):
    def test_sha256_file_matches_bytes(self):
        data = os.urandom(300_000)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        self.addCleanup(os.remove, path)
        self.assertEqual(sha256_file(path), sha256_bytes(data))

    def test_known_vector(self):
        # SHA-256 of the empty string is a well known constant.
        self.assertEqual(
            sha256_bytes(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )


class SanitisationTests(unittest.TestCase):
    def test_strips_directories(self):
        self.assertEqual(safe_filename("../../etc/passwd"), "passwd")
        self.assertEqual(safe_filename(r"C:\Windows\System32\evil.exe"),
                         "evil.exe")

    def test_strips_dangerous_chars(self):
        self.assertNotIn("|", safe_filename("a|b|c.txt"))

    def test_empty_falls_back(self):
        self.assertEqual(safe_filename("///"), "input.bin")


class MiscTests(unittest.TestCase):
    def test_human_bytes(self):
        self.assertEqual(human_bytes(512), "512.0 B")
        self.assertEqual(human_bytes(2048), "2.0 KiB")

    def test_msg_types_are_unique(self):
        values = list(MsgType)
        self.assertEqual(len(values), len(set(values)))


if __name__ == "__main__":
    unittest.main()
