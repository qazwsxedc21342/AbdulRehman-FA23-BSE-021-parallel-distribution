"""
tests/test_integrity.py
-----------------------
Adversarial wire-level tests for Task 4 (file integrity validation):

* upload declared with the **wrong SHA-256** -> worker must reject it,
* truncated upload (declared size > bytes sent) -> must fail cleanly,
* protocol version mismatch -> handshake must be refused.

These drive the raw protocol directly (bypassing the client helpers) so the
server-side validation is what is actually under test.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    MsgType,
    recv_frame,
    send_frame,
    send_json,
    sha256_bytes,
)
from server.daemon import WorkerDaemon  # noqa: E402


def _decode(payload: bytes) -> dict:
    return json.loads(payload.decode("utf-8"))


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workdir = tempfile.mkdtemp(prefix="lab4_integrity_")
        cls.daemon = WorkerDaemon(host="127.0.0.1", port=0, workers=1,
                                  workdir=cls.workdir)
        cls.daemon.start()
        cls.port = cls.daemon.port

    @classmethod
    def tearDownClass(cls):
        cls.daemon.stop()
        shutil.rmtree(cls.workdir, ignore_errors=True)

    # -- helpers ---------------------------------------------------------- #
    def _raw_session(self) -> socket.socket:
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        send_json(sock, MsgType.HELLO, {
            "protocol_version": PROTOCOL_VERSION,
            "client_id": "integrity-test",
            "nonce": "n-1",
            "timestamp": time.time(),
        })
        mtype, payload = recv_frame(sock)
        self.assertEqual(mtype, MsgType.HELLO_ACK)
        self.addCleanup(sock.close)
        return sock

    def _submit(self, sock: socket.socket, data: bytes,
                declared_sha: str) -> str:
        send_json(sock, MsgType.JOB_SUBMIT, {
            "task": "synthetic",
            "params": {"work_units": 3},
            "input_name": "payload.bin",
            "input_size": len(data),
            "input_sha256": declared_sha,
            "output_name": "payload_out.bin",
            "client_id": "integrity-test",
        })
        mtype, payload = recv_frame(sock)
        if mtype == MsgType.JOB_REJECTED:
            raise AssertionError(f"rejected: {_decode(payload)}")
        self.assertEqual(mtype, MsgType.JOB_ACCEPTED)
        return _decode(payload)["job_id"]
    # -- tests ------------------------------------------------------------ #
    def test_wrong_checksum_is_rejected(self):
        sock = self._raw_session()
        data = os.urandom(200_000)
        job_id = self._submit(sock, data, declared_sha=sha256_bytes(data))

        send_json(sock, MsgType.FILE_BEGIN,
                  {"job_id": job_id, "size": len(data), "sha256": "0" * 64})
        send_frame(sock, MsgType.FILE_CHUNK, data)
        send_json(sock, MsgType.FILE_END,
                  {"job_id": job_id, "size": len(data), "sha256": "0" * 64})

        # Drain progress/log/heartbeat frames until the transfer verdict.
        ack = None
        for _ in range(64):
            mtype, payload = recv_frame(sock)
            if mtype == MsgType.FILE_ACK:
                ack = _decode(payload)
                break
        self.assertIsNotNone(ack, "worker must answer FILE_ACK")
        self.assertFalse(ack["ok"])
        self.assertIn("checksum", ack["error"].lower())
        # The corrupted artefact must be deleted, not executed.
        self.assertFalse(
            [f for f in os.listdir(self.workdir) if f.startswith(job_id)],
            "rejected upload must not be kept on the worker")

    def test_truncated_upload_is_rejected(self):
        sock = self._raw_session()
        data = os.urandom(100_000)
        declared = len(data) + 4096  # lie about the size
        job_id = self._submit(sock, data, declared_sha=sha256_bytes(data))

        send_json(sock, MsgType.FILE_BEGIN, {"job_id": job_id, "size": declared,
                                             "sha256": sha256_bytes(data)})
        send_frame(sock, MsgType.FILE_CHUNK, data)
        # Socket closes before the declared size arrives.
        sock.shutdown(socket.SHUT_WR)

        # The worker either times out or reports a failed transfer.
        deadline = time.time() + config_io_timeout() + 2
        ack = None
        while time.time() < deadline:
            try:
                mtype, payload = recv_frame(sock)
            except (OSError, ConnectionError):
                break
            if mtype == MsgType.FILE_ACK:
                ack = _decode(payload)
                break
            if mtype in (MsgType.PROGRESS, MsgType.HEARTBEAT, MsgType.LOG):
                continue
        if ack is not None:
            self.assertFalse(ack.get("ok", True))

    def test_protocol_version_mismatch_refused(self):
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        self.addCleanup(sock.close)
        send_json(sock, MsgType.HELLO, {"protocol_version": 999,
                                        "client_id": "old-client"})
        mtype, payload = recv_frame(sock)
        self.assertEqual(mtype, MsgType.ERROR)
        self.assertIn("version", _decode(payload)["error"].lower())

    def test_oversize_submission_rejected(self):
        sock = self._raw_session()
        send_json(sock, MsgType.JOB_SUBMIT, {
            "task": "synthetic",
            "params": {},
            "input_name": "huge.bin",
            "input_size": 1 << 40,  # 1 TiB - beyond the guard rail
            "input_sha256": "0" * 64,
            "output_name": "out.bin",
        })
        mtype, payload = recv_frame(sock)
        self.assertEqual(mtype, MsgType.JOB_REJECTED)
        self.assertIn("limit", _decode(payload)["error"].lower())

    def test_unknown_output_request_is_reported(self):
        sock = self._raw_session()
        send_json(sock, MsgType.OUTPUT_REQUEST, {"job_id": "deadbeef"})
        mtype, payload = recv_frame(sock)
        self.assertEqual(mtype, MsgType.ERROR)


def config_io_timeout() -> float:
    from common import config

    return config.IO_TIMEOUT


if __name__ == "__main__":
    unittest.main()
