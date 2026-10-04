"""
client/api.py
-------------
Core client library for the Distributed Task Offloading system.

Implements the client half of the protocol:

* ``connect()``   - TCP connect with timeout + HELLO/HELLO_ACK handshake
                    including protocol version negotiation and capability
                    exchange (Task 1).
* ``ping()``      - multi-sample latency (RTT) measurement with min/avg/max
                    statistics before any job is submitted (Task 1).
* ``submit()``    - job submission, chunked SHA-256 verified upload (Task 4).
* ``run_job()``   - full lifecycle: submit -> upload -> wait for asynchronous
                    PROGRESS/LOG/HEARTBEAT events -> download verified output.
* robustness      - connect retries, socket timeouts, checksum validation of
                    both the input and the output, and graceful
                    ``OffloadError`` reporting on disconnects (Task 4).

The GUI and the CLI both sit on top of this class, so the exact same code
path is exercised by the desktop app, the benchmark harness and the tests.
"""

from __future__ import annotations

import os
import queue
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from common import config
from common.protocol import (
    PROTOCOL_VERSION,
    MsgType,
    ProtocolError,
    human_bytes,
    recv_frame,
    send_frame,
    send_json,
    set_socket_options,
    sha256_file,
)


class OffloadError(RuntimeError):
    """Any recoverable-but-fatal client-side failure (reported to the UI)."""


@dataclass
class ProgressEvent:
    percent: float
    stage: str
    elapsed: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass
class LogEvent:
    message: str
    level: str = "info"
    ts: float = 0.0


@dataclass
class JobResult:
    job_id: str
    output_path: str
    output_bytes: int
    output_sha256: str
    duration_s: float
    result: dict
    progress: List[ProgressEvent] = field(default_factory=list)
    logs: List[LogEvent] = field(default_factory=list)
    transfer_bytes: int = 0
    transfer_seconds: float = 0.0
    rtt_ms: float = 0.0

    @property
    def speedup_note(self) -> str:
        return f"{human_bytes(self.transfer_bytes)} up in {self.transfer_seconds:.2f}s"


class OffloadClient:
    """
    Blocking client with an event callback for asynchronous progress.

    Typical use::

        client = OffloadClient("192.168.1.1", 5000)
        info = client.connect()          # handshake + capabilities
        rtt  = client.ping()             # latency probe
        result = client.run_job("video", "clip.mp4", params={...},
                                on_progress=..., on_log=...)
        client.close()
    """

    def __init__(self, host: str = config.DEFAULT_HOST, port: int = config.DEFAULT_PORT,
                 client_id: Optional[str] = None,
                 on_progress: Optional[Callable[[ProgressEvent], None]] = None,
                 on_log: Optional[Callable[[LogEvent], None]] = None,
                 on_heartbeat: Optional[Callable[[dict], None]] = None,
                 download_dir: str = config.CLIENT_DOWNLOAD_DIR):
        self.host = host
        self.port = port
        self.client_id = client_id or f"client-{uuid.uuid4().hex[:8]}"
        self.on_progress = on_progress
        self.on_log = on_log
        self.on_heartbeat = on_heartbeat
        self.download_dir = download_dir

        self.sock: Optional[socket.socket] = None
        self.server_info: Optional[dict] = None
        self.rtt_ms: float = 0.0
        self.connected = False

        # Single reader thread demultiplexes events into these queues so a
        # GUI can poll without blocking its main loop.
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._send_lock = threading.Lock()
        self._current_job: Optional[str] = None
        self._job_result: Optional[dict] = None
        self._job_error: Optional[str] = None
        self._download: Optional[dict] = None
        self._download_ready = threading.Event()
        self._job_done = threading.Event()
        self._file_ack: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # Connection & handshake (Task 1)
    # ------------------------------------------------------------------ #
    def connect(self, retries: int = config.MAX_SUBMIT_ATTEMPTS,
                timeout: float = config.CONNECT_TIMEOUT) -> dict:
        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                sock = socket.create_connection((self.host, self.port), timeout=timeout)
                set_socket_options(sock)
                sock.settimeout(config.HANDSHAKE_TIMEOUT)
                self.sock = sock
                self.connected = True

                nonce = uuid.uuid4().hex
                t0 = time.perf_counter()
                send_json(sock, MsgType.HELLO, {
                    "protocol_version": PROTOCOL_VERSION,
                    "client_id": self.client_id,
                    "client_name": socket.gethostname(),
                    "nonce": nonce,
                    "timestamp": time.time(),
                    "user": os.environ.get("USERNAME") or os.environ.get("USER") or "student",
                })
                msg_type, ack = self._recv_json()
                if msg_type == MsgType.ERROR:
                    raise OffloadError(f"handshake rejected: {ack.get('error')}")
                if msg_type != MsgType.HELLO_ACK:
                    raise ProtocolError(f"expected HELLO_ACK, got {msg_type}")
                if ack.get("echo_nonce") != nonce:
                    raise ProtocolError("handshake nonce mismatch (peer is not our worker)")
                if int(ack.get("protocol_version", -1)) != PROTOCOL_VERSION:
                    raise ProtocolError("protocol version mismatch with worker")

                self.handshake_ms = (time.perf_counter() - t0) * 1000.0
                self.server_info = ack
                sock.settimeout(config.IO_TIMEOUT)

                # Reader thread: pushes asynchronous events while we block.
                self._stop.clear()
                self._reader = threading.Thread(target=self._read_loop,
                                                name="net-reader", daemon=True)
                self._reader.start()

                self.log(f"Handshake OK with {ack.get('server_id')} "
                         f"(session {ack.get('session_id')}, "
                         f"{self.handshake_ms:.1f} ms)")
                env = (ack.get("environment") or {})
                self.log(f"Worker capabilities: {env.get('engine_summary', 'unknown')}")
                return ack
            except (OSError, ProtocolError, OffloadError) as exc:
                last_error = exc
                self.connected = False
                self._teardown_socket()
                if attempt < retries:
                    delay = 0.5 * attempt
                    self.log(f"Connection attempt {attempt} failed ({exc}); "
                             f"retrying in {delay:.1f}s...", level="warning")
                    time.sleep(delay)
        raise OffloadError(f"cannot reach worker {self.host}:{self.port} "
                           f"after {retries} attempts: {last_error}")

    def _recv_json(self):
        assert self.sock is not None
        msg_type, payload = recv_frame(self.sock)
        import json
        try:
            obj = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise ProtocolError(f"malformed JSON from worker: {exc}") from exc
        return msg_type, obj

    # ------------------------------------------------------------------ #
    # Latency probe (Task 1)
    # ------------------------------------------------------------------ #
    def ping(self, samples: int = config.PING_PAYLOAD_COUNT) -> dict:
        """Measure RTT to the worker; returns stats and marks it unavailable on failure."""
        if not self.sock:
            raise OffloadError("not connected")
        rtts: List[float] = []
        for seq in range(samples):
            t0 = time.time()
            try:
                with self._send_lock:
                    send_json(self.sock, MsgType.PING,
                              {"seq": seq, "client_timestamp": t0})
                # PONG is consumed by the reader thread; wait for it there.
                msg_type, obj = self._wait_event("pong", timeout=config.HANDSHAKE_TIMEOUT)
                t1 = time.time()
                if msg_type == "pong":
                    rtts.append((t1 - t0) * 1000.0)
            except OffloadError as exc:
                # one lost PONG counts as packet loss; abort only if none arrive
                self.log(f"PING seq={seq} lost: {exc}")
            time.sleep(0.05)

        if not rtts:
            raise OffloadError("worker did not answer PING (unavailable)")
        if len(rtts) > 1:
            jitter = sum(abs(rtts[i] - rtts[i - 1]) for i in range(1, len(rtts))) \
                / (len(rtts) - 1)
        else:
            jitter = 0.0
        stats = {
            "min_ms": min(rtts),
            "avg_ms": sum(rtts) / len(rtts),
            "max_ms": max(rtts),
            "jitter_ms": jitter,
            "loss_pct": max(0.0, (samples - len(rtts)) / samples * 100.0),
            "samples": len(rtts),
            "requested": samples,
        }
        self.rtt_ms = stats["avg_ms"]
        self.log(f"Latency to worker: avg {stats['avg_ms']:.1f} ms "
                 f"(min {stats['min_ms']:.1f} / max {stats['max_ms']:.1f}, "
                 f"jitter {stats['jitter_ms']:.1f} ms, {len(rtts)}/{samples} replies)")
        return stats

    def _wait_event(self, kind: str, timeout: float = config.IO_TIMEOUT):
        deadline = time.time() + timeout
        stash: List[tuple] = []
        try:
            while time.time() < deadline:
                try:
                    item = self.events.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item[0] == kind:
                    return item
                stash.append(item)
            raise OffloadError(f"timed out waiting for '{kind}' event")
        finally:
            for item in stash:
                self.events.put(item)

    # ------------------------------------------------------------------ #
    # Reader loop (asynchronous event demultiplexer, Task 4)
    # ------------------------------------------------------------------ #
    def _read_loop(self) -> None:
        import json
        sock = self.sock
        if sock is None:
            return
        while not self._stop.is_set():
            try:
                sock.settimeout(config.IDLE_TIMEOUT)
                msg_type, payload = recv_frame(sock)
            except socket.timeout:
                continue
            except (ConnectionError, OSError, ProtocolError) as exc:
                if not self._stop.is_set():
                    self.events.put(("disconnected", {"error": str(exc)}))
                break

            obj: dict = {}
            if payload:
                try:
                    obj = json.loads(payload.decode("utf-8"))
                except Exception:
                    obj = {}

            if msg_type == MsgType.PROGRESS:
                if obj.get("stage") == "upload":
                    self.events.put(("upload", obj))
                    continue
                ev = ProgressEvent(
                    percent=float(obj.get("percent", 0.0)),
                    stage=str(obj.get("stage", "")),
                    elapsed=float(obj.get("elapsed", 0.0)),
                    raw=obj,
                )
                # Queued only: run_job() re-dispatches to the caller's
                # callback so handlers fire exactly once.
                self.events.put(("progress", ev))
            elif msg_type == MsgType.LOG:
                ev = LogEvent(message=str(obj.get("message", "")),
                              level=str(obj.get("level", "info")),
                              ts=float(obj.get("ts", 0.0)))
                self.events.put(("log", ev))
            elif msg_type == MsgType.HEARTBEAT:
                self.events.put(("heartbeat", obj))
                if self.on_heartbeat:
                    try:
                        self.on_heartbeat(obj)
                    except Exception:
                        pass
            elif msg_type == MsgType.PONG:
                self.events.put(("pong", obj))
            elif msg_type == MsgType.JOB_DONE:
                self._job_result = obj
                self._job_done.set()
                self.events.put(("done", obj))
            elif msg_type == MsgType.JOB_FAILED:
                self._job_error = str(obj.get("error", "unknown worker error"))
                self._job_done.set()
                self.events.put(("failed", obj))
            elif msg_type == MsgType.FILE_ACK:
                self._file_ack = obj
                self.events.put(("file_ack", obj))
            elif msg_type in (MsgType.OUTPUT_BEGIN, MsgType.OUTPUT_CHUNK,
                              MsgType.OUTPUT_END):
                self.events.put(("output", (msg_type, obj, payload)))
            elif msg_type == MsgType.JOB_ACCEPTED:
                self.events.put(("job_accepted", obj))
            elif msg_type == MsgType.JOB_REJECTED:
                self.events.put(("error", obj))
            elif msg_type == MsgType.ERROR:
                self.events.put(("error", obj))
            else:
                self.events.put(("other", (msg_type, obj)))

    # ------------------------------------------------------------------ #
    # Upload with integrity verification (Task 4)
    # ------------------------------------------------------------------ #
    def submit(self, task: str, input_path: str, params: Optional[dict] = None,
               output_name: Optional[str] = None, timeout: float = config.IO_TIMEOUT) -> dict:
        """Send a JOB_SUBMIT; returns the worker's JOB_ACCEPTED payload (job_id...)."""
        if not self.sock:
            raise OffloadError("not connected")
        if not os.path.isfile(input_path):
            raise OffloadError(f"input file not found: {input_path}")

        size = os.path.getsize(input_path)
        if size > config.MAX_INPUT_BYTES:
            raise OffloadError(f"input too large ({human_bytes(size)} > "
                               f"{human_bytes(config.MAX_INPUT_BYTES)})")
        name = output_name or self._default_output_name(input_path, task)

        self.log(f"Computing SHA-256 of {os.path.basename(input_path)} "
                 f"({human_bytes(size)})...")
        digest = sha256_file(input_path)

        spec = {
            "task": task,
            "params": params or {},
            "input_name": os.path.basename(input_path),
            "input_size": size,
            "input_sha256": digest,
            "output_name": name,
            "client_id": self.client_id,
            "submitted_at": time.time(),
        }
        return self._submit_spec(spec, timeout=timeout)

    # -- lower level submit helpers ------------------------------------- #
    def _submit_spec(self, spec: dict, timeout: float = config.IO_TIMEOUT) -> dict:
        """Send JOB_SUBMIT and block until JOB_ACCEPTED / JOB_REJECTED."""
        with self._send_lock:
            send_json(self.sock, MsgType.JOB_SUBMIT, spec)
        deadline = time.time() + timeout
        stash: List[tuple] = []
        try:
            while time.time() < deadline:
                try:
                    item = self.events.get(timeout=0.2)
                except queue.Empty:
                    continue
                kind, payload = item
                if kind == "job_accepted":
                    return payload
                if kind in ("error", "failed"):
                    raise OffloadError(payload.get("error", "worker rejected the job"))
                stash.append(item)
            raise OffloadError("timed out waiting for JOB_ACCEPTED")
        finally:
            for item in stash:
                self.events.put(item)

    @staticmethod
    def _default_output_name(input_path: str, task: str) -> str:
        stem, ext = os.path.splitext(os.path.basename(input_path))
        if task == "video":
            return f"{stem}_gpu{ext if ext else '.mp4'}"
        if task == "tensor":
            return f"{stem}_tensor.npy"
        return f"{stem}_render{ext if ext else '.bin'}"

    def upload(self, job_id: str, input_path: str,
               on_progress: Optional[Callable[[float, int, int], None]] = None) -> dict:
        """Chunked upload; blocks for FILE_ACK and verifies the worker's checksum."""
        size = os.path.getsize(input_path)
        digest = sha256_file(input_path)
        chunk_size = config.CHUNK_SIZE

        with self._send_lock:
            send_json(self.sock, MsgType.FILE_BEGIN,
                      {"job_id": job_id, "size": size, "sha256": digest})
            sent = 0
            t0 = time.perf_counter()
            with open(input_path, "rb") as handle:
                while True:
                    block = handle.read(chunk_size)
                    if not block:
                        break
                    send_frame(self.sock, MsgType.FILE_CHUNK, block)
                    sent += len(block)
                    if on_progress and sent % (8 * chunk_size) < chunk_size:
                        on_progress(100.0 * sent / max(size, 1), sent, size)
            send_json(self.sock, MsgType.FILE_END,
                      {"job_id": job_id, "size": size, "sha256": digest})
        upload_seconds = time.perf_counter() - t0
        if on_progress:
            on_progress(100.0, sent, size)

        # Wait for FILE_ACK
        deadline = time.time() + config.IO_TIMEOUT
        stash: List[tuple] = []
        try:
            while time.time() < deadline:
                try:
                    item = self.events.get(timeout=0.2)
                except queue.Empty:
                    continue
                kind, payload = item
                if kind == "file_ack":
                    if not payload.get("ok"):
                        raise OffloadError(
                            f"worker rejected the transfer: {payload.get('error')}")
                    if payload.get("sha256") != digest:
                        raise OffloadError("checksum mismatch reported by worker")
                    return {
                        "bytes": sent,
                        "seconds": upload_seconds,
                        "mbps": (sent / 1e6) / max(upload_seconds, 1e-6),
                        "sha256": digest,
                    }
                if kind == "disconnected":
                    raise OffloadError(f"connection lost during upload: {payload}")
                stash.append(item)
            raise OffloadError("timed out waiting for FILE_ACK")
        finally:
            for item in stash:
                self.events.put(item)

    # ------------------------------------------------------------------ #
    # Full job lifecycle
    # ------------------------------------------------------------------ #
    def run_job(self, task: str, input_path: str, params: Optional[dict] = None,
                output_name: Optional[str] = None,
                wait_timeout: float = 1800.0,
                download_dir: Optional[str] = None,
                on_progress: Optional[Callable[[ProgressEvent], None]] = None,
                on_log: Optional[Callable[[LogEvent], None]] = None) -> JobResult:
        if not self.sock:
            raise OffloadError("not connected - call connect() first")

        name = None
        size = os.path.getsize(input_path)
        digest = sha256_file(input_path)

        spec = {
            "task": task,
            "params": params or {},
            "input_name": os.path.basename(input_path),
            "input_size": size,
            "input_sha256": digest,
            "output_name": output_name or self._default_output_name(input_path, task),
            "client_id": self.client_id,
            "submitted_at": time.time(),
        }
        accepted = self._submit_spec(spec)
        job_id = accepted["job_id"]
        self._current_job = job_id
        self._job_done.clear()
        self._job_error = None
        self._job_result = None
        self.log(f"Job {job_id} accepted (queue position {accepted.get('queue_position')})")

        up = self.upload(job_id, input_path,
                         on_progress=lambda p, s, t: self.events.put(
                             ("upload", {"percent": p, "received": s, "total": t})))
        self.log(f"Upload complete: {human_bytes(up['bytes'])} in "
                 f"{up['seconds']:.2f}s ({up['mbps']:.1f} Mbit/s), "
                 f"sha256 verified")

        progress_events: List[ProgressEvent] = []
        log_events: List[LogEvent] = []

        # Drain asynchronous events until JOB_DONE / JOB_FAILED.
        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            if self._job_done.is_set():
                break
            try:
                kind, payload = self.events.get(timeout=0.25)
            except queue.Empty:
                continue
            if kind == "progress":
                progress_events.append(payload)
                if on_progress:
                    on_progress(payload)
            elif kind == "log":
                log_events.append(payload)
                if on_log:
                    on_log(payload)
            elif kind == "heartbeat":
                pass
            elif kind == "disconnected":
                raise OffloadError(f"connection lost while waiting for job: "
                                   f"{payload.get('error')}")
            elif kind == "failed":
                raise OffloadError(payload.get("error", "job failed on worker"))
            elif kind == "error":
                raise OffloadError(payload.get("error", "worker error"))

        if not self._job_done.is_set():
            raise OffloadError(f"job {job_id} timed out after {wait_timeout:.0f}s")
        if self._job_error:
            raise OffloadError(self._job_error)

        done = self._job_result or {}
        out_meta = done.get("output") or {}

        # Download the artefact (verified).
        out_path = self.download(job_id, out_meta.get("name") or spec["output_name"],
                                 download_dir or self.download_dir, expected=out_meta)

        return JobResult(
            job_id=job_id,
            output_path=out_path,
            output_bytes=int(out_meta.get("size", os.path.getsize(out_path))),
            output_sha256=str(out_meta.get("sha256", "")),
            duration_s=float(done.get("duration_s", 0.0)),
            result=done.get("result") or {},
            progress=progress_events,
            logs=log_events,
            transfer_bytes=up["bytes"],
            transfer_seconds=up["seconds"],
            rtt_ms=self.rtt_ms,
        )

    # ------------------------------------------------------------------ #
    # Output download with integrity verification (Task 4)
    # ------------------------------------------------------------------ #
    def download(self, job_id: str, output_name: str, dest_dir: str,
                 expected: Optional[dict] = None,
                 on_progress: Optional[Callable[[float, int, int], None]] = None) -> str:
        if not self.sock:
            raise OffloadError("not connected")
        os.makedirs(dest_dir, exist_ok=True)

        with self._send_lock:
            send_json(self.sock, MsgType.OUTPUT_REQUEST, {"job_id": job_id})

        meta: Optional[dict] = None
        expected_size = int((expected or {}).get("size", -1))
        expected_sha = str((expected or {}).get("sha256", ""))

        dest_path = os.path.join(dest_dir, os.path.basename(output_name))
        received = 0
        deadline = time.time() + config.IO_TIMEOUT * 4
        handle = None
        try:
            while time.time() < deadline:
                try:
                    item = self.events.get(timeout=0.5)
                except queue.Empty:
                    if handle:  # streaming in progress -> keep waiting
                        continue
                    continue
                kind, payload = item
                if kind != "output":
                    if kind in ("disconnected", "error"):
                        raise OffloadError(str(payload))
                    continue
                msg_type, obj, raw = payload

                if msg_type == MsgType.OUTPUT_BEGIN:
                    meta = obj
                    expected_size = int(obj.get("size", expected_size))
                    expected_sha = str(obj.get("sha256", expected_sha))
                    dest_path = os.path.join(dest_dir,
                                             os.path.basename(str(obj.get("name", output_name))))
                    tmp = dest_path + ".part"
                    handle = open(tmp, "wb")
                    received = 0
                    self.log(f"Receiving output {obj.get('name')} "
                             f"({human_bytes(expected_size)})...")
                elif msg_type == MsgType.OUTPUT_CHUNK:
                    if handle is None:
                        raise ProtocolError("OUTPUT_CHUNK before OUTPUT_BEGIN")
                    handle.write(raw)
                    received += len(raw)
                    if on_progress and expected_size > 0:
                        on_progress(100.0 * received / expected_size, received, expected_size)
                elif msg_type == MsgType.OUTPUT_END:
                    if handle:
                        handle.close()
                        handle = None
                    tmp = dest_path + ".part"
                    actual_size = os.path.getsize(tmp)
                    actual_sha = sha256_file(tmp)
                    if expected_size > 0 and actual_size != expected_size:
                        os.remove(tmp)
                        raise OffloadError(
                            f"output size mismatch: expected {expected_size}, "
                            f"got {actual_size}")
                    if expected_sha and actual_sha != expected_sha:
                        os.remove(tmp)
                        raise OffloadError(
                            f"output SHA-256 mismatch: expected {expected_sha[:16]}..., "
                            f"got {actual_sha[:16]}...")
                    os.replace(tmp, dest_path)
                    self.log(f"Output verified and saved: {dest_path}")
                    with self._send_lock:
                        send_json(self.sock, MsgType.OUTPUT_ACK,
                                  {"job_id": job_id, "ok": True, "sha256": actual_sha})
                    return dest_path
        finally:
            if handle:
                handle.close()

        raise OffloadError("timed out while downloading the output")

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    def log(self, message: str, level: str = "info") -> None:
        """Local (client-side) log line - delivered straight to the callback."""
        if self.on_log:
            self.on_log(LogEvent(message=message, level=level, ts=time.time()))
        else:
            print(f"[{level}] {message}", flush=True)

    def capabilities(self) -> dict:
        return (self.server_info or {}).get("environment") or {}

    def close(self) -> None:
        self._stop.set()
        self._teardown_socket()
        self.connected = False

    def _teardown_socket(self) -> None:
        sock, self.sock = self.sock, None
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def __enter__(self) -> "OffloadClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
