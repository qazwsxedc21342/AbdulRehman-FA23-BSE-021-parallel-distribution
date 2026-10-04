"""
server/daemon.py
----------------
Headless worker daemon for the Distributed Task Offloading system (Tasks 1, 2 & 4).

Responsibilities
----------------
* Accept TCP connections and perform the HELLO / HELLO_ACK handshake with
  protocol version negotiation and capability exchange (GPU, encoders, CUDA).
* Serve latency probes (PING -> PONG) so the client can measure RTT before
  submitting any job.
* Validate and receive input files with SHA-256 integrity checks.
* Queue accepted jobs and execute them on a bounded worker pool (the
  "Task Queue Manager" from the architecture diagram).
* Stream asynchronous PROGRESS / LOG / HEARTBEAT events back to the owning
  client while the job runs.
* Stream the rendered artefact back with a SHA-256 integrity check.

Run with::

    python -m server.daemon --host 0.0.0.0 --port 5000 --workers 2
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import socket
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from common import config
from common.protocol import (
    PROTOCOL_VERSION,
    MsgType,
    ProtocolError,
    recv_frame,
    safe_filename,
    send_frame,
    set_socket_options,
    sha256_file,
)
from server.environment import detect
from server.executors import ExecutorError, get_executor

log = logging.getLogger("server.daemon")


# --------------------------------------------------------------------------- #
# Job model
# --------------------------------------------------------------------------- #

@dataclass
class Job:
    job_id: str
    task: str
    params: dict
    input_name: str
    input_size: int
    input_sha256: str
    output_name: str
    owner: "ClientSession"
    state: str = "queued"            # queued | running | done | failed | cancelled
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    percent: float = 0.0
    stage: str = "queued"
    result: Optional[dict] = None
    error: Optional[str] = None
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    @property
    def elapsed(self) -> float:
        start = self.started_at or self.created_at
        end = self.finished_at or time.time()
        return end - start


# --------------------------------------------------------------------------- #
# Session: one TCP client connection
# --------------------------------------------------------------------------- #

class ClientSession(threading.Thread):
    """Handles a single connected client, running on its own thread."""

    def __init__(self, server: "WorkerDaemon", sock: socket.socket, addr):
        super().__init__(daemon=True, name=f"session-{addr[0]}:{addr[1]}")
        self.server = server
        self.sock = sock
        self.addr = addr
        self.session_id = uuid.uuid4().hex[:12]
        self.client_id = "unknown"
        self.send_lock = threading.Lock()
        self.jobs: Dict[str, Job] = {}
        self.alive = True
        self.current_job_id: Optional[str] = None
        self._last_activity = time.time()

    # -- outbound helpers ------------------------------------------------- #
    def send(self, msg_type: int, payload: bytes = b"") -> None:
        with self.send_lock:
            send_frame(self.sock, msg_type, payload)

    def send_obj(self, msg_type: int, obj: dict) -> None:
        import json

        self.send(msg_type, json.dumps(obj, separators=(",", ":")).encode("utf-8"))

    def close(self) -> None:
        self.alive = False
        for job in self.jobs.values():
            if job.state in ("queued", "running"):
                job.cancel_event.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    # -- main loop -------------------------------------------------------- #
    def run(self) -> None:
        log.info("CONNECT  %s:%s session=%s", self.addr[0], self.addr[1], self.session_id)
        set_socket_options(self.sock)
        self.sock.settimeout(config.IDLE_TIMEOUT)
        try:
            while self.alive:
                try:
                    msg_type, payload = recv_frame(self.sock)
                except socket.timeout:
                    log.info("IDLE     %s session=%s (no traffic for %.0fs)",
                             self.addr[0], self.session_id, config.IDLE_TIMEOUT)
                    break
                self._last_activity = time.time()
                self._dispatch(msg_type, payload)
        except (ConnectionError, OSError, ProtocolError) as exc:
            log.warning("DISCONNECT %s session=%s reason=%s",
                        self.addr[0], self.session_id, exc)
        finally:
            self.close()
            self.server.unregister(self)
            log.info("CLOSED   %s session=%s", self.addr[0], self.session_id)

    # -- dispatch --------------------------------------------------------- #
    def _dispatch(self, msg_type: int, payload: bytes) -> None:
        import json

        if msg_type == MsgType.HELLO:
            self._handle_hello(payload)
        elif msg_type == MsgType.PING:
            self._handle_ping(payload)
        elif msg_type == MsgType.JOB_SUBMIT:
            self._handle_submit(payload)
        elif msg_type == MsgType.FILE_BEGIN:
            self._handle_file_transfer(payload)
        elif msg_type == MsgType.OUTPUT_REQUEST:
            self._handle_output_request(payload)
        elif msg_type == MsgType.JOB_CANCEL:
            self._handle_cancel(payload)
        elif msg_type == MsgType.OUTPUT_ACK:
            pass  # informational
        else:
            self.send_obj(MsgType.ERROR, {"error": f"unexpected message type {msg_type}"})

    # -- Task 1: handshake & latency ------------------------------------- #
    def _handle_hello(self, payload: bytes) -> None:
        import json

        try:
            hello = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_obj(MsgType.ERROR, {"error": "malformed HELLO"})
            return

        client_proto = int(hello.get("protocol_version", -1))
        if client_proto != PROTOCOL_VERSION:
            self.send_obj(MsgType.ERROR, {
                "error": "protocol version mismatch",
                "server_protocol": PROTOCOL_VERSION,
                "client_protocol": client_proto,
            })
            return

        self.client_id = str(hello.get("client_id", "unknown"))[:64]
        env = detect()
        self.send_obj(MsgType.HELLO_ACK, {
            "protocol_version": PROTOCOL_VERSION,
            "session_id": self.session_id,
            "server_id": env.hostname,
            "server_time": time.time(),
            "echo_nonce": hello.get("nonce"),
            "environment": env.to_dict(),
            "supported_tasks": list(config.SUPPORTED_TASKS),
            "max_chunk_size": config.CHUNK_SIZE,
            "queue_limit": config.MAX_QUEUE_DEPTH,
        })
        log.info("HANDSHAKE client=%s session=%s proto=%s",
                 self.client_id, self.session_id, client_proto)

    def _handle_ping(self, payload: bytes) -> None:
        import json

        try:
            ping = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            ping = {}
        self.send_obj(MsgType.PONG, {
            "seq": ping.get("seq", 0),
            "client_timestamp": ping.get("client_timestamp"),
            "server_timestamp": time.time(),
        })

    # -- Task 2/4: submission -------------------------------------------- #
    def _handle_submit(self, payload: bytes) -> None:
        import json

        try:
            spec = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_obj(MsgType.JOB_REJECTED, {"error": "malformed JOB_SUBMIT"})
            return

        task = str(spec.get("task", ""))
        if task not in config.SUPPORTED_TASKS:
            self.send_obj(MsgType.JOB_REJECTED, {
                "error": f"unsupported task '{task}'",
                "supported": list(config.SUPPORTED_TASKS),
            })
            return

        input_name = safe_filename(str(spec.get("input_name", "input.bin")))
        input_size = int(spec.get("input_size", 0))
        if input_size < 0 or input_size > config.MAX_INPUT_BYTES:
            self.send_obj(MsgType.JOB_REJECTED, {
                "error": f"input size {input_size} exceeds limit",
            })
            return

        if self.server.queue_depth() >= config.MAX_QUEUE_DEPTH:
            self.send_obj(MsgType.JOB_REJECTED, {"error": "worker queue is full, retry shortly"})
            return

        output_name = safe_filename(str(spec.get("output_name") or f"output_{uuid.uuid4().hex[:8]}.bin"))
        job = Job(
            job_id=uuid.uuid4().hex[:12],
            task=task,
            params=spec.get("params") or {},
            input_name=input_name,
            input_size=input_size,
            input_sha256=str(spec.get("input_sha256", "")).lower(),
            output_name=output_name,
            owner=self,
        )
        self.jobs[job.job_id] = job
        self.server.register_job(job)

        self.send_obj(MsgType.JOB_ACCEPTED, {
            "job_id": job.job_id,
            "queue_position": self.server.queue_depth(),
            "input_name": input_name,
            "message": "send FILE_BEGIN to upload the input asset",
        })
        log.info("ACCEPT   job=%s task=%s client=%s size=%s",
                 job.job_id, task, self.client_id, input_size)

    # -- input transfer with integrity check ------------------------------ #
    def _handle_file_transfer(self, payload: bytes) -> None:
        import json

        try:
            begin = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_obj(MsgType.FILE_ACK, {"ok": False, "error": "malformed FILE_BEGIN"})
            return

        job_id = str(begin.get("job_id", ""))
        job = self.jobs.get(job_id)
        if job is None:
            self.send_obj(MsgType.FILE_ACK, {"ok": False, "error": f"unknown job {job_id}"})
            return
        if job.state != "queued":
            self.send_obj(MsgType.FILE_ACK, {"ok": False, "error": f"job already {job.state}"})
            return

        declared_size = int(begin.get("size", -1))
        declared_sha = str(begin.get("sha256", "")).lower()
        if declared_size < 0 or declared_size > config.MAX_INPUT_BYTES:
            self.send_obj(MsgType.FILE_ACK, {"ok": False, "error": "illegal size"})
            return

        job.stage = "receiving"
        os.makedirs(config.SERVER_WORK_DIR, exist_ok=True)
        tmp_path = os.path.join(config.SERVER_WORK_DIR, f"{job.job_id}_{job.input_name}.part")
        received = 0
        self.sock.settimeout(config.IO_TIMEOUT)
        try:
            with open(tmp_path, "wb") as handle:
                while received < declared_size:
                    msg_type, chunk = recv_frame(self.sock)
                    if msg_type == MsgType.JOB_CANCEL:
                        raise ExecutorError("upload cancelled by client")
                    if msg_type != MsgType.FILE_CHUNK:
                        raise ProtocolError(f"expected FILE_CHUNK, got type {msg_type}")
                    if not chunk:
                        raise ProtocolError("empty chunk")
                    handle.write(chunk)
                    received += len(chunk)
                    if received > config.MAX_INPUT_BYTES:
                        raise ProtocolError("transfer exceeded size limit")
                    if received % (32 * config.CHUNK_SIZE) < config.CHUNK_SIZE:
                        pct = 100.0 * received / max(declared_size, 1)
                        self.send_obj(MsgType.PROGRESS, {
                            "job_id": job_id, "percent": round(min(pct, 99.0), 2),
                            "stage": "upload", "received": received, "total": declared_size,
                        })

                # Trailing FILE_END frame
                msg_type, end_payload = recv_frame(self.sock)
                if msg_type != MsgType.FILE_END:
                    raise ProtocolError("expected FILE_END after chunks")
                try:
                    end = json.loads(end_payload.decode("utf-8"))
                except Exception:
                    end = {}
                declared_sha = str(end.get("sha256", declared_sha)).lower()
        except (ConnectionError, OSError, ProtocolError, ExecutorError) as exc:
            self.send_obj(MsgType.FILE_ACK, {"ok": False, "job_id": job_id, "error": str(exc)})
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            job.state = "failed"
            job.error = f"upload failed: {exc}"
            return
        finally:
            self.sock.settimeout(config.IDLE_TIMEOUT)

        actual_sha = sha256_file(tmp_path)
        actual_size = os.path.getsize(tmp_path)
        ok = (actual_size == declared_size) and (actual_sha == declared_sha)
        if not ok:
            self.send_obj(MsgType.FILE_ACK, {
                "ok": False, "job_id": job_id,
                "error": "checksum/size mismatch",
                "expected_size": declared_size, "actual_size": actual_size,
                "expected_sha256": declared_sha, "actual_sha256": actual_sha,
            })
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            job.state = "failed"
            job.error = "input integrity check failed"
            log.warning("CHECKSUM job=%s FAILED expected=%s actual=%s",
                        job_id, declared_sha[:12], actual_sha[:12])
            return

        job.input_path = tmp_path
        job.stage = "queued"
        self.send_obj(MsgType.FILE_ACK, {
            "ok": True, "job_id": job_id,
            "size": actual_size, "sha256": actual_sha,
            "message": "integrity verified, job queued for execution",
        })
        log.info("UPLOAD   job=%s bytes=%s sha256=%s OK", job_id, actual_size, actual_sha[:16])
        self.server.enqueue(job)

    # -- output download --------------------------------------------------- #
    def _handle_output_request(self, payload: bytes) -> None:
        import json

        try:
            req = json.loads(payload.decode("utf-8"))
        except Exception:
            req = {}
        job_id = str(req.get("job_id", ""))
        job = self.jobs.get(job_id)
        if job is None or job.state != "done" or not job.output_path:
            self.send_obj(MsgType.ERROR, {"error": f"no completed output for job {job_id}"})
            return
        if not os.path.exists(job.output_path):
            self.send_obj(MsgType.ERROR, {"error": "output file missing on worker"})
            return

        size = os.path.getsize(job.output_path)
        digest = sha256_file(job.output_path)
        self.send_obj(MsgType.OUTPUT_BEGIN, {
            "job_id": job_id, "name": job.output_name,
            "size": size, "sha256": digest,
        })
        sent = 0
        with self.send_lock:
            try:
                with open(job.output_path, "rb") as handle:
                    while True:
                        block = handle.read(config.CHUNK_SIZE)
                        if not block:
                            break
                        send_frame(self.sock, MsgType.OUTPUT_CHUNK, block)
                        sent += len(block)
                send_frame(self.sock, MsgType.OUTPUT_END, json.dumps({
                    "job_id": job_id, "size": sent, "sha256": digest,
                }).encode("utf-8"))
            except OSError as exc:
                log.warning("OUTPUT   job=%s failed: %s", job_id, exc)
                return
        log.info("DOWNLOAD job=%s bytes=%s sha256=%s", job_id, sent, digest[:16])

    def _handle_cancel(self, payload: bytes) -> None:
        import json

        try:
            req = json.loads(payload.decode("utf-8"))
        except Exception:
            req = {}
        job = self.jobs.get(str(req.get("job_id", "")))
        if job and job.state in ("queued", "running"):
            job.cancel_event.set()
            job.state = "cancelled"
            self.send_obj(MsgType.LOG, {"job_id": job.job_id, "level": "warning",
                                        "message": "cancellation requested by client"})

    # -- called from job runner ------------------------------------------- #
    def emit_progress(self, job: Job, percent: float, stage: str, info: Optional[dict] = None) -> None:
        if not self.alive:
            return
        job.percent = max(job.percent, percent) if stage != "upload" else percent
        job.stage = stage
        try:
            self.send_obj(MsgType.PROGRESS, {
                "job_id": job.job_id,
                "percent": round(percent, 2),
                "stage": stage,
                "elapsed": round(job.elapsed, 2),
                **(info or {}),
            })
        except OSError:
            self.alive = False

    def emit_log(self, job_id: str, message: str, level: str = "info") -> None:
        if not self.alive:
            return
        try:
            self.send_obj(MsgType.LOG, {"job_id": job_id, "level": level,
                                        "message": message, "ts": time.time()})
        except OSError:
            self.alive = False

    def emit_heartbeat(self, job: Job) -> None:
        if not self.alive:
            return
        try:
            self.send_obj(MsgType.HEARTBEAT, {
                "job_id": job.job_id, "state": job.state,
                "stage": job.stage, "percent": round(job.percent, 2),
                "queue_depth": self.server.queue_depth(),
            })
        except OSError:
            self.alive = False


# --------------------------------------------------------------------------- #
# Worker daemon
# --------------------------------------------------------------------------- #

class WorkerDaemon:
    """Threaded TCP server with a bounded background worker pool."""

    def __init__(self, host: str = "0.0.0.0", port: int = config.DEFAULT_PORT,
                 workers: int = 2, workdir: Optional[str] = None):
        self.host = host
        self.port = port
        self.workers = max(1, workers)
        self.workdir = workdir or config.SERVER_WORK_DIR
        self.env = detect()

        self.jobs_lock = threading.Lock()
        self.jobs: Dict[str, Job] = {}
        self.queue: "queue.Queue[Job]" = queue.Queue(maxsize=config.MAX_QUEUE_DEPTH)
        self.sessions: List[ClientSession] = []

        self._listener: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._worker_threads: List[threading.Thread] = []
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # -- lifecycle -------------------------------------------------------- #
    def start(self) -> None:
        os.makedirs(self.workdir, exist_ok=True)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(16)
        listener.settimeout(1.0)
        self._listener = listener
        if self.port == 0:
            self.port = listener.getsockname()[1]

        for i in range(self.workers):
            t = threading.Thread(target=self._worker_loop, name=f"worker-{i}", daemon=True)
            t.start()
            self._worker_threads.append(t)

        self._accept_thread = threading.Thread(target=self._accept_loop,
                                               name="accept", daemon=True)
        self._accept_thread.start()

        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop,
                                                  name="heartbeat", daemon=True)
        self._heartbeat_thread.start()

        log.info("=" * 72)
        log.info("  CSC-334 Distributed Task Offloading - Worker Daemon")
        log.info("  listening on %s:%s  (workers=%d)", self.host, self.port, self.workers)
        log.info("  GPU      : %s", self.env.gpu_label)
        log.info("  Encoders : %s", ", ".join(self.env.encoders) or "none (CPU only)")
        log.info("  FFmpeg   : %s", self.env.ffmpeg_path or "not found")
        log.info("  CUDA     : %s", self.env.cuda_available)
        log.info("  Tasks    : %s", ", ".join(config.SUPPORTED_TASKS))
        log.info("=" * 72)

    def stop(self) -> None:
        self._stop.set()
        if self._listener:
            try:
                self._listener.close()
            except OSError:
                pass
        for session in list(self.sessions):
            session.close()
        log.info("daemon stopped")

    def serve_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            log.info("Ctrl+C received, shutting down...")
            self.stop()

    # -- accept loop ------------------------------------------------------ #
    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                sock, addr = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            session = ClientSession(self, sock, addr)
            self.sessions.append(session)
            session.start()

    def unregister(self, session: ClientSession) -> None:
        if session in self.sessions:
            self.sessions.remove(session)

    # -- job registry / queue --------------------------------------------- #
    def register_job(self, job: Job) -> None:
        with self.jobs_lock:
            self.jobs[job.job_id] = job

    def queue_depth(self) -> int:
        return self.queue.qsize()

    def enqueue(self, job: Job) -> None:
        self.queue.put(job)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._execute(job)
            self.queue.task_done()

    # -- execution --------------------------------------------------------- #
    def _execute(self, job: Job) -> None:
        session = job.owner
        if job.cancel_event.is_set() or not session.alive:
            job.state = "cancelled"
            return

        job.state = "running"
        job.started_at = time.time()
        job.stage = "running"
        session.emit_log(job.job_id, f"worker started task '{job.task}' "
                                     f"(job {job.job_id}) on {self.env.hostname}")

        os.makedirs(self.workdir, exist_ok=True)
        output_path = os.path.join(self.workdir, f"{job.job_id}_{job.output_name}")

        last_heartbeat = 0.0

        def progress_cb(percent: float, info: dict) -> None:
            nonlocal last_heartbeat
            session.emit_progress(job, float(percent), job.task, info)
            now = time.time()
            if now - last_heartbeat >= config.HEARTBEAT_INTERVAL:
                last_heartbeat = now
                session.emit_heartbeat(job)

        def log_cb(message: str) -> None:
            session.emit_log(job.job_id, message)

        try:
            executor = get_executor(job.task)
            result = executor.run(
                input_path=job.input_path or "",
                output_path=output_path,
                params=job.params,
                progress_cb=progress_cb,
                log_cb=log_cb,
                cancel_event=job.cancel_event,
            )
            # Some executors write <path>.npy - normalise to the real file.
            if not os.path.exists(output_path) and os.path.exists(output_path + ".npy"):
                output_path += ".npy"
            if not os.path.exists(output_path):
                raise ExecutorError("executor produced no output artefact")

            job.output_path = output_path
            job.state = "done"
            job.percent = 100.0
            job.stage = "done"
            job.result = result
            job.finished_at = time.time()

            size = os.path.getsize(output_path)
            digest = sha256_file(output_path)
            session.emit_progress(job, 100.0, "done", {
                "engine": result.get("engine"), "elapsed": round(job.elapsed, 2)})
            session.send_obj(MsgType.JOB_DONE, {
                "job_id": job.job_id,
                "state": "done",
                "duration_s": round(job.elapsed, 3),
                "output": {
                    "name": job.output_name if output_path.endswith(job.output_name)
                    else os.path.basename(output_path),
                    "size": size,
                    "sha256": digest,
                },
                "result": result,
            })
            log.info("DONE     job=%s task=%s engine=%s %.2fs bytes=%s",
                     job.job_id, job.task, result.get("engine"), job.elapsed, size)

        except ExecutorError as exc:
            if job.cancel_event.is_set():
                job.state = "cancelled"
                job.error = "cancelled"
                session.send_obj(MsgType.JOB_FAILED,
                                 {"job_id": job.job_id, "state": "cancelled",
                                  "error": "job cancelled by client"})
                log.info("CANCEL   job=%s", job.job_id)
                return
            self._fail(job, str(exc))
        except Exception as exc:  # noqa: BLE001 - report any executor crash
            self._fail(job, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}")
        finally:
            if job.input_path and os.path.exists(job.input_path):
                try:
                    os.remove(job.input_path)
                except OSError:
                    pass
            job.finished_at = job.finished_at or time.time()

    def _fail(self, job: Job, error: str) -> None:
        job.state = "failed"
        job.error = error
        job.finished_at = time.time()
        session = job.owner
        try:
            session.send_obj(MsgType.JOB_FAILED,
                             {"job_id": job.job_id, "state": "failed",
                              "error": error[:4000]})
            session.emit_log(job.job_id, f"job failed: {error.splitlines()[0]}", level="error")
        except OSError:
            session.alive = False
        log.error("FAILED   job=%s error=%s", job.job_id, error.splitlines()[0] if error else "?")

    # -- keep-alive for queued jobs ---------------------------------------- #
    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(config.HEARTBEAT_INTERVAL)
            with self.jobs_lock:
                pending = [j for j in self.jobs.values() if j.state == "queued"]
            for job in pending:
                job.owner.emit_heartbeat(job)

    # -- introspection ------------------------------------------------------ #
    def stats(self) -> dict:
        with self.jobs_lock:
            states: Dict[str, int] = {}
            for job in self.jobs.values():
                states[job.state] = states.get(job.state, 0) + 1
        return {
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "queue_depth": self.queue_depth(),
            "active_sessions": len(self.sessions),
            "jobs": states,
            "environment": self.env.to_dict(),
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CSC-334 Distributed Task Offloading - remote GPU worker daemon")
    parser.add_argument("--host", default="0.0.0.0",
                        help="bind address (default 0.0.0.0 = all interfaces)")
    parser.add_argument("--port", type=int, default=config.DEFAULT_PORT,
                        help=f"TCP port (default {config.DEFAULT_PORT})")
    parser.add_argument("--workers", type=int, default=2,
                        help="number of concurrent job workers (default 2)")
    parser.add_argument("--workdir", default=config.SERVER_WORK_DIR,
                        help="directory for received inputs and rendered outputs")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(threadName)-11s | %(message)s",
        datefmt="%H:%M:%S",
    )

    daemon = WorkerDaemon(host=args.host, port=args.port,
                          workers=args.workers, workdir=args.workdir)
    daemon.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
