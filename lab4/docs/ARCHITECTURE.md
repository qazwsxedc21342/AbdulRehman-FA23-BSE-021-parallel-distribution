# Architecture & Design Notes

**Project:** Custom Distributed Task Offloading & Remote GPU Rendering System
**Course:** CSC-334 — Parallel and Distributed Computing

This document explains *why* the system is built the way it is. For setup and
usage see the [README](../README.md).

---

## 1. High-level design

```
┌──────────────────────── CLIENT NODE ────────────────────────┐
│  client/gui.py (CustomTkinter)      client/cli.py           │
│        │  widgets on the Tk thread       │                  │
│        ▼                                 ▼                  │
│  queue.Queue(events)  ◄── pump ──  reader thread             │
│        ▲                                 │                  │
│        └──────── client/api.py ──────────┤  OffloadClient   │
│                    (blocking socket) ◄───┘                  │
└───────────────────────┬─────────────────────────────────────┘
                        │  TCP :5000  (custom framed protocol)
┌───────────────────────▼────────────── SERVER NODE ──────────┐
│  server/daemon.py                                             │
│    accept loop ──► ClientSession (thread)                     │
│                      │  handshake / frames / uploads         │
│                      ▼                                       │
│                job queue (Queue) ──► worker threads (N)       │
│                                        │                     │
│                        server/executors.py                   │
│                        ├─ VideoExecutor  (FFmpeg, NVENC→x264)│
│                        ├─ TensorExecutor (torch-CUDA→numpy)  │
│                        └─ SyntheticExecutor (scaled work)    │
│                        server/environment.py (capability     │
│                        probe: ffmpeg / encoders / GPU / torch)│
└──────────────────────────────────────────────────────────────┘
```

**One process, thread-per-connection, thread-per-worker.** The workload is
I/O-bound (socket reads) plus a handful of long-running subprocess/Compute jobs,
so OS threads with blocking sockets are simpler and fast enough; no asyncio
complexity, easy cancellation via `Process.kill()`.

---

## 2. Wire protocol (`common/protocol.py`)

- **Framing:** `struct.pack('>IB', payload_len, msg_type)` — a 5-byte header
  (big-endian uint32 length + uint8 type) followed by the payload.
  Length-prefixing removes all ambiguity about message boundaries and lets the
  reader `recv_exact()` a full frame before dispatch.
- **Control vs bulk:** JSON payloads for control messages (handshake, job spec,
  progress, logs), raw bytes for file chunks. Same framing for both.
- **Versioning:** `PROTOCOL_VERSION = 1` is exchanged inside `HELLO` /
  `HELLO_ACK`; a mismatch refuses the connection *before* any work happens.
- **Nonce:** the client sends a random nonce; the server echoes it back (plus its
  own). This proves we are talking to our own daemon and defeats stale/replayed
  handshakes from a previous session.
- **Limits:** `MAX_PAYLOAD = 16 MiB` per frame — a malicious or corrupt length
  field cannot make the peer allocate unbounded memory.
- **Sanitising:** all remote-supplied filenames pass `safe_filename()` (strip
  separators, reject `..`, dot/empty names) so a job can never write outside
  `--workdir`.

---

## 3. Handshake & session lifecycle (`server/daemon.py`, `client/api.py`)

```
connect (with retries/backoff)
   → HELLO {version, nonce, client_id}
   ← HELLO_ACK {server_nonce, version, caps{encoders, gpu, torch, cpu…}}
   → PING / ← PONG            (any number of times, RTT stats)
   → JOB_SUBMIT {task, params} → JOB_ACCEPTED | ERROR
   → FILE_BEGIN … FILE_CHUNK* … FILE_END {sha256}
   ← FILE_ACK {ok}                     (after re-hashing server-side)
   ← PROGRESS / LOG / HEARTBEAT*       (streamed, tagged with job id)
   ← JOB_DONE {output, sha256, stats} | JOB_FAILED {reason}
   → OUTPUT_BEGIN … ← OUTPUT_CHUNK* …   (client re-hashes before use)
   → DISCONNECT                        (server cancels jobs, joins threads)
```

Capabilities (`caps`) are probed **once at daemon start** by
`server/environment.py`: locate an FFmpeg binary (PATH → `imageio_ffmpeg`),
enumerate `-encoders` and probe each candidate (`h264_nvenc` first, then
`libx264`), detect `nvidia-smi`/CUDA and `torch`. The client therefore knows
*before submitting* whether the worker is GPU-capable.

---

## 4. Threading model & backpressure

**Server**

- `accept` loop thread → one `ClientSession` thread per connection.
- Each session reads frames, validates, and enqueues jobs onto a bounded
  `Queue`; `--workers` worker threads pull from it. The queue is the natural
  back-pressure point: a flood of submissions cannot spawn unbounded threads.
- Progress/log frames are emitted **from the worker thread** while FFmpeg runs,
  driven by parsing `-progress pipe:1` output — no polling of the process.
- `HEARTBEAT` frames (CPU/RAM via `psutil`) keep the link provably alive during
  long jobs even when no progress changes.

**Client**

- All blocking network I/O happens on a background thread(s); the Tk main loop
  only touches widgets.
- The reader thread **demultiplexes** server frames into a `queue.Queue`; a
  periodic `after(80)` pump on the Tk thread drains the queue and updates
  widgets. Tkinter is not thread-safe — this is the standard, correct pattern.
- Job events are re-dispatched exactly once in `run_job` (no duplicate progress
  callbacks), and user callbacks are invoked outside the reader thread.

---

## 5. Executors & graceful degradation (`server/executors.py`)

| Executor | Preferred path | Automatic fallback |
|---|---|---|
| **Video** | `ffmpeg -c:v h264_nvenc` (preset/cq from job spec) | `-h encoder=` probe fails or runtime `Cannot load nvcuda.dll` → **libx264**, logged as a `LOG` frame |
| **Tensor** | `torch` CUDA GEMM | no torch/CUDA → NumPy BLAS GEMM, saved as `.npy` |
| **Synthetic** | size-scaled deterministic compute | always CPU |

Rationale: hardware selection is *attempted, then verified*, never assumed —
`h264_nvenc` can appear in the encoder list yet fail at load time on a machine
without the NVIDIA driver. `_run_once` retries with the CPU encoder so a job
never fails for a reason the machine can handle.

FFmpeg progress comes from `-progress pipe:1` (machine-readable key=value
lines) rather than scraping stderr, so `PROGRESS` frames are exact and
stage-labelled.

---

## 6. Integrity & robustness (Task 4)

- **Upload:** client hashes while sending; `FILE_END` carries the digest; the
  server re-hashes the reassembled file and compares. Mismatch ⇒ `FILE_ACK {ok:false}`,
  the partial file is **deleted**, and the job never executes.
- **Download:** server hashes the artefact at job end; the client re-hashes the
  received stream and refuses to surface an output that does not match.
- **Timeouts:** every blocking operation (connect, handshake, frame reads,
  per-chunk acks, job wait) has a deadline from `common/config.py`; expiry
  raises `OffloadError` instead of hanging forever.
- **Graceful disconnect:** `DISCONNECT`/EOF makes the session kill its running
  child process, delete partial outputs, and mark in-flight jobs as reclaimed —
  verified by `tests/test_end_to_end.py`.
- **Cancel:** `JOB_CANCEL` → `Process.kill()` server-side + `JOB_FAILED`-style
  terminal state so client and server agree on the outcome.
- **Malformed input:** unknown message types, oversize lengths, bad JSON, and
  wrong-version hellos are answered with `ERROR` (or closed) without ever
  reaching the job queue.

---

## 7. GUI (`client/gui.py`)

- **Connection bar** — host/port/Connect, status badge (● Online/Offline), last
  RTT, worker GPU/encoder line (from `HELLO_ACK` caps).
- **Job config** — file browser, task selector, and per-task parameter widgets
  (resolution, bitrate, preset, encoder, quality slider, matrix size, …) plus an
  extra-parameters escape hatch.
- **Execution** — Start/Cancel; progress bar + percent + stage; live status line
  (`Job running — 42% (h264_nvenc)`); scrollable log terminal fed by `LOG`
  frames; Open-Output-Folder once verified.
- **State** — last connection persisted to `.client_state`, so relaunching the
  GUI reconnects to the same worker.

---

## 8. Benchmarking (`benchmarks/run_benchmark.py`)

Each case runs the *same executor + input* twice: **local in-process baseline**
vs **remote offload** (client-observed wall time including handshake, upload,
execution, verified download). Reported metrics: speedup, transfer overhead
(`T_remote − T_worker`), upload throughput, per-stage timings. Results are
written to `docs/BENCHMARK_REPORT.md` + `docs/BENCHMARK_RESULTS.csv` with both
hosts recorded, so loopback numbers are never mistaken for LAN numbers.

---

## 9. Test strategy (`tests/`)

- **Unit:** frame codec, hashing, sanitising, limits (`test_protocol.py`).
- **Integration:** a real daemon is booted *in-process on an ephemeral port*
  and exercised through the public `OffloadClient` API — handshake failures,
  ping stats, full job round-trips with verified downloads, retries, and
  disconnect cleanup (`test_end_to_end.py`).
- **Adversarial:** corrupt checksums, truncated uploads, version mismatches,
  oversize payloads, unknown output requests (`test_integrity.py`).

Everything is runnable offline with `python -m unittest discover -s tests`.
