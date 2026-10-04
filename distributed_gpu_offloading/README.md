# Custom Distributed Task Offloading & Remote GPU Rendering System

**Course:** CSC-334 — Parallel and Distributed Computing
**Assignment:** Lab 04 — Custom Distributed Task Offloading & Remote GPU Rendering System (100 marks)
**Author:** Abdul Rehman (FA23-BSE-021, Section A)

A complete client/server system that offloads heavy multimedia and compute jobs
(video transcoding, tensor math, synthetic rendering) from a resource-constrained
client laptop to a remote worker node over a TCP network, with a live GUI,
streaming progress, integrity-verified transfers, and a benchmarking harness.


---
Outputs:
<img width="1600" height="1039" alt="WhatsApp Image 2026-10-04 at 1 28 52 PM (3)" src="https://github.com/user-attachments/assets/d29d56df-9d37-44b3-b209-7eff5ad3f054" />
<img width="1600" height="1039" alt="WhatsApp Image 2026-10-04 at 1 28 52 PM (2)" src="https://github.com/user-attachments/assets/4f3f5273-d9cf-45c0-93e6-08a50c05fa69" />
<img width="1600" height="1039" alt="WhatsApp Image 2026-10-04 at 1 28 52 PM (1)" src="https://github.com/user-attachments/assets/fd782b99-6813-4cf3-b30d-30326a3094bd" />
<img width="1000" height="649" alt="WhatsApp Image 2026-10-04 at 1 28 52 PM" src="https://github.com/user-attachments/assets/ebac6683-7aa9-483c-8372-bbc5e119b2c2" />


## Table of Contents

1. [Problem Statement & Motivation](#1-problem-statement--motivation)
2. [Features / Assignment Task Map](#2-features--assignment-task-map)
3. [Repository Structure](#3-repository-structure)
4. [Setup Instructions (Client & Server)](#4-setup-instructions-client--server)
5. [Network Configuration Guide](#5-network-configuration-guide)
6. [Execution Guide](#6-execution-guide)
7. [Demo Screenshots & GIF](#7-demo-screenshots--gif)
8. [Protocol Overview](#8-protocol-overview)
9. [Testing](#9-testing)
10. [Benchmarking (Task 5)](#10-benchmarking-task-5)
11. [Troubleshooting](#11-troubleshooting)
12. [Submission Checklist](#12-submission-checklist)

---

## 1. Problem Statement & Motivation

A client laptop with an integrated or low-end GPU cannot reliably transcode
high-resolution video or run heavy matrix workloads: jobs take minutes, the
machine thermally throttles, and large inputs can exhaust memory entirely
(out-of-memory failures).

**Distributed task offloading** solves this without new hardware: a fast local
network link (direct CAT6 cable or high-throughput Wi-Fi) connects the weak
client to a powerful worker node that owns the discrete GPU. The client only
packages the job, ships it, streams progress, and downloads the finished
artefact — the heavy lifting happens remotely.

This project implements that pipeline end to end with a **custom binary
protocol** (no gRPC/HTTP), a **GPU-aware worker daemon**, a **desktop GUI**, and
a **benchmark harness** that proves the offload behaviour with numbers.

---

## 2. Features / Assignment Task Map

| Task | Requirement | Where it lives |
|---|---|---|
| **Task 1** — Networking & Handshake | TCP socket programming, handshake with version + nonce validation, latency ping (min/avg/max RTT, jitter, packet loss) | [common/protocol.py](common/protocol.py), [client/api.py](client/api.py), [server/daemon.py](server/daemon.py) (`PING`/`PONG`), `client.cli ping` |
| **Task 2** — GPU/FFmpeg Rendering Daemon | Worker daemon with job queue, bounded worker pool, FFmpeg + NVENC (auto-fallback to libx264), torch-CUDA (CPU fallback), capability reporting, chunked file receive | [server/daemon.py](server/daemon.py), [server/executors.py](server/executors.py), [server/environment.py](server/environment.py) |
| **Task 3** — Client GUI | CustomTkinter desktop app: connect bar, job configuration, encoder/resolution presets, live progress bar, scrollable log terminal, cancel, open-output | [client/gui.py](client/gui.py) |
| **Task 4** — Async Progress, Timeouts, Checksums, Graceful Disconnect | Streaming `PROGRESS`/`LOG`/`HEARTBEAT` frames from a reader thread, per-operation timeouts, SHA-256 checksums on upload *and* download, clean `JOB_CANCEL`, disconnect cleanup (orphaned jobs reclaimed) | [client/api.py](client/api.py), [server/daemon.py](server/daemon.py), [tests/test_integrity.py](tests/test_integrity.py) |
| **Task 5** — Benchmarking & Analysis | Local-vs-remote matrix (video/tensor/synthetic), speedup, transfer overhead, upload throughput, generated Markdown + CSV report | [benchmarks/run_benchmark.py](benchmarks/run_benchmark.py), [docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md), [docs/BENCHMARK_RESULTS.csv](docs/BENCHMARK_RESULTS.csv) |
| **Submission** | `client/` + `server/` directories, README, screenshots, tests | this repository |

Additional hardening included:

- **Framed protocol**: 5-byte header (`>IB` = payload length + message type), versioned, size-capped (16 MiB/frame).
- **Integrity both directions**: SHA-256 computed while uploading/downloading; wrong checksum ⇒ job rejected and file deleted, never executed.
- **Graceful degradation**: `h264_nvenc` is probed at runtime; if the driver/`nvcuda.dll` is unavailable the executor transparently falls back to `libx264` and logs it.
- **CLI client**: `ping`, `info`, `run`, `bench` — usable without the GUI (good for automation and the demo video).
- **30 automated tests**, all passing, including an in-process daemon end-to-end suite.

---

## 3. Repository Structure

```
LAB 4(Custom Distributed Task Offloading & Remote GPU Rendering System)/
├── README.md                  ← this file
├── common/                    ← shared by client and server
│   ├── protocol.py            ← framing, message types, SHA-256, filename sanitising
│   └── config.py              ← hosts, ports, timeouts, limits, presets
├── client/                    ← CLIENT NODE
│   ├── api.py                 ← OffloadClient: handshake, ping, upload, run_job, download
│   ├── gui.py                 ← CustomTkinter desktop GUI (Task 3)
│   ├── cli.py                 ← command-line client (ping/info/run/bench)
│   └── samples/               ← generated demo inputs (video + binary blob)
├── server/                    ← WORKER NODE
│   ├── daemon.py              ← WorkerDaemon + ClientSession (Task 1/2/4)
│   ├── executors.py           ← video / tensor / synthetic executors (Task 2)
│   └── environment.py         ← ffmpeg, encoder, GPU, torch capability detection
├── benchmarks/
│   └── run_benchmark.py       ← Task 5 harness (writes docs/BENCHMARK_REPORT.md)
├── tests/                     ← 30 unittest tests (protocol, end-to-end, integrity)
├── tools/
│   ├── make_sample_assets.py  ← generates sample video + input blob
│   └── capture_gui.py         ← captures the demo screenshots (GDI PrintWindow)
├── docs/
│   ├── ARCHITECTURE.md        ← design notes, protocol diagrams, threading model
│   ├── BENCHMARK_REPORT.md    ← generated Task 5 report
│   └── BENCHMARK_RESULTS.csv  ← raw benchmark rows
└── screenshots/               ← demo screenshots + animated GIF
```

---

## 4. Setup Instructions (Client & Server)

The same codebase runs on both nodes; only the launch command differs.

### 4.1 Prerequisites

- **Python 3.10+** (developed and tested on Python 3.13, Windows)
- Two machines on the same LAN (or one machine for loopback testing)
- Optional on the worker: an NVIDIA GPU + driver for hardware encoding

### 4.2 Install dependencies

From the project root (the directory containing this README):

```bash
python -m pip install customtkinter imageio-ffmpeg psutil pillow numpy
```

| Package | Why |
|---|---|
| `customtkinter` | modern Tk-based GUI (Task 3) |
| `imageio-ffmpeg` | ships a **bundled static FFmpeg binary** — no system FFmpeg needed |
| `psutil` | CPU/RAM telemetry for the daemon heartbeat and benchmarks |
| `pillow` | screenshot capture tool, image handling |
| `numpy` | CPU tensor executor (mandatory) |
| `torch` *(optional)* | enables the CUDA tensor path on GPU workers: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |

> FFmpeg is discovered via `imageio_ffmpeg.get_ffmpeg_exe()` automatically, so a
> system-wide FFmpeg install is **not** required. If you *do* have FFmpeg on
> `PATH`, it is detected first and preferred.

### 4.3 Verify the install

```bash
python -m unittest discover -s tests -v     # 30 tests should pass
python -m client.cli --help                 # CLI is importable
```

---

## 5. Network Configuration Guide

### 5.1 Quick start (same machine / loopback testing)

Defaults work out of the box — no network configuration required:

```bash
python -m server.daemon --host 127.0.0.1 --port 5050
python -m client.gui                       # pre-filled with 127.0.0.1:5050 state
```

### 5.2 Two-machine setup over Ethernet (direct CAT6) or Wi-Fi

**Step 1 — Give the worker a static IP (Windows):**

Settings → Network & Internet → (Ethernet/Wi-Fi) → your adapter →
**IP assignment → Edit → Manual**:

| Field | Worker (server) | Client |
|---|---|---|
| IPv4 address | `192.168.1.10` | `192.168.1.11` |
| Subnet mask | `255.255.255.0` | `255.255.255.0` |
| Gateway | *(blank for direct cable)* | *(blank for direct cable)* |
| DNS | `8.8.8.8` (or blank) | `8.8.8.8` (or blank) |

For a **direct cable link** (no router), these addresses work as-is; if you use
a router/AP instead, use addresses inside the router's subnet (e.g.
`192.168.1.x`) and set the router as gateway.

**Step 2 — Open the firewall on the worker** (Administrator PowerShell):

```powershell
New-NetFirewallRule -DisplayName "CSC334 Offload Daemon" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
```

**Step 3 — Confirm connectivity** from the client:

```bash
ping 192.168.1.10
python -m client.cli --host 192.168.1.10 --port 5000 ping
```

**Step 4 — Point the GUI/CLI at the worker:** type `192.168.1.10` and `5000`
in the connection bar (GUI) or pass `--host/--port` (CLI). The client remembers
the last connection in `.client_state`.

> **Command Prompt users:** if `python` is not found, run `py -m ...` instead,
> or add Python to `PATH` during installation.

---

## 6. Execution Guide

### 6.1 Start the worker daemon (SERVER node)

```bash
python -m server.daemon --host 0.0.0.0 --port 5000 --workers 2
```

```
usage: daemon.py [-h] [--host HOST] [--port PORT] [--workers WORKERS]
                 [--workdir WORKDIR] [--log-level {DEBUG,INFO,WARNING,ERROR}]
```

- `--host 0.0.0.0` binds all interfaces (use `127.0.0.1` for local-only testing)
- `--workers N` concurrent jobs (default 2)
- `--workdir` where received inputs/outputs land (default `server/workdir`)
- On start it probes FFmpeg, available encoders (NVENC → libx264), GPU, and torch,
  and advertises them to clients in the handshake.

### 6.2 Launch the GUI (CLIENT node)

```bash
python -m client.gui
```

1. Enter the worker IP/port → **Connect** — the badge turns **● Online** and the
   bar shows the worker's RTT and capabilities (`video=h264_nvenc, cpu=8 logical`).
2. **Browse…** to pick an input file (a sample video ships in `client/samples/`).
3. Choose the **task** (Video transcode / Tensor GEMM / Synthetic render) and its
   parameters: resolution, preset, bitrate, encoder, quality, matrix size, …
4. Press **Start Job** — watch the progress bar, stage label, and live log stream.
5. **Cancel Job** aborts cleanly (server-side process is killed, partial output removed).
6. **Open Output Folder** jumps to the verified downloaded artefact.

### 6.3 CLI without the GUI

```bash
# Task 1: latency measurement (RTT min/avg/max, jitter, loss)
python -m client.cli --host 127.0.0.1 --port 5050 ping --count 10

# Worker capabilities (encoders, GPU, torch)
python -m client.cli --host 127.0.0.1 --port 5050 info

# Full job round-trip: upload → execute → stream progress → verified download
python -m client.cli --host 127.0.0.1 --port 5050 run \
    --task video --input client/samples/sample_720p.mp4 \
    --resolution 1280x720 --preset fast

python -m client.cli --host 127.0.0.1 --port 5050 run \
    --task tensor --input client/samples/sample_input.bin --matrix-size 512

# Task 5: regenerate the benchmark report
python -m client.cli bench
```

### 6.4 Sample assets & screenshots

```bash
python tools/make_sample_assets.py                      # rebuild client/samples/*
python tools/capture_gui.py --host 127.0.0.1 --port 5050   # refresh screenshots/
```

---

## 7. Demo Screenshots & GIF

### Connected — handshake complete, worker capabilities shown

![GUI connected](screenshots/01_gui_connected.png)

### Live progress streaming — progress bar, stage label, streamed log frames

![GUI progress streaming](screenshots/02_gui_progress.png)

### Job complete — output verified and downloaded locally

![GUI job done](screenshots/03_gui_job_done.png)

### Animated demo (connect → stream → done)

![Demo](screenshots/demo.gif)

---

## 8. Protocol Overview

Every frame is `uint32 payload_len | uint8 msg_type | payload`, big-endian
(`struct.pack('>IB', …)`), payload ≤ 16 MiB, JSON for control frames and raw
bytes for bulk transfers.

```
Client                                     Server
  │ ── HELLO {version, nonce, client_id} ──▶│   version+nonce checked
  │◀── HELLO_ACK {server_nonce, caps…} ────│   capabilities advertised
  │ ── PING {seq, t0} ────────────────────▶│
  │◀── PONG {seq, t0, server_t} ───────────│   RTT statistics
  │ ── JOB_SUBMIT {task, params, sha256} ─▶│   validated → queue → JOB_ACCEPTED
  │ ── FILE_BEGIN / FILE_CHUNK*n / FILE_END▶│   reassembled, SHA-256 verified
  │◀── PROGRESS {pct, stage} ──────────────│   streamed while executing
  │◀── LOG {line} ─────────────────────────│   FFmpeg/executor output
  │◀── HEARTBEAT {cpu, mem} ───────────────│   every 2 s
  │◀── JOB_DONE {output, sha256, stats} ───│   (or JOB_FAILED {reason})
  │ ── OUTPUT_BEGIN / CHUNK*n ────────────▶│   download request
  │◀── OUTPUT_CHUNK*n ─────────────────────│   client re-verifies SHA-256
  │ ── JOB_CANCEL / DISCONNECT ───────────▶│   child process killed, job reclaimed
```

Message types, limits, and defaults live in
[common/protocol.py](common/protocol.py) and [common/config.py](common/config.py).
Deeper design notes (threading model, executor fallback, failure handling) are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 9. Testing

```bash
python -m unittest discover -s tests -v
```

**30 tests, all passing** (~2–3 s):

| Suite | Covers |
|---|---|
| `tests/test_protocol.py` | frame encode/decode round-trips, length-prefix errors, SHA-256 helpers, filename sanitising, payload caps |
| `tests/test_end_to_end.py` | boots a daemon on an ephemeral port: handshake + version/nonce rejection, ping stats, video/tensor/synthetic round-trips with verified downloads, connect retries, graceful-disconnect job cleanup |
| `tests/test_integrity.py` | wrong checksum rejected **and file deleted**, truncated upload detected, protocol-version mismatch refused, oversize submit refused, unknown output request refused, progress frames drained before `FILE_ACK` |

---

## 10. Benchmarking (Task 5)

```bash
python -m client.cli bench            # or: python -m benchmarks.run_benchmark
```

For each case the harness runs the **same executor and input twice**: once
in-process on the client (local baseline) and once offloaded to the daemon,
then computes:

```
speedup           = T_local / T_remote
transfer overhead = T_remote − T_worker
upload throughput = (input_bytes × 8) / upload_seconds
```

Outputs:

- [docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md) — methodology, results table, analysis
- [docs/BENCHMARK_RESULTS.csv](docs/BENCHMARK_RESULTS.csv) — raw rows for plotting

**Reference result (loopback, this machine):**

| Case | Local (s) | Remote wall (s) | Overhead (s) | Speedup | Engine |
|---|---:|---:|---:|---:|---|
| video 480p transcode | 1.527 | 1.626 | 0.087 | x0.939 | libx264 |
| tensor GEMM 1024 | 10.883 | 11.228 | 0.203 | x0.969 | numpy-cpu |
| synthetic render 8 MiB | 0.082 | 0.279 | 0.204 | x0.294 | python-cpu |

> **Interpretation:** on loopback the network cost is ~0, so a speedup ≈ 1 is the
> expected result — it proves the offload path adds only tens of milliseconds of
> handshake/transfer overhead. Real gains appear when the *worker* has a discrete
> GPU (NVENC/CUDA) while the client does not; run the benchmark across two
> machines to reproduce that comparison (the report header records both hosts).

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` | daemon not running, wrong IP/port, or firewall rule missing (§5.2 Step 2) |
| Badge stays **● Offline** | check IP/port; try `python -m client.cli --host … ping` first |
| `h264_nvenc` fails (`Cannot load nvcuda.dll`) | expected without an NVIDIA driver — the executor logs it and falls back to `libx264` automatically |
| No FFmpeg found | `python -m pip install imageio-ffmpeg` (provides a bundled binary) |
| `torch` not available | tensor task runs NumPy/CPU automatically; install torch for CUDA |
| Upload rejected: `checksum mismatch` | network corruption or a file changed mid-upload — re-run; the bad file is deleted server-side |
| GUI screenshots come out black | use `tools/capture_gui.py` (PrintWindow) instead of plain `ImageGrab` on a locked session |
| `python` not found in CMD | use `py -m …` or re-install Python with "Add to PATH" |

---

## 12. Submission Checklist

- [x] Public GitHub repository with full source code
- [x] Clear separation of **`client/`** and **`server/`** directories
- [x] Comprehensive **README.md** with step-by-step setup, network configuration
      guide (static IP over Ethernet/Wi-Fi), and execution guide (daemon + GUI)
- [x] **Task 1:** socket programming, handshake, latency ping
- [x] **Task 2:** GPU/FFmpeg rendering daemon with capability detection
- [x] **Task 3:** CustomTkinter GUI with progress display
- [x] **Task 4:** asynchronous progress streaming, timeouts, SHA-256 checksums,
      graceful disconnect/cancel
- [x] **Task 5:** benchmarking report with analysis
- [x] Execution screenshots + animated demo GIF
- [x] Automated test suite (30 tests, all passing)
