"""
benchmarks/run_benchmark.py
----------------------------
Task 5: Performance Benchmarking & Analysis.

Compares **local execution on the client laptop** against **remote execution on
the worker node** for the same job, across several input sizes / resolutions,
and reports:

* local wall time vs remote wall time
* speedup factor            = T_local / T_remote
* network transfer overhead = upload + download seconds
* effective throughput (Mbit/s) and RTT
* queue / handshake overhead (remote wall - worker compute time)

Usage::

    python -m benchmarks.run_benchmark                 # quick matrix
    python -m benchmarks.run_benchmark --full          # full matrix
    python -m benchmarks.run_benchmark --host 192.168.1.1 --json out.json

Results are written to ``docs/BENCHMARK_RESULTS.csv`` and a Markdown report
is rendered to ``docs/BENCHMARK_REPORT.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common import config  # noqa: E402
from client.api import OffloadClient, OffloadError, LogEvent, ProgressEvent  # noqa: E402
from server.environment import detect  # noqa: E402
from server.executors import get_executor, probe_duration  # noqa: E402

DOCS_DIR = os.path.join(ROOT, "docs")
SAMPLE_DIR = os.path.join(ROOT, "client", "samples")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class BenchRow:
    case: str
    task: str
    input_bytes: int
    resolution: str
    local_seconds: float
    remote_seconds: float          # full client-observed wall time
    worker_seconds: float          # compute time measured on the worker
    upload_seconds: float
    download_seconds: float
    rtt_ms: float
    engine: str
    speedup: float = 0.0
    overhead_seconds: float = 0.0
    upload_mbps: float = 0.0
    notes: str = ""

    def finalize(self) -> "BenchRow":
        if self.remote_seconds > 0:
            self.speedup = round(self.local_seconds / self.remote_seconds, 3)
        self.overhead_seconds = round(
            self.remote_seconds - self.worker_seconds, 3)
        if self.upload_seconds > 0:
            self.upload_mbps = round(
                (self.input_bytes * 8) / (self.upload_seconds * 1e6), 1)
        return self


# --------------------------------------------------------------------------- #
# Local baseline (same executors, run in-process on the client)
# --------------------------------------------------------------------------- #

def run_local(task: str, input_path: str, params: dict,
              out_dir: Optional[str] = None) -> float:
    """Execute the same executor locally and return wall seconds."""
    out_dir = out_dir or tempfile.mkdtemp(prefix="bench_local_")
    os.makedirs(out_dir, exist_ok=True)
    ext = {"video": ".mp4", "tensor": ".npy", "synthetic": ".bin"}[task]
    output = os.path.join(out_dir, f"local_{int(time.time())}{ext}")

    executor = get_executor(task)
    t0 = time.perf_counter()
    try:
        executor.run(input_path, output, params,
                     progress_cb=lambda p, i: None,
                     log_cb=lambda m: None)
    except Exception as exc:  # noqa: BLE001 - recorded in the report
        return -1.0 if "ffmpeg" in str(exc).lower() else -2.0
    return time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Remote run
# --------------------------------------------------------------------------- #

def run_remote(host: str, port: int, task: str, input_path: str,
               params: dict) -> tuple:
    """Returns (remote_wall, worker_seconds, upload_s, download_s, rtt, engine)."""
    client = OffloadClient(host, port, on_log=lambda e: None)
    client.connect()
    rtt = client.ping()
    t0 = time.perf_counter()
    result = client.run_job(task, input_path, params=params)
    wall = time.perf_counter() - t0
    client.close()
    download_s = max(wall - result.duration_s - result.transfer_seconds, 0.0)
    return {
        "wall": wall,
        "worker": result.duration_s,
        "upload": result.transfer_seconds,
        "download": download_s,
        "rtt": rtt["avg_ms"],
        "engine": str(result.result.get("engine", "?")),
        "job_id": result.job_id,
    }


# --------------------------------------------------------------------------- #
# Case matrix
# --------------------------------------------------------------------------- #

def build_cases(quick: bool) -> List[dict]:
    video = os.path.join(SAMPLE_DIR, "sample_720p.mp4")
    blob = os.path.join(SAMPLE_DIR, "sample_input.bin")
    cases: List[dict] = []

    # --- Video transcoding across resolutions -------------------------- #
    for res, label in (("854x480", "480p"), ("1280x720", "720p")):
        if quick and res == "1280x720":
            continue
        cases.append({
            "case": f"video {label} transcode",
            "task": "video",
            "input": video,
            "resolution": res,
            "params": {"resolution": res, "bitrate": "4M",
                       "preset": "veryfast", "encoder": "auto", "quality": 23},
        })
    if not quick:
        cases.append({
            "case": "video 360p transcode",
            "task": "video",
            "input": video,
            "resolution": "640x360",
            "params": {"resolution": "640x360", "bitrate": "2M",
                       "preset": "veryfast", "encoder": "auto", "quality": 23},
        })

    # --- Tensor (compute) workload -------------------------------------- #
    cases.append({
        "case": "tensor GEMM 1024 (compute offload)",
        "task": "tensor",
        "input": blob,
        "resolution": "-",
        "params": {"matrix_size": 1024, "iterations": 60 if not quick else 30,
                   "batch_size": 32, "device": "auto"},
    })

    # --- Synthetic pipeline / transfer-dominated job ---------------------- #
    cases.append({
        "case": "synthetic render 8MiB input",
        "task": "synthetic",
        "input": blob,
        "resolution": "-",
        "params": {"work_units": 200 if not quick else 80},
    })
    return cases


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_matrix(host: str = "127.0.0.1", port: int = config.DEFAULT_PORT,
               quick: bool = True) -> List[dict]:
    rows: List[dict] = []
    for case in build_cases(quick):
        input_path = case["input"]
        if not os.path.exists(input_path):
            print(f"[skip] missing asset {input_path}")
            continue
        size = os.path.getsize(input_path)
        print(f"\n=== {case['case']} ({size} bytes) ===")

        local_s = run_local(case["task"], input_path, dict(case["params"]))
        print(f"  local  : {local_s:.2f}s")

        try:
            remote = run_remote(host, port, case["task"], input_path,
                                dict(case["params"]))
        except (OffloadError, OSError) as exc:
            print(f"  remote : FAILED ({exc})")
            rows.append(BenchRow(
                case=case["case"], task=case["task"], input_bytes=size,
                resolution=case["resolution"],
                local_seconds=round(max(local_s, 0.0), 3), remote_seconds=0.0,
                worker_seconds=0.0, upload_seconds=0.0, download_seconds=0.0,
                rtt_ms=0.0, engine="n/a", notes=f"remote failed: {exc}",
            ).finalize().__dict__)
            continue

        row = BenchRow(
            case=case["case"],
            task=case["task"],
            input_bytes=size,
            resolution=case["resolution"],
            local_seconds=round(max(local_s, 0.0), 3),
            remote_seconds=round(remote["wall"], 3),
            worker_seconds=round(remote["worker"], 3),
            upload_seconds=round(remote["upload"], 3),
            download_seconds=round(remote["download"], 3),
            rtt_ms=round(remote["rtt"], 2),
            engine=remote["engine"],
            notes="" if local_s > 0 else "local baseline unavailable (no ffmpeg)",
        ).finalize()
        print(f"  remote : {remote['wall']:.2f}s "
              f"(worker {remote['worker']:.2f}s, upload {remote['upload']:.2f}s) "
              f"-> speedup x{row.speedup}")
        rows.append(row.__dict__)
    return rows


def write_csv(rows: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: List[dict], path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# Performance Benchmark & Analysis Report",
        "",
        "**Course:** CSC-334 Parallel and Distributed Computing  ",
        "**Task:** Task 5 - Performance Benchmarking & Analysis  ",
        f"**Generated:** {meta.get('generated', '')}  ",
        f"**Worker node:** {meta.get('worker', 'unknown')}  ",
        f"**Client node:** {meta.get('client', 'unknown')}  ",
        f"**Worker capabilities:** {meta.get('capabilities', '')}  ",
        f"**Average RTT:** {meta.get('rtt_ms', 'n/a')} ms  ",
        "",
        "## 1. Methodology",
        "",
        "Each case runs the *same* executor and the *same* input twice:",
        "",
        "1. **Local baseline** - executed in-process on the client laptop.",
        "2. **Remote offload** - executed by the worker daemon; the client-observed",
        "   wall time includes handshake, SHA-256 upload, execution, and the",
        "   verified download of the artefact.",
        "",
        ("> **Note:** this run used a loopback/localhost target, so the network"
         " term is ~0 and the speedup hovers around x1 - it isolates protocol"
         " overhead. Re-run against the real worker IP"
         " (`--host 192.168.1.1`) over a LAN to measure genuine offload speedup."
         if str(meta.get("host", "")).startswith(("127.", "localhost")) else
         "> **Note:** measured against a real remote worker over the LAN."),
        "",
        "```\n"
        "speedup            = T_local / T_remote\n"
        "transfer overhead  = T_remote - T_worker\n"
        "upload throughput  = (input_bytes x 8) / upload_seconds   [bit/s]\n"
        "```",
        "",
        "## 2. Results",
        "",
        "| Case | Input | Local (s) | Remote wall (s) | Worker compute (s) | "
        "Upload (s) | Download (s) | Overhead (s) | Speedup | Engine |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['input_bytes']:,} | {row['local_seconds']} | "
            f"{row['remote_seconds']} | {row['worker_seconds']} | "
            f"{row['upload_seconds']} | {row['download_seconds']} | "
            f"{row['overhead_seconds']} | "
            f"{('x' + str(row['speedup'])) if row['speedup'] else '-'} | "
            f"{row['engine']} |"
        )

    succeeded = [r for r in rows if r["remote_seconds"] > 0 and r["local_seconds"] > 0]
    lines += ["", "## 3. Analysis", ""]
    if succeeded:
        mean_speedup = statistics.mean(r["speedup"] for r in succeeded)
        mean_overhead = statistics.mean(r["overhead_seconds"] for r in succeeded)
        best = max(succeeded, key=lambda r: r["speedup"])
        lines += [
            f"* **Mean speedup factor:** x{mean_speedup:.2f} across "
            f"{len(succeeded)} comparable case(s).",
            f"* **Best case:** {best['case']} at x{best['speedup']} "
            f"({best['local_seconds']}s local -> {best['remote_seconds']}s remote).",
            f"* **Mean network overhead** (handshake + upload + download): "
            f"{mean_overhead:.2f}s.",
            "",
        ]
        gpu_rows = [r for r in succeeded if r["engine"] in
                    ("h264_nvenc", "hevc_nvenc", "h264_qsv", "torch-cuda")]
        cpu_rows = [r for r in succeeded if r["engine"] in
                    ("libx264", "numpy-cpu", "python-cpu")]
        if gpu_rows:
            lines.append(
                f"* **Hardware acceleration:** GPU engines averaged "
                f"x{statistics.mean(r['speedup'] for r in gpu_rows):.2f} speedup."
            )
        if cpu_rows:
            lines.append(
                f"* **CPU fallback engines** averaged "
                f"x{statistics.mean(r['speedup'] for r in cpu_rows):.2f} speedup."
            )
    else:
        lines.append("* No directly comparable local/remote pair completed; "
                     "see the `notes` column.")

    lines += [
        "",
        "### Transfer overhead vs job size",
        "",
        "| Case | Input bytes | Upload (s) | Throughput (Mbit/s) | "
        "Overhead share of wall time |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        share = (row["overhead_seconds"] / row["remote_seconds"] * 100.0
                 if row["remote_seconds"] else 0.0)
        lines.append(
            f"| {row['case']} | {row['input_bytes']:,} | {row['upload_seconds']} | "
            f"{row['upload_mbps']} | {share:.1f}% |"
        )

    lines += [
        "",
        "## 4. Conclusions",
        "",
        "* Offloading wins when the **compute-to-transfer ratio** is high: heavy",
        "  encodes and tensor workloads amortise the upload/download cost easily.",
        "  On a 1 Gbit/s LAN a 100 MiB asset moves in well under a second, so the",
        "  fixed ~RTT + handshake cost dominates the overhead term.",
        "* Transfer-dominated, cheap jobs (tiny synthetic payloads) show the",
        "  overhead floor: this is the price of offloading and is visible in the",
        "  `Overhead share` column.",
        "* Real GPU workers (NVENC / CUDA) push the speedup far beyond the CPU",
        "  numbers reported here when the worker node has a dedicated GPU; the",
        "  harness automatically records whichever engine the worker selected.",
        "",
        "---",
        "*Raw data:* `docs/BENCHMARK_RESULTS.csv` - regenerate with "
        "`python -m benchmarks.run_benchmark`.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Local vs remote benchmark")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=config.DEFAULT_PORT)
    parser.add_argument("--full", action="store_true", help="run the full matrix")
    parser.add_argument("--json", default=None, help="also write raw JSON here")
    args = parser.parse_args(argv)

    rows = run_matrix(host=args.host, port=args.port, quick=not args.full)

    csv_path = os.path.join(DOCS_DIR, "BENCHMARK_RESULTS.csv")
    report_path = os.path.join(DOCS_DIR, "BENCHMARK_REPORT.md")
    env = detect()
    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "worker": env.hostname,
        "host": args.host,
        "client": os.environ.get("COMPUTERNAME", "client"),
        "capabilities": env.engine_summary,
        "rtt_ms": "see rows",
    }
    write_csv(rows, csv_path)
    write_report(rows, report_path, meta)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"meta": meta, "rows": rows}, handle, indent=2)

    print(f"\n[ok] wrote {csv_path}")
    print(f"[ok] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
