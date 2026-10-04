"""
client/cli.py
-------------
Headless command-line client (same code path as the GUI).

Useful for servers without a display, for the benchmark harness and for
smoke-testing the whole pipeline:

    python -m client.cli ping  --host 127.0.0.1
    python -m client.cli info  --host 127.0.0.1
    python -m client.cli run   --host 127.0.0.1 --task video --input clip.mp4 \\
                               --resolution 1280x720 --bitrate 4M
    python -m client.cli bench --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

from common import config
from client.api import LogEvent, OffloadClient, OffloadError, ProgressEvent


def _print_log(event: LogEvent) -> None:
    tag = {"warning": "WARN", "error": "ERROR"}.get(event.level, "INFO")
    print(f"  [{tag}] {event.message}", flush=True)


def _print_progress(event: ProgressEvent) -> None:
    bar_len = 32
    filled = int(bar_len * event.percent / 100.0)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r  [{bar}] {event.percent:6.2f}%  {event.stage:<10}", end="", flush=True)
    if event.percent >= 100.0:
        print()


def _build_params(args: argparse.Namespace) -> dict:
    if args.task == "video":
        return {
            "resolution": args.resolution,
            "bitrate": args.bitrate,
            "preset": args.preset,
            "encoder": args.encoder,
            "quality": args.quality,
            "audio_bitrate": args.audio_bitrate,
        }
    if args.task == "tensor":
        return {
            "matrix_size": args.matrix_size,
            "iterations": args.iterations,
            "batch_size": args.batch_size,
            "device": args.device,
        }
    return {"work_units": args.work_units}


def cmd_ping(args: argparse.Namespace) -> int:
    client = OffloadClient(args.host, args.port, on_log=_print_log)
    try:
        client.connect(retries=args.retries)
        stats = client.ping(samples=args.count)
        print(f"RTT avg={stats['avg_ms']:.2f} ms  min={stats['min_ms']:.2f} ms  "
              f"max={stats['max_ms']:.2f} ms  jitter={stats['jitter_ms']:.2f} ms  "
              f"loss={stats['loss_pct']:.1f}%  samples={stats['samples']}/{stats['requested']}")
        return 0
    except OffloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        client.close()


def cmd_info(args: argparse.Namespace) -> int:
    client = OffloadClient(args.host, args.port, on_log=_print_log)
    try:
        info = client.connect(retries=args.retries)
        client.ping(samples=3)
        env = info.get("environment", {})
        print(json.dumps({
            "server_id": env.get("hostname"),
            "session_id": info.get("session_id"),
            "rtt_ms": round(client.rtt_ms, 2),
            "supported_tasks": info.get("supported_tasks"),
            "environment": env,
        }, indent=2))
        return 0
    except OffloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        client.close()


def cmd_run(args: argparse.Namespace) -> int:
    client = OffloadClient(args.host, args.port, on_log=_print_log)
    try:
        client.connect(retries=args.retries)
        client.ping()
        t0 = time.perf_counter()
        result = client.run_job(
            args.task, args.input, params=_build_params(args),
            output_name=args.output, wait_timeout=args.timeout,
            on_progress=_print_progress, on_log=_print_log,
        )
        wall = time.perf_counter() - t0
        print()
        print("=" * 64)
        print(f"  job id      : {result.job_id}")
        print(f"  engine      : {result.result.get('engine')}")
        print(f"  worker time : {result.duration_s:.2f} s")
        print(f"  wall time   : {wall:.2f} s")
        print(f"  upload      : {result.transfer_bytes} bytes in "
              f"{result.transfer_seconds:.2f} s")
        print(f"  output      : {result.output_path} "
              f"({result.output_bytes} bytes, sha256 {result.output_sha256[:16]}...)")
        print("=" * 64)
        return 0
    except OffloadError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 3
    finally:
        client.close()


def cmd_bench(args: argparse.Namespace) -> int:
    from benchmarks.run_benchmark import run_matrix  # local import keeps CLI light
    rows = run_matrix(host=args.host, port=args.port, quick=not args.full)
    print(json.dumps(rows, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="client.cli",
        description="CSC-334 Distributed Task Offloading - command line client")
    parser.add_argument("--host", default=config.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=config.DEFAULT_PORT)
    parser.add_argument("--retries", type=int, default=config.MAX_SUBMIT_ATTEMPTS)

    sub = parser.add_subparsers(dest="command", required=True)

    ping = sub.add_parser("ping", help="measure latency to the worker")
    ping.add_argument("--count", type=int, default=5,
                      help="number of PING probes (default 5)")
    sub.add_parser("info", help="print worker capabilities as JSON")

    run = sub.add_parser("run", help="submit and download a job")
    run.add_argument("--task", default="video", choices=config.SUPPORTED_TASKS)
    run.add_argument("--input", required=True, help="path to the input asset")
    run.add_argument("--output", default=None, help="remote output filename")
    run.add_argument("--timeout", type=float, default=1800.0)
    run.add_argument("--resolution", default="1280x720", choices=list(config.RESOLUTIONS))
    run.add_argument("--bitrate", default="4M")
    run.add_argument("--preset", default="p4")
    run.add_argument("--encoder", default="auto")
    run.add_argument("--quality", type=int, default=23)
    run.add_argument("--audio-bitrate", default="192k")
    run.add_argument("--matrix-size", type=int, default=1024)
    run.add_argument("--iterations", type=int, default=60)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    run.add_argument("--work-units", type=int, default=0)

    bench = sub.add_parser("bench", help="run the local-vs-remote benchmark matrix")
    bench.add_argument("--full", action="store_true", help="run the full matrix")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ping":
        return cmd_ping(args)
    if args.command == "info":
        return cmd_info(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "bench":
        return cmd_bench(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
