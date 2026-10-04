"""
server/executors.py
-------------------
Execution engines running on the worker node (Task 2).

Three task types are implemented, all sharing one callback contract so the
daemon can stream progress asynchronously to the client (Task 4):

    video      -> FFmpeg transcode, hardware accelerated with NVENC when the
                  machine supports it, automatic libx264 CPU fallback
                  otherwise. Progress is parsed from ``ffmpeg -progress``.
    tensor     -> PyTorch CUDA GEMM when CUDA is available, NumPy GEMM on the
                  CPU otherwise. Iteration-level progress callbacks.
    synthetic  -> pure-Python CPU workload used for benchmarking the pipeline
                  itself (transfer + queueing) without any external binary.

Every executor returns a result dict and reports progress through
``progress_cb(percent, info_dict)`` and log lines through ``log_cb(text)``.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Optional

from common.config import VIDEO_ENCODER_PRIORITY
from server.environment import Environment, detect

ProgressCB = Callable[[float, dict], None]
LogCB = Callable[[str], None]


class ExecutorError(RuntimeError):
    """Raised when a task cannot be executed on this worker."""


def _null_progress(percent: float, info: dict) -> None:
    return None


def _null_log(text: str) -> None:
    return None


# --------------------------------------------------------------------------- #
# Probe helpers
# --------------------------------------------------------------------------- #

def probe_duration(path: str) -> Optional[float]:
    """Return media duration in seconds via ffprobe (or ffmpeg stderr)."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            proc = subprocess.run(
                [
                    ffprobe, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "json", path,
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=False,
            )
            if proc.returncode == 0:
                import json

                data = json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
                dur = data.get("format", {}).get("duration")
                if dur not in (None, "N/A"):
                    return float(dur)
        except Exception:
            pass

    ffmpeg = detect().ffmpeg_path
    if not ffmpeg:
        return None
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=False,
        )
    except Exception:
        return None
    text = proc.stderr.decode("utf-8", "replace")
    # e.g. "  Duration: 00:00:10.00, start: 0.000000, bitrate: 1234 kb/s"
    for line in text.splitlines():
        if "Duration:" in line and "N/A" not in line:
            try:
                hh, mm, rest = line.split("Duration:", 1)[1].strip().split(",")[0].split(":")
                return int(hh) * 3600 + int(mm) * 60 + float(rest)
            except (ValueError, IndexError):
                continue
    return None


# --------------------------------------------------------------------------- #
# Video transcode executor
# --------------------------------------------------------------------------- #

class VideoExecutor:
    """FFmpeg-based transcoding with hardware acceleration when available."""

    TASK = "video"

    def __init__(self, env: Optional[Environment] = None):
        self.env = env or detect()

    # -- encoder negotiation -------------------------------------------- #
    def pick_encoder(self, requested: str = "auto") -> str:
        available = self.env.all_encoders
        if not available:
            raise ExecutorError(
                "FFmpeg was not found on this worker. Install ffmpeg "
                "(https://ffmpeg.org/download.html) and restart the daemon."
            )
        if requested and requested != "auto":
            if requested in available:
                return requested
            # Fall through to automatic selection with a clear log line.
        for name, _label in VIDEO_ENCODER_PRIORITY:
            if name in available:
                return name
        return "libx264" if "libx264" in available else available[0]

    def _encoder_args(self, encoder: str, params: dict) -> list:
        """Build encoder-specific argument lists."""
        bitrate = str(params.get("bitrate", "4M"))
        args: list = ["-c:v", encoder]

        if encoder.endswith("nvenc"):
            preset = params.get("preset", "p4")
            if preset not in ("p1", "p2", "p3", "p4", "p5"):
                preset = "p4"
            args += ["-preset", preset, "-rc", "vbr", "-b:v", bitrate]
            if params.get("profile"):
                args += ["-profile:v", str(params["profile"])]
            # Quality knob: cq (constant quality) maps from the 0-100 slider.
            quality = int(params.get("quality", 23))
            cq = max(1, min(51, quality))
            args += ["-cq", str(cq)]
        elif encoder in ("h264_qsv", "hevc_qsv"):
            args += ["-b:v", bitrate, "-quality", str(params.get("qsv_quality", "speed"))]
        elif encoder == "h264_vaapi":
            args = ["-c:v", "h264_vaapi", "-b:v", bitrate]
        else:  # software x264 family
            preset = params.get("preset", "medium")
            if preset not in ("ultrafast", "superfast", "veryfast", "faster",
                              "fast", "medium", "slow", "slower", "veryslow"):
                preset = "medium"
            args += ["-preset", preset, "-b:v", bitrate,
                     "-crf", str(params.get("crf", params.get("quality", 23)))]
        return args

    def _scale_filter(self, params: dict) -> Optional[str]:
        resolution = str(params.get("resolution", "source"))
        if not resolution or resolution == "source":
            return None
        try:
            width, height = resolution.lower().split("x")
            int(width), int(height)
        except ValueError:
            return None
        return f"scale={int(width)}:{int(height)}"

    # -- execution -------------------------------------------------------- #
    def run(self, input_path: str, output_path: str, params: dict,
            progress_cb: ProgressCB = _null_progress,
            log_cb: LogCB = _null_log,
            cancel_event: Optional[threading.Event] = None) -> dict:
        if not self.env.ffmpeg_path:
            raise ExecutorError("FFmpeg is not installed on this worker node.")

        encoder = self.pick_encoder(params.get("encoder", "auto"))
        requested_hw = encoder not in ("libx264", "libx265", "mpeg4")
        log_cb(f"FFmpeg: {self.env.ffmpeg_version or 'unknown version'}")
        log_cb(f"Encoder selected: {encoder} "
               f"({'hardware/GPU' if requested_hw else 'software/CPU'})")

        # If a *hardware* encoder fails at runtime (driver missing, no GPU on
        # this node, NVENC session limit), transparently retry once on the
        # CPU encoder so the job still completes instead of dying.
        try:
            return self._run_once(input_path, output_path, params, encoder,
                                  progress_cb, log_cb, cancel_event)
        except ExecutorError as exc:
            if not requested_hw:
                raise
            log_cb(f"Hardware encoder '{encoder}' failed: {exc}")
            log_cb("Falling back to software encoder libx264 (CPU) ...")
            progress_cb(0.0, {"engine": "libx264", "hardware": False,
                              "note": "encoder fallback"})
            return self._run_once(input_path, output_path, params, "libx264",
                                  progress_cb, log_cb, cancel_event)

    def _run_once(self, input_path: str, output_path: str, params: dict,
                  encoder: str, progress_cb: ProgressCB, log_cb: LogCB,
                  cancel_event: Optional[threading.Event]) -> dict:
        hw = encoder not in ("libx264", "libx265", "mpeg4")
        duration = probe_duration(input_path)

        cmd = [self.env.ffmpeg_path, "-hide_banner", "-y", "-nostdin", "-i", input_path]
        scale = self._scale_filter(params)
        if scale:
            cmd += ["-vf", scale]
        cmd += self._encoder_args(encoder, params)
        # Audio: re-encode to AAC at the requested audio bitrate when a stream exists.
        cmd += ["-c:a", "aac", "-b:a", str(params.get("audio_bitrate", "192k"))]
        cmd += ["-movflags", "+faststart"]
        # Machine readable progress on stdout:  key=value lines.
        cmd += ["-progress", "pipe:1", "-nostats", output_path]

        log_cb("Command: " + " ".join(cmd))
        started = time.perf_counter()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise ExecutorError(f"failed to launch ffmpeg: {exc}") from exc

        stderr_tail: list = []

        def _drain_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                line = line.rstrip()
                if not line:
                    continue
                stderr_tail.append(line)
                if len(stderr_tail) > 200:
                    del stderr_tail[:50]
                # Surface the interesting lines (resolution/fps/bitrate).
                if any(tag in line for tag in ("frame=", "speed=", "Stream #", "Output #")):
                    log_cb(line)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        percent = 0.0
        last_emit = 0.0
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key == "out_time_ms" and duration and duration > 0:
                try:
                    seconds = int(value) / 1_000_000.0
                except ValueError:
                    continue
                percent = max(0.0, min(99.5, (seconds / duration) * 100.0))
            elif key == "speed" and value not in ("N/A", "0x"):
                pass  # carried below
            elif key == "progress" and value == "end":
                percent = 100.0
            elif key == "fps":
                pass

            now = time.perf_counter()
            if now - last_emit >= 0.25 or percent >= 100.0:
                last_emit = now
                progress_cb(
                    round(percent, 2),
                    {
                        "engine": encoder,
                        "hardware": hw,
                        "elapsed": round(now - started, 2),
                        "duration": duration,
                    },
                )
            if cancel_event is not None and cancel_event.is_set():
                proc.kill()
                raise ExecutorError("job cancelled by client")

        proc.wait()
        stderr_thread.join(timeout=2.0)
        elapsed = time.perf_counter() - started

        if proc.returncode != 0:
            tail = "\n".join(stderr_tail[-15:])
            raise ExecutorError(f"ffmpeg exited with code {proc.returncode}:\n{tail}")

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ExecutorError("ffmpeg produced no output file")

        progress_cb(100.0, {"engine": encoder, "hardware": hw, "elapsed": round(elapsed, 2)})
        return {
            "engine": encoder,
            "hardware_accelerated": hw,
            "duration_s": round(elapsed, 3),
            "source_duration_s": duration,
            "output_bytes": os.path.getsize(output_path),
        }


# --------------------------------------------------------------------------- #
# Tensor executor (PyTorch CUDA when available, NumPy otherwise)
# --------------------------------------------------------------------------- #

class TensorExecutor:
    """Batched GEMM benchmark/training-style workload on GPU or CPU."""

    TASK = "tensor"

    def run(self, input_path: str, output_path: str, params: dict,
            progress_cb: ProgressCB = _null_progress,
            log_cb: LogCB = _null_log,
            cancel_event: Optional[threading.Event] = None) -> dict:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise ExecutorError("NumPy is required for tensor tasks") from exc

        iterations = int(params.get("iterations", 60))
        size = int(params.get("matrix_size", 1024))
        batch = int(params.get("batch_size", 32))

        use_torch = False
        torch = None
        device = "cpu"
        if params.get("device", "auto") in ("auto", "cuda"):
            try:
                import torch as _torch  # type: ignore
                torch = _torch
                if _torch.cuda.is_available():
                    use_torch = True
                    device = "cuda"
            except Exception:
                torch = None

        log_cb(f"Tensor workload: {size}x{size} GEMM, {iterations} iterations, "
               f"batch={batch}, device={device if use_torch else 'cpu (numpy)'}")

        started = time.perf_counter()
        checksum = 0.0

        if use_torch:
            assert torch is not None
            log_cb(f"CUDA device: {torch.cuda.get_device_name(0)}")
            a = torch.randn(size, size, device="cuda")
            b = torch.randn(size, size, device="cuda")
            torch.cuda.synchronize()
            for i in range(iterations):
                if cancel_event is not None and cancel_event.is_set():
                    raise ExecutorError("job cancelled by client")
                for _ in range(batch):
                    c = a @ b
                    c = c @ c if i % 2 else c
                torch.cuda.synchronize()
                checksum += float(c.float().mean().item())
                progress_cb(
                    round((i + 1) / iterations * 100.0, 2),
                    {"engine": "torch-cuda", "device": device,
                     "elapsed": round(time.perf_counter() - started, 2)},
                )
            metrics = {
                "engine": "torch-cuda",
                "device": "cuda",
                "gpu_name": torch.cuda.get_device_name(0),
            }
            # Persist the resulting tensor for the client to download.
            np.save(output_path if output_path.endswith(".npy") else output_path + ".npy",
                    c.detach().cpu().numpy())
        else:
            rng = np.random.default_rng(1234)
            a = rng.standard_normal((size, size), dtype=np.float32)
            b = rng.standard_normal((size, size), dtype=np.float32)
            for i in range(iterations):
                if cancel_event is not None and cancel_event.is_set():
                    raise ExecutorError("job cancelled by client")
                for _ in range(batch):
                    c = a @ b
                checksum += float(c.mean())
                progress_cb(
                    round((i + 1) / iterations * 100.0, 2),
                    {"engine": "numpy-cpu", "device": "cpu",
                     "elapsed": round(time.perf_counter() - started, 2)},
                )
            metrics = {"engine": "numpy-cpu", "device": "cpu"}
            np.save(output_path if output_path.endswith(".npy") else output_path + ".npy",
                    c)

        final_path = output_path if output_path.endswith(".npy") else output_path + ".npy"
        elapsed = time.perf_counter() - started
        progress_cb(100.0, {"engine": metrics["engine"], "elapsed": round(elapsed, 2)})
        metrics.update(
            {
                "duration_s": round(elapsed, 3),
                "iterations": iterations,
                "matrix_size": size,
                "checksum": round(checksum, 6),
                "output_path": final_path,
            }
        )
        return metrics


# --------------------------------------------------------------------------- #
# Synthetic CPU executor (pipeline / benchmark harness)
# --------------------------------------------------------------------------- #

class SyntheticExecutor:
    """
    Deterministic CPU workload used to benchmark the *offloading pipeline*
    (transfer + queueing + streaming) on machines without FFmpeg/GPU.
    The workload size scales with the input file so transfer overhead shows
    up realistically in the benchmark report.
    """

    TASK = "synthetic"

    def run(self, input_path: str, output_path: str, params: dict,
            progress_cb: ProgressCB = _null_progress,
            log_cb: LogCB = _null_log,
            cancel_event: Optional[threading.Event] = None) -> dict:
        in_bytes = os.path.getsize(input_path) if os.path.exists(input_path) else 0
        work_units = int(params.get("work_units", 0)) or max(20, in_bytes // (256 * 1024))
        work_units = min(work_units, 4000)

        log_cb(f"Synthetic workload: {work_units} units "
               f"(input {in_bytes} bytes) on {os.cpu_count()} logical CPUs")

        started = time.perf_counter()
        acc = 0.0
        for i in range(work_units):
            if cancel_event is not None and cancel_event.is_set():
                raise ExecutorError("job cancelled by client")
            # ~30ms of pure-python math per unit on a typical laptop core.
            acc += sum(math.sqrt(j) * math.sin(j) for j in range(4000))
            progress_cb(
                round((i + 1) / work_units * 100.0, 2),
                {"engine": "python-cpu", "elapsed": round(time.perf_counter() - started, 2)},
            )

        elapsed = time.perf_counter() - started
        # Produce an artefact of comparable size to the input (transform report).
        report = (
            f"synthetic-render report\n"
            f"input_bytes={in_bytes}\nwork_units={work_units}\n"
            f"accumulator={acc:.6f}\nwall_seconds={elapsed:.3f}\n"
        ).encode("utf-8")
        with open(output_path, "wb") as handle:
            # Pad the report so output size scales with input size as well.
            handle.write(report)
            handle.write(b"\x00" * min(in_bytes, 64 * 1024 * 1024))

        progress_cb(100.0, {"engine": "python-cpu", "elapsed": round(elapsed, 2)})
        return {
            "engine": "python-cpu",
            "hardware_accelerated": False,
            "duration_s": round(elapsed, 3),
            "work_units": work_units,
            "accumulator": round(acc, 6),
            "output_bytes": os.path.getsize(output_path),
        }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

EXECUTORS = {
    "video": VideoExecutor,
    "tensor": TensorExecutor,
    "synthetic": SyntheticExecutor,
}


def get_executor(task: str):
    cls = EXECUTORS.get(task)
    if cls is None:
        raise ExecutorError(
            f"unsupported task '{task}' (supported: {', '.join(EXECUTORS)})"
        )
    return cls()


if __name__ == "__main__":  # pragma: no cover
    env = detect()
    print("ffmpeg:", env.ffmpeg_path)
    print("encoders:", env.encoders)
    print("best:", env.best_video_encoder)
    print("gpus:", env.gpus)
    print("cuda:", env.cuda_available)
