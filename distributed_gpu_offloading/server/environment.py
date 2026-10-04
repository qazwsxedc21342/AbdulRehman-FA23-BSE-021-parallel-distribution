"""
server/environment.py
---------------------
Capability discovery for the worker node: FFmpeg binary, hardware encoders
(NVENC / Quick Sync / VA-API), NVIDIA GPU models and CUDA/PyTorch availability.

Everything here is *best effort*: a headless worker without a GPU must still
start and fall back to CPU encoders so the system degrades gracefully.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import List, Optional

from common.config import VIDEO_ENCODER_PRIORITY


def _run(cmd: List[str], timeout: float = 8.0) -> Optional[str]:
    """Run a command and return stdout, or None on any failure."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace")


@dataclass
class Environment:
    """Snapshot of what this worker node can actually execute."""

    hostname: str
    ffmpeg_path: Optional[str] = None
    ffmpeg_version: Optional[str] = None
    encoders: List[str] = field(default_factory=list)          # hardware only
    all_encoders: List[str] = field(default_factory=list)
    gpus: List[dict] = field(default_factory=list)
    cuda_available: bool = False
    torch_version: Optional[str] = None
    cpu_count: int = 1
    python_version: str = ""

    # ------------------------------------------------------------------ #
    @property
    def best_video_encoder(self) -> Optional[str]:
        """Highest priority encoder that this machine actually supports."""
        for name, _label in VIDEO_ENCODER_PRIORITY:
            if name in self.all_encoders:
                return name
        return None

    @property
    def gpu_label(self) -> str:
        if self.gpus:
            return "; ".join(
                f"{g.get('name', 'GPU')} ({g.get('memory', '?')})" for g in self.gpus
            )
        return "No dedicated GPU detected (CPU mode)"

    @property
    def engine_summary(self) -> str:
        enc = self.best_video_encoder
        parts = []
        if enc:
            parts.append(f"video={enc}")
        if self.cuda_available:
            parts.append(f"cuda=torch {self.torch_version}")
        parts.append(f"cpu={self.cpu_count} logical")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["best_video_encoder"] = self.best_video_encoder
        data["gpu_label"] = self.gpu_label
        data["engine_summary"] = self.engine_summary
        return data


def _detect_ffmpeg(env: Environment) -> None:
    path = shutil.which("ffmpeg")
    if not path:
        # Optional fallback: the `imageio-ffmpeg` wheel ships a static build.
        # This keeps the worker usable on machines without a system FFmpeg.
        try:
            import imageio_ffmpeg  # type: ignore

            path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            path = None
    if not path:
        return
    env.ffmpeg_path = path
    out = _run([path, "-version"])
    if out:
        env.ffmpeg_version = out.splitlines()[0].strip() if out.splitlines() else None
    # `ffmpeg -hide_banner -encoders` lists every compiled-in encoder.
    enc_out = _run([path, "-hide_banner", "-encoders"], timeout=12.0)
    if enc_out:
        names = []
        for line in enc_out.splitlines():
            parts = line.split()
            # Lines look like:  " V....D h264_nvenc           NVIDIA NVENC H.264 encoder"
            if (len(parts) >= 2 and parts[0] and parts[0][0] in "VAS"
                    and parts[0].replace(".", "")
                    and parts[1].replace("_", "").isalnum()
                    and not parts[0].startswith("---")):
                names.append(parts[1])
        env.all_encoders = names
        hardware = {"h264_nvenc", "hevc_nvenc", "h264_qsv", "hevc_qsv", "h264_vaapi", "h264_videotoolbox"}
        env.encoders = [n for n in names if n in hardware]
        return
    # Some minimal builds omit `-encoders`; probe the known list one by one.
    probed = []
    for name, _label in VIDEO_ENCODER_PRIORITY:
        if _run([path, "-hide_banner", "-h", f"encoder={name}"], timeout=6.0):
            probed.append(name)
    env.all_encoders = probed or (["libx264"] if path else [])
    hardware = {"h264_nvenc", "hevc_nvenc", "h264_qsv", "hevc_qsv", "h264_vaapi", "h264_videotoolbox"}
    env.encoders = [n for n in env.all_encoders if n in hardware]


def _detect_gpus(env: Environment) -> None:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=8.0,
    )
    if not out:
        return
    for line in out.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) >= 4:
            env.gpus.append(
                {
                    "index": cols[0],
                    "name": cols[1],
                    "memory": f"{cols[2]} MiB",
                    "driver": cols[3],
                }
            )


def _detect_torch(env: Environment) -> None:
    try:
        import torch  # type: ignore
    except Exception:
        return
    env.torch_version = getattr(torch, "__version__", None)
    try:
        env.cuda_available = bool(torch.cuda.is_available())
    except Exception:
        env.cuda_available = False
    if env.cuda_available and not env.gpus:
        try:
            env.gpus.append(
                {
                    "index": "0",
                    "name": torch.cuda.get_device_name(0),
                    "memory": "via torch",
                    "driver": "n/a",
                }
            )
        except Exception:
            pass


@lru_cache(maxsize=1)
def detect() -> Environment:
    """Detect (and cache) the capabilities of this machine."""
    import platform
    import sys

    env = Environment(
        hostname=platform.node() or "worker",
        cpu_count=os.cpu_count() or 1,
        python_version=platform.python_version(),
    )
    _detect_ffmpeg(env)
    _detect_gpus(env)
    _detect_torch(env)
    return env


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    print(json.dumps(detect().to_dict(), indent=2))
