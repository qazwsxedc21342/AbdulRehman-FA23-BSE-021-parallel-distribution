"""
tools/make_sample_assets.py
----------------------------
Generates demo input assets so the project works out of the box:

* ``client/samples/sample_720p.mp4``  - 8 second 1280x720 synthetic test clip
* ``client/samples/sample_input.bin`` - deterministic binary blob for the
  synthetic (pipeline) task and for benchmarking transfer overhead.

Run from the repository root::

    python tools/make_sample_assets.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import config  # noqa: E402
from server.environment import detect  # noqa: E402

SAMPLE_DIR = os.path.join(ROOT, "client", "samples")


def make_video(path: str, seconds: int = 8) -> bool:
    """Build an MP4 with ffmpeg's built-in testsrc2 source."""
    env = detect()
    if not env.ffmpeg_path:
        print("[skip] ffmpeg not available - no sample video generated")
        return False
    cmd = [
        env.ffmpeg_path, "-hide_banner", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", path,
    ]
    import subprocess

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("[fail] sample video generation failed:")
        print(proc.stderr.decode("utf-8", "replace")[-1500:])
        return False
    print(f"[ok] {path} ({os.path.getsize(path)} bytes)")
    return True


def make_blob(path: str, size: int = 8 * 1024 * 1024) -> None:
    block = bytes(range(256))
    with open(path, "wb") as handle:
        written = 0
        while written < size:
            handle.write(block)
            written += len(block)
    print(f"[ok] {path} ({os.path.getsize(path)} bytes)")


def main() -> int:
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    make_video(os.path.join(SAMPLE_DIR, "sample_720p.mp4"))
    make_blob(os.path.join(SAMPLE_DIR, "sample_input.bin"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
