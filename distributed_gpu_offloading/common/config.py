"""
Shared configuration defaults for client and server daemon.
"""

from __future__ import annotations

import os

# --- Network --------------------------------------------------------------- #
DEFAULT_HOST = "192.168.1.1"          # static peer-to-peer address of the worker node
DEFAULT_PORT = 5000
LOOPBACK_HOST = "127.0.0.1"

CONNECT_TIMEOUT = 5.0                 # seconds to establish TCP connection
HANDSHAKE_TIMEOUT = 5.0               # seconds to complete HELLO/HELLO_ACK
IO_TIMEOUT = 30.0                     # socket read timeout during bulk transfer
IDLE_TIMEOUT = 600.0                  # drop a connection silent for this long
HEARTBEAT_INTERVAL = 2.0              # server keep-alive cadence while queued/running
PING_PAYLOAD_COUNT = 5                # latency samples taken during handshake
MAX_SUBMIT_ATTEMPTS = 3               # client retry budget for flaky links

# --- Transfer -------------------------------------------------------------- #
CHUNK_SIZE = 1024 * 1024              # 1 MiB
MAX_INPUT_BYTES = 8 * 1024 * 1024 * 1024   # 8 GiB guard rail
MAX_QUEUE_DEPTH = 64

# --- Paths ------------------------------------------------------------------ #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_WORK_DIR = os.path.join(PROJECT_ROOT, "server", "workdir")
CLIENT_DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "client", "downloads")
DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT, "client", "samples")

# --- Engine selection ------------------------------------------------------- #
SUPPORTED_TASKS = ("video", "tensor", "synthetic")

VIDEO_ENCODER_PRIORITY = (
    ("h264_nvenc", "NVIDIA NVENC H.264 (GPU)"),
    ("hevc_nvenc", "NVIDIA NVENC H.265 (GPU)"),
    ("h264_qsv", "Intel Quick Sync (GPU)"),
    ("h264_vaapi", "VA-API (Linux GPU)"),
    ("libx264", "libx264 (CPU fallback)"),
)

FFMPEG_PRESETS = (
    "p1", "p2", "p3", "p4", "p5",      # NVENC quality presets
    "veryfast", "faster", "fast", "medium",  # libx264 presets
)

RESOLUTIONS = (
    "1920x1080",
    "1280x720",
    "854x480",
    "640x360",
    "source",
)
