"""
tools/capture_gui.py
--------------------
Grabs screenshots of the desktop GUI for the README / submission evidence.

``PIL.ImageGrab`` returns a black frame on locked or headless sessions, so this
uses the Win32 ``PrintWindow`` API with ``PW_RENDERFULLCONTENT`` to read the
window's own back-buffer instead of the screen.

It drives the real application end to end: connect -> latency probe ->
submit a job -> wait for progress -> capture the finished state.

    python tools/capture_gui.py --host 127.0.0.1 --port 5050
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import threading
import time
from ctypes import wintypes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import config  # noqa: E402

PW_RENDERFULLCONTENT = 3
DIB_RGB_COLORS = 0


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


def capture_window(hwnd: int):
    """Return a PIL image of the window's client rendering, or None."""
    from PIL import Image

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None

    hdc = user32.GetWindowDC(hwnd)
    mem_dc = gdi32.CreateCompatibleDC(hdc)
    bitmap = gdi32.CreateCompatibleBitmap(hdc, width, height)
    gdi32.SelectObject(mem_dc, bitmap)

    # PW_RENDERFULLCONTENT is Windows 8.1+; fall back to plain PrintWindow.
    ok = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)
    if not ok:
        ok = user32.PrintWindow(hwnd, mem_dc, 0)
    if not ok:
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, hdc)
        return None

    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = width
    bmi.biHeight = -height  # top-down
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0

    buf = ctypes.create_string_buffer(width * height * 4)
    gdi32.GetDIBits(mem_dc, bitmap, 0, height, buf, ctypes.byref(bmi), DIB_RGB_COLORS)

    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(mem_dc)
    user32.ReleaseDC(hwnd, hdc)

    img = Image.frombuffer("RGBA", (width, height), buf, "raw", "BGRA", 0, 1)
    return img.convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=config.DEFAULT_PORT)
    parser.add_argument("--out", default=os.path.join(ROOT, "screenshots"))
    args = parser.parse_args()

    import client.gui as g  # deferred: needs a display

    os.makedirs(args.out, exist_ok=True)
    app = g.OffloadGUI()
    app.ip_var.set(args.host)
    app.port_var.set(str(args.port))
    sample = os.path.join(config.DEFAULT_INPUT_DIR, "sample_720p.mp4")
    if os.path.exists(sample):
        app.file_var.set(sample)

    shots: list = []
    errors: list = []
    done = threading.Event()

    def pump(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            app.update()
            time.sleep(0.05)

    def shot(name: str, timeout: float = 10.0) -> None:
        """Capture on the Tk thread (PrintWindow sends window messages, so
        calling it from a worker thread can deadlock the event loop)."""
        result: dict = {}
        finished = threading.Event()

        def _capture() -> None:
            try:
                app.update_idletasks()
                frame = str(app.wm_frame() or "")
                hwnd = int(frame, 16) if frame.startswith("0x") else int(frame or 0, 0)
                if not hwnd:
                    hwnd = int(app.winfo_id())
                img = capture_window(hwnd)
                if img is None:
                    raise RuntimeError("PrintWindow returned nothing")
                path = os.path.join(args.out, name)
                img.save(path)
                result["path"] = path
            except Exception as exc:  # noqa: BLE001
                result["error"] = repr(exc)
            finally:
                finished.set()

        # flow() runs on the Tk thread, so execute the capture inline.
        _capture()
        if not finished.wait(timeout):
            errors.append(f"{name}: capture timed out")
            return
        if "error" in result:
            errors.append(f"{name}: {result['error']}")
        else:
            shots.append(result["path"])

    def flow() -> None:
        try:
            app.on_connect()
            for _ in range(120):
                pump(0.25)
                if app.connected:
                    break
            shot("01_gui_connected.png")

            app.on_start()
            # capture once while progress is mid-flight (time-boxed)
            seen_partial = False
            deadline = time.time() + 120
            while time.time() < deadline:
                pump(0.2)
                pct = app.percent_label.cget("text")
                if not seen_partial and "%" in pct:
                    try:
                        value = float(pct.replace("%", "").strip())
                    except ValueError:
                        value = 0.0
                    if 5.0 <= value < 99.0:
                        shot("02_gui_progress.png")
                        seen_partial = True
                if app.start_btn.cget("state") == "normal" and seen_partial:
                    break
            pump(1.0)
            shot("03_gui_job_done.png")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"flow: {exc!r}")
        finally:
            done.set()

    # flow() runs on the Tk thread: pump() drives the event loop, and
    # PrintWindow/after callbacks therefore never deadlock.
    flow()
    app.destroy()
    done.wait(5)

    print("captured:")
    for path in shots:
        print("  ", path)
    if errors:
        print("errors:")
        for err in errors:
            print("  ", err)
    return 0 if shots and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
