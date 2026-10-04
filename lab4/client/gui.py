"""
client/gui.py
-------------
Desktop client application (Task 3) built with CustomTkinter.

Layout
------
    +------------------------------------------------------------------+
    |  Worker Node   [ IP ] [ Port ]  [ Connect ]   RTT / GPU badge     |
    +------------------------------------------------------------------+
    |  Input asset   [ path........................ ] [Browse...]        |
    |  Task          [video ▾]  Resolution [1280x720 ▾]  Preset [p4 ▾]   |
    |  Bitrate       [4M   ▾]  Quality slider [====|======]  Encoder[auto]|
    |                        [  Start Remote Job  ] [Cancel]            |
    +------------------------------------------------------------------+
    |  Progress  [#################-----------]  62.5%   stage=video     |
    |  Status    Job 4f2a... running on DESKTOP-WORKER                   |
    +------------------------------------------------------------------+
    |  Live Log Terminal (scrollback)                                    |
    +------------------------------------------------------------------+

Threading model: the Tk main loop never blocks. Network work runs on a
worker thread; events are pushed into a ``queue.Queue`` and drained with
``root.after(...)`` so the UI stays responsive (and never freezes on a
slow network).
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

# Allow running as a script:  python client/gui.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common import config  # noqa: E402
from client.api import LogEvent, OffloadClient, OffloadError, ProgressEvent  # noqa: E402

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT = "#3b8edf"
OK = "#2fb344"
WARN = "#f0a020"
ERR = "#e05252"


class OffloadGUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("CSC-334 | Distributed Task Offloading & Remote GPU Rendering")
        self.geometry("1120x780")
        self.minsize(940, 680)

        self.client: OffloadClient | None = None
        self.ui_queue: "queue.Queue[tuple]" = queue.Queue()
        self.job_thread: threading.Thread | None = None
        self.cancel_flag = threading.Event()
        self.server_env: dict = {}
        self.connected = False

        self._build_ui()
        self.after(80, self._drain_ui_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Convenience: prefill the last used server from a tiny state file.
        self._load_state()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        # --- Row 0: connection ----------------------------------------- #
        conn = ctk.CTkFrame(self, corner_radius=10)
        conn.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 6))
        conn.grid_columnconfigure(7, weight=1)

        ctk.CTkLabel(conn, text="Worker Node", font=ctk.CTkFont(size=14, weight="bold")) \
            .grid(row=0, column=0, padx=(12, 8), pady=10)

        ctk.CTkLabel(conn, text="Server IP").grid(row=0, column=1, padx=(6, 2))
        self.ip_var = tk.StringVar(value=config.DEFAULT_HOST)
        ctk.CTkEntry(conn, textvariable=self.ip_var, width=150,
                     placeholder_text="192.168.1.1").grid(row=0, column=2, padx=4)

        ctk.CTkLabel(conn, text="Port").grid(row=0, column=3, padx=(10, 2))
        self.port_var = tk.StringVar(value=str(config.DEFAULT_PORT))
        ctk.CTkEntry(conn, textvariable=self.port_var, width=70).grid(row=0, column=4, padx=4)

        self.connect_btn = ctk.CTkButton(conn, text="Connect", width=110,
                                         command=self.on_connect)
        self.connect_btn.grid(row=0, column=5, padx=10)

        self.badge = ctk.CTkLabel(conn, text="● Offline", text_color=ERR,
                                  font=ctk.CTkFont(size=13, weight="bold"))
        self.badge.grid(row=0, column=6, padx=(10, 6), sticky="w")

        self.rtt_label = ctk.CTkLabel(conn, text="RTT: -", text_color="#9aa4b2")
        self.rtt_label.grid(row=0, column=7, padx=6, sticky="w")
        self.gpu_label = ctk.CTkLabel(conn, text="GPU: -", text_color="#9aa4b2")
        self.gpu_label.grid(row=0, column=8, padx=(6, 12), sticky="e")

        # --- Row 1: job configuration ----------------------------------- #
        job = ctk.CTkFrame(self, corner_radius=10)
        job.grid(row=1, column=0, sticky="ew", padx=14, pady=6)
        for col in range(10):
            job.grid_columnconfigure(col, weight=0)
        job.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(job, text="Input asset",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, padx=(12, 6), pady=(10, 4), sticky="w")

        self.file_var = tk.StringVar(value=os.path.join(config.DEFAULT_INPUT_DIR,
                                                        "sample_720p.mp4"))
        ctk.CTkEntry(job, textvariable=self.file_var).grid(
            row=0, column=1, columnspan=4, padx=4, pady=(10, 4), sticky="ew")
        ctk.CTkButton(job, text="Browse...", width=90, fg_color="#44556b",
                      command=self.on_browse).grid(row=0, column=5, padx=4, pady=(10, 4))

        ctk.CTkLabel(job, text="Task").grid(row=1, column=0, padx=(12, 6), pady=4, sticky="w")
        self.task_var = tk.StringVar(value="video")
        ctk.CTkOptionMenu(job, variable=self.task_var, width=140,
                          values=list(config.SUPPORTED_TASKS),
                          command=self._on_task_change).grid(
            row=1, column=1, padx=4, pady=4, sticky="w")

        ctk.CTkLabel(job, text="Resolution").grid(row=1, column=2, padx=(14, 6), pady=4, sticky="w")
        self.res_var = tk.StringVar(value="1280x720")
        ctk.CTkOptionMenu(job, variable=self.res_var, width=130,
                          values=list(config.RESOLUTIONS)).grid(
            row=1, column=3, padx=4, pady=4, sticky="w")

        ctk.CTkLabel(job, text="Preset").grid(row=1, column=4, padx=(14, 6), pady=4, sticky="w")
        self.preset_var = tk.StringVar(value="p4")
        self.preset_menu = ctk.CTkOptionMenu(job, variable=self.preset_var, width=120,
                                             values=list(config.FFMPEG_PRESETS))
        self.preset_menu.grid(row=1, column=5, padx=4, pady=4, sticky="w")

        ctk.CTkLabel(job, text="Bitrate").grid(row=2, column=0, padx=(12, 6), pady=4, sticky="w")
        self.bitrate_var = tk.StringVar(value="4M")
        ctk.CTkOptionMenu(job, variable=self.bitrate_var, width=110,
                          values=["1M", "2M", "4M", "6M", "8M", "12M", "16M"]).grid(
            row=2, column=1, padx=4, pady=4, sticky="w")

        ctk.CTkLabel(job, text="Encoder").grid(row=2, column=2, padx=(14, 6), pady=4, sticky="w")
        self.encoder_var = tk.StringVar(value="auto")
        ctk.CTkOptionMenu(job, variable=self.encoder_var, width=140,
                          values=["auto", "h264_nvenc", "hevc_nvenc", "h264_qsv",
                                  "h264_vaapi", "libx264"]).grid(
            row=2, column=3, padx=4, pady=4, sticky="w")

        ctk.CTkLabel(job, text="Quality").grid(row=2, column=4, padx=(14, 6), pady=4, sticky="w")
        self.quality_var = tk.IntVar(value=23)
        ctk.CTkSlider(job, from_=1, to=51, variable=self.quality_var, width=140,
                      number_of_steps=50).grid(row=2, column=5, padx=4, pady=4, sticky="w")

        # Extra params for tensor / synthetic tasks
        self.extra_label = ctk.CTkLabel(job, text="", text_color="#9aa4b2")
        self.extra_label.grid(row=3, column=0, columnspan=2, padx=(12, 6), pady=2, sticky="w")
        self.extra_entry = ctk.CTkEntry(job, width=220,
                                        placeholder_text="iterations=60, matrix_size=1024")
        self.extra_entry.grid(row=3, column=2, columnspan=3, padx=4, pady=2, sticky="w")

        # --- Row 2: actions --------------------------------------------- #
        actions = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 4))

        self.start_btn = ctk.CTkButton(
            master=actions,
            text="Start Remote Job", height=40, width=190,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled", command=self.on_start)
        self.start_btn.pack(side="left", padx=(0, 10))

        self.cancel_btn = ctk.CTkButton(
            master=actions,
            text="Cancel Job", height=40, width=120, fg_color="#7a3b3b",
            hover_color="#945050", state="disabled", command=self.on_cancel)
        self.cancel_btn.pack(side="left")

        self.open_out_btn = ctk.CTkButton(
            master=actions,
            text="Open Output Folder", height=40, width=160, fg_color="#44556b",
            hover_color="#546a84", command=self.on_open_output)
        self.open_out_btn.pack(side="left", padx=10)

        self.status_label = ctk.CTkLabel(
            master=actions, text="Not connected", text_color="#9aa4b2",
            font=ctk.CTkFont(size=13))
        self.status_label.pack(side="left", padx=14)

        # --- Row 3: progress -------------------------------------------- #
        prog = ctk.CTkFrame(self, corner_radius=10)
        prog.grid(row=3, column=0, sticky="ew", padx=14, pady=6)
        prog.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(prog, text="Progress",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, padx=(12, 8), pady=10, sticky="w")
        self.progress = ctk.CTkProgressBar(prog, height=18, progress_color=ACCENT)
        self.progress.grid(row=0, column=1, sticky="ew", padx=4, pady=10)
        self.progress.set(0)
        self.percent_label = ctk.CTkLabel(prog, text="0.00%", width=80,
                                          font=ctk.CTkFont(size=13, weight="bold"))
        self.percent_label.grid(row=0, column=2, padx=6)
        self.stage_label = ctk.CTkLabel(prog, text="stage: idle", text_color="#9aa4b2", width=170)
        self.stage_label.grid(row=0, column=3, padx=6)

        # --- Row 4: log terminal ---------------------------------------- #
        log_frame = ctk.CTkFrame(self, corner_radius=10)
        log_frame.grid(row=4, column=0, sticky="nsew", padx=14, pady=(6, 14))
        log_frame.grid_rowconfigure(1, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(log_frame, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 0))
        ctk.CTkLabel(head, text="Live Log Terminal",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        ctk.CTkButton(head, text="Clear", width=70, fg_color="#44556b",
                      command=lambda: self.log_box.delete("1.0", "end")).pack(side="right", padx=4)
        ctk.CTkButton(head, text="Save Log", width=90, fg_color="#44556b",
                      command=self.on_save_log).pack(side="right", padx=4)

        self.log_box = ctk.CTkTextbox(log_frame, font=ctk.CTkFont(family="Consolas", size=12),
                                      text_color="#d5dbe5", fg_color="#12161c")
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self._tag_colors()

    def _tag_colors(self) -> None:
        # CTkTextbox wraps tkinter Text; configure tag colours for levels.
        try:
            self.log_box.tag_config("error", foreground=ERR)
            self.log_box.tag_config("warning", foreground=WARN)
            self.log_box.tag_config("ok", foreground=OK)
            self.log_box.tag_config("dim", foreground="#7b8593")
        except tk.TclError:
            pass

    def _on_task_change(self, task: str) -> None:
        if task == "video":
            self.extra_label.configure(text="")
            self.extra_entry.configure(placeholder_text="",
                                       state="normal")
            self.quality_var.set(23)
        elif task == "tensor":
            self.extra_label.configure(text="Tensor params:")
            self.extra_entry.configure(
                placeholder_text="iterations=60, matrix_size=1024, device=auto",
                state="normal")
        else:
            self.extra_label.configure(text="Synthetic params:")
            self.extra_entry.configure(placeholder_text="work_units=200", state="normal")

    # ------------------------------------------------------------------ #
    # Logging helpers (thread-safe via ui_queue)
    # ------------------------------------------------------------------ #
    def log(self, message: str, level: str = "info") -> None:
        self.ui_queue.put(("log", (message, level)))

    def _append_log(self, message: str, level: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        tag = {"error": "error", "warning": "warning"}.get(level, None)
        line = f"[{stamp}] {message}\n"
        self.log_box.insert("end", line, tag or ())
        self.log_box.see("end")

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #
    def on_connect(self) -> None:
        if self.connected and self.client:
            self.client.close()
            self.client = None
            self.connected = False
            self.connect_btn.configure(text="Connect")
            self.badge.configure(text="● Offline", text_color=ERR)
            self.start_btn.configure(state="disabled")
            self.status_label.configure(text="Disconnected")
            self.log("Disconnected from worker", "warning")
            return

        host = self.ip_var.get().strip()
        try:
            port = int(self.port_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid port", "Port must be a number (e.g. 5000)")
            return

        self.connect_btn.configure(state="disabled", text="Connecting...")
        self.badge.configure(text="● Connecting...", text_color=WARN)
        self.log(f"Connecting to {host}:{port} ...")
        self._save_state()

        threading.Thread(target=self._connect_worker,
                         args=(host, port), daemon=True).start()

    def _connect_worker(self, host: str, port: int) -> None:
        client = OffloadClient(host, port,
                               on_log=lambda ev: self.log(ev.message, ev.level))
        try:
            info = client.connect()
            rtt = client.ping()
            env = info.get("environment", {})
            self.ui_queue.put(("connected", (client, info, rtt)))
        except OffloadError as exc:
            client.close()
            self.ui_queue.put(("connect_failed", str(exc)))

    # ------------------------------------------------------------------ #
    # Job execution
    # ------------------------------------------------------------------ #
    def _parse_extra_params(self) -> dict:
        raw = self.extra_entry.get().strip()
        params: dict = {}
        if raw:
            for part in raw.replace(";", ",").split(","):
                if "=" not in part:
                    continue
                key, _, value = part.partition("=")
                key, value = key.strip(), value.strip()
                try:
                    params[key] = int(value)
                except ValueError:
                    try:
                        params[key] = float(value)
                    except ValueError:
                        params[key] = value
        return params

    def on_start(self) -> None:
        if not self.connected or not self.client:
            messagebox.showwarning("Not connected", "Connect to a worker node first.")
            return

        path = self.file_var.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Missing input", f"Input asset not found:\n{path}")
            return

        task = self.task_var.get()
        params = self._parse_extra_params()
        if task == "video":
            params.update({
                "resolution": self.res_var.get(),
                "bitrate": self.bitrate_var.get(),
                "preset": self.preset_var.get(),
                "encoder": self.encoder_var.get(),
                "quality": int(self.quality_var.get()),
            })
        elif task == "synthetic" and "work_units" not in params:
            params["work_units"] = 0  # auto-size from the input file

        self.cancel_flag.clear()
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.set(0)
        self.percent_label.configure(text="0.00%")
        self.status_label.configure(text="Submitting job...")
        self.log(f"Submitting {task} job for {os.path.basename(path)}")

        self.job_thread = threading.Thread(
            target=self._job_worker, args=(task, path, params),
            daemon=True)
        self.job_thread.start()

    def _job_worker(self, task: str, path: str, params: dict) -> None:
        client = self.client
        if client is None:
            return
        try:
            def on_progress(ev: ProgressEvent) -> None:
                self.ui_queue.put(("progress", ev))

            def on_log(ev: LogEvent) -> None:
                self.ui_queue.put(("log", (ev.message, ev.level)))

            result = client.run_job(task, path, params=params,
                                    on_progress=on_progress, on_log=on_log,
                                    download_dir=config.CLIENT_DOWNLOAD_DIR)
            self.ui_queue.put(("done", result))
        except OffloadError as exc:
            self.ui_queue.put(("failed", str(exc)))
        except Exception as exc:  # noqa: BLE001
            self.ui_queue.put(("failed", f"{type(exc).__name__}: {exc}"))

    def on_cancel(self) -> None:
        self.log("Cancellation requested - closing session", "warning")
        self.cancel_btn.configure(state="disabled")
        if self.client:
            # Closing the socket makes the daemon mark the job cancelled.
            threading.Thread(target=self.client.close, daemon=True).start()

    # ------------------------------------------------------------------ #
    # UI event pump
    # ------------------------------------------------------------------ #
    def _drain_ui_queue(self) -> None:
        try:
            for _ in range(200):  # bounded so the UI never stalls
                kind, payload = self.ui_queue.get_nowait()
                self._handle_event(kind, payload)
        except queue.Empty:
            pass
        self.after(80, self._drain_ui_queue)

    def _handle_event(self, kind: str, payload) -> None:
        if kind == "log":
            message, level = payload
            self._append_log(message, level)
        elif kind == "connected":
            client, info, rtt = payload
            self.client = client
            self.connected = True
            env = info.get("environment", {})
            self.server_env = env
            self.connect_btn.configure(state="normal", text="Disconnect")
            self.badge.configure(text="● Online", text_color=OK)
            self.rtt_label.configure(
                text=f"RTT: {rtt['avg_ms']:.1f} ms")
            gpus = env.get("gpus") or []
            self.gpu_label.configure(
                text=f"GPU: {gpus[0].get('name')}" if gpus else "GPU: none (CPU mode)")
            self.start_btn.configure(state="normal")
            self.status_label.configure(
                text=f"{env.get('hostname', '')} | session {info.get('session_id')} | "
                     f"{env.get('engine_summary', '')}")
            self.log(f"Connected. {env.get('engine_summary', '')}", "ok")
            if not env.get("ffmpeg_path") and self.task_var.get() == "video":
                self.log("FFmpeg not found on worker - video jobs will fail there", "warning")
        elif kind == "connect_failed":
            self.connect_btn.configure(state="normal", text="Connect")
            self.badge.configure(text="● Offline", text_color=ERR)
            self.log(f"Connection failed: {payload}", "error")
            messagebox.showerror("Connection failed", str(payload))
        elif kind == "progress":
            ev: ProgressEvent = payload
            self.progress.set(max(0.0, min(1.0, ev.percent / 100.0)))
            self.percent_label.configure(text=f"{ev.percent:6.2f}%")
            engine = ev.raw.get("engine")
            stage = ev.stage + (f" [{engine}]" if engine else "")
            self.stage_label.configure(text=f"stage: {stage}")
            if ev.percent >= 100 and ev.stage == "done":
                self.status_label.configure(text="Transferring verified output...")
            elif ev.stage not in ("done", "idle"):
                engine = ev.raw.get("engine")
                suffix = f" ({engine})" if engine else ""
                self.status_label.configure(
                    text=f"Job running - {ev.percent:.1f}%{suffix}")
        elif kind == "done":
            result = payload
            self.progress.set(1.0)
            self.percent_label.configure(text="100.00%")
            self.stage_label.configure(text="stage: done")
            self.status_label.configure(
                text=f"Job {result.job_id} complete in {result.duration_s:.2f}s")
            self.cancel_btn.configure(state="disabled")
            self.start_btn.configure(state="normal")
            engine = result.result.get("engine", "?")
            hw = result.result.get("hardware_accelerated")
            self.log(f"Job {result.job_id} finished in {result.duration_s:.2f}s "
                     f"using {engine} ({'GPU' if hw else 'CPU'})", "ok")
            self.log(f"Output verified: {result.output_path} "
                     f"({result.output_bytes} bytes)", "ok")
        elif kind == "failed":
            self.progress.set(0)
            self.percent_label.configure(text="0.00%")
            self.stage_label.configure(text="stage: failed")
            self.status_label.configure(text="Job failed")
            self.cancel_btn.configure(state="disabled")
            self.start_btn.configure(state="normal")
            self.log(f"Job failed: {payload}", "error")
        elif kind == "disconnected":
            self.connected = False
            self.start_btn.configure(state="disabled")
            self.connect_btn.configure(text="Connect")
            self.badge.configure(text="● Offline", text_color=ERR)
            self.log(f"Connection lost: {payload}", "error")

    # ------------------------------------------------------------------ #
    # Misc actions
    # ------------------------------------------------------------------ #
    def on_browse(self) -> None:
        initial = config.DEFAULT_INPUT_DIR if os.path.isdir(config.DEFAULT_INPUT_DIR) else ROOT
        path = filedialog.askopenfilename(
            title="Select input asset",
            initialdir=initial,
            filetypes=[
                ("Media & assets", "*.mp4 *.mkv *.mov *.avi *.webm *.png *.jpg *.bin"),
                ("All files", "*.*"),
            ])
        if path:
            self.file_var.set(path)
            self.log(f"Selected input: {path}")

    def on_open_output(self) -> None:
        os.makedirs(config.CLIENT_DOWNLOAD_DIR, exist_ok=True)
        path = config.CLIENT_DOWNLOAD_DIR
        try:
            if os.name == "nt":
                os.startfile(path)  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except OSError as exc:
            messagebox.showerror("Cannot open folder", str(exc))

    def on_save_log(self) -> None:
        content = self.log_box.get("1.0", "end")
        if not content.strip():
            return
        path = filedialog.asksaveasfilename(
            title="Save log", defaultextension=".txt",
            initialfile=f"offload_log_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            filetypes=[("Text", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
            self.log(f"Log saved to {path}", "ok")

    def _save_state(self) -> None:
        try:
            with open(os.path.join(ROOT, ".client_state"), "w", encoding="utf-8") as h:
                h.write(f"{self.ip_var.get().strip()}\n{self.port_var.get().strip()}\n")
        except OSError:
            pass

    def _load_state(self) -> None:
        try:
            with open(os.path.join(ROOT, ".client_state"), "r", encoding="utf-8") as h:
                host, port = h.read().splitlines()[:2]
                if host.strip():
                    self.ip_var.set(host.strip())
                if port.strip().isdigit():
                    self.port_var.set(port.strip())
        except (OSError, ValueError):
            pass

    def _on_close(self) -> None:
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
        self.destroy()


def main() -> int:
    app = OffloadGUI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
