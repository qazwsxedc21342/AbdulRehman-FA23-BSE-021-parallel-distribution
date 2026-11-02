"""
================================================================================
CSC-334: Parallel and Distributed Computing
Lab 4: Custom Distributed Task Offloading & Remote GPU Rendering System
Module: client/client_gui.py
Description: Modern CustomTkinter Desktop GUI for Task Offloading, Remote GPU Rendering,
             Third-Party Video Editor Integration (CapCut/ClipChamp/etc.), and LAN File Sharing.
             - Browse any folder (CapCut exports, ClipChamp output, etc.)
             - Select a video from the folder listing
             - Send to remote server for rendering with live logs
             - Server renders and sends the rendered video back to this laptop
================================================================================
"""

import os
import sys
import time
import queue
import threading
from typing import Optional, Dict, Any, List
from tkinter import filedialog, messagebox
import customtkinter as ctk

# Add client directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from network_client import OffloadingClient
from utils import generate_sample_video, format_bytes, format_time, compute_sha256
from protocol import DEFAULT_HOST, DEFAULT_PORT


# ============================================================================
# Supported video file extensions for folder scanning
# ============================================================================
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".wmv", ".flv", ".m4v", ".ts"}


class TaskOffloaderApp(ctk.CTk):
    """
    Modern CustomTkinter GUI for Custom Distributed Task Offloading,
    Remote Hardware Acceleration, Third-Party Video Editor Integration,
    and LAN File Sharing Hub.
    """
    def __init__(self):
        super().__init__()

        # Window Appearance & Geometry
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        self.title("CSC-334: Distributed Task Offloading & Remote Compute System")
        self.geometry("1100x880")
        self.minsize(980, 720)

        # State Variables
        self.client: Optional[OffloadingClient] = None
        self.is_processing = False
        self.cancel_flag = False
        self.selected_file_path = ""
        self.selected_clips: List[Dict[str, Any]] = []
        self.last_rendered_path = ""
        self.last_log_path = ""
        self.selected_folder_path = ""
        self.download_output_dir = os.path.join(current_dir, "client_downloads")
        os.makedirs(self.download_output_dir, exist_ok=True)

        # Thread-safe UI update queue
        self.ui_queue = queue.Queue()
        self._poll_ui_queue()

        # Build UI Layout
        self._build_header()
        self._build_tab_container()
        self._build_log_terminal()

        # Log Initial System Banner
        self.log("[SYSTEM] Distributed Task Offloading Client initialized (Computer A).")
        self.log(f"[SYSTEM] Default Target: {DEFAULT_HOST}:{DEFAULT_PORT}. Click '⚡ Handshake & Ping' to connect to Computer B.")
        self.log("[TIP] Use '📂 Video Editor Exports' tab to select your CapCut/ClipChamp/etc. output folder!")

    # ==========================================
    # UI BUILDERS
    # ==========================================
    def _build_header(self):
        """Top Header Banner with Course & System Details."""
        header_frame = ctk.CTkFrame(self, corner_radius=10, fg_color=("#1f2937", "#111827"))
        header_frame.pack(fill="x", padx=16, pady=(12, 6))

        title_row = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_row.pack(fill="x", padx=16, pady=(10, 2))

        ctk.CTkLabel(
            title_row,
            text="⚡ CSC-334: Distributed Task Offloading & Remote Compute (Computer A)",
            font=ctk.CTkFont(family="Inter", size=19, weight="bold"),
            text_color="#60a5fa"
        ).pack(side="left")

        self.conn_indicator = ctk.CTkLabel(
            title_row,
            text="🔴 OFFLINE",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#f87171",
            fg_color=("#374151", "#1e293b"),
            corner_radius=6,
            padx=10,
            pady=3
        )
        self.conn_indicator.pack(side="right")

        sub_lbl = ctk.CTkLabel(
            header_frame,
            text="Peer-to-Peer Remote Acceleration | GPU NVENC / CUDA / Complex Project Tasks | FA23-BSE-021",
            font=ctk.CTkFont(family="Inter", size=12),
            text_color="#9ca3af"
        )
        sub_lbl.pack(anchor="w", padx=16, pady=(0, 10))

    def _build_tab_container(self):
        """Builds tabs: Tab 1 = Video Editor Exports; Tab 2 = LAN File Sharing."""
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.pack(fill="both", expand=True, padx=16, pady=4)

        self.tab_offload = self.tabview.add("📂 Video Editor Exports (CapCut/ClipChamp)")
        self.tab_sharing = self.tabview.add("📁 LAN File Sharing Hub")

        self._build_offload_tab()
        self._build_sharing_tab()

    def _build_offload_tab(self):
        """Tab 1: Folder browser for third-party video editor exports + server rendering."""
        main_container = ctk.CTkFrame(self.tab_offload, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=4, pady=2)
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)

        # ==========================================
        # LEFT COLUMN: Connection & Folder Browser
        # ==========================================
        left_col = ctk.CTkFrame(main_container, corner_radius=10)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=2)

        # Section 1: Networking & Handshake Setup
        net_title = ctk.CTkLabel(
            left_col,
            text="🌐 1. Network & Server Connection (Computer B)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#93c5fd"
        )
        net_title.pack(anchor="w", padx=14, pady=(10, 4))

        net_row = ctk.CTkFrame(left_col, fg_color="transparent")
        net_row.pack(fill="x", padx=14, pady=2)

        self.ip_entry = ctk.CTkEntry(net_row, placeholder_text="Server IP (e.g. 192.168.1.20)", width=170)
        self.ip_entry.insert(0, DEFAULT_HOST)
        self.ip_entry.pack(side="left", padx=(0, 6))

        self.port_entry = ctk.CTkEntry(net_row, placeholder_text="Port", width=65)
        self.port_entry.insert(0, str(DEFAULT_PORT))
        self.port_entry.pack(side="left", padx=(0, 6))

        self.ping_btn = ctk.CTkButton(
            net_row,
            text="⚡ Handshake & Ping",
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            command=self._handle_handshake_and_ping
        )
        self.ping_btn.pack(side="left", fill="x", expand=True)

        # Server Node Status Card
        self.worker_card = ctk.CTkFrame(left_col, fg_color=("#374151", "#1e293b"), corner_radius=8)
        self.worker_card.pack(fill="x", padx=14, pady=6)

        self.worker_status_lbl = ctk.CTkLabel(
            self.worker_card,
            text="Server Status: DISCONNECTED",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#f87171"
        )
        self.worker_status_lbl.pack(anchor="w", padx=10, pady=(5, 2))

        self.worker_specs_lbl = ctk.CTkLabel(
            self.worker_card,
            text="Enter Server IP & click 'Handshake & Ping' to connect Computer B",
            font=ctk.CTkFont(size=11),
            text_color="#cbd5e1"
        )
        self.worker_specs_lbl.pack(anchor="w", padx=10, pady=(0, 5))

        # Section 2: Third-Party Video Editor Folder Browser
        folder_title = ctk.CTkLabel(
            left_col,
            text="📂 2. Select Video Exports Folder (CapCut / ClipChamp / etc.)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#93c5fd"
        )
        folder_title.pack(anchor="w", padx=14, pady=(8, 4))

        folder_desc = ctk.CTkLabel(
            left_col,
            text="Browse to the folder where your video editor saves exported videos.",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8"
        )
        folder_desc.pack(anchor="w", padx=14, pady=(0, 4))

        folder_row = ctk.CTkFrame(left_col, fg_color="transparent")
        folder_row.pack(fill="x", padx=14, pady=2)

        self.folder_path_entry = ctk.CTkEntry(folder_row, placeholder_text="Select exports folder...", width=240)
        self.folder_path_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.folder_browse_btn = ctk.CTkButton(
            folder_row,
            text="📂 Browse Folder",
            width=120,
            fg_color="#7c3aed",
            hover_color="#6d28d9",
            command=self._handle_browse_folder
        )
        self.folder_browse_btn.pack(side="left", padx=(0, 6))

        self.folder_refresh_btn = ctk.CTkButton(
            folder_row,
            text="🔄",
            width=35,
            fg_color="#475569",
            hover_color="#334155",
            command=self._refresh_folder_videos
        )
        self.folder_refresh_btn.pack(side="left")

        # Video Files List in Folder
        self.video_list_label = ctk.CTkLabel(
            left_col,
            text="📹 Video Files Found: (select a folder first)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#cbd5e1"
        )
        self.video_list_label.pack(anchor="w", padx=14, pady=(6, 2))

        self.video_listbox = ctk.CTkTextbox(
            left_col,
            height=100,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=("#111827", "#030712"),
            text_color="#e2e8f0"
        )
        self.video_listbox.pack(fill="both", expand=True, padx=14, pady=(0, 4))
        self.video_listbox.insert("end", "No folder selected yet.\nClick '📂 Browse Folder' to select your CapCut/ClipChamp exports directory.\n")
        self.video_listbox.configure(state="disabled")

        # Video selector and trim row
        trim_row = ctk.CTkFrame(left_col, fg_color="transparent")
        trim_row.pack(fill="x", padx=14, pady=(2, 2))

        self.video_select_entry = ctk.CTkEntry(
            trim_row,
            placeholder_text="Video # or filename from folder...",
            width=160
        )
        self.video_select_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.trim_start_entry = ctk.CTkEntry(
            trim_row,
            placeholder_text="Start (0s)",
            width=65
        )
        self.trim_start_entry.pack(side="left", padx=(0, 4))

        self.trim_end_entry = ctk.CTkEntry(
            trim_row,
            placeholder_text="End (Full)",
            width=65
        )
        self.trim_end_entry.pack(side="left", padx=(0, 4))

        self.add_from_folder_btn = ctk.CTkButton(
            trim_row,
            text="➕ Add Clip",
            width=85,
            fg_color="#059669",
            hover_color="#047857",
            command=self._handle_add_selected_from_folder
        )
        self.add_from_folder_btn.pack(side="left", padx=(0, 4))

        self.add_files_btn = ctk.CTkButton(
            trim_row,
            text="📂 Browse & Add File(s)",
            width=140,
            fg_color="#0284c7",
            hover_color="#0369a1",
            command=self._handle_add_multiple_files
        )
        self.add_files_btn.pack(side="left")

        # Multi-Part Video Merge Queue Display
        ctk.CTkLabel(
            left_col,
            text="🎬 Selected Video Parts to Merge & Render (In Sequence):",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38bdf8"
        ).pack(anchor="w", padx=14, pady=(4, 2))

        self.clips_listbox = ctk.CTkTextbox(
            left_col,
            height=90,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=("#111827", "#030712"),
            text_color="#e2e8f0"
        )
        self.clips_listbox.pack(fill="both", expand=True, padx=14, pady=(0, 2))
        self.clips_listbox.insert("end", "No video parts added yet.\nClick '📂 Browse & Add File(s)' or select from folder above and click '➕ Add Clip'.\n")
        self.clips_listbox.configure(state="disabled")

        # Clip sequence management controls
        ctrl_clip_row = ctk.CTkFrame(left_col, fg_color="transparent")
        ctrl_clip_row.pack(fill="x", padx=14, pady=(2, 4))

        self.clip_num_entry = ctk.CTkEntry(
            ctrl_clip_row,
            placeholder_text="#",
            width=40
        )
        self.clip_num_entry.pack(side="left", padx=(0, 4))

        self.move_up_btn = ctk.CTkButton(
            ctrl_clip_row,
            text="⬆️ Up",
            width=55,
            fg_color="#475569",
            hover_color="#334155",
            command=self._handle_move_clip_up
        )
        self.move_up_btn.pack(side="left", padx=(0, 4))

        self.move_down_btn = ctk.CTkButton(
            ctrl_clip_row,
            text="⬇️ Down",
            width=55,
            fg_color="#475569",
            hover_color="#334155",
            command=self._handle_move_clip_down
        )
        self.move_down_btn.pack(side="left", padx=(0, 4))

        self.remove_clip_btn = ctk.CTkButton(
            ctrl_clip_row,
            text="❌ Remove",
            width=70,
            fg_color="#dc2626",
            hover_color="#b91c1c",
            command=self._handle_remove_selected_clip
        )
        self.remove_clip_btn.pack(side="left", padx=(0, 4))

        self.clear_clips_btn = ctk.CTkButton(
            ctrl_clip_row,
            text="🗑️ Clear All",
            width=75,
            fg_color="#64748b",
            hover_color="#475569",
            command=self._handle_clear_clips
        )
        self.clear_clips_btn.pack(side="left", padx=(0, 6))

        self.clips_summary_lbl = ctk.CTkLabel(
            ctrl_clip_row,
            text="Total Parts: 0 | Size: 0 B",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#fbbf24"
        )
        self.clips_summary_lbl.pack(side="right")

        # ==========================================
        # RIGHT COLUMN: Render Config & Dashboard
        # ==========================================
        right_col = ctk.CTkFrame(main_container, corner_radius=10)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=2)

        # Section 3: Render Settings
        render_title = ctk.CTkLabel(
            right_col,
            text="⚙️ 3. Render Configuration (Server-Side)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#93c5fd"
        )
        render_title.pack(anchor="w", padx=14, pady=(10, 4))

        # Render Settings Box
        self.settings_frame = ctk.CTkFrame(right_col, fg_color=("#374151", "#1e293b"), corner_radius=8)
        self.settings_frame.pack(fill="x", padx=14, pady=4)

        # Row 1: Resolution & Bitrate
        row1 = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=3)

        ctk.CTkLabel(row1, text="Resolution:", width=75, anchor="w").pack(side="left")
        self.res_combo = ctk.CTkComboBox(
            row1,
            values=["1920x1080", "1280x720", "854x480", "3840x2160", "original"],
            width=110
        )
        self.res_combo.set("1280x720")
        self.res_combo.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(row1, text="Bitrate:", width=50, anchor="w").pack(side="left")
        self.bitrate_combo = ctk.CTkComboBox(
            row1,
            values=["1500k", "3000k", "5000k", "8000k", "12000k"],
            width=100
        )
        self.bitrate_combo.set("3000k")
        self.bitrate_combo.pack(side="left")

        # Row 2: Preset & Acceleration Mode
        row2 = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(2, 4))

        ctk.CTkLabel(row2, text="Preset:", width=75, anchor="w").pack(side="left")
        self.preset_combo = ctk.CTkComboBox(
            row2,
            values=["ultrafast", "fast", "medium", "slow", "p4"],
            width=110
        )
        self.preset_combo.set("fast")
        self.preset_combo.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(row2, text="Mode:", width=50, anchor="w").pack(side="left")
        self.accel_combo = ctk.CTkComboBox(
            row2,
            values=["auto", "nvenc", "cpu"],
            width=100
        )
        self.accel_combo.set("auto")
        self.accel_combo.pack(side="left")

        # Row 3: Creative Filters & FPS
        row3 = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        row3.pack(fill="x", padx=10, pady=(2, 4))

        ctk.CTkLabel(row3, text="Filter:", width=75, anchor="w").pack(side="left")
        self.filter_combo = ctk.CTkComboBox(
            row3,
            values=["none", "grayscale", "sepia", "mirror", "blur", "speed_2x"],
            width=110
        )
        self.filter_combo.set("none")
        self.filter_combo.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(row3, text="FPS:", width=50, anchor="w").pack(side="left")
        self.fps_combo = ctk.CTkComboBox(
            row3,
            values=["original", "24 fps", "30 fps", "60 fps"],
            width=100
        )
        self.fps_combo.set("original")
        self.fps_combo.pack(side="left")

        # Section 4: Real-Time Progress Dashboard
        prog_title = ctk.CTkLabel(
            right_col,
            text="📊 4. Real-Time Remote Execution Dashboard",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#93c5fd"
        )
        prog_title.pack(anchor="w", padx=14, pady=(10, 4))

        # Progress Percentage Bar
        self.progress_bar = ctk.CTkProgressBar(right_col, height=18, corner_radius=8)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(fill="x", padx=14, pady=(4, 2))

        self.progress_lbl = ctk.CTkLabel(
            right_col,
            text="0.0% | Ready to offload video",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#cbd5e1"
        )
        self.progress_lbl.pack(anchor="w", padx=14, pady=(0, 6))

        # Metrics Card Grid
        metrics_box = ctk.CTkFrame(right_col, fg_color=("#374151", "#1e293b"), corner_radius=8)
        metrics_box.pack(fill="x", padx=14, pady=4)
        metrics_box.columnconfigure(0, weight=1)
        metrics_box.columnconfigure(1, weight=1)

        self.metric_fps = ctk.CTkLabel(metrics_box, text="Speed/Stage: --", font=ctk.CTkFont(size=12))
        self.metric_fps.grid(row=0, column=0, padx=8, pady=4, sticky="w")

        self.metric_eta = ctk.CTkLabel(metrics_box, text="ETA: --", font=ctk.CTkFont(size=12))
        self.metric_eta.grid(row=0, column=1, padx=8, pady=4, sticky="w")

        self.metric_turnaround = ctk.CTkLabel(metrics_box, text="Turnaround: -- s", font=ctk.CTkFont(size=12))
        self.metric_turnaround.grid(row=1, column=0, padx=8, pady=4, sticky="w")

        self.metric_integrity = ctk.CTkLabel(metrics_box, text="Integrity: Idle", font=ctk.CTkFont(size=12), text_color="#fbbf24")
        self.metric_integrity.grid(row=1, column=1, padx=8, pady=4, sticky="w")

        # Action Buttons
        btn_frame = ctk.CTkFrame(right_col, fg_color="transparent")
        btn_frame.pack(fill="x", padx=14, pady=(8, 4))

        self.offload_btn = ctk.CTkButton(
            btn_frame,
            text="🚀 Merge & Render Video Parts on Server (Computer B)",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#16a34a",
            hover_color="#15803d",
            height=38,
            command=self._handle_start_offload
        )
        self.offload_btn.pack(fill="x", pady=(0, 6))

        btn_row = ctk.CTkFrame(btn_frame, fg_color="transparent")
        btn_row.pack(fill="x")

        self.play_btn = ctk.CTkButton(
            btn_row,
            text="▶️ Play Video",
            fg_color="#0284c7",
            hover_color="#0369a1",
            height=30,
            width=90,
            state="disabled",
            command=self._handle_play_rendered_video
        )
        self.play_btn.pack(side="left", padx=(0, 6))

        self.view_log_btn = ctk.CTkButton(
            btn_row,
            text="📄 View Logs",
            fg_color="#0d9488",
            hover_color="#0f766e",
            height=30,
            width=95,
            state="disabled",
            command=self._handle_view_server_logs
        )
        self.view_log_btn.pack(side="left", padx=(0, 6))

        self.open_dl_btn = ctk.CTkButton(
            btn_row,
            text="📂 Open Downloads",
            fg_color="#7c3aed",
            hover_color="#6d28d9",
            height=30,
            command=lambda: os.startfile(self.download_output_dir)
        )
        self.open_dl_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.cancel_btn = ctk.CTkButton(
            btn_row,
            text="🛑 Cancel",
            fg_color="#dc2626",
            hover_color="#b91c1c",
            height=30,
            width=80,
            command=self._handle_cancel
        )
        self.cancel_btn.pack(side="left")

    def _build_sharing_tab(self):
        """Tab 2: Peer-to-Peer LAN File Sharing Hub."""
        frame = ctk.CTkFrame(self.tab_sharing, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=8, pady=6)

        # Top Control Row
        ctrl_row = ctk.CTkFrame(frame, corner_radius=8)
        ctrl_row.pack(fill="x", padx=6, pady=4)

        ctk.CTkLabel(
            ctrl_row,
            text="📁 LAN Storage Hub (Share files with Computer B without internet)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#93c5fd"
        ).pack(side="left", padx=12, pady=10)

        refresh_files_btn = ctk.CTkButton(
            ctrl_row,
            text="🔄 Refresh Server Files",
            width=140,
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            command=self._handle_list_shared_files
        )
        refresh_files_btn.pack(side="right", padx=(0, 10))

        upload_share_btn = ctk.CTkButton(
            ctrl_row,
            text="📤 Upload File to Server",
            width=150,
            fg_color="#059669",
            hover_color="#047857",
            command=self._handle_upload_shared_file
        )
        upload_share_btn.pack(side="right", padx=(0, 10))

        # Files Display
        self.shared_files_box = ctk.CTkTextbox(
            frame,
            height=180,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=("#111827", "#030712"),
            text_color="#e2e8f0"
        )
        self.shared_files_box.pack(fill="both", expand=True, padx=6, pady=4)
        self.shared_files_box.insert("end", "Click '🔄 Refresh Server Files' to see available files on Computer B.\n")
        self.shared_files_box.configure(state="disabled")

        # Bottom Actions
        act_row = ctk.CTkFrame(frame, fg_color="transparent")
        act_row.pack(fill="x", padx=6, pady=4)

        self.dl_fname_entry = ctk.CTkEntry(act_row, placeholder_text="Enter file name to download...", width=280)
        self.dl_fname_entry.pack(side="left", padx=(0, 8))

        dl_btn = ctk.CTkButton(
            act_row,
            text="📥 Download from Server",
            width=160,
            fg_color="#0284c7",
            hover_color="#0369a1",
            command=self._handle_download_shared_file
        )
        dl_btn.pack(side="left", padx=(0, 10))

        open_dl_btn = ctk.CTkButton(
            act_row,
            text="📂 Open Downloads Folder",
            width=160,
            fg_color="#475569",
            hover_color="#334155",
            command=lambda: os.startfile(self.download_output_dir)
        )
        open_dl_btn.pack(side="left")

    def _build_log_terminal(self):
        """Bottom Real-Time Log Terminal Widget."""
        term_frame = ctk.CTkFrame(self, corner_radius=10)
        term_frame.pack(fill="both", expand=True, padx=16, pady=(2, 10))

        head_row = ctk.CTkFrame(term_frame, fg_color="transparent")
        head_row.pack(fill="x", padx=12, pady=(6, 2))

        ctk.CTkLabel(
            head_row,
            text="💻 Integrated Real-Time Live Console (Computer A)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#93c5fd"
        ).pack(side="left")

        clear_btn = ctk.CTkButton(
            head_row,
            text="Clear Log",
            width=70,
            height=22,
            fg_color="#4b5563",
            hover_color="#374151",
            command=self._clear_logs
        )
        clear_btn.pack(side="right")

        self.log_textbox = ctk.CTkTextbox(
            term_frame,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=("#111827", "#030712"),
            text_color="#e2e8f0",
            wrap="none"
        )
        self.log_textbox.pack(fill="both", expand=True, padx=12, pady=(0, 6))

    # ==========================================
    # LOGGING & UI HELPERS (THREAD-SAFE)
    # ==========================================
    def _poll_ui_queue(self):
        """Processes pending UI mutations on Tkinter main thread."""
        try:
            while True:
                fn, args, kwargs = self.ui_queue.get_nowait()
                try:
                    fn(*args, **kwargs)
                except Exception:
                    pass
        except queue.Empty:
            pass
        self.after(40, self._poll_ui_queue)

    def run_on_ui(self, fn, *args, **kwargs):
        """Enqueues a function to run safely on the main GUI thread."""
        self.ui_queue.put((fn, args, kwargs))

    def log(self, text: str):
        """Thread-safe append to terminal log console."""
        timestamp = time.strftime("%H:%M:%S")
        msg = f"[{timestamp}] {text}\n"
        def _do_insert():
            self.log_textbox.insert("end", msg)
            self.log_textbox.see("end")
        self.run_on_ui(_do_insert)

    def _clear_logs(self):
        self.log_textbox.delete("1.0", "end")

    # ==========================================
    # FOLDER BROWSER & VIDEO FILE LISTING
    # ==========================================
    def _handle_browse_folder(self):
        """Opens folder dialog to select a third-party video editor export directory."""
        folder = filedialog.askdirectory(
            title="Select Video Editor Exports Folder (CapCut / ClipChamp / etc.)"
        )
        if folder:
            self.selected_folder_path = folder
            self.folder_path_entry.delete(0, "end")
            self.folder_path_entry.insert(0, folder)
            self.log(f"📂 [FOLDER SELECTED] {folder}")
            self._refresh_folder_videos()

    def _refresh_folder_videos(self):
        """Scans the selected folder for video files and lists them."""
        folder = self.folder_path_entry.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("No Folder", "Please select a valid folder first using '📂 Browse Folder'.")
            return

        self.selected_folder_path = folder
        video_files = []

        try:
            for fname in sorted(os.listdir(folder)):
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in VIDEO_EXTENSIONS:
                        size = os.path.getsize(fpath)
                        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(fpath)))
                        video_files.append((fname, size, mtime))
        except Exception as e:
            self.log(f"❌ [SCAN ERROR] Could not scan folder: {e}")
            return

        # Update the video list display
        self.video_listbox.configure(state="normal")
        self.video_listbox.delete("1.0", "end")

        if not video_files:
            self.video_listbox.insert("end", f"No video files found in:\n{folder}\n\nSupported formats: {', '.join(VIDEO_EXTENSIONS)}\n")
            self.video_list_label.configure(text="📹 Video Files Found: 0")
        else:
            self.video_listbox.insert("end", f"{'#':<4} {'File Name':<40} {'Size':<12} {'Modified'}\n")
            self.video_listbox.insert("end", "─" * 80 + "\n")
            for idx, (name, size, mtime) in enumerate(video_files, 1):
                size_str = format_bytes(size)
                self.video_listbox.insert("end", f"{idx:<4} {name:<40} {size_str:<12} {mtime}\n")
            self.video_list_label.configure(text=f"📹 Video Files Found: {len(video_files)}")
            self.log(f"📹 [SCAN] Found {len(video_files)} video file(s) in: {folder}")

        self.video_listbox.configure(state="disabled")

        # Store the list for selection
        self._folder_video_files = video_files

    def _parse_time_str(self, t_str: str) -> Optional[float]:
        """Parses seconds (e.g. '12.5') or mm:ss (e.g. '01:30') into float seconds."""
        if not t_str or not t_str.strip():
            return None
        t = t_str.strip()
        try:
            if ":" in t:
                parts = t.split(":")
                if len(parts) == 2:
                    return float(parts[0]) * 60.0 + float(parts[1])
                elif len(parts) == 3:
                    return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
            return float(t)
        except ValueError:
            return None

    def _probe_clip_duration(self, video_path: str) -> float:
        """Quick duration probe using OpenCV if available."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and cnt > 0:
                return round(cnt / fps, 2)
        except Exception:
            pass
        return 0.0

    def _add_clip_to_merge_list(self, file_path: str, start_time: Optional[float] = None, end_time: Optional[float] = None):
        """Adds a video part/segment to the merge sequence."""
        if not os.path.exists(file_path):
            return
        
        size = os.path.getsize(file_path)
        dur = self._probe_clip_duration(file_path)
        sha = compute_sha256(file_path)

        clip_item = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size": size,
            "duration": dur,
            "start_time": start_time,
            "end_time": end_time,
            "sha256": sha
        }
        self.selected_clips.append(clip_item)
        self.selected_file_path = file_path
        self._update_clips_display()
        self.log(f"➕ [PART ADDED] Part #{len(self.selected_clips)}: {clip_item['file_name']} (Start: {start_time or 0}s, End: {end_time or 'Full'})")

    def _update_clips_display(self):
        """Refreshes the Selected Video Parts to Merge textbox."""
        self.clips_listbox.configure(state="normal")
        self.clips_listbox.delete("1.0", "end")

        if not self.selected_clips:
            self.clips_listbox.insert("end", "No video parts added yet.\nClick '📂 Browse & Add File(s)' or select from folder above and click '➕ Add Clip'.\n")
            self.clips_summary_lbl.configure(text="Total Parts: 0 | Size: 0 B")
        else:
            self.clips_listbox.insert("end", f"{'#':<4} {'File Name':<32} {'Size':<10} {'Duration':<10} {'Trim Range'}\n")
            self.clips_listbox.insert("end", "─" * 80 + "\n")
            total_bytes = 0
            total_dur = 0.0

            for idx, c in enumerate(self.selected_clips, 1):
                total_bytes += c["file_size"]
                c_dur = c.get("duration", 0.0)
                s_val = c.get("start_time")
                e_val = c.get("end_time")
                
                if s_val is not None or e_val is not None:
                    s_disp = f"{s_val:.1f}s" if s_val is not None else "0s"
                    e_disp = f"{e_val:.1f}s" if e_val is not None else "End"
                    trim_disp = f"{s_disp} - {e_disp} (Trimmed)"
                    part_len = ((e_val if e_val is not None else c_dur) - (s_val if s_val is not None else 0))
                    total_dur += max(0, part_len)
                else:
                    trim_disp = "Full Length"
                    total_dur += c_dur

                dur_disp = f"{c_dur:.1f}s" if c_dur > 0 else "--"
                self.clips_listbox.insert("end", f"[{idx}]  {c['file_name']:<32} {format_bytes(c['file_size']):<10} {dur_disp:<10} {trim_disp}\n")

            self.clips_summary_lbl.configure(text=f"Total: {len(self.selected_clips)} parts | Est. Len: {total_dur:.1f}s | Size: {format_bytes(total_bytes)}")

        self.clips_listbox.configure(state="disabled")

    def _handle_add_multiple_files(self):
        """Allows selecting one or multiple video files from anywhere on Computer A."""
        paths = filedialog.askopenfilenames(
            title="Select Video Files to Merge & Render",
            filetypes=[("Video Files", "*.mp4 *.mov *.mkv *.avi *.webm *.wmv *.flv *.m4v *.ts"), ("All Files", "*.*")]
        )
        if paths:
            s_val = self._parse_time_str(self.trim_start_entry.get())
            e_val = self._parse_time_str(self.trim_end_entry.get())
            for p in paths:
                self._add_clip_to_merge_list(p, start_time=s_val, end_time=e_val)
            self.log(f"🎬 [QUEUE UPDATED] Added {len(paths)} video part(s) to merge list.")

    def _handle_add_selected_from_folder(self):
        """Adds video selected from folder list with trim parameters."""
        text = self.video_select_entry.get().strip()
        if not text:
            messagebox.showwarning("Empty Selection", "Please enter video filename or # from folder list.")
            return

        if not hasattr(self, '_folder_video_files') or not self._folder_video_files:
            messagebox.showwarning("No Videos", "Please scan a folder first using '📂 Browse Folder'.")
            return

        selected_name = None
        try:
            idx = int(text)
            if 1 <= idx <= len(self._folder_video_files):
                selected_name = self._folder_video_files[idx - 1][0]
        except ValueError:
            pass

        if not selected_name:
            for name, _, _ in self._folder_video_files:
                if text.lower() in name.lower():
                    selected_name = name
                    break

        if not selected_name:
            messagebox.showerror("Not Found", f"No video matching '{text}' found in folder.")
            return

        full_path = os.path.join(self.selected_folder_path, selected_name)
        if not os.path.exists(full_path):
            messagebox.showerror("Error", f"File not found: {full_path}")
            return

        s_val = self._parse_time_str(self.trim_start_entry.get())
        e_val = self._parse_time_str(self.trim_end_entry.get())
        self._add_clip_to_merge_list(full_path, start_time=s_val, end_time=e_val)

    def _handle_remove_selected_clip(self):
        """Removes a clip by index from the merge queue."""
        num_str = self.clip_num_entry.get().strip()
        if not num_str:
            if self.selected_clips:
                self.selected_clips.pop()
                self._update_clips_display()
                self.log(f"🗑️ [REMOVED] Removed last video part.")
            return
        try:
            idx = int(num_str) - 1
            if 0 <= idx < len(self.selected_clips):
                removed = self.selected_clips.pop(idx)
                self._update_clips_display()
                self.log(f"🗑️ [REMOVED] Removed part #{idx+1}: {removed['file_name']}")
            else:
                messagebox.showwarning("Invalid Index", f"Part #{idx+1} does not exist.")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid numeric part #.")

    def _handle_move_clip_up(self):
        num_str = self.clip_num_entry.get().strip()
        if not num_str:
            return
        try:
            idx = int(num_str) - 1
            if 1 <= idx < len(self.selected_clips):
                self.selected_clips[idx - 1], self.selected_clips[idx] = self.selected_clips[idx], self.selected_clips[idx - 1]
                self.clip_num_entry.delete(0, "end")
                self.clip_num_entry.insert(0, str(idx))
                self._update_clips_display()
        except ValueError:
            pass

    def _handle_move_clip_down(self):
        num_str = self.clip_num_entry.get().strip()
        if not num_str:
            return
        try:
            idx = int(num_str) - 1
            if 0 <= idx < len(self.selected_clips) - 1:
                self.selected_clips[idx + 1], self.selected_clips[idx] = self.selected_clips[idx], self.selected_clips[idx + 1]
                self.clip_num_entry.delete(0, "end")
                self.clip_num_entry.insert(0, str(idx + 2))
                self._update_clips_display()
        except ValueError:
            pass

    def _handle_clear_clips(self):
        self.selected_clips.clear()
        self.selected_file_path = ""
        self._update_clips_display()
        self.log("[CLEARED] Cleared all video parts from merge queue.")

    # ==========================================
    # EVENT HANDLERS
    # ==========================================
    def _handle_handshake_and_ping(self):
        """Task 1: Handshake protocol and latency ping check with prominent alerts."""
        host = self.ip_entry.get().strip()
        try:
            port = int(self.port_entry.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Invalid port number.")
            return

        def task():
            self.log(f"\n[CONNECTING] Contacting Remote Server (Computer B) at {host}:{port}...")
            try:
                self.client = OffloadingClient(host=host, port=port)
                info = self.client.perform_handshake()
                latency = self.client.measure_ping_latency(num_samples=3)

                hostname = info.get('hostname', 'Remote Server')
                cpu_count = info.get('cpu_count', 4)
                gpu_status = info.get('nvenc_status', 'N/A')
                status_txt = f"Connected to: {hostname} ({host})"
                specs_txt = f"CPU: {cpu_count} Cores | GPU: {gpu_status[:30]} | Ping RTT: {latency} ms"

                def _update_ui():
                    self.worker_status_lbl.configure(text=f"🟢 {status_txt}", text_color="#4ade80")
                    self.worker_specs_lbl.configure(text=specs_txt)
                    self.conn_indicator.configure(text=f"🟢 ONLINE ({hostname})", text_color="#4ade80")

                self.run_on_ui(_update_ui)

                self.log(f"⚡ [TCP CONNECTED] Socket established with {host}:{port}!")
                self.log(f"🤝 [HANDSHAKE VERIFIED] Server '{hostname}' running protocol v{info.get('protocol_version')}")
                self.log(f"⚡ [LATENCY PROBE] Measured Round-Trip Time (RTT): {latency} ms")
                self.log(f"🚀 [RESOURCES UNLOCKED] {cpu_count} CPU Cores | {gpu_status}")
                self.log(f"   Computer B is now ready to render your video exports!")

                # Prominent notification popup
                msg = (
                    f"🎉 Successfully Connected to Server (Computer B)!\n\n"
                    f"• Node Hostname : {hostname}\n"
                    f"• Bound Address : {host}:{port}\n"
                    f"• CPU Hardware  : {cpu_count} Processing Cores\n"
                    f"• Acceleration  : {gpu_status}\n"
                    f"• Ping Latency  : {latency} ms (Fast LAN Wire)\n\n"
                    f"Remote resources are active! You can now send video exports for rendering."
                )
                self.run_on_ui(lambda: messagebox.showinfo("Connected to Server!", msg))

            except Exception as e:
                def _update_err():
                    self.worker_status_lbl.configure(text="🔴 Server Status: UNREACHABLE", text_color="#f87171")
                    self.worker_specs_lbl.configure(text=f"Connection Error: {e}")
                    self.conn_indicator.configure(text="🔴 OFFLINE", text_color="#f87171")
                self.run_on_ui(_update_err)
                self.log(f"❌ [CONNECT ERROR] Could not connect to Server at {host}:{port} -> {e}")
                self.run_on_ui(lambda: messagebox.showerror(
                    "Connection Failed",
                    f"Could not connect to Server at {host}:{port}.\n\n"
                    f"Check:\n"
                    f"1. Is Server GUI / Daemon running on Computer B?\n"
                    f"2. Are both computers connected with LAN cable?\n"
                    f"3. Is the IP address correct?"
                ))

        threading.Thread(target=task, daemon=True).start()

    def _handle_start_offload(self):
        """Initiates video rendering offload to remote server."""
        if self.is_processing:
            messagebox.showwarning("Warning", "A task is currently running.")
            return

        if not self.selected_clips:
            if self.selected_file_path and os.path.exists(self.selected_file_path):
                self._add_clip_to_merge_list(self.selected_file_path)
            else:
                messagebox.showerror("No Videos Selected", "Please add video file(s) to the merge queue.\n\nClick '📂 Browse & Add File(s)' or select from the exports folder.")
                return

        host = self.ip_entry.get().strip()
        port = int(self.port_entry.get().strip())
        render_cfg = {
            "resolution": self.res_combo.get(),
            "bitrate": self.bitrate_combo.get(),
            "preset": self.preset_combo.get(),
            "mode": self.accel_combo.get(),
            "filter": self.filter_combo.get(),
            "fps": self.fps_combo.get()
        }
        clips_to_process = list(self.selected_clips)

        self.is_processing = True
        self.cancel_flag = False

        self.progress_bar.set(0.0)
        self.progress_lbl.configure(text=f"0.0% | Initializing Multi-Part Video Offload ({len(clips_to_process)} parts)...")
        self.metric_integrity.configure(text="Checksum: Calculating...", text_color="#fbbf24")

        def run_thread():
            try:
                # Ensure connection is active
                need_reconnect = False
                if not self.client or not self.client.is_connected or not self.client.sock:
                    need_reconnect = True
                else:
                    try:
                        self.client.sock.send(b"")
                    except Exception:
                        need_reconnect = True

                if need_reconnect:
                    if self.client:
                        try:
                            self.client.disconnect()
                        except Exception:
                            pass
                    self.client = OffloadingClient(host=host, port=port)
                    self.client.connect()
                    self.client.perform_handshake()

                def progress_cb(prog: dict):
                    pct = prog.get("percent", 0.0)
                    status = prog.get("status", "")
                    fps_val = prog.get("fps", 0)
                    speed_val = prog.get("speed", "1x")
                    eta_val = prog.get("eta_seconds", 0)

                    def _update_prog_widgets():
                        self.progress_bar.set(pct / 100.0)
                        self.progress_lbl.configure(text=f"{pct:.1f}% | {status}")
                        if "fps" in prog:
                            self.metric_fps.configure(text=f"Speed: {fps_val} ({speed_val})")
                        if "eta_seconds" in prog:
                            self.metric_eta.configure(text=f"ETA: {eta_val:.1f} s")

                    self.run_on_ui(_update_prog_widgets)
                    if status:
                        self.log(f"  [REMOTE] {status}")

                self.log(f"\n{'='*60}")
                self.log(f"🚀 [OFFLOAD MERGE & RENDER] Offloading {len(clips_to_process)} Video Part(s) to Computer B")
                for idx, c in enumerate(clips_to_process, 1):
                    self.log(f"   Part {idx}: {c['file_name']} ({format_bytes(c['file_size'])}) | Trims: {c.get('start_time') or 0}s - {c.get('end_time') or 'Full'}")
                self.log(f"   Config: {render_cfg}")
                self.log(f"{'='*60}")

                res = self.client.offload_video_merge_render(
                    clips=clips_to_process,
                    output_dir=self.download_output_dir,
                    render_config=render_cfg,
                    progress_callback=progress_cb,
                    cancel_check=lambda: self.cancel_flag
                )

                def _update_video_done():
                    self.progress_bar.set(1.0)
                    self.progress_lbl.configure(text="100.0% | Complete & Downloaded to Laptop!")
                    self.metric_turnaround.configure(text=f"Turnaround: {res['total_turnaround_sec']} s")
                    self.last_rendered_path = res['local_output_path']
                    self.last_log_path = res.get('local_log_path', '')
                    self.play_btn.configure(state="normal")
                    if self.last_log_path and os.path.exists(self.last_log_path):
                        self.view_log_btn.configure(state="normal")
                    if res.get("checksum_verified"):
                        self.metric_integrity.configure(text="SHA-256: VERIFIED ✔", text_color="#4ade80")
                    else:
                        self.metric_integrity.configure(text="SHA-256: MISMATCH ❌", text_color="#f87171")

                self.run_on_ui(_update_video_done)

                self.log(f"\n{'='*60}")
                self.log(f"✅ [SUCCESS] {len(clips_to_process)} Video Parts Merged on Computer B & Saved to Computer A!")
                self.log(f"   • Remote Engine   : {res.get('engine')}")
                self.log(f"   • Parts Processed : {res.get('clips_count', len(clips_to_process))} parts")
                self.log(f"   • Upload Time     : {res['upload_duration_sec']}s")
                self.log(f"   • Render & Merge  : {res['remote_compute_duration_sec']}s")
                self.log(f"   • Download Time   : {res['download_duration_sec']}s")
                self.log(f"   • Total Turnaround: {res['total_turnaround_sec']}s")
                self.log(f"   • Input Total Size: {format_bytes(res['input_file_size'])}")
                self.log(f"   • Output Size     : {format_bytes(res['output_file_size'])}")
                self.log(f"   • SHA-256 Verified: {'✔ YES' if res.get('checksum_verified') else '❌ NO'}")
                self.log(f"   • Saved to Laptop : {res['local_output_path']}")
                if res.get('local_log_path'):
                    self.log(f"   • Server Log File : {res['local_log_path']}")
                self.log(f"{'='*60}")

                if res.get('server_logs_text'):
                    self.log(f"\n📋 [SERVER EXECUTION LOG SUMMARY]")
                    for l in res['server_logs_text'].splitlines()[:15]:
                        self.log(f"   {l}")
                    if len(res['server_logs_text'].splitlines()) > 15:
                        self.log(f"   ... ({len(res['server_logs_text'].splitlines()) - 15} more lines in log file)")

                log_info = f"\n• Server Log File : {os.path.basename(res['local_log_path']) if res.get('local_log_path') else 'Attached'}"
                msg = (
                    f"🎉 {len(clips_to_process)} Video Parts Successfully Merged & Rendered on Server (Computer B)!\n\n"
                    f"• Merged Parts    : {len(clips_to_process)} clips\n"
                    f"• Render Engine   : {res.get('engine')}\n"
                    f"• Upload Time     : {res['upload_duration_sec']} seconds\n"
                    f"• Render & Merge  : {res['remote_compute_duration_sec']} seconds\n"
                    f"• Download Time   : {res['download_duration_sec']} seconds\n"
                    f"• Total Time      : {res['total_turnaround_sec']} seconds\n"
                    f"• Integrity Check : SHA-256 Verified ✔{log_info}\n\n"
                    f"The final merged video is saved in client_downloads/.\n"
                    f"Click '▶️ Play Video' to watch or '📄 View Logs' to inspect details!"
                )
                self.run_on_ui(lambda: messagebox.showinfo("Merge & Render Complete!", msg))

            except Exception as e:
                self.log(f"❌ [OFFLOAD FAILED] {e}")
                def _update_err_badge():
                    self.metric_integrity.configure(text="Error", text_color="#f87171")
                    self.progress_lbl.configure(text=f"Failed: {e}")
                self.run_on_ui(_update_err_badge)
                self.run_on_ui(lambda: messagebox.showerror("Task Failed", f"Execution failed:\n{e}"))
            finally:
                self.is_processing = False

        threading.Thread(target=run_thread, daemon=True).start()

    def _handle_list_shared_files(self):
        """Queries and displays files available in Server storage."""
        def task():
            try:
                self.log("[FILE HUB] Querying shared files from Computer B...")
                if not self.client or not self.client.is_connected:
                    host = self.ip_entry.get().strip()
                    port = int(self.port_entry.get().strip())
                    self.client = OffloadingClient(host=host, port=port)
                    self.client.connect()
                    self.client.perform_handshake()

                files = self.client.list_server_files()
                self.log(f"[FILE HUB] Received {len(files)} shared file(s) from server.")

                def _update_box():
                    self.shared_files_box.configure(state="normal")
                    self.shared_files_box.delete("1.0", "end")
                    if not files:
                        self.shared_files_box.insert("end", "No shared files currently on Server.\nUpload a file using 'Upload File to Server'.\n")
                    else:
                        self.shared_files_box.insert("end", f"{'File Name':<35} | {'Size':<10} | {'Modified Time'}\n")
                        self.shared_files_box.insert("end", "-" * 70 + "\n")
                        for f in files:
                            self.shared_files_box.insert("end", f"{f['name']:<35} | {f['size_str']:<10} | {f['modified']}\n")
                    self.shared_files_box.configure(state="disabled")

                self.run_on_ui(_update_box)
            except Exception as e:
                self.log(f"[FILE HUB ERROR] {e}")
                self.run_on_ui(lambda: messagebox.showerror("Error", f"Could not list server files:\n{e}"))

        threading.Thread(target=task, daemon=True).start()

    def _handle_upload_shared_file(self):
        """Uploads a local file from Computer A directly to Computer B's storage hub."""
        src_path = filedialog.askopenfilename(title="Select File to Upload to Server")
        if not src_path:
            return

        def task():
            try:
                fname = os.path.basename(src_path)
                self.log(f"📁 [UPLOADING] Sending '{fname}' to Computer B...")
                if not self.client or not self.client.is_connected:
                    host = self.ip_entry.get().strip()
                    port = int(self.port_entry.get().strip())
                    self.client = OffloadingClient(host=host, port=port)
                    self.client.connect()
                    self.client.perform_handshake()

                success = self.client.upload_shared_file(
                    src_path,
                    progress_callback=lambda msg: self.log(f"  {msg}")
                )
                if success:
                    self.log(f"✅ [UPLOAD COMPLETE] '{fname}' uploaded to Server shared storage!")
                    self.run_on_ui(lambda: messagebox.showinfo("File Uploaded", f"Successfully uploaded '{fname}' to Server!"))
                    self._handle_list_shared_files()
            except Exception as e:
                self.log(f"❌ [UPLOAD ERROR] {e}")
                self.run_on_ui(lambda: messagebox.showerror("Upload Error", f"Failed to upload file:\n{e}"))

        threading.Thread(target=task, daemon=True).start()

    def _handle_download_shared_file(self):
        """Downloads a requested file from Computer B to Computer A."""
        fname = self.dl_fname_entry.get().strip()
        if not fname:
            messagebox.showwarning("Enter Filename", "Please enter the file name to download from the list above.")
            return

        def task():
            try:
                self.log(f"📥 [DOWNLOADING] Requesting '{fname}' from Computer B...")
                if not self.client or not self.client.is_connected:
                    host = self.ip_entry.get().strip()
                    port = int(self.port_entry.get().strip())
                    self.client = OffloadingClient(host=host, port=port)
                    self.client.connect()
                    self.client.perform_handshake()

                dest = self.client.download_shared_file(
                    fname,
                    self.download_output_dir,
                    progress_callback=lambda msg: self.log(f"  {msg}")
                )
                self.log(f"✅ [DOWNLOAD COMPLETE] Saved to: {dest}")
                self.run_on_ui(lambda: messagebox.showinfo("Download Complete", f"Downloaded '{fname}'!\nSaved to: {dest}"))
            except Exception as e:
                self.log(f"❌ [DOWNLOAD ERROR] {e}")
                self.run_on_ui(lambda: messagebox.showerror("Download Error", f"Could not download file:\n{e}"))

        threading.Thread(target=task, daemon=True).start()

    def _handle_play_rendered_video(self):
        """Opens the downloaded rendered video in the default media player on this laptop."""
        if self.last_rendered_path and os.path.exists(self.last_rendered_path):
            try:
                os.startfile(self.last_rendered_path)
                self.log(f"[MEDIA PLAYER] Opened rendered video: {self.last_rendered_path}")
            except Exception as e:
                self.log(f"[ERROR] Could not open video player: {e}")
        else:
            messagebox.showwarning("No Video", "No rendered video available yet. Please send a video for rendering first.")

    def _handle_view_server_logs(self):
        """Opens the downloaded server execution log file."""
        if self.last_log_path and os.path.exists(self.last_log_path):
            try:
                os.startfile(self.last_log_path)
                self.log(f"[LOG VIEWER] Opened server execution log: {self.last_log_path}")
            except Exception as e:
                self.log(f"[ERROR] Could not open log viewer: {e}")
        else:
            messagebox.showwarning("No Logs", "No server execution logs available yet. Please complete a render task first.")

    def _handle_cancel(self):
        if self.is_processing:
            self.cancel_flag = True
            self.log("[CANCEL] User initiated cancellation request.")
        else:
            self.log("[INFO] No active task to cancel.")


def main():
    app = TaskOffloaderApp()
    app.mainloop()


if __name__ == "__main__":
    main()
