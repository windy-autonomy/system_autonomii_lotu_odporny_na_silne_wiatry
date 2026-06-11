from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from openscale_processing import (
    KG_TO_NEWTONS,
    RawSample,
    make_raw_sample,
    write_raw_sample,
    write_raw_session_header,
)


class OpenScaleRecorderApp(tk.Tk):
    def __init__(self, port: str, baudrate: int, output_dir: Path):
        super().__init__()
        self.title("OpenScale Recorder")
        self.ui_scale = self._detect_ui_scale()
        self.tk.call("tk", "scaling", self.ui_scale)
        self.geometry(f"{int(1440 * self.ui_scale)}x{int(920 * self.ui_scale)}")
        self.minsize(int(1100 * self.ui_scale), int(720 * self.ui_scale))

        self.port_var = tk.StringVar(value=port)
        self.baudrate_var = tk.StringVar(value=str(baudrate))
        self.output_dir_var = tk.StringVar(value=str(output_dir))
        self.status_var = tk.StringVar(value="Disconnected")
        self.session_var = tk.StringVar(value="No active session")
        self.current_var = tk.StringVar(value="Waiting for data...")
        self.raw_line_var = tk.StringVar(value="Raw line: -")
        self.connection_button_text = tk.StringVar(value="Connect")
        self.start_button_text = tk.StringVar(value="Start")
        self.stop_button_text = tk.StringVar(value="Stop")

        self.serial_module = None
        self.serial_connection = None
        self.reader_thread: threading.Thread | None = None
        self.reader_stop_event = threading.Event()
        self.poll_interval_ms = 50
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.recording = False
        self.connected = False
        self.session_start_unix = 0.0
        self.session_file = None
        self.session_path: Path | None = None
        self.samples_since_last_draw = 0

        self.live_timestamps: list[float] = []
        self.live_values: list[float] = []

        self._build_style()
        self._build_layout()
        self.after(self.poll_interval_ms, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _detect_ui_scale(self) -> float:
        try:
            pixels_per_inch = float(self.winfo_fpixels("1i"))
        except Exception:
            pixels_per_inch = 96.0

        scale = pixels_per_inch / 96.0
        return max(1.15, min(scale, 2.2))

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        base_font = max(11, int(11 * self.ui_scale))
        header_font = max(22, int(18 * self.ui_scale))
        value_font = max(20, int(16 * self.ui_scale))
        status_font = max(12, int(10 * self.ui_scale))

        style.configure("App.TFrame", background="#f3f0ea")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Header.TLabel", background="#f3f0ea", font=("TkDefaultFont", header_font, "bold"))
        style.configure("Subtle.TLabel", background="#f3f0ea", foreground="#555555", font=("TkDefaultFont", base_font))
        style.configure("Status.TLabel", background="#ffffff", foreground="#1f3b4d", font=("TkDefaultFont", status_font, "bold"))
        style.configure("Value.TLabel", background="#ffffff", font=("TkDefaultFont", value_font, "bold"))
        style.configure("Raw.TLabel", background="#ffffff", foreground="#555555", font=("TkDefaultFont", base_font))
        style.configure("TLabel", font=("TkDefaultFont", base_font))
        style.configure("TButton", font=("TkDefaultFont", base_font, "bold"), padding=(12, 8))
        style.configure("TEntry", padding=(10, 8))
        style.configure("TCheckbutton", font=("TkDefaultFont", base_font))

    def _build_layout(self) -> None:
        outer_pad = int(18 * self.ui_scale)
        inner_pad = int(14 * self.ui_scale)
        control_width = int(28 * self.ui_scale)

        root = ttk.Frame(self, padding=outer_pad, style="App.TFrame")
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root, style="App.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="OpenScale Recorder", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Connect, record, and inspect live OpenScale readings with raw session logging.",
            style="Subtle.TLabel",
        ).pack(anchor="w", pady=(2, 12))

        body = ttk.Frame(root, style="App.TFrame")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        controls = ttk.Frame(body, padding=inner_pad, style="Panel.TFrame")
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, outer_pad))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Connection", style="Status.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, int(10 * self.ui_scale)))
        ttk.Label(controls, text="Port", style="Raw.TLabel").grid(row=1, column=0, sticky="w", pady=int(5 * self.ui_scale))
        ttk.Entry(controls, textvariable=self.port_var, width=control_width).grid(row=1, column=1, sticky="ew", pady=int(5 * self.ui_scale))
        ttk.Label(controls, text="Baudrate", style="Raw.TLabel").grid(row=2, column=0, sticky="w", pady=int(5 * self.ui_scale))
        ttk.Entry(controls, textvariable=self.baudrate_var, width=control_width).grid(row=2, column=1, sticky="ew", pady=int(5 * self.ui_scale))
        ttk.Label(controls, text="Output dir", style="Raw.TLabel").grid(row=3, column=0, sticky="w", pady=int(5 * self.ui_scale))
        ttk.Entry(controls, textvariable=self.output_dir_var, width=control_width).grid(row=3, column=1, sticky="ew", pady=int(5 * self.ui_scale))
        ttk.Button(controls, text="Browse", command=self._browse_output_dir).grid(row=4, column=1, sticky="e", pady=(0, int(10 * self.ui_scale)))

        ttk.Separator(controls).grid(row=5, column=0, columnspan=2, sticky="ew", pady=int(12 * self.ui_scale))
        ttk.Label(controls, text="Controls", style="Status.TLabel").grid(row=6, column=0, columnspan=2, sticky="w", pady=(0, int(10 * self.ui_scale)))

        self.connect_button = ttk.Button(controls, textvariable=self.connection_button_text, command=self._toggle_connection)
        self.connect_button.grid(row=7, column=0, columnspan=2, sticky="ew", pady=int(5 * self.ui_scale))

        self.start_button = ttk.Button(controls, textvariable=self.start_button_text, command=self.start_recording, state="disabled")
        self.start_button.grid(row=8, column=0, columnspan=2, sticky="ew", pady=int(5 * self.ui_scale))

        self.stop_button = ttk.Button(controls, textvariable=self.stop_button_text, command=self.stop_recording, state="disabled")
        self.stop_button.grid(row=9, column=0, columnspan=2, sticky="ew", pady=int(5 * self.ui_scale))

        ttk.Separator(controls).grid(row=10, column=0, columnspan=2, sticky="ew", pady=int(12 * self.ui_scale))
        ttk.Label(controls, text="Status", style="Status.TLabel").grid(row=11, column=0, columnspan=2, sticky="w")
        ttk.Label(controls, textvariable=self.status_var, style="Raw.TLabel", wraplength=int(300 * self.ui_scale), justify="left").grid(
            row=12, column=0, columnspan=2, sticky="w", pady=(2, int(10 * self.ui_scale))
        )
        ttk.Label(controls, text="Session", style="Status.TLabel").grid(row=13, column=0, columnspan=2, sticky="w")
        ttk.Label(controls, textvariable=self.session_var, style="Raw.TLabel", wraplength=int(300 * self.ui_scale), justify="left").grid(
            row=14, column=0, columnspan=2, sticky="w", pady=(2, int(10 * self.ui_scale))
        )
        ttk.Label(controls, text="Current reading", style="Status.TLabel").grid(row=15, column=0, columnspan=2, sticky="w")
        ttk.Label(controls, textvariable=self.current_var, style="Value.TLabel", wraplength=int(300 * self.ui_scale), justify="left").grid(
            row=16, column=0, columnspan=2, sticky="w", pady=(2, int(6 * self.ui_scale))
        )
        ttk.Label(controls, textvariable=self.raw_line_var, style="Raw.TLabel", wraplength=int(300 * self.ui_scale), justify="left").grid(
            row=17, column=0, columnspan=2, sticky="w"
        )

        plot_panel = ttk.Frame(body, padding=inner_pad, style="Panel.TFrame")
        plot_panel.grid(row=0, column=1, sticky="nsew")
        plot_panel.rowconfigure(0, weight=1)
        plot_panel.columnconfigure(0, weight=1)

        self.figure = Figure(
            figsize=(9.5 * self.ui_scale, 6.6 * self.ui_scale),
            dpi=140 * self.ui_scale,
            facecolor="#ffffff",
        )
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Live raw readings")
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Force [N]")
        self.ax.tick_params(labelsize=max(10, int(10 * self.ui_scale)))
        self.ax.title.set_size(max(14, int(13 * self.ui_scale)))
        self.ax.xaxis.label.set_size(max(12, int(11 * self.ui_scale)))
        self.ax.yaxis.label.set_size(max(12, int(11 * self.ui_scale)))
        self.ax.grid(True, alpha=0.25, linewidth=0.8)

        (self.live_line,) = self.ax.plot([], [], color="#2b6cb0", linewidth=2.4, marker="o", markersize=max(4.0, 3.5 * self.ui_scale))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_panel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.draw()

    def _browse_output_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.output_dir_var.get() or str(Path.cwd()))
        if selected:
            self.output_dir_var.set(selected)

    def _toggle_connection(self) -> None:
        if self.connected:
            self.disconnect()
        else:
            self.connect()

    def connect(self) -> None:
        if self.connected:
            return

        try:
            import serial  # type: ignore
        except ImportError:
            messagebox.showerror(
                "Missing dependency",
                "pyserial is not installed. Install it with: pip install pyserial",
            )
            self.status_var.set("pyserial is missing")
            return

        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning("Missing port", "Enter a serial port first.")
            return

        try:
            baudrate = int(self.baudrate_var.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid baudrate", "Baudrate must be a number.")
            return

        try:
            connection = serial.Serial(port, baudrate, timeout=0.25)
        except Exception as exc:  # pragma: no cover - hardware dependent
            messagebox.showerror("Connection failed", f"Could not open {port}:\n{exc}")
            self.status_var.set(f"Connection failed: {exc}")
            return

        time.sleep(2.0)

        self.serial_module = serial
        self.serial_connection = connection
        self.reader_stop_event.clear()
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()
        self.connected = True
        self.connection_button_text.set("Disconnect")
        self.start_button.configure(state="normal")
        self.status_var.set(f"Connected to {port} at {baudrate} baud")
        self.session_var.set("No active session")

    def disconnect(self) -> None:
        self.stop_recording()
        self.reader_stop_event.set()
        reader_thread = self.reader_thread
        connection = self.serial_connection
        self.serial_connection = None
        self.reader_thread = None
        self.connected = False
        self.connection_button_text.set("Connect")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")

        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

        if reader_thread is not None and reader_thread.is_alive():
            reader_thread.join(timeout=1.0)

        self.status_var.set("Disconnected")

    def start_recording(self) -> None:
        if not self.connected or self.serial_connection is None:
            messagebox.showinfo("Not connected", "Connect to the OpenScale before starting a session.")
            return
        if self.recording:
            return

        try:
            baudrate = int(self.baudrate_var.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid baudrate", "Baudrate must be a number.")
            return

        output_dir = Path(self.output_dir_var.get().strip() or "recordings")
        output_dir.mkdir(parents=True, exist_ok=True)
        session_name = datetime.now().strftime("openscale_%Y%m%d_%H%M%S.csv")
        self.session_path = output_dir / session_name
        self.session_start_unix = time.time()
        self.live_timestamps.clear()
        self.live_values.clear()
        self.samples_since_last_draw = 0
        self._clear_plot()

        self.session_file = self.session_path.open("w", newline="")
        write_raw_session_header(
            self.session_file,
            port=self.port_var.get().strip(),
            baudrate=baudrate,
            started_at_unix=self.session_start_unix,
        )
        self.session_file.flush()
        self.recording = True
        self.stop_button.configure(state="normal")
        self.status_var.set("Recording live data")
        self.session_var.set(str(self.session_path))

    def stop_recording(self) -> None:
        if not self.recording:
            self.stop_button.configure(state="disabled")
            return

        if self.session_file is not None:
            try:
                self.session_file.flush()
            finally:
                self.session_file.close()
        self.session_file = None
        self.recording = False
        self.stop_button.configure(state="disabled")
        if self.connected:
            self.status_var.set("Connected, recording stopped")
        if self.session_path is not None:
            self.session_var.set(f"Saved raw session: {self.session_path}")

    def _reader_loop(self) -> None:
        assert self.serial_connection is not None
        while not self.reader_stop_event.is_set():
            try:
                raw_bytes = self.serial_connection.readline()
            except Exception as exc:  # pragma: no cover - hardware dependent
                self.queue.put(("error", exc))
                break

            if not raw_bytes:
                continue

            try:
                raw_line = raw_bytes.decode("utf-8", errors="replace").strip()
            except Exception:
                raw_line = raw_bytes.decode(errors="replace").strip()

            sample = make_raw_sample(time.time(), 0.0, raw_line)
            self.queue.put(("sample", sample))

        self.queue.put(("reader_stopped", None))

    def _poll_queue(self) -> None:
        drained = 0
        while True:
            try:
                kind, payload = self.queue.get_nowait()
            except queue.Empty:
                break

            drained += 1
            if kind == "sample":
                self._handle_sample(payload)  # type: ignore[arg-type]
            elif kind == "error":
                self._handle_reader_error(payload)  # type: ignore[arg-type]
            elif kind == "reader_stopped":
                if self.connected:
                    self.status_var.set("Serial reader stopped")

        if drained and self.recording:
            self._redraw_plot()

        self.after(self.poll_interval_ms, self._poll_queue)

    def _handle_sample(self, sample: RawSample) -> None:
        elapsed_s = max(0.0, sample.timestamp_unix - self.session_start_unix) if self.recording else 0.0
        self.raw_line_var.set(f"Raw line: {sample.raw_text}")

        if self.recording:
            if self.session_file is not None:
                write_raw_sample(
                    self.session_file,
                    replace(sample, elapsed_s=elapsed_s),
                )
                self.session_file.flush()
            if sample.value_kg is not None:
                force_n = sample.value_kg * KG_TO_NEWTONS
                self.current_var.set(f"{force_n:.3f} N")
                self.live_timestamps.append(elapsed_s)
                self.live_values.append(force_n)
                self.samples_since_last_draw += 1
        elif sample.value_kg is not None:
            force_n = sample.value_kg * KG_TO_NEWTONS
            self.current_var.set(f"{force_n:.3f} N")

    def _handle_reader_error(self, exc: Exception) -> None:
        self.status_var.set(f"Reader error: {exc}")
        self.disconnect()

    def _clear_plot(self) -> None:
        self.live_line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def _redraw_plot(self) -> None:
        if not self.live_timestamps:
            self._clear_plot()
            return

        self.live_line.set_data(self.live_timestamps, self.live_values)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def _on_close(self) -> None:
        self.reader_stop_event.set()
        self.stop_recording()
        reader_thread = self.reader_thread
        if self.serial_connection is not None:
            try:
                self.serial_connection.close()
            except Exception:
                pass
        if reader_thread is not None and reader_thread.is_alive():
            reader_thread.join(timeout=1.0)
        self.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenScale live recorder")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port for the OpenScale")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument(
        "--output-dir",
        default="recordings",
        help="Directory where raw session CSV files are stored",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = OpenScaleRecorderApp(
        port=args.port,
        baudrate=args.baudrate,
        output_dir=Path(args.output_dir),
    )
    app.mainloop()


if __name__ == "__main__":
    main()
