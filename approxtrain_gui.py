import sys
import queue
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

REPO_ROOT = Path(__file__).resolve().parent

# Maps friendly LUT name → filename
LUT_FILES = {p.name: p for p in sorted((REPO_ROOT / "lut").glob("*.bin"))}

# Maps friendly model name → script filename
MODELS = {
    "LeNet-300-100": "lenet300100.py",
}

DATASETS = ["MNIST"]


# ------------------------------------------------------------------ #
# Shared runner mixin — gives any frame its own process + output queue
# ------------------------------------------------------------------ #

class RunnerMixin:
    """Adds run/stop/poll machinery to a Frame subclass."""

    def _init_runner(self, log_widget):
        self._log = log_widget
        self._queue = queue.Queue()
        self._process = None
        self._poll()

    def _poll(self):
        while not self._queue.empty():
            self._write(self._queue.get())
        self.after(100, self._poll)

    def _write(self, text):
        self._log.insert(tk.END, text)
        self._log.see(tk.END)

    def _run(self, cmd):
        if self._process is not None and self._process.poll() is None:
            messagebox.showwarning("Already running", "A process is already running.")
            return
        self._write(f"\n$ {' '.join(cmd)}\n")
        threading.Thread(target=self._run_thread, args=(cmd,), daemon=True).start()

    def _run_thread(self, cmd):
        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
            assert self._process.stdout is not None
            for line in self._process.stdout:
                self._queue.put(line)
            code = self._process.wait()
            self._queue.put(f"\n[Process exited with code {code}]\n")
        except Exception as e:
            self._queue.put(f"\n[Error: {e}]\n")
        finally:
            self._process = None

    def _stop(self):
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            self._write("\n[Stop requested]\n")


# ------------------------------------------------------------------ #
# App controller
# ------------------------------------------------------------------ #

class ApproxTrainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ApproxTrain")
        self.root.geometry("900x620")

        self.container = tk.Frame(self.root)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for FrameClass in (MainMenuFrame, QuickStartFrame, BuildFrame,
                           CreditsFrame, OptionsFrame):
            frame = FrameClass(self.container, self)
            self.frames[FrameClass] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(MainMenuFrame)

    def show_frame(self, frame_class):
        self.frames[frame_class].tkraise()


# ------------------------------------------------------------------ #
# Frames
# ------------------------------------------------------------------ #

class MainMenuFrame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        tk.Label(
            self,
            text="ApproxTrain",
            font=("Helvetica", 48, "bold"),
        ).grid(row=0, column=0, pady=(80, 20))

        btn_frame = tk.Frame(self)
        btn_frame.grid(row=1, column=0)

        btn_cfg = dict(width=14, height=2)
        tk.Button(btn_frame, text="Credits",
                  command=lambda: self.app.show_frame(CreditsFrame),
                  **btn_cfg).pack(side="left", padx=16)
        tk.Button(btn_frame, text="Quick Start",
                  command=lambda: self.app.show_frame(QuickStartFrame),
                  **btn_cfg).pack(side="left", padx=16)
        tk.Button(btn_frame, text="Build",
                  command=lambda: self.app.show_frame(BuildFrame),
                  **btn_cfg).pack(side="left", padx=16)
        tk.Button(btn_frame, text="Options",
                  command=lambda: self.app.show_frame(OptionsFrame),
                  **btn_cfg).pack(side="left", padx=16)


class QuickStartFrame(RunnerMixin, tk.Frame):
    def __init__(self, parent, app):
        tk.Frame.__init__(self, parent)
        self.app = app
        self._script_var = tk.StringVar(value="lenet300100.py")
        self._mul_var = tk.StringVar(value=str(REPO_ROOT / "lut" / "MBM_7.bin"))
        self._approx_var = tk.BooleanVar(value=True)
        self._build()

    def _build(self):
        main = ttk.Frame(self, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Button(main, text="← Menu",
                   command=lambda: self.app.show_frame(MainMenuFrame)).pack(anchor="w")

        controls = ttk.LabelFrame(main, text="Run Settings", padding=10)
        controls.pack(fill="x", pady=(6, 0))

        ttk.Label(controls, text="Script").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Combobox(
            controls,
            textvariable=self._script_var,
            values=["lenet300100.py", "mnist_example.py"],
            state="readonly",
            width=30,
        ).grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(controls, text="Multiplier LUT").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(controls, textvariable=self._mul_var, width=60).grid(
            row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(controls, text="Browse", command=self._browse).grid(
            row=1, column=2, padx=5, pady=5)

        ttk.Checkbutton(controls, text="Use approximate mode",
                        variable=self._approx_var).grid(
            row=2, column=1, sticky="w", padx=5, pady=5)

        controls.columnconfigure(1, weight=1)

        btns = ttk.Frame(main)
        btns.pack(fill="x", pady=(10, 6))
        ttk.Button(btns, text="Run", command=self._do_run).pack(side="left", padx=5)
        ttk.Button(btns, text="Stop", command=self._stop).pack(side="left", padx=5)
        ttk.Button(btns, text="Clear Log",
                   command=lambda: self._log.delete("1.0", tk.END)).pack(side="left", padx=5)

        log_frame = ttk.LabelFrame(main, text="Output", padding=8)
        log_frame.pack(fill="both", expand=True)
        log = tk.Text(log_frame, wrap="word", height=25)
        log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(log_frame, command=log.yview)
        sb.pack(side="right", fill="y")
        log.configure(yscrollcommand=sb.set)

        self._init_runner(log)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select LUT file",
            filetypes=[("Binary files", "*.bin"), ("All files", "*.*")],
        )
        if path:
            self._mul_var.set(path)

    def _do_run(self):
        script = self._script_var.get()
        script_path = REPO_ROOT / script
        if not script_path.exists():
            messagebox.showerror("Error", f"Script not found: {script_path}")
            return
        cmd = [sys.executable, str(script_path)]
        if script == "lenet300100.py":
            cmd += ["--mul", self._mul_var.get()]
            if self._approx_var.get():
                cmd += ["--approx"]
        self._run(cmd)


class BuildFrame(RunnerMixin, tk.Frame):
    """Guided training builder — choose model, dataset, and multiplier via dropdowns."""

    def __init__(self, parent, app):
        tk.Frame.__init__(self, parent)
        self.app = app

        self._model_var = tk.StringVar(value=next(iter(MODELS)))
        self._dataset_var = tk.StringVar(value=DATASETS[0])
        lut_default = "MBM_7.bin" if "MBM_7.bin" in LUT_FILES else next(iter(LUT_FILES))
        self._lut_var = tk.StringVar(value=lut_default)
        self._approx_var = tk.BooleanVar(value=True)
        self._preview_var = tk.StringVar()

        self._build()
        self._refresh_preview()

    def _build(self):
        main = ttk.Frame(self, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Button(main, text="← Menu",
                   command=lambda: self.app.show_frame(MainMenuFrame)).pack(anchor="w")

        # ── Configuration panel ──────────────────────────────────────────
        cfg = ttk.LabelFrame(main, text="Build Configuration", padding=10)
        cfg.pack(fill="x", pady=(6, 0))
        cfg.columnconfigure(1, weight=1)

        # Model
        ttk.Label(cfg, text="Model").grid(row=0, column=0, sticky="w", padx=5, pady=6)
        ttk.Combobox(cfg, textvariable=self._model_var,
                     values=list(MODELS.keys()), state="readonly", width=28).grid(
            row=0, column=1, sticky="ew", padx=5, pady=6)

        # Dataset
        ttk.Label(cfg, text="Dataset").grid(row=1, column=0, sticky="w", padx=5, pady=6)
        ttk.Combobox(cfg, textvariable=self._dataset_var,
                     values=DATASETS, state="readonly", width=28).grid(
            row=1, column=1, sticky="ew", padx=5, pady=6)

        # Multiplier LUT
        ttk.Label(cfg, text="Multiplier").grid(row=2, column=0, sticky="w", padx=5, pady=6)
        lut_box = ttk.Combobox(cfg, textvariable=self._lut_var,
                               values=list(LUT_FILES.keys()), state="readonly", width=28)
        lut_box.grid(row=2, column=1, sticky="ew", padx=5, pady=6)
        ttk.Label(cfg, text="(from lut/)").grid(row=2, column=2, sticky="w", padx=4)

        # Approx mode
        ttk.Checkbutton(cfg, text="Approximate mode",
                        variable=self._approx_var).grid(
            row=3, column=1, sticky="w", padx=5, pady=6)

        # Bind all controls to refresh the preview
        for var in (self._model_var, self._dataset_var,
                    self._lut_var, self._approx_var):
            var.trace_add("write", lambda *_: self._refresh_preview())

        # ── Command preview ──────────────────────────────────────────────
        prev_frame = ttk.LabelFrame(main, text="Command Preview", padding=6)
        prev_frame.pack(fill="x", pady=(8, 0))
        ttk.Label(prev_frame, textvariable=self._preview_var,
                  font=("Courier", 9), anchor="w", wraplength=820).pack(fill="x")

        # ── Action buttons ───────────────────────────────────────────────
        btns = ttk.Frame(main)
        btns.pack(fill="x", pady=(8, 4))
        ttk.Button(btns, text="Run", command=self._do_run).pack(side="left", padx=5)
        ttk.Button(btns, text="Stop", command=self._stop).pack(side="left", padx=5)
        ttk.Button(btns, text="Clear Log",
                   command=lambda: self._log.delete("1.0", tk.END)).pack(side="left", padx=5)

        # ── Output log ───────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(main, text="Output", padding=8)
        log_frame.pack(fill="both", expand=True, pady=(4, 0))
        log = tk.Text(log_frame, wrap="word", height=18)
        log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(log_frame, command=log.yview)
        sb.pack(side="right", fill="y")
        log.configure(yscrollcommand=sb.set)

        self._init_runner(log)

    def _build_cmd(self):
        script = MODELS[self._model_var.get()]
        script_path = REPO_ROOT / script
        lut_path = LUT_FILES[self._lut_var.get()]
        cmd = [sys.executable, str(script_path), "--mul", str(lut_path)]
        if self._approx_var.get():
            cmd += ["--approx"]
        return cmd

    def _refresh_preview(self):
        try:
            cmd = self._build_cmd()
            self._preview_var.set(" ".join(cmd))
        except Exception as e:
            self._preview_var.set(f"(error: {e})")

    def _do_run(self):
        try:
            cmd = self._build_cmd()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        lut_path = LUT_FILES.get(self._lut_var.get())
        if lut_path and not lut_path.exists():
            messagebox.showerror("Error", f"LUT file not found: {lut_path}")
            return
        self._run(cmd)


class CreditsFrame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        ttk.Button(self, text="← Menu",
                   command=lambda: self.app.show_frame(MainMenuFrame)).grid(
            row=0, column=0, sticky="w", padx=12, pady=8)

        tk.Label(self, text="Credits", font=("Helvetica", 24, "bold")).grid(row=1, column=0)


class OptionsFrame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        ttk.Button(self, text="← Menu",
                   command=lambda: self.app.show_frame(MainMenuFrame)).grid(
            row=0, column=0, sticky="w", padx=12, pady=8)

        tk.Label(self, text="Options", font=("Helvetica", 24, "bold")).grid(row=1, column=0)


if __name__ == "__main__":
    root = tk.Tk()
    app = ApproxTrainGUI(root)
    root.mainloop()
