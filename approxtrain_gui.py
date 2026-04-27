import os
import sys
import re
import math
import time
import queue
import threading
import subprocess
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

REPO_ROOT = Path(__file__).resolve().parent

LUT_FILES = {p.name: p for p in sorted((REPO_ROOT / "lut").glob("*.bin"))}
MODELS    = {"LeNet-300-100": "lenet300100.py"}
DATASETS  = ["MNIST"]

LAYER_DEFS = {
    "Dense": [
        {"name": "units",      "label": "Units",      "type": "int",    "default": 128},
        {"name": "activation", "label": "Activation", "type": "choice", "default": "relu",
         "choices": ["relu", "sigmoid", "softmax", "tanh", "linear"]},
    ],
    "Flatten": [],
    "Dropout": [
        {"name": "rate", "label": "Rate", "type": "float", "default": 0.5},
    ],
    "Conv2D": [
        {"name": "filters",     "label": "Filters",    "type": "int",    "default": 32},
        {"name": "kernel_size", "label": "Kernel",     "type": "int",    "default": 3},
        {"name": "activation",  "label": "Activation", "type": "choice", "default": "relu",
         "choices": ["relu", "sigmoid", "tanh", "linear"]},
    ],
    "MaxPooling2D": [
        {"name": "pool_size", "label": "Pool Size", "type": "int", "default": 2},
    ],
    "BatchNormalization": [],
    "Input (Flatten 28×28)": [],
}
LAYER_NAMES = list(LAYER_DEFS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Metrics collection
# ─────────────────────────────────────────────────────────────────────────────

METRIC_KEYS = ["epoch", "elapsed_s", "loss", "accuracy", "val_loss", "val_accuracy"]


class MetricsCollector:
    """Parses Keras stdout lines and accumulates per-epoch metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.data  = {k: [] for k in METRIC_KEYS}
        self._epoch = 0
        self._t0    = None
        self.dirty  = False

    def feed(self, line: str):
        # "Epoch 3/10"
        m = re.match(r'Epoch (\d+)/\d+', line.strip())
        if m:
            self._epoch = int(m.group(1))
            if self._t0 is None:
                self._t0 = time.time()
            return

        # Final step summary: contains "/step" but not "ETA"
        if self._epoch > 0 and '/step' in line and 'ETA' not in line:
            vals = {}
            for key in ('loss', 'accuracy', 'val_loss', 'val_accuracy'):
                m2 = re.search(rf'\b{re.escape(key)}: ([0-9.eE+\-]+)', line)
                if m2:
                    try:
                        vals[key] = float(m2.group(1))
                    except ValueError:
                        pass
            if vals:
                elapsed = round(time.time() - self._t0, 2) if self._t0 else 0.0
                self.data["epoch"].append(self._epoch)
                self.data["elapsed_s"].append(elapsed)
                for key in ('loss', 'accuracy', 'val_loss', 'val_accuracy'):
                    self.data[key].append(vals.get(key, float('nan')))
                self.dirty = True


# ─────────────────────────────────────────────────────────────────────────────
# Plot panel
# ─────────────────────────────────────────────────────────────────────────────

_Y2_NONE = "— none —"


class PlotPanel(tk.Frame):
    """Embedded matplotlib figure with user-selectable X / Y / Y2 axes."""

    def __init__(self, parent, metrics: MetricsCollector):
        super().__init__(parent)
        self._metrics = metrics
        self._build()

    def _build(self):
        ctrl = tk.Frame(self)
        ctrl.pack(fill="x", padx=6, pady=4)

        for label, attr, default, extra in [
            ("X",  "_x_var",  "epoch",    []),
            ("Y",  "_y_var",  "loss",     []),
            ("Y2", "_y2_var", _Y2_NONE,   [_Y2_NONE]),
        ]:
            tk.Label(ctrl, text=label).pack(side="left", padx=(8, 2))
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            ttk.Combobox(ctrl, textvariable=var,
                         values=extra + METRIC_KEYS,
                         state="readonly", width=13).pack(side="left")

        ttk.Button(ctrl, text="Refresh", command=self.refresh).pack(side="left", padx=10)

        fig = Figure(figsize=(5, 3.2), dpi=90, tight_layout=True)
        self._ax     = fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(fig, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        self.refresh()

    def refresh(self):
        d = self._metrics.data
        x_key  = self._x_var.get()
        y_key  = self._y_var.get()
        y2_key = self._y2_var.get()

        def pairs(xk, yk):
            return [
                (x, y) for x, y in zip(d.get(xk, []), d.get(yk, []))
                if not (isinstance(x, float) and math.isnan(x))
                and not (isinstance(y, float) and math.isnan(y))
            ]

        self._ax.clear()
        plotted = False

        p1 = pairs(x_key, y_key)
        if p1:
            xs, ys = zip(*p1)
            self._ax.plot(xs, ys, marker='o', linewidth=1.5, label=y_key)
            plotted = True

        if y2_key != _Y2_NONE:
            p2 = pairs(x_key, y2_key)
            if p2:
                xs2, ys2 = zip(*p2)
                self._ax.plot(xs2, ys2, marker='s', linewidth=1.5,
                              linestyle='--', label=y2_key)
                plotted = True

        if plotted:
            self._ax.set_xlabel(x_key)
            self._ax.set_ylabel("value" if y2_key != _Y2_NONE else y_key)
            self._ax.grid(True, alpha=0.3)
            self._ax.legend(fontsize=8)
        else:
            self._ax.text(0.5, 0.5,
                          "No data yet.\nRun training to populate the plot.",
                          transform=self._ax.transAxes,
                          ha='center', va='center', color='gray', fontsize=10)

        self._canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Shared runner mixin
# ─────────────────────────────────────────────────────────────────────────────

class RunnerMixin:
    """Subprocess management + metrics feeding + auto plot refresh."""

    def _init_runner(self, log_widget: tk.Text, metrics: MetricsCollector):
        self._log            = log_widget
        self._queue          = queue.Queue()
        self._process        = None
        self._metrics        = metrics
        self._plot_panel     = None   # set by frame after PlotPanel is created
        self._last_plot_time = 0.0
        self._poll()

    def _poll(self):
        while not self._queue.empty():
            line = self._queue.get()
            self._write(line)
            self._metrics.feed(line)

        # Refresh plot at most every 2 s when new data is available
        now = time.time()
        if (self._plot_panel is not None
                and self._metrics.dirty
                and now - self._last_plot_time > 2.0):
            self._plot_panel.refresh()
            self._metrics.dirty  = False
            self._last_plot_time = now

        self.after(100, self._poll)

    def _write(self, text: str):
        self._log.insert(tk.END, text)
        self._log.see(tk.END)

    def _run(self, cmd):
        if self._process is not None and self._process.poll() is None:
            messagebox.showwarning("Already running", "A process is already running.")
            return
        self._metrics.reset()
        self._write(f"\n$ {' '.join(cmd)}\n")
        threading.Thread(target=self._run_thread, args=(cmd,), daemon=True).start()

    def _run_thread(self, cmd):
        try:
            self._process = subprocess.Popen(
                cmd, cwd=REPO_ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1,
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

    def _make_output_notebook(self, parent) -> tk.Text:
        """
        Build a ttk.Notebook with Log and Plot tabs.
        Returns the log Text widget. Sets self._plot_panel as a side-effect.
        Call this after self._metrics exists.
        """
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        # Log tab
        log_frame = tk.Frame(nb)
        nb.add(log_frame, text="Log")
        log = tk.Text(log_frame, wrap="word", font=("Courier", 9))
        log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(log_frame, command=log.yview)
        sb.pack(side="right", fill="y")
        log.configure(yscrollcommand=sb.set)

        # Plot tab
        plot_frame = tk.Frame(nb)
        nb.add(plot_frame, text="Plot")
        panel = PlotPanel(plot_frame, self._metrics)
        panel.pack(fill="both", expand=True)
        self._plot_panel = panel

        return log


# ─────────────────────────────────────────────────────────────────────────────
# App controller
# ─────────────────────────────────────────────────────────────────────────────

class ApproxTrainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ApproxTrain")
        self.root.geometry("960x680")

        self.container = tk.Frame(self.root)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for cls in (MainMenuFrame, QuickStartFrame, BuildFrame,
                    ModelMakerFrame, CreditsFrame, OptionsFrame):
            f = cls(self.container, self)
            self.frames[cls] = f
            f.grid(row=0, column=0, sticky="nsew")

        self.show_frame(MainMenuFrame)

    def show_frame(self, cls):
        self.frames[cls].tkraise()


# ─────────────────────────────────────────────────────────────────────────────
# Main menu
# ─────────────────────────────────────────────────────────────────────────────

class MainMenuFrame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        tk.Label(self, text="ApproxTrain",
                 font=("Helvetica", 48, "bold")).grid(row=0, column=0, pady=(80, 20))

        btn_frame = tk.Frame(self)
        btn_frame.grid(row=1, column=0)

        for label, cls in [
            ("Credits",     CreditsFrame),
            ("Quick Start", QuickStartFrame),
            ("Build",       BuildFrame),
            ("Model Maker", ModelMakerFrame),
            ("Options",     OptionsFrame),
        ]:
            tk.Button(btn_frame, text=label, width=12, height=2,
                      command=lambda c=cls: self.app.show_frame(c)
                      ).pack(side="left", padx=12)


# ─────────────────────────────────────────────────────────────────────────────
# Quick Start
# ─────────────────────────────────────────────────────────────────────────────

class QuickStartFrame(RunnerMixin, tk.Frame):
    def __init__(self, parent, app):
        tk.Frame.__init__(self, parent)
        self.app = app
        self._script_var = tk.StringVar(value="lenet300100.py")
        self._mul_var    = tk.StringVar(value=str(REPO_ROOT / "lut" / "MBM_7.bin"))
        self._approx_var = tk.BooleanVar(value=True)
        self._metrics    = MetricsCollector()
        self._build()

    def _build(self):
        main = ttk.Frame(self, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Button(main, text="← Menu",
                   command=lambda: self.app.show_frame(MainMenuFrame)).pack(anchor="w")

        cfg = ttk.LabelFrame(main, text="Run Settings", padding=10)
        cfg.pack(fill="x", pady=(6, 0))

        ttk.Label(cfg, text="Script").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Combobox(cfg, textvariable=self._script_var,
                     values=["lenet300100.py", "mnist_example.py"],
                     state="readonly", width=30
                     ).grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(cfg, text="Multiplier LUT").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(cfg, textvariable=self._mul_var, width=50
                  ).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(cfg, text="Browse", command=self._browse
                   ).grid(row=1, column=2, padx=5, pady=5)

        ttk.Checkbutton(cfg, text="Use approximate mode",
                        variable=self._approx_var
                        ).grid(row=2, column=1, sticky="w", padx=5, pady=5)
        cfg.columnconfigure(1, weight=1)

        btns = ttk.Frame(main)
        btns.pack(fill="x", pady=(8, 4))
        ttk.Button(btns, text="Run",  command=self._do_run).pack(side="left", padx=5)
        ttk.Button(btns, text="Stop", command=self._stop).pack(side="left", padx=5)
        ttk.Button(btns, text="Clear Log",
                   command=lambda: self._log.delete("1.0", tk.END)).pack(side="left", padx=5)

        log = self._make_output_notebook(main)
        self._init_runner(log, self._metrics)

    def _browse(self):
        p = filedialog.askopenfilename(
            title="Select LUT file",
            filetypes=[("Binary files", "*.bin"), ("All files", "*.*")])
        if p:
            self._mul_var.set(p)

    def _do_run(self):
        script = self._script_var.get()
        path   = REPO_ROOT / script
        if not path.exists():
            messagebox.showerror("Error", f"Script not found: {path}")
            return
        cmd = [sys.executable, str(path)]
        if script == "lenet300100.py":
            cmd += ["--mul", self._mul_var.get()]
            if self._approx_var.get():
                cmd += ["--approx"]
        self._run(cmd)


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────

class BuildFrame(RunnerMixin, tk.Frame):
    def __init__(self, parent, app):
        tk.Frame.__init__(self, parent)
        self.app = app
        self._model_var   = tk.StringVar(value=next(iter(MODELS)))
        self._dataset_var = tk.StringVar(value=DATASETS[0])
        lut_default = "MBM_7.bin" if "MBM_7.bin" in LUT_FILES else next(iter(LUT_FILES))
        self._lut_var     = tk.StringVar(value=lut_default)
        self._approx_var  = tk.BooleanVar(value=True)
        self._preview_var = tk.StringVar()
        self._metrics     = MetricsCollector()
        self._build()
        self._refresh_preview()

    def _build(self):
        main = ttk.Frame(self, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Button(main, text="← Menu",
                   command=lambda: self.app.show_frame(MainMenuFrame)).pack(anchor="w")

        cfg = ttk.LabelFrame(main, text="Build Configuration", padding=10)
        cfg.pack(fill="x", pady=(6, 0))
        cfg.columnconfigure(1, weight=1)

        ttk.Label(cfg, text="Model").grid(row=0, column=0, sticky="w", padx=5, pady=6)
        ttk.Combobox(cfg, textvariable=self._model_var,
                     values=list(MODELS.keys()), state="readonly", width=28
                     ).grid(row=0, column=1, sticky="ew", padx=5, pady=6)

        ttk.Label(cfg, text="Dataset").grid(row=1, column=0, sticky="w", padx=5, pady=6)
        ttk.Combobox(cfg, textvariable=self._dataset_var,
                     values=DATASETS, state="readonly", width=28
                     ).grid(row=1, column=1, sticky="ew", padx=5, pady=6)

        ttk.Label(cfg, text="Multiplier").grid(row=2, column=0, sticky="w", padx=5, pady=6)
        ttk.Combobox(cfg, textvariable=self._lut_var,
                     values=list(LUT_FILES.keys()), state="readonly", width=28
                     ).grid(row=2, column=1, sticky="ew", padx=5, pady=6)
        ttk.Label(cfg, text="(from lut/)").grid(row=2, column=2, sticky="w", padx=4)

        ttk.Checkbutton(cfg, text="Approximate mode",
                        variable=self._approx_var
                        ).grid(row=3, column=1, sticky="w", padx=5, pady=6)

        for v in (self._model_var, self._dataset_var, self._lut_var, self._approx_var):
            v.trace_add("write", lambda *_: self._refresh_preview())

        prev = ttk.LabelFrame(main, text="Command Preview", padding=6)
        prev.pack(fill="x", pady=(8, 0))
        ttk.Label(prev, textvariable=self._preview_var,
                  font=("Courier", 9), anchor="w", wraplength=900).pack(fill="x")

        btns = ttk.Frame(main)
        btns.pack(fill="x", pady=(8, 4))
        ttk.Button(btns, text="Run",  command=self._do_run).pack(side="left", padx=5)
        ttk.Button(btns, text="Stop", command=self._stop).pack(side="left", padx=5)
        ttk.Button(btns, text="Clear Log",
                   command=lambda: self._log.delete("1.0", tk.END)).pack(side="left", padx=5)

        log = self._make_output_notebook(main)
        self._init_runner(log, self._metrics)

    def _build_cmd(self):
        script   = MODELS[self._model_var.get()]
        lut_path = LUT_FILES[self._lut_var.get()]
        cmd = [sys.executable, str(REPO_ROOT / script), "--mul", str(lut_path)]
        if self._approx_var.get():
            cmd += ["--approx"]
        return cmd

    def _refresh_preview(self):
        try:
            self._preview_var.set(" ".join(self._build_cmd()))
        except Exception as e:
            self._preview_var.set(f"(error: {e})")

    def _do_run(self):
        try:
            cmd = self._build_cmd()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self._run(cmd)


# ─────────────────────────────────────────────────────────────────────────────
# Model Maker — layer row
# ─────────────────────────────────────────────────────────────────────────────

class LayerRow(tk.Frame):
    def __init__(self, parent, on_change, on_remove, initial_type=None):
        super().__init__(parent, relief="groove", bd=1, padx=4, pady=3)
        self._on_change  = on_change
        self._on_remove  = on_remove
        self._param_vars = {}

        self._type_var = tk.StringVar(value=initial_type or LAYER_NAMES[0])
        self._type_var.trace_add("write", lambda *_: self._rebuild_params())

        ttk.Combobox(self, textvariable=self._type_var,
                     values=LAYER_NAMES, state="readonly", width=20).pack(side="left", padx=(0, 6))

        self._param_frame = tk.Frame(self)
        self._param_frame.pack(side="left", fill="x", expand=True)

        tk.Button(self, text="×", fg="red", width=2,
                  relief="flat", command=self._on_remove).pack(side="right", padx=(6, 0))

        self._rebuild_params()

    def _rebuild_params(self):
        for w in self._param_frame.winfo_children():
            w.destroy()
        self._param_vars.clear()

        for p in LAYER_DEFS.get(self._type_var.get(), []):
            tk.Label(self._param_frame, text=p["label"]).pack(side="left", padx=(6, 2))
            var = tk.StringVar(value=str(p["default"]))
            self._param_vars[p["name"]] = var
            if p["type"] == "choice":
                w = ttk.Combobox(self._param_frame, textvariable=var,
                                 values=p["choices"], state="readonly", width=10)
            else:
                w = ttk.Entry(self._param_frame, textvariable=var, width=7)
            w.pack(side="left", padx=2)
            var.trace_add("write", lambda *_: self._on_change())

        self._on_change()

    def get_config(self):
        return {"type": self._type_var.get(),
                "params": {k: v.get() for k, v in self._param_vars.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# Model Maker frame
# ─────────────────────────────────────────────────────────────────────────────

class ModelMakerFrame(RunnerMixin, tk.Frame):
    def __init__(self, parent, app):
        tk.Frame.__init__(self, parent)
        self.app      = app
        self._rows = []
        self._metrics = MetricsCollector()
        self._build()
        for t in ("Input (Flatten 28×28)", "Dense", "Dense", "Dense"):
            self._add_row(initial_type=t)

    def _build(self):
        top = ttk.Frame(self, padding=(12, 8, 12, 0))
        top.pack(fill="x")
        ttk.Button(top, text="← Menu",
                   command=lambda: self.app.show_frame(MainMenuFrame)).pack(side="left")
        tk.Label(top, text="Model Maker",
                 font=("Helvetica", 16, "bold")).pack(side="left", padx=16)

        paned = tk.PanedWindow(self, orient="horizontal",
                               sashrelief="raised", sashwidth=5)
        paned.pack(fill="both", expand=True, padx=12, pady=8)

        # ── Left: layer stack ─────────────────────────────────────────────
        left = ttk.Frame(paned)
        paned.add(left, minsize=360)

        ttk.Label(left, text="Layers",
                  font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(0, 4))

        canvas_outer = tk.Frame(left, relief="sunken", bd=1)
        canvas_outer.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(canvas_outer, highlightthickness=0)
        vsb = ttk.Scrollbar(canvas_outer, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._stack = tk.Frame(self._canvas)
        self._cwin  = self._canvas.create_window((0, 0), window=self._stack, anchor="nw")
        self._stack.bind("<Configure>",
                         lambda _: self._canvas.configure(
                             scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>",
                          lambda e: self._canvas.itemconfig(self._cwin, width=e.width))

        ttk.Button(left, text="+ Add Layer", command=self._add_row).pack(pady=6)

        # ── Right: code preview + log/plot notebook ───────────────────────
        right = ttk.Frame(paned)
        paned.add(right, minsize=340)

        ttk.Label(right, text="Generated Script",
                  font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(0, 4))

        self._code_box = tk.Text(right, wrap="none", font=("Courier", 9),
                                 state="disabled", height=14)
        self._code_box.pack(fill="x")
        xsb = ttk.Scrollbar(right, orient="horizontal", command=self._code_box.xview)
        xsb.pack(fill="x")
        self._code_box.configure(xscrollcommand=xsb.set)

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=(6, 4))
        ttk.Button(btns, text="Run",         command=self._do_run).pack(side="left", padx=5)
        ttk.Button(btns, text="Stop",        command=self._stop).pack(side="left", padx=5)
        ttk.Button(btns, text="Save Script", command=self._save).pack(side="left", padx=5)
        ttk.Button(btns, text="Clear Log",
                   command=lambda: self._log.delete("1.0", tk.END)).pack(side="left", padx=5)

        log = self._make_output_notebook(right)
        self._init_runner(log, self._metrics)

    # ── Layer management ──────────────────────────────────────────────────────

    def _add_row(self, initial_type=None):
        row = LayerRow(self._stack,
                       on_change=self._refresh_code,
                       on_remove=lambda r=None: self._remove_row(row),
                       initial_type=initial_type)
        row.pack(fill="x", pady=2, padx=2)
        self._rows.append(row)
        self._refresh_code()

    def _remove_row(self, row):
        if row in self._rows:
            self._rows.remove(row)
            row.destroy()
            self._refresh_code()

    # ── Code generation ───────────────────────────────────────────────────────

    def _generate(self):
        cfgs     = [r.get_config() for r in self._rows]
        has_conv = any(c["type"] == "Conv2D" for c in cfgs)

        if has_conv:
            reshape = ("x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0\n"
                       "x_test  = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0")
        else:
            reshape = ("x_train = x_train.reshape(-1, 784).astype('float32') / 255.0\n"
                       "x_test  = x_test.reshape(-1, 784).astype('float32') / 255.0")

        layer_lines = "\n".join(
            "    " + self._layer_code(c) + "," for c in cfgs
        ) or "    # no layers defined"

        return (
f"""import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
{reshape}

model = tf.keras.Sequential([
{layer_lines}
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
loss, acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {{acc:.4f}}")
"""
        )

    @staticmethod
    def _layer_code(cfg):
        t, p = cfg["type"], cfg["params"]
        if t == "Dense":
            return f"tf.keras.layers.Dense({p['units']}, activation='{p['activation']}')"
        if t == "Conv2D":
            k = p["kernel_size"]
            return (f"tf.keras.layers.Conv2D({p['filters']}, ({k}, {k}), "
                    f"activation='{p['activation']}')")
        if t == "Flatten":
            return "tf.keras.layers.Flatten()"
        if t == "Input (Flatten 28×28)":
            return "tf.keras.layers.Flatten(input_shape=(28, 28))"
        if t == "Dropout":
            return f"tf.keras.layers.Dropout({p['rate']})"
        if t == "MaxPooling2D":
            s = p["pool_size"]
            return f"tf.keras.layers.MaxPooling2D(({s}, {s}))"
        if t == "BatchNormalization":
            return "tf.keras.layers.BatchNormalization()"
        return f"# unknown layer: {t}"

    def _refresh_code(self):
        code = self._generate()
        self._code_box.configure(state="normal")
        self._code_box.delete("1.0", tk.END)
        self._code_box.insert(tk.END, code)
        self._code_box.configure(state="disabled")

    # ── Run / save ────────────────────────────────────────────────────────────

    def _do_run(self):
        code = self._generate()
        tmp  = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir=REPO_ROOT, prefix="_approxtrain_maker_")
        tmp.write(code)
        tmp.close()
        self._run([sys.executable, tmp.name])

    def _save(self):
        path = filedialog.asksaveasfilename(
            title="Save generated script", defaultextension=".py",
            filetypes=[("Python files", "*.py")], initialdir=REPO_ROOT)
        if path:
            Path(path).write_text(self._generate())


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder frames
# ─────────────────────────────────────────────────────────────────────────────

class CreditsFrame(tk.Frame):
    CREDITS_FILE = os.path.join(os.path.dirname(__file__), "credits.txt")

    def __init__(self, parent, app):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=3)
        ttk.Button(self, text="← Menu",
                   command=lambda: app.show_frame(MainMenuFrame)
                   ).grid(row=0, column=0, sticky="w", padx=12, pady=8)
        tk.Label(self, text="Credits",
                 font=("Helvetica", 24, "bold")).grid(row=1, column=0, pady=(0, 8))

        text = self._load_credits()
        tk.Label(self, text=text, font=("Helvetica", 12),
                 justify="center", wraplength=600).grid(row=2, column=0, padx=20)

    def _load_credits(self):
        try:
            with open(self.CREDITS_FILE, "r") as f:
                return f.read()
        except FileNotFoundError:
            return "(credits.txt not found)"


class OptionsFrame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        ttk.Button(self, text="← Menu",
                   command=lambda: app.show_frame(MainMenuFrame)
                   ).grid(row=0, column=0, sticky="w", padx=12, pady=8)
        tk.Label(self, text="Options",
                 font=("Helvetica", 24, "bold")).grid(row=1, column=0)


if __name__ == "__main__":
    root = tk.Tk()
    app  = ApproxTrainGUI(root)
    root.mainloop()
