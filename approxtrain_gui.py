import sys
import queue
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

REPO_ROOT = Path(__file__).resolve().parent

class ApproxTrainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ApproxTrain GUI")
        self.root.geometry("900x600")

        self.output_queue = queue.Queue()
        self.process = None

        self.script_var = tk.StringVar(value="lenet300100.py")
        self.mul_var = tk.StringVar(value=str(REPO_ROOT / "lut" / "MBM_7.bin"))
        self.approx_var = tk.BooleanVar(value=True)

        self.build_ui()
        self.poll_output()

    def build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(main, text="Run Settings", padding=10)
        controls.pack(fill="x")

        ttk.Label(controls, text="Script").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        script_box = ttk.Combobox(
            controls,
            textvariable=self.script_var,
            values=["lenet300100.py", "mnist_example.py"],
            state="readonly",
            width=30
        )
        script_box.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(controls, text="Multiplier LUT").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(controls, textvariable=self.mul_var, width=60).grid(
            row=1, column=1, sticky="ew", padx=5, pady=5
        )
        ttk.Button(controls, text="Browse", command=self.browse_mul).grid(
            row=1, column=2, padx=5, pady=5
        )

        ttk.Checkbutton(
            controls,
            text="Use approximate mode",
            variable=self.approx_var
        ).grid(row=2, column=1, sticky="w", padx=5, pady=5)

        controls.columnconfigure(1, weight=1)

        buttons = ttk.Frame(main)
        buttons.pack(fill="x", pady=(10, 10))

        ttk.Button(buttons, text="Run", command=self.run_script).pack(side="left", padx=5)
        ttk.Button(buttons, text="Stop", command=self.stop_script).pack(side="left", padx=5)
        ttk.Button(buttons, text="Clear Log", command=self.clear_log).pack(side="left", padx=5)

        log_frame = ttk.LabelFrame(main, text="Output", padding=8)
        log_frame.pack(fill="both", expand=True)

        self.log = tk.Text(log_frame, wrap="word", height=25)
        self.log.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        scrollbar.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scrollbar.set)

    def browse_mul(self):
        path = filedialog.askopenfilename(
            title="Select LUT file",
            filetypes=[("Binary files", "*.bin"), ("All files", "*.*")]
        )
        if path:
            self.mul_var.set(path)

    def build_command(self):
        script = self.script_var.get()
        script_path = REPO_ROOT / script

        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        cmd = [sys.executable, str(script_path)]

        # Best with patched lenet300100.py using action='store_true'
        if script == "lenet300100.py":
            cmd += ["--mul", self.mul_var.get()]
            if self.approx_var.get():
                cmd += ["--approx"]

        return cmd

    def run_script(self):
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("Already running", "A process is already running.")
            return

        try:
            cmd = self.build_command()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.write_log(f"\n$ {' '.join(cmd)}\n")

        thread = threading.Thread(target=self._run_in_thread, args=(cmd,), daemon=True)
        thread.start()

    def _run_in_thread(self, cmd):
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.output_queue.put(line)

            return_code = self.process.wait()
            self.output_queue.put(f"\n[Process exited with code {return_code}]\n")
        except Exception as e:
            self.output_queue.put(f"\n[Error: {e}]\n")
        finally:
            self.process = None

    def stop_script(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.write_log("\n[Stop requested]\n")

    def clear_log(self):
        self.log.delete("1.0", tk.END)

    def write_log(self, text):
        self.log.insert(tk.END, text)
        self.log.see(tk.END)

    def poll_output(self):
        while not self.output_queue.empty():
            self.write_log(self.output_queue.get())
        self.root.after(100, self.poll_output)

if __name__ == "__main__":
    root = tk.Tk()
    app = ApproxTrainGUI(root)
    root.mainloop()
