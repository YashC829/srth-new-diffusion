from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

import rospy

log = logging.getLogger(__name__)


class InferenceGuiRuntime(Protocol):
    _shutdown_event: threading.Event
    _control_update_event: threading.Event
    _inference_thread: threading.Thread | None

    def get_runtime_controls(self) -> tuple[float, float, float, float, str]:
        ...

    def update_runtime_controls(
        self,
        prediction_frequency_hz=None,
        action_execution_hz=None,
        command=None,
        request_reprediction: bool = True,
    ) -> None:
        ...

    def _format_runtime_controls(self) -> str:
        ...

    def _run_inference_loop(self) -> None:
        ...

    def stop_action_execution(self) -> None:
        ...

    def is_paused(self) -> bool:
        ...

    def is_manual_pause_enabled(self) -> bool:
        ...

    def set_manual_pause(self, paused: bool) -> None:
        ...

    def get_pause_status_text(self) -> str:
        ...

    def get_available_commands(self) -> list[str]:
        ...


def _close_gui_window(root) -> None:
    try:
        if root.winfo_exists():
            root.quit()
            root.destroy()
    except Exception:
        pass


def run_inference_gui(runtime: InferenceGuiRuntime) -> bool:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        log.warning("Tkinter is unavailable; continuing inference without GUI: %s", exc)
        return False

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        log.warning("Could not start inference GUI; continuing without it: %s", exc)
        return False

    root.title("Low-Level Policy Inference")
    root.resizable(False, False)

    prediction_frequency_hz, _, action_execution_hz, _, command = runtime.get_runtime_controls()
    available_commands = runtime.get_available_commands()
    prediction_frequency_var = tk.StringVar(value=f"{prediction_frequency_hz:.3f}")
    action_execution_hz_var = tk.StringVar(value=f"{action_execution_hz:.3f}")
    command_var = tk.StringVar(value=command)
    status_var = tk.StringVar(
        value=(
            "Inference starts paused. Click Resume Inference to begin."
            if runtime.is_manual_pause_enabled()
            else "Apply changes to stop the current rollout and re-predict immediately."
        )
    )
    runtime_controls_var = tk.StringVar(value=runtime._format_runtime_controls())
    robot_state_var = tk.StringVar(value=runtime.get_pause_status_text())
    pause_button_text = tk.StringVar(
        value="Resume Inference" if runtime.is_manual_pause_enabled() else "Pause Inference"
    )

    def apply_controls():
        try:
            runtime.update_runtime_controls(
                prediction_frequency_hz=prediction_frequency_var.get(),
                action_execution_hz=action_execution_hz_var.get(),
                command=command_var.get(),
            )
        except ValueError as exc:
            status_var.set(str(exc))
            return

        runtime_controls_var.set(runtime._format_runtime_controls())
        status_var.set(f"Applied at {time.strftime('%H:%M:%S')}")

    def poll_runtime_state():
        if runtime._shutdown_event.is_set() or rospy.is_shutdown():
            _close_gui_window(root)
            return

        runtime_controls_var.set(runtime._format_runtime_controls())
        robot_state_var.set(runtime.get_pause_status_text())
        pause_button_text.set(
            "Resume Inference" if runtime.is_manual_pause_enabled() else "Pause Inference"
        )
        root.after(200, poll_runtime_state)

    def toggle_pause():
        new_pause_state = not runtime.is_manual_pause_enabled()
        runtime.set_manual_pause(new_pause_state)
        robot_state_var.set(runtime.get_pause_status_text())
        pause_button_text.set(
            "Resume Inference" if runtime.is_manual_pause_enabled() else "Pause Inference"
        )
        status_var.set(
            f"{'Paused' if new_pause_state else 'Resumed'} inference at {time.strftime('%H:%M:%S')}"
        )

    def handle_close():
        status_var.set("Shutting down inference...")
        runtime._shutdown_event.set()
        runtime._control_update_event.set()
        runtime.stop_action_execution()
        _close_gui_window(root)

    container = ttk.Frame(root, padding=12)
    container.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    container.columnconfigure(1, weight=1)

    ttk.Label(container, text="Prediction frequency (Hz)").grid(
        row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 6)
    )
    ttk.Entry(container, textvariable=prediction_frequency_var, width=18).grid(
        row=0, column=1, sticky="ew", pady=(0, 6)
    )

    ttk.Label(container, text="Action execution (Hz)").grid(
        row=1, column=0, sticky="w", padx=(0, 8), pady=(0, 6)
    )
    ttk.Entry(container, textvariable=action_execution_hz_var, width=18).grid(
        row=1, column=1, sticky="ew", pady=(0, 6)
    )

    ttk.Label(container, text="Command").grid(
        row=2, column=0, sticky="w", padx=(0, 8), pady=(0, 6)
    )
    command_selector = ttk.Combobox(
        container,
        textvariable=command_var,
        values=available_commands,
        width=30,
        state="normal",
    )
    command_selector.grid(row=2, column=1, sticky="ew", pady=(0, 6))

    ttk.Button(container, text="Apply", command=apply_controls).grid(
        row=3, column=0, columnspan=2, sticky="ew", pady=(4, 10)
    )

    ttk.Button(
        container,
        textvariable=pause_button_text,
        command=toggle_pause,
    ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 10))

    ttk.Label(container, text="Runtime controls").grid(
        row=5, column=0, sticky="nw", padx=(0, 8)
    )
    ttk.Label(
        container,
        textvariable=runtime_controls_var,
        wraplength=320,
        justify="left",
    ).grid(row=5, column=1, sticky="w")

    ttk.Label(container, text="Robot state").grid(
        row=6, column=0, sticky="w", padx=(0, 8), pady=(8, 0)
    )
    ttk.Label(container, textvariable=robot_state_var).grid(
        row=6, column=1, sticky="w", pady=(8, 0)
    )

    ttk.Label(
        container,
        textvariable=status_var,
        wraplength=420,
        justify="left",
    ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(10, 0))

    root.protocol("WM_DELETE_WINDOW", handle_close)

    runtime._inference_thread = threading.Thread(
        target=runtime._run_inference_loop,
        daemon=True,
    )
    runtime._inference_thread.start()
    root.after(200, poll_runtime_state)

    try:
        root.mainloop()
    finally:
        runtime._shutdown_event.set()
        runtime._control_update_event.set()
        runtime.stop_action_execution()
        if runtime._inference_thread is not None and runtime._inference_thread.is_alive():
            runtime._inference_thread.join()
        runtime._inference_thread = None

    return True
