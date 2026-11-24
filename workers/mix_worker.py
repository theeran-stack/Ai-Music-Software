"""Background QThread Worker for Smart Mix Studio in Audio Studio AI.
Handles asynchronous acoustic analysis, AI transition planning, preview generation,
and master continuous DJ set rendering.
"""
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from PySide6.QtCore import QThread, Signal

from ai.mixing.mix_planner import MixPlanner
from ai.mixing.mix_plan import MixPlan
from ai.mixing.mix_renderer import MixRenderer

class SmartMixWorker(QThread):
    progress_changed = Signal(float, str)   # (progress_ratio, stage_description)
    plan_completed = Signal(object)         # MixPlan object
    preview_ready = Signal(str, dict)       # (preview_path, result_dict)
    render_completed = Signal(dict)         # Result dict
    failed = Signal(str)                   # Error message
    cancelled = Signal()                   # Cancelled signal

    def __init__(
        self,
        mode: str,                          # "plan", "preview", or "render"
        tracks: Optional[List[Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        plan: Optional[MixPlan] = None,
        output_path: Optional[str] = None,
        export_bitrate: str = "320k",
        parent=None
    ):
        super().__init__(parent)
        self._mode = mode
        self._tracks = tracks or []
        self._settings = settings or {}
        self._plan = plan
        self._output_path = output_path
        self._export_bitrate = export_bitrate

        self._cancel_token = threading.Event()

    def cancel(self):
        self._cancel_token.set()

    def run(self):
        try:
            def _progress_cb(pct: float, stage: str):
                self.progress_changed.emit(pct, stage)

            if self._mode == "plan":
                plan = MixPlanner.plan_mix(
                    tracks=self._tracks,
                    settings=self._settings,
                    progress_callback=_progress_cb
                )
                self.plan_completed.emit(plan)

            elif self._mode == "preview":
                if not self._plan or not self._output_path:
                    raise ValueError("MixPlan and output_path required for preview mode.")
                result = MixRenderer.render_mix(
                    plan=self._plan,
                    output_path=self._output_path,
                    export_bitrate=self._export_bitrate,
                    cancel_token=self._cancel_token,
                    progress_cb=_progress_cb
                )
                self.preview_ready.emit(result["output_path"], result)

            elif self._mode == "render":
                if not self._plan or not self._output_path:
                    raise ValueError("MixPlan and output_path required for render mode.")
                result = MixRenderer.render_mix(
                    plan=self._plan,
                    output_path=self._output_path,
                    export_bitrate=self._export_bitrate,
                    cancel_token=self._cancel_token,
                    progress_cb=_progress_cb
                )
                self.render_completed.emit(result)

        except InterruptedError:
            self.cancelled.emit()
        except Exception as e:
            self.failed.emit(str(e))
