"""Audio Visualizer Workspace for Audio Studio AI.
High-FPS audio visualization workspace featuring Spectrum Bars, Circular Visualizer,
Oscilloscope Waveform, and Frequency Analyser modes with real-time sensitivity controls.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QFrame, QComboBox, QStyledItemDelegate
)
from PySide6.QtCore import Qt

from ui.themes.theme_manager import ThemeManager
from ui.components.card import PremiumCard
from ui.components.button import PremiumButton
from ui.components.visualizer_widget import VisualizerWidget
from core.playback_engine import PlaybackEngine

class AudioVisualizerWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._playback = PlaybackEngine.get_instance()
        self._theme_mgr = ThemeManager.get_instance()

        self._build_ui()
        self._connect_signals()
        self._theme_mgr.theme_changed.connect(lambda _: self._update_style())
        self._update_style()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 28, 28, 28)
        main_layout.setSpacing(16)

        # Header Title & Mode Buttons
        header_box = QHBoxLayout()
        header_box.setSpacing(16)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("Audio Visualizer")
        title.setObjectName("WorkspaceTitle")
        subtitle = QLabel("Real-time GPU-accelerated spectral and waveform animation synchronized to playback.")
        subtitle.setObjectName("WorkspaceSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_box.addLayout(title_box, 1)

        # Mode Selection Buttons
        self.btn_mode_spectrum = PremiumButton("📊 Spectrum Bars", variant="primary")
        self.btn_mode_spectrum.clicked.connect(lambda: self._set_mode("spectrum", self.btn_mode_spectrum))

        self.btn_mode_circular = PremiumButton("⭕ Circular Visualizer", variant="secondary")
        self.btn_mode_circular.clicked.connect(lambda: self._set_mode("circular", self.btn_mode_circular))

        self.btn_mode_osc = PremiumButton("〰️ Oscilloscope", variant="secondary")
        self.btn_mode_osc.clicked.connect(lambda: self._set_mode("oscilloscope", self.btn_mode_osc))

        self.btn_mode_freq = PremiumButton("📈 Frequency Curve", variant="secondary")
        self.btn_mode_freq.clicked.connect(lambda: self._set_mode("frequency", self.btn_mode_freq))

        header_box.addWidget(self.btn_mode_spectrum)
        header_box.addWidget(self.btn_mode_circular)
        header_box.addWidget(self.btn_mode_osc)
        header_box.addWidget(self.btn_mode_freq)
        main_layout.addLayout(header_box)

        # Main Visualizer Card
        self.vis_card = PremiumCard()
        vis_layout = QVBoxLayout()
        vis_layout.setContentsMargins(0, 0, 0, 0)

        self.visualizer = VisualizerWidget(mode="spectrum")
        self.visualizer.setMinimumHeight(400)
        self.visualizer.set_active(True)  # ambient animation active

        vis_layout.addWidget(self.visualizer)
        self.vis_card.add_layout(vis_layout)
        main_layout.addWidget(self.vis_card, 1)

        # Controls & Customization Toolbar
        ctrl_card = PremiumCard()
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(20)

        # Sensitivity Slider
        sens_box = QHBoxLayout()
        sens_lbl = QLabel("Sensitivity:")
        sens_lbl.setObjectName("FieldLabel")
        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(5, 25)
        self.sens_slider.setValue(12)
        self.sens_slider.setFixedWidth(120)
        self.sens_slider.valueChanged.connect(lambda val: self.visualizer.set_sensitivity(val / 10.0))
        sens_box.addWidget(sens_lbl)
        sens_box.addWidget(self.sens_slider)
        ctrl_layout.addLayout(sens_box)

        # Frequency Focus
        focus_box = QHBoxLayout()
        focus_lbl = QLabel("Focus Range:")
        focus_lbl.setObjectName("FieldLabel")
        self.focus_combo = QComboBox()
        self.focus_combo.setItemDelegate(QStyledItemDelegate(self.focus_combo))
        self.focus_combo.addItems(["Full Spectrum (20Hz - 20kHz)", "Sub & Bass (20Hz - 250Hz)", "Mid & Vocals (250Hz - 4kHz)", "High Presence (4kHz - 20kHz)"])
        focus_box.addWidget(focus_lbl)
        focus_box.addWidget(self.focus_combo)
        ctrl_layout.addLayout(focus_box)

        ctrl_layout.addStretch()

        # Freeze / Active Toggle
        self.btn_freeze = PremiumButton("⏸ Freeze Animation", variant="outline")
        self.btn_freeze.clicked.connect(self._toggle_freeze)
        ctrl_layout.addWidget(self.btn_freeze)

        ctrl_card.add_layout(ctrl_layout)
        main_layout.addWidget(ctrl_card)

    def _set_mode(self, mode: str, active_btn: PremiumButton):
        self.visualizer.set_mode(mode)
        for btn in [self.btn_mode_spectrum, self.btn_mode_circular, self.btn_mode_osc, self.btn_mode_freq]:
            # Reset button variant
            btn._variant = "primary" if btn == active_btn else "secondary"
            btn._update_style()

    def _toggle_freeze(self):
        new_state = not self.visualizer._is_active
        self.visualizer.set_active(new_state)
        self.btn_freeze.setText("▶ Resume Visualizer" if not new_state else "⏸ Freeze Animation")

    def _connect_signals(self):
        self._playback.state_changed.connect(lambda is_playing: self.visualizer.set_active(True))

    def _update_style(self):
        c = self._theme_mgr.get_current_theme().get("colors", {})
        text_primary = c.get("text_primary", "#F5F5F5")
        text_secondary = c.get("text_secondary", "#969BA3")

        self.setStyleSheet(f"""
            QLabel#WorkspaceTitle {{
                color: {text_primary};
                font-size: 22px;
                font-weight: 700;
            }}
            QLabel#WorkspaceSubtitle {{
                color: {text_secondary};
                font-size: 13px;
            }}
            QLabel#FieldLabel {{
                color: {text_secondary};
                font-size: 12px;
                font-weight: 600;
            }}
        """)

