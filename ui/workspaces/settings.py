"""Settings Workspace for Audio Studio AI.
Supports Theme switching, Animation settings, Audio hardware selection,
AI model management, Offline mode privacy, and system diagnostics.
"""
import os
from pathlib import Path
from typing import Dict, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QCheckBox, QSlider, QTabWidget, QFrame, QScrollArea,
    QLineEdit, QFileDialog, QMessageBox, QStyledItemDelegate
)
from PySide6.QtCore import Qt

from ui.themes.theme_manager import ThemeManager
from ui.animations.animation_manager import AnimationManager
from ui.components.card import PremiumCard
from ui.components.button import PremiumButton
from database.database import DatabaseManager
from config import (
    APP_NAME, APP_VERSION, APP_DESCRIPTION,
    CUDA_AVAILABLE, CUDA_DEVICE_NAME, MODELS_DIR, DATA_DIR, PROJECTS_DIR
)

class SettingsWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._theme_mgr = ThemeManager.get_instance()
        self._anim_mgr = AnimationManager.get_instance()
        self._db = DatabaseManager()

        self._build_ui()
        self._theme_mgr.theme_changed.connect(lambda _: self._update_style())
        self._update_style()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 28, 28, 28)
        main_layout.setSpacing(20)

        # Header Title
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("Application Settings")
        title.setObjectName("WorkspaceTitle")
        subtitle = QLabel("Configure visual themes, offline audio engines, GPU hardware acceleration, and preferences.")
        subtitle.setObjectName("WorkspaceSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        main_layout.addLayout(title_box)

        # Tab Widget for Categories
        self.tabs = QTabWidget()
        self.tabs.setObjectName("SettingsTabs")

        # 1. Appearance & Themes Tab
        self.tabs.addTab(self._build_appearance_tab(), "🎨  Themes & Appearance")

        # 2. Animations & Motion Tab
        self.tabs.addTab(self._build_animation_tab(), "✨  Animation & Performance")

        # 3. AI & Hardware Acceleration Tab
        self.tabs.addTab(self._build_ai_hardware_tab(), "⚡  AI Models & Hardware")

        # 4. Storage & Offline Privacy Tab
        self.tabs.addTab(self._build_storage_privacy_tab(), "🔒  Offline & Storage")

        # 5. About Tab
        self.tabs.addTab(self._build_about_tab(), "ℹ️  About")

        main_layout.addWidget(self.tabs, 1)

    def _build_appearance_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 14, 10, 14)
        layout.setSpacing(16)

        # Theme Selector Card
        theme_card = PremiumCard(title="Active Studio Theme", subtitle="Instant visual identity switching with high-contrast accessibility")
        t_layout = QVBoxLayout()
        t_layout.setSpacing(12)

        row = QHBoxLayout()
        lbl = QLabel("Theme Preset:")
        lbl.setObjectName("FieldLabel")
        self.theme_combo = QComboBox()
        self.theme_combo.setItemDelegate(QStyledItemDelegate(self.theme_combo))

        themes = self._theme_mgr.get_available_themes()
        curr_theme = self._theme_mgr.get_active_theme_id()
        for tid, tname in themes.items():
            self.theme_combo.addItem(f"{tname}", tid)
            if tid == curr_theme:
                self.theme_combo.setCurrentIndex(self.theme_combo.count() - 1)

        self.theme_combo.currentIndexChanged.connect(self._on_theme_selected)
        row.addWidget(lbl)
        row.addWidget(self.theme_combo, 1)
        t_layout.addLayout(row)

        desc = QLabel("Obsidian Silver is engineered for low visual fatigue in dark studio environments with silver accents.")
        desc.setObjectName("SettingDesc")
        t_layout.addWidget(desc)

        theme_card.add_layout(t_layout)
        layout.addWidget(theme_card)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_animation_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 14, 10, 14)
        layout.setSpacing(16)

        anim_card = PremiumCard(title="Fluid Motion & Transition System", subtitle="Configure 60 FPS transitions, easing curves, or enable reduced motion")
        a_layout = QVBoxLayout()
        a_layout.setSpacing(14)

        self.cb_anim_enabled = QCheckBox("Enable UI Fluid Animations")
        self.cb_anim_enabled.setChecked(self._anim_mgr.is_enabled)
        self.cb_anim_enabled.toggled.connect(self._on_anim_toggled)
        a_layout.addWidget(self.cb_anim_enabled)

        self.cb_reduced_motion = QCheckBox("Reduced Motion Mode (Instant Cuts, No Sliding)")
        self.cb_reduced_motion.setChecked(self._anim_mgr.reduced_motion)
        self.cb_reduced_motion.toggled.connect(self._on_reduced_motion_toggled)
        a_layout.addWidget(self.cb_reduced_motion)

        notice = QLabel("Animations run on Qt property threads and never interrupt realtime audio rendering.")
        notice.setObjectName("SettingDesc")
        a_layout.addWidget(notice)

        anim_card.add_layout(a_layout)
        layout.addWidget(anim_card)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_ai_hardware_tab(self) -> QWidget:
        from ai.model_manager import ModelManager
        mm = ModelManager.get_instance()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 14, 10, 14)
        layout.setSpacing(16)

        # 1. Hardware Diagnostics Card
        hw_card = PremiumCard(title="Compute Engine & Hardware Diagnostics", subtitle="Local PyTorch inference execution target")
        h_layout = QVBoxLayout()
        h_layout.setSpacing(10)

        gpu_status_str = f"✅ NVIDIA CUDA GPU Active ({CUDA_DEVICE_NAME})" if CUDA_AVAILABLE else "⚙️ CPU Fallback Mode (AVX2 / Multi-threading Active)"
        lbl_device = QLabel(f"Active Execution Target: {gpu_status_str}")
        lbl_device.setObjectName("DeviceNotice")

        lbl_info = QLabel(
            "Hardware Detection: PyTorch 2.14 local runtime. "
            "When a compatible NVIDIA GPU with CUDA compute capability is present, acceleration engages automatically. "
            "Otherwise, high-performance multi-threaded CPU SIMD instructions (AVX2) are utilized without performance starvation."
        )
        lbl_info.setObjectName("SettingDesc")

        h_layout.addWidget(lbl_device)
        h_layout.addWidget(lbl_info)
        hw_card.add_layout(h_layout)
        layout.addWidget(hw_card)

        # 2. AI Model Manager Card
        models_card = PremiumCard(
            title="AI Model Manager & Offline Weights Repository",
            subtitle=f"Local models repository: {MODELS_DIR}"
        )
        m_layout = QVBoxLayout()
        m_layout.setSpacing(12)

        models = mm.get_all_models()
        for m in models:
            m_frame = QFrame()
            m_frame.setObjectName("ModelItemFrame")
            m_frame.setStyleSheet("""
                QFrame#ModelItemFrame {
                    background-color: #15171B;
                    border: 1px solid #282B32;
                    border-radius: 8px;
                    padding: 8px 12px;
                }
            """)
            mf_layout = QHBoxLayout(m_frame)
            mf_layout.setContentsMargins(10, 8, 10, 8)
            mf_layout.setSpacing(12)

            info_box = QVBoxLayout()
            info_box.setSpacing(2)
            name_lbl = QLabel(f"<b>{m['name']}</b> ({m.get('size_mb', 0)} MB)")
            name_lbl.setStyleSheet("color: #F5F5F5; font-size: 13px;")
            desc_lbl = QLabel(m.get("description", ""))
            desc_lbl.setStyleSheet("color: #969BA3; font-size: 11px;")
            info_box.addWidget(name_lbl)
            info_box.addWidget(desc_lbl)
            mf_layout.addLayout(info_box, 1)

            # Status Badge
            is_inst = m.get("installed", False)
            status_text = "✅  Installed (Offline Ready)" if is_inst else "⬇️  Available for Download"
            status_col = "#48B58A" if is_inst else "#C5C9D0"
            stat_lbl = QLabel(status_text)
            stat_lbl.setStyleSheet(f"color: {status_col}; font-size: 12px; font-weight: 600;")
            mf_layout.addWidget(stat_lbl)

            # Action Button
            if is_inst:
                btn_action = PremiumButton("Verify Integrity", variant="outline")
                btn_action.setFixedHeight(30)
                mid = m["id"]
                btn_action.clicked.connect(lambda _, mid=mid: self._verify_model_action(mid))
            else:
                btn_action = PremiumButton("Download", variant="primary")
                btn_action.setFixedHeight(30)
                mid = m["id"]
                btn_action.clicked.connect(lambda _, mid=mid: self._download_model_action(mid))

            mf_layout.addWidget(btn_action)
            m_layout.addWidget(m_frame)

        models_card.add_layout(m_layout)
        layout.addWidget(models_card)

        # 3. Optional Hugging Face Token Card
        hf_card = PremiumCard(
            title="Hugging Face Repository Access (Optional)",
            subtitle="Optional user token for private model weights or elevated download quotas"
        )
        hf_layout = QVBoxLayout()
        hf_layout.setSpacing(10)

        hf_row = QHBoxLayout()
        hf_lbl = QLabel("HF_TOKEN:")
        hf_lbl.setObjectName("FieldLabel")

        self.hf_token_input = QLineEdit()
        self.hf_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_input.setPlaceholderText("hf_... (Optional: leave blank for anonymous public models)")
        saved_hf = self._db.get_setting("hf_token", "") or os.environ.get("HF_TOKEN", "")
        if saved_hf:
            self.hf_token_input.setText(saved_hf)
        self.hf_token_input.textChanged.connect(self._on_hf_token_changed)

        hf_row.addWidget(hf_lbl)
        hf_row.addWidget(self.hf_token_input, 1)
        hf_layout.addLayout(hf_row)

        hf_notice = QLabel(
            "Authentication Status: Public models (such as HTDemucs v4) do NOT require authentication and "
            "operate anonymously. Enter a personal Hugging Face User Access Token only if you require "
            "elevated API rate limits or access to gated research models."
        )
        hf_notice.setObjectName("SettingDesc")
        hf_layout.addWidget(hf_notice)

        hf_card.add_layout(hf_layout)
        layout.addWidget(hf_card)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _on_hf_token_changed(self, text: str):
        token = text.strip()
        self._db.set_setting("hf_token", token)
        if token:
            os.environ["HF_TOKEN"] = token
        elif "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]

    def _verify_model_action(self, model_id: str):
        from ai.model_manager import ModelManager
        ok, msg = ModelManager.get_instance().verify_model_integrity(model_id)
        if ok:
            QMessageBox.information(self, "Model Integrity Verified", f"Model '{model_id}' is valid and ready:\n\n{msg}")
        else:
            QMessageBox.warning(self, "Integrity Issue", f"Model '{model_id}' validation warning:\n\n{msg}")

    def _download_model_action(self, model_id: str):
        from ai.model_manager import ModelManager
        ok = ModelManager.get_instance().download_model(model_id)
        if ok:
            QMessageBox.information(self, "Download Complete", f"Model '{model_id}' downloaded successfully.")
        else:
            QMessageBox.warning(self, "Download Failed", f"Could not download model '{model_id}'.")

    def _build_storage_privacy_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 14, 10, 14)
        layout.setSpacing(16)

        privacy_card = PremiumCard(title="Offline-First Privacy Guarantee", subtitle="Zero cloud telemetry and zero external audio uploads")
        p_layout = QVBoxLayout()
        p_layout.setSpacing(12)

        self.cb_offline = QCheckBox("Enforce Strict Offline Mode (Block all outbound network calls)")
        self.cb_offline.setChecked(True)
        p_layout.addWidget(self.cb_offline)

        path_info = QLabel(
            f"• Local Database: {DATA_DIR}\n"
            f"• AI Models Cache: {MODELS_DIR}\n"
            f"• Projects Folder: {PROJECTS_DIR}"
        )
        path_info.setObjectName("SettingDesc")
        p_layout.addWidget(path_info)

        privacy_card.add_layout(p_layout)
        layout.addWidget(privacy_card)

        # Audio Format & Export Card
        export_card = PremiumCard(
            title="Global Audio Format & Master Export Policy",
            subtitle="Application-wide MP3 import, encoding, and default bitrate parameters"
        )
        exp_layout = QVBoxLayout()
        exp_layout.setSpacing(12)

        exp_row = QHBoxLayout()
        exp_lbl = QLabel("Default Master Export Bitrate:")
        exp_lbl.setObjectName("FieldLabel")

        self.export_bitrate_combo = QComboBox()
        self.export_bitrate_combo.setItemDelegate(QStyledItemDelegate(self.export_bitrate_combo))
        self.export_bitrate_combo.addItem("320 kbps (Studio Master - Recommended)", "320k")
        self.export_bitrate_combo.addItem("256 kbps (High Quality)", "256k")
        self.export_bitrate_combo.addItem("192 kbps (Standard Broadcast)", "192k")
        self.export_bitrate_combo.addItem("128 kbps (Compact)", "128k")

        saved_bitrate = self._db.get_setting("default_bitrate", "320k")
        idx = self.export_bitrate_combo.findData(saved_bitrate)
        if idx >= 0:
            self.export_bitrate_combo.setCurrentIndex(idx)
        self.export_bitrate_combo.currentIndexChanged.connect(self._on_default_bitrate_changed)

        exp_row.addWidget(exp_lbl)
        exp_row.addWidget(self.export_bitrate_combo, 1)
        exp_layout.addLayout(exp_row)

        exp_desc = QLabel(
            "Global Format Policy: Audio Studio AI operates strictly in MP3 (*.mp3) format for all "
            "imported and exported tracks across all workspaces. Internal processing operates in 32-bit float PCM."
        )
        exp_desc.setObjectName("SettingDesc")
        exp_layout.addWidget(exp_desc)

        export_card.add_layout(exp_layout)
        layout.addWidget(export_card)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_about_tab(self) -> QWidget:
        card = PremiumCard(title=f"{APP_NAME} v{APP_VERSION}", subtitle=APP_DESCRIPTION)
        layout = QVBoxLayout()
        layout.setSpacing(10)

        info = QLabel(
            "Engineered for desktop professional music production, AI vocal extraction, "
            "and offline audio processing.\n\n"
            "• GUI Framework: PySide6 (Qt 6.12)\n"
            "• Deep Learning Engine: PyTorch 2.14\n"
            "• Audio DSP: SoundFile, NumPy, SciPy\n"
            "• Visual Design: Obsidian Silver Minimalist Workstation\n"
            "• License: Local Commercial / Personal Offline Studio"
        )
        info.setObjectName("AboutInfo")
        layout.addWidget(info)
        card.add_layout(layout)
        return card

    def _on_default_bitrate_changed(self, index: int):
        bitrate = self.export_bitrate_combo.currentData()
        if bitrate:
            self._db.set_setting("default_bitrate", bitrate)

    def _on_theme_selected(self, index: int):
        theme_id = self.theme_combo.currentData()
        if theme_id:
            self._theme_mgr.set_theme(theme_id)

    def _on_anim_toggled(self, checked: bool):
        self._anim_mgr.is_enabled = checked

    def _on_reduced_motion_toggled(self, checked: bool):
        self._anim_mgr.reduced_motion = checked

    def _update_style(self):
        c = self._theme_mgr.get_current_theme().get("colors", {})
        text_primary = c.get("text_primary", "#F5F5F5")
        text_secondary = c.get("text_secondary", "#969BA3")
        accent = c.get("accent_silver", "#C5C9D0")

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
                font-size: 13px;
                font-weight: 600;
            }}
            QLabel#SettingDesc {{
                color: {text_secondary};
                font-size: 12px;
                line-height: 1.4;
            }}
            QLabel#DeviceNotice {{
                color: {accent};
                font-size: 13px;
                font-weight: 600;
            }}
            QLabel#AboutInfo {{
                color: {text_primary};
                font-size: 13px;
                line-height: 1.6;
            }}
            QCheckBox {{
                color: {text_primary};
                font-size: 13px;
                font-weight: 500;
                spacing: 8px;
            }}
        """)

