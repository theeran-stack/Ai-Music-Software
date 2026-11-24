"""Theme Manager for Audio Studio AI.
Provides runtime theme switching, stylesheet generation (QSS), custom theme persistence,
and color tokens for custom-painted widgets.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor

from config import THEMES_DIR, DEFAULT_THEME
from database.database import DatabaseManager

class ThemeManager(QObject):
    theme_changed = Signal(str)  # theme_id

    _instance = None

    @classmethod
    def get_instance(cls) -> "ThemeManager":
        if cls._instance is None:
            cls._instance = ThemeManager()
        return cls._instance

    def __init__(self, parent=None):
        super().__init__(parent)
        self._themes_dir = THEMES_DIR
        self._themes: Dict[str, Dict[str, Any]] = {}
        self._active_theme_id: str = DEFAULT_THEME
        self._current_theme_data: Dict[str, Any] = {}
        self._load_all_themes()

        # Restore saved theme from database if available
        saved_theme = DatabaseManager().get_setting("active_theme", DEFAULT_THEME)
        if saved_theme in self._themes:
            self._active_theme_id = saved_theme
        self._current_theme_data = self._themes.get(self._active_theme_id, self._themes[DEFAULT_THEME])

    def _load_all_themes(self):
        for json_file in self._themes_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    theme_id = data.get("id", json_file.stem)
                    self._themes[theme_id] = data
            except Exception as e:
                print(f"Error loading theme {json_file}: {e}")

    def get_available_themes(self) -> Dict[str, str]:
        """Returns dict of theme_id -> theme_name"""
        return {tid: tdata.get("name", tid) for tid, tdata in self._themes.items()}

    def get_active_theme_id(self) -> str:
        return self._active_theme_id

    def get_current_theme(self) -> Dict[str, Any]:
        return self._current_theme_data

    def get_color(self, key: str, fallback: str = "#FFFFFF") -> str:
        return self._current_theme_data.get("colors", {}).get(key, fallback)

    def get_qcolor(self, key: str, fallback: str = "#FFFFFF") -> QColor:
        color_str = self.get_color(key, fallback)
        if color_str.startswith("rgba"):
            # Parse rgba(r, g, b, a)
            parts = color_str.replace("rgba(", "").replace(")", "").split(",")
            r, g, b = int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip())
            a = float(parts[3].strip())
            return QColor(r, g, b, int(a * 255))
        return QColor(color_str)

    def set_theme(self, theme_id: str) -> bool:
        if theme_id not in self._themes:
            return False
        self._active_theme_id = theme_id
        self._current_theme_data = self._themes[theme_id]
        DatabaseManager().set_setting("active_theme", theme_id)
        self.theme_changed.emit(theme_id)
        return True

    def save_custom_theme(self, theme_id: str, name: str, colors: Dict[str, str]) -> bool:
        theme_data = {
            "name": name,
            "id": theme_id,
            "type": "custom",
            "colors": colors,
            "metrics": self._current_theme_data.get("metrics", {})
        }
        target_path = self._themes_dir / f"{theme_id}.json"
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(theme_data, f, indent=2)
            self._themes[theme_id] = theme_data
            return True
        except Exception as e:
            print(f"Error saving theme {theme_id}: {e}")
            return False

    def generate_qss(self) -> str:
        c = self._current_theme_data.get("colors", {})
        m = self._current_theme_data.get("metrics", {})

        bg_main = c.get("background_main", "#101114")
        bg_sidebar = c.get("background_sidebar", "#151619")
        bg_card = c.get("background_card", "#1B1D21")
        bg_elevated = c.get("background_elevated", "#22252A")
        bg_input = c.get("background_input", "#181A1E")
        bg_hover = c.get("background_hover", "#26292F")
        border_subtle = c.get("border_subtle", "#25282F")
        border_default = c.get("border_default", "#30333A")
        border_focus = c.get("border_focus", "#C5C9D0")
        text_primary = c.get("text_primary", "#F5F5F5")
        text_secondary = c.get("text_secondary", "#969BA3")
        text_muted = c.get("text_muted", "#60646D")
        accent_silver = c.get("accent_silver", "#C5C9D0")
        accent_hover = c.get("accent_hover", "#D8DCE3")
        accent_active = c.get("accent_active", "#B0B4BC")
        player_bg = c.get("player_bg", "#141518")

        radius_sm = m.get("border_radius_sm", 6)
        radius_md = m.get("border_radius_md", 10)
        radius_card = m.get("border_radius_card", 12)

        return f"""
            /* Global Application Styling */
            * {{
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Inter', Roboto, sans-serif;
                outline: none;
            }}

            QMainWindow, QWidget#CentralWidget {{
                background-color: {bg_main};
                color: {text_primary};
            }}

            QWidget#Sidebar {{
                background-color: {bg_sidebar};
                border-right: 1px solid {border_subtle};
            }}

            QWidget#TopBar {{
                background-color: {bg_main};
                border-bottom: 1px solid {border_subtle};
            }}

            QWidget#BottomPlayer {{
                background-color: {player_bg};
                border-top: 1px solid {border_default};
            }}

            /* Scrollbars - Luxury minimal design */
            QScrollBar:vertical {{
                background: transparent;
                width: 8px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {border_default};
                border-radius: 4px;
                min-height: 28px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {accent_silver};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background: transparent;
                height: 8px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background: {border_default};
                border-radius: 4px;
                min-width: 28px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {accent_silver};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}

            /* Inputs & LineEdits */
            QLineEdit {{
                background-color: {bg_input};
                color: {text_primary};
                border: 1px solid {border_default};
                border-radius: {radius_sm}px;
                padding: 8px 12px;
                font-size: 13px;
                selection-background-color: {accent_silver};
                selection-color: {bg_main};
            }}
            QLineEdit:focus {{
                border: 1px solid {border_focus};
                background-color: {bg_elevated};
            }}
            QLineEdit:hover:!focus {{
                border: 1px solid {accent_active};
            }}

            /* ComboBox */
            QComboBox {{
                background-color: {bg_input};
                color: {text_primary};
                border: 1px solid {border_default};
                border-radius: {radius_sm}px;
                padding: 6px 12px;
                font-size: 9.5pt;
                min-height: 20px;
            }}
            QComboBox:hover {{
                border: 1px solid {accent_active};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 24px;
                border-left-width: 0px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {bg_elevated};
                color: {text_primary};
                border: 1px solid {border_default};
                border-radius: {radius_sm}px;
                selection-background-color: {bg_hover};
                selection-color: {accent_silver};
                padding: 4px;
                font-size: 9.5pt;
            }}

            /* Context Menus & Dropdown Menus */
            QMenu {{
                background-color: {bg_elevated};
                color: {text_primary};
                border: 1px solid {border_default};
                border-radius: {radius_sm}px;
                padding: 4px;
                font-size: 9.5pt;
            }}
            QMenu::item {{
                background: transparent;
                color: {text_primary};
                padding: 6px 16px;
                border-radius: 4px;
                font-size: 9.5pt;
            }}
            QMenu::item:selected {{
                background-color: {bg_hover};
                color: {accent_silver};
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {border_default};
                margin: 4px 8px;
            }}

            /* Sliders */
            QSlider::groove:horizontal {{
                height: 4px;
                background: {border_default};
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {accent_silver};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {text_primary};
                border: 1px solid {border_default};
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #FFFFFF;
                border: 1px solid {accent_silver};
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}

            /* Tooltips */
            QToolTip {{
                background-color: {bg_elevated};
                color: {text_primary};
                border: 1px solid {border_default};
                border-radius: 4px;
                padding: 5px 8px;
                font-size: 12px;
            }}

            /* Tab Bar */
            QTabWidget::pane {{
                border: 1px solid {border_subtle};
                background: {bg_main};
                border-radius: {radius_md}px;
            }}
            QTabBar::tab {{
                background: {bg_card};
                color: {text_secondary};
                border: 1px solid {border_subtle};
                padding: 8px 16px;
                border-top-left-radius: {radius_sm}px;
                border-top-right-radius: {radius_sm}px;
                margin-right: 4px;
            }}
            QTabBar::tab:selected {{
                background: {bg_elevated};
                color: {text_primary};
                border-bottom: 2px solid {accent_silver};
            }}
            QTabBar::tab:hover:!selected {{
                background: {bg_hover};
                color: {text_primary};
            }}
        """
