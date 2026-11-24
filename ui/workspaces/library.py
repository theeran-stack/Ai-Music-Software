"""Music Library Workspace for Audio Studio AI.
Supports folder selection, local scanning, metadata indexing, search, favorites,
and direct playback.
"""
import os
import subprocess
from pathlib import Path
from typing import List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QMessageBox, QFrame, QMenu
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor, QAction

from ui.themes.theme_manager import ThemeManager
from ui.components.card import PremiumCard
from ui.components.button import PremiumButton
from database.database import DatabaseManager
from database.models import Track
from core.playback_engine import PlaybackEngine
from config import SUPPORTED_AUDIO_EXTENSIONS, AUDIO_FILE_FILTER
from core.audio_format import validate_mp3_file

class LibraryWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._db = DatabaseManager()
        self._playback = PlaybackEngine.get_instance()
        self._theme_mgr = ThemeManager.get_instance()
        self._tracks: List[Track] = []
        self._filter_favorites_only = False

        self._build_ui()
        self.load_tracks()
        self._theme_mgr.theme_changed.connect(lambda _: self._update_style())
        self._update_style()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 28, 28, 28)
        main_layout.setSpacing(20)

        # Header Bar
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("Music Library")
        title.setObjectName("WorkspaceTitle")
        subtitle = QLabel("Local offline collection with instant metadata search and playback.")
        subtitle.setObjectName("WorkspaceSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box, 1)

        # Action Buttons
        self.btn_import = PremiumButton("➕  Import MP3", variant="secondary")
        self.btn_import.clicked.connect(self._import_mp3_dialog)

        self.btn_scan = PremiumButton("📁  Scan MP3 Folder", variant="primary")
        self.btn_scan.clicked.connect(self._scan_folder_dialog)

        self.btn_fav_filter = PremiumButton("★  Favorites", variant="outline")
        self.btn_fav_filter.clicked.connect(self._toggle_fav_filter)

        header_layout.addWidget(self.btn_fav_filter)
        header_layout.addWidget(self.btn_import)
        header_layout.addWidget(self.btn_scan)
        main_layout.addLayout(header_layout)

        # Search Bar Row
        search_card = PremiumCard()
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.setSpacing(12)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍  Search by title, artist, album, or format...")
        self.search_input.textChanged.connect(self._on_search_changed)
        search_layout.addWidget(self.search_input, 1)

        self.track_count_label = QLabel("0 tracks")
        self.track_count_label.setObjectName("TrackCountLabel")
        search_layout.addWidget(self.track_count_label)

        search_card.add_layout(search_layout)
        main_layout.addWidget(search_card)

        # Tracks Table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Title", "Artist", "Album", "Duration", "Format", "Fav", "Actions"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setShowGrid(False)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.table.cellDoubleClicked.connect(self._on_row_double_clicked)

        main_layout.addWidget(self.table, 1)

    def load_tracks(self):
        if self._filter_favorites_only:
            self._tracks = self._db.get_favorite_tracks()
        else:
            self._tracks = self._db.get_all_tracks()
        self._populate_table(self._tracks)

    def _populate_table(self, tracks: List[Track]):
        self.table.setRowCount(len(tracks))
        self.track_count_label.setText(f"{len(tracks)} tracks")

        for row, track in enumerate(tracks):
            # Title
            item_title = QTableWidgetItem(track.title or Path(track.file_path).stem)
            item_title.setToolTip(track.file_path)

            # Artist
            item_artist = QTableWidgetItem(track.artist or "Unknown")

            # Album
            item_album = QTableWidgetItem(track.album or "Unknown")

            # Duration
            mins = int(track.duration_seconds // 60)
            secs = int(track.duration_seconds % 60)
            item_dur = QTableWidgetItem(f"{mins:02d}:{secs:02d}")
            item_dur.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # Format
            fmt = track.format.upper().replace(".", "") if track.format else "AUDIO"
            item_fmt = QTableWidgetItem(fmt)
            item_fmt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # Fav item
            fav_str = "★" if track.is_favorite else "☆"
            item_fav = QTableWidgetItem(fav_str)
            item_fav.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # Action widget (Play button)
            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(4, 2, 4, 2)
            action_layout.setSpacing(6)

            btn_play = PremiumButton("▶ Play", variant="secondary")
            btn_play.setFixedHeight(26)
            btn_play.clicked.connect(lambda checked=False, t=track: self._playback.play_track(t))
            action_layout.addWidget(btn_play)

            self.table.setItem(row, 0, item_title)
            self.table.setItem(row, 1, item_artist)
            self.table.setItem(row, 2, item_album)
            self.table.setItem(row, 3, item_dur)
            self.table.setItem(row, 4, item_fmt)
            self.table.setItem(row, 5, item_fav)
            self.table.setCellWidget(row, 6, action_widget)

    def _on_search_changed(self, text: str):
        query = text.strip()
        if not query:
            self._populate_table(self._tracks)
        else:
            filtered = [
                t for t in self._tracks
                if query.lower() in t.title.lower() or
                   query.lower() in t.artist.lower() or
                   query.lower() in t.album.lower() or
                   query.lower() in t.format.lower()
            ]
            self._populate_table(filtered)

    def _toggle_fav_filter(self):
        self._filter_favorites_only = not self._filter_favorites_only
        self.btn_fav_filter.setText("★ Showing Favorites" if self._filter_favorites_only else "★  Favorites")
        self.load_tracks()

    def _on_row_double_clicked(self, row: int, col: int):
        if 0 <= row < len(self._tracks):
            track = self._tracks[row]
            self._playback.play_track(track)

    def _show_context_menu(self, pos):
        row = self.table.rowAt(pos.y())
        if 0 <= row < len(self._tracks):
            track = self._tracks[row]
            menu = QMenu(self)

            act_play = QAction("▶ Play Track", self)
            act_play.triggered.connect(lambda: self._playback.play_track(track))
            menu.addAction(act_play)

            fav_label = "Remove from Favorites" if track.is_favorite else "Add to Favorites"
            act_fav = QAction(fav_label, self)
            act_fav.triggered.connect(lambda: self._toggle_favorite(track))
            menu.addAction(act_fav)

            menu.addSeparator()
            act_open_dir = QAction("📂 Open Containing Folder", self)
            act_open_dir.triggered.connect(lambda: self._open_file_location(track.file_path))
            menu.addAction(act_open_dir)

            menu.exec(self.table.viewport().mapToGlobal(pos))

    def _toggle_favorite(self, track: Track):
        if track.id:
            new_state = self._db.toggle_favorite(track.id)
            track.is_favorite = new_state
            self.load_tracks()

    def _open_file_location(self, file_path_str: str):
        p = Path(file_path_str)
        if p.exists():
            subprocess.run(["explorer", f"/select,{p.resolve()}"])

    def _import_mp3_dialog(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Import MP3 Music Track",
            "",
            AUDIO_FILE_FILTER
        )
        if fpath:
            if not fpath.lower().endswith(".mp3"):
                QMessageBox.warning(self, "Invalid Format", "Only MP3 audio files are supported.")
                return
            valid, err = validate_mp3_file(fpath)
            if not valid:
                QMessageBox.warning(self, "Invalid Format", f"Only MP3 audio files are supported.\n\n{err}")
                return

            item = Path(fpath)
            stat = item.stat()
            artist = "Unknown Artist"
            album = "Unknown Album"
            title = item.stem
            duration = 0.0
            try:
                import mutagen
                meta = mutagen.File(str(item))
                if meta and meta.info:
                    duration = float(meta.info.length)
                if meta and meta.tags:
                    if "title" in meta.tags:
                        title = str(meta.tags["title"][0])
                    elif "TIT2" in meta.tags:
                        title = str(meta.tags["TIT2"])
                    if "artist" in meta.tags:
                        artist = str(meta.tags["artist"][0])
                    elif "TPE1" in meta.tags:
                        artist = str(meta.tags["TPE1"])
                    if "album" in meta.tags:
                        album = str(meta.tags["album"][0])
                    elif "TALB" in meta.tags:
                        album = str(meta.tags["TALB"])
            except Exception:
                pass

            track = Track(
                file_path=str(item.resolve()),
                title=title,
                artist=artist,
                album=album,
                duration_seconds=duration,
                format=item.suffix.lower(),
                file_size_bytes=stat.st_size,
                file_modified_time=stat.st_mtime
            )
            self._db.add_or_update_track(track)
            self.load_tracks()

    def _scan_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Music Folder to Scan")
        if not folder:
            return

        self._db.add_library_folder(folder)
        scanned_count = self._scan_directory(Path(folder))
        self.load_tracks()
        QMessageBox.information(
            self,
            "Scan Complete",
            f"Successfully scanned folder.\nIndexed {scanned_count} audio files."
        )

    def _scan_directory(self, folder_path: Path) -> int:
        count = 0
        try:
            for item in folder_path.rglob("*"):
                if item.is_file() and item.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                    stat = item.stat()
                    # Try basic metadata with mutagen fallback
                    artist = "Unknown Artist"
                    album = "Unknown Album"
                    title = item.stem
                    duration = 0.0

                    try:
                        import mutagen
                        meta = mutagen.File(str(item))
                        if meta and meta.info:
                            duration = float(meta.info.length)
                        if meta and meta.tags:
                            if "title" in meta.tags:
                                title = str(meta.tags["title"][0])
                            elif "TIT2" in meta.tags:
                                title = str(meta.tags["TIT2"])
                            if "artist" in meta.tags:
                                artist = str(meta.tags["artist"][0])
                            elif "TPE1" in meta.tags:
                                artist = str(meta.tags["TPE1"])
                            if "album" in meta.tags:
                                album = str(meta.tags["album"][0])
                            elif "TALB" in meta.tags:
                                album = str(meta.tags["TALB"])
                    except Exception:
                        pass

                    track = Track(
                        file_path=str(item.resolve()),
                        title=title,
                        artist=artist,
                        album=album,
                        duration_seconds=duration,
                        format=item.suffix.lower(),
                        file_size_bytes=stat.st_size,
                        file_modified_time=stat.st_mtime
                    )
                    self._db.add_or_update_track(track)
                    count += 1
        except Exception as e:
            print(f"Error scanning folder {folder_path}: {e}")
        return count

    def _update_style(self):
        c = self._theme_mgr.get_current_theme().get("colors", {})
        bg_card = c.get("background_card", "#1B1D21")
        bg_elevated = c.get("background_elevated", "#22252A")
        bg_hover = c.get("background_hover", "#26292F")
        border = c.get("border_default", "#30333A")
        border_subtle = c.get("border_subtle", "#25282F")
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
            QLabel#TrackCountLabel {{
                color: {text_secondary};
                font-size: 13px;
                font-weight: 500;
            }}
            QTableWidget {{
                background-color: {bg_card};
                border: 1px solid {border};
                border-radius: 10px;
                gridline-color: transparent;
                color: {text_primary};
                font-size: 13px;
            }}
            QTableWidget::item {{
                padding: 10px 12px;
                border-bottom: 1px solid {border_subtle};
            }}
            QTableWidget::item:selected {{
                background-color: {bg_hover};
                color: {accent};
            }}
            QHeaderView::section {{
                background-color: {bg_elevated};
                color: {text_secondary};
                padding: 8px 12px;
                border: none;
                border-bottom: 1px solid {border};
                font-weight: 600;
                font-size: 12px;
            }}
        """)
