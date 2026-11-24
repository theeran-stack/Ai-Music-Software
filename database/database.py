"""Database manager for Audio Studio AI using SQLite.
Thread-safe and supports schema migrations, tracks, playlists, settings and history.
"""
import sqlite3
import threading
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from config import DB_PATH
from database.models import Track, Playlist, ProcessingTask

class DatabaseManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Path = DB_PATH):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseManager, cls).__new__(cls)
                cls._instance._db_path = db_path
                cls._instance._local = threading.local()
                cls._instance._init_db()
            return cls._instance

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._local.conn = sqlite3.connect(
                str(self._db_path),
                timeout=30.0,
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn.execute("PRAGMA foreign_keys=ON;")
        return self._local.conn

    def _init_db(self):
        conn = self._get_connection()
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    artist TEXT DEFAULT 'Unknown Artist',
                    album TEXT DEFAULT 'Unknown Album',
                    genre TEXT DEFAULT '',
                    duration_seconds REAL DEFAULT 0.0,
                    bpm REAL,
                    musical_key TEXT,
                    sample_rate INTEGER DEFAULT 44100,
                    channels INTEGER DEFAULT 2,
                    bitrate INTEGER DEFAULT 0,
                    format TEXT DEFAULT '',
                    waveform_cached INTEGER DEFAULT 0,
                    is_favorite INTEGER DEFAULT 0,
                    play_count INTEGER DEFAULT 0,
                    last_played REAL,
                    date_added REAL NOT NULL,
                    file_size_bytes INTEGER DEFAULT 0,
                    file_modified_time REAL DEFAULT 0.0
                );

                CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist);
                CREATE INDEX IF NOT EXISTS idx_tracks_album ON tracks(album);
                CREATE INDEX IF NOT EXISTS idx_tracks_favorite ON tracks(is_favorite);
                CREATE INDEX IF NOT EXISTS idx_tracks_last_played ON tracks(last_played);

                CREATE TABLE IF NOT EXISTS library_folders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    folder_path TEXT UNIQUE NOT NULL,
                    date_added REAL NOT NULL,
                    last_scanned REAL
                );

                CREATE TABLE IF NOT EXISTS playlists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS playlist_tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    playlist_id INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    track_order INTEGER NOT NULL,
                    FOREIGN KEY(playlist_id) REFERENCES playlists(id) ON DELETE CASCADE,
                    FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS processing_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    input_file TEXT NOT NULL,
                    output_files TEXT DEFAULT '',
                    status TEXT NOT NULL,
                    progress REAL DEFAULT 0.0,
                    status_message TEXT DEFAULT '',
                    error_message TEXT,
                    model_name TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    completed_at REAL
                );

                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    project_type TEXT NOT NULL,
                    project_path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
            """)

    # --- Tracks Operations ---
    def add_or_update_track(self, track: Track) -> int:
        conn = self._get_connection()
        with conn:
            cursor = conn.execute("""
                INSERT INTO tracks (
                    file_path, title, artist, album, genre,
                    duration_seconds, bpm, musical_key, sample_rate,
                    channels, bitrate, format, waveform_cached,
                    is_favorite, play_count, last_played, date_added,
                    file_size_bytes, file_modified_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    title=excluded.title,
                    artist=excluded.artist,
                    album=excluded.album,
                    genre=excluded.genre,
                    duration_seconds=excluded.duration_seconds,
                    sample_rate=excluded.sample_rate,
                    channels=excluded.channels,
                    bitrate=excluded.bitrate,
                    format=excluded.format,
                    file_size_bytes=excluded.file_size_bytes,
                    file_modified_time=excluded.file_modified_time
            """, (
                track.file_path, track.title, track.artist, track.album, track.genre,
                track.duration_seconds, track.bpm, track.musical_key, track.sample_rate,
                track.channels, track.bitrate, track.format, 1 if track.waveform_cached else 0,
                1 if track.is_favorite else 0, track.play_count, track.last_played,
                track.date_added, track.file_size_bytes, track.file_modified_time
            ))
            return cursor.lastrowid

    def get_all_tracks(self) -> List[Track]:
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM tracks ORDER BY title COLLATE NOCASE ASC")
        rows = cursor.fetchall()
        return [self._row_to_track(r) for r in rows]

    def get_favorite_tracks(self) -> List[Track]:
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM tracks WHERE is_favorite = 1 ORDER BY title COLLATE NOCASE ASC")
        return [self._row_to_track(r) for r in cursor.fetchall()]

    def get_recently_played_tracks(self, limit: int = 10) -> List[Track]:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM tracks WHERE last_played IS NOT NULL ORDER BY last_played DESC LIMIT ?",
            (limit,)
        )
        return [self._row_to_track(r) for r in cursor.fetchall()]

    def search_tracks(self, query: str) -> List[Track]:
        conn = self._get_connection()
        pattern = f"%{query}%"
        cursor = conn.execute("""
            SELECT * FROM tracks
            WHERE title LIKE ? OR artist LIKE ? OR album LIKE ? OR genre LIKE ?
            ORDER BY title COLLATE NOCASE ASC
        """, (pattern, pattern, pattern, pattern))
        return [self._row_to_track(r) for r in cursor.fetchall()]

    def toggle_favorite(self, track_id: int) -> bool:
        conn = self._get_connection()
        with conn:
            cursor = conn.execute("SELECT is_favorite FROM tracks WHERE id = ?", (track_id,))
            row = cursor.fetchone()
            if not row:
                return False
            new_state = 0 if row["is_favorite"] else 1
            conn.execute("UPDATE tracks SET is_favorite = ? WHERE id = ?", (new_state, track_id))
            return bool(new_state)

    def record_track_played(self, track_id: int):
        conn = self._get_connection()
        with conn:
            now = time.time()
            conn.execute("""
                UPDATE tracks
                SET play_count = play_count + 1, last_played = ?
                WHERE id = ?
            """, (now, track_id))

    def get_library_statistics(self) -> Dict[str, Any]:
        conn = self._get_connection()
        total_tracks = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
        total_duration = conn.execute("SELECT SUM(duration_seconds) FROM tracks").fetchone()[0] or 0.0
        total_artists = conn.execute("SELECT COUNT(DISTINCT artist) FROM tracks").fetchone()[0]
        total_albums = conn.execute("SELECT COUNT(DISTINCT album) FROM tracks").fetchone()[0]
        favorites_count = conn.execute("SELECT COUNT(*) FROM tracks WHERE is_favorite = 1").fetchone()[0]
        return {
            "total_tracks": total_tracks,
            "total_duration": total_duration,
            "total_artists": total_artists,
            "total_albums": total_albums,
            "favorites_count": favorites_count
        }

    # --- Library Folders ---
    def get_library_folders(self) -> List[str]:
        conn = self._get_connection()
        cursor = conn.execute("SELECT folder_path FROM library_folders ORDER BY date_added ASC")
        return [r["folder_path"] for r in cursor.fetchall()]

    def add_library_folder(self, folder_path: str):
        conn = self._get_connection()
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO library_folders (folder_path, date_added) VALUES (?, ?)",
                (folder_path, time.time())
            )

    def remove_library_folder(self, folder_path: str):
        conn = self._get_connection()
        with conn:
            conn.execute("DELETE FROM library_folders WHERE folder_path = ?", (folder_path,))

    # --- Settings Operations ---
    def get_setting(self, key: str, default: Any = None) -> Any:
        conn = self._get_connection()
        cursor = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            try:
                return json.loads(row["value"])
            except Exception:
                return row["value"]
        return default

    def set_setting(self, key: str, value: Any):
        conn = self._get_connection()
        val_str = json.dumps(value)
        with conn:
            conn.execute("""
                INSERT INTO app_settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """, (key, val_str))

    # --- Processing Tasks ---
    def add_processing_task(self, task: ProcessingTask) -> int:
        conn = self._get_connection()
        with conn:
            cursor = conn.execute("""
                INSERT INTO processing_tasks (
                    task_type, title, input_file, output_files,
                    status, progress, status_message, error_message,
                    model_name, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.task_type, task.title, task.input_file, task.output_files,
                task.status, task.progress, task.status_message, task.error_message,
                task.model_name, task.created_at, task.completed_at
            ))
            return cursor.lastrowid

    def get_recent_tasks(self, limit: int = 10) -> List[ProcessingTask]:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM processing_tasks ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        return [self._row_to_task(r) for r in cursor.fetchall()]

    def _row_to_track(self, r: sqlite3.Row) -> Track:
        return Track(
            id=r["id"],
            file_path=r["file_path"],
            title=r["title"],
            artist=r["artist"],
            album=r["album"],
            genre=r["genre"],
            duration_seconds=r["duration_seconds"],
            bpm=r["bpm"],
            musical_key=r["musical_key"],
            sample_rate=r["sample_rate"],
            channels=r["channels"],
            bitrate=r["bitrate"],
            format=r["format"],
            waveform_cached=bool(r["waveform_cached"]),
            is_favorite=bool(r["is_favorite"]),
            play_count=r["play_count"],
            last_played=r["last_played"],
            date_added=r["date_added"],
            file_size_bytes=r["file_size_bytes"],
            file_modified_time=r["file_modified_time"]
        )

    def _row_to_task(self, r: sqlite3.Row) -> ProcessingTask:
        return ProcessingTask(
            id=r["id"],
            task_type=r["task_type"],
            title=r["title"],
            input_file=r["input_file"],
            output_files=r["output_files"],
            status=r["status"],
            progress=r["progress"],
            status_message=r["status_message"],
            error_message=r["error_message"],
            model_name=r["model_name"],
            created_at=r["created_at"],
            completed_at=r["completed_at"]
        )
