"""Audio Studio AI - Global Configuration & Path Management.
Implements Windows 11 Desktop Storage Architecture (Phase 4),
read-only bundled asset management, hardware acceleration detection,
and AI model repository paths.
"""
import os
import sys
import shutil
from pathlib import Path

# Application Metadata
APP_NAME = "Audio Studio AI"
APP_VERSION = "1.0.0"
APP_ORGANIZATION = "AudioStudioAI"
APP_PUBLISHER = "Audio Studio AI"
APP_DESCRIPTION = "Next-Generation AI-Powered Music Production & Audio Workstation"

# Packaging and Runtime Mode
IS_FROZEN = getattr(sys, "frozen", False)

# Application Installation & Bundle Directory
if IS_FROZEN:
    APP_DIR = Path(sys.executable).resolve().parent
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR)).resolve()
else:
    APP_DIR = Path(__file__).resolve().parent
    BUNDLE_DIR = APP_DIR

BASE_DIR = APP_DIR
USER_HOME = Path(os.path.expanduser("~"))

# Read-only Application Resources (bundled inside installation directory)
ASSETS_DIR = BUNDLE_DIR / "assets"
THEMES_DIR = BUNDLE_DIR / "ui" / "themes"

# Bundled Binary Tools (FFmpeg, FFprobe)
BUNDLED_BIN = BUNDLE_DIR / "bin"
if BUNDLED_BIN.exists():
    os.environ["PATH"] = str(BUNDLED_BIN) + os.pathsep + os.environ.get("PATH", "")

# PHASE 4 — WINDOWS STORAGE ARCHITECTURE
# Never store mutable user data in Program Files
LOCAL_APPDATA = Path(os.environ.get("LOCALAPPDATA", str(USER_HOME / "AppData" / "Local")))
APP_DATA_ROOT = LOCAL_APPDATA / "AudioStudioAI"

# In production (frozen standalone) or when explicitly configured, use %LOCALAPPDATA%\AudioStudioAI
if IS_FROZEN or os.environ.get("AUDIO_STUDIO_STORAGE", "").lower() == "appdata":
    DATA_DIR = APP_DATA_ROOT / "Data"
    DB_DIR = APP_DATA_ROOT / "Database"
    DB_PATH = DB_DIR / "audio_studio.db"
    LOGS_DIR = APP_DATA_ROOT / "Logs"
    MODELS_DIR = APP_DATA_ROOT / "Models"
    CACHE_DIR = APP_DATA_ROOT / "Cache"
    SETTINGS_DIR = APP_DATA_ROOT / "Settings"
    PLUGINS_DIR = APP_DATA_ROOT / "Plugins"
    PROJECTS_DIR = APP_DATA_ROOT / "Projects"
else:
    # Development mode: default to repository data folder, or AppData if requested
    DATA_DIR = APP_DIR / "data"
    DB_DIR = DATA_DIR
    DB_PATH = DATA_DIR / "audio_studio.db"
    LOGS_DIR = DATA_DIR / "logs"
    MODELS_DIR = APP_DIR / "models"
    CACHE_DIR = DATA_DIR / "cache"
    SETTINGS_DIR = DATA_DIR / "settings"
    PLUGINS_DIR = APP_DIR / "plugins"
    PROJECTS_DIR = APP_DIR / "projects"

# Ensure essential runtime folders exist
for folder in [DATA_DIR, DB_DIR, PROJECTS_DIR, MODELS_DIR, PLUGINS_DIR, ASSETS_DIR, CACHE_DIR, LOGS_DIR, SETTINGS_DIR]:
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# Preserve existing user databases and settings when updating or first run
try:
    if not DB_PATH.exists():
        legacy_db = APP_DIR / "data" / "audio_studio.db"
        if legacy_db.exists() and legacy_db != DB_PATH:
            try:
                shutil.copy2(legacy_db, DB_PATH)
            except Exception:
                pass
except Exception:
    pass

# PHASE 3 — AI MODEL ARCHITECTURE & STORAGE
# Store models outside installation directory (%LOCALAPPDATA%\AudioStudioAI\Models)
HF_CACHE_DIR = MODELS_DIR / "huggingface"
TORCH_CACHE_DIR = MODELS_DIR / "torch_hub"
try:
    (HF_CACHE_DIR / "hub").mkdir(parents=True, exist_ok=True)
    TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# Detect and preserve existing downloaded models from ~/.cache if present
existing_htdemucs = USER_HOME / ".cache" / "huggingface" / "hub" / "models--adefossez--HTDemucs"
target_htdemucs = HF_CACHE_DIR / "hub" / "models--adefossez--HTDemucs"
if existing_htdemucs.exists() and not target_htdemucs.exists():
    try:
        shutil.copytree(existing_htdemucs, target_htdemucs)
    except Exception:
        pass

# Set environment caches for HuggingFace and Torch to local appdata models directory
if target_htdemucs.exists() or (HF_CACHE_DIR / "hub").exists():
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)

# Initialize optional Hugging Face authentication token if saved in user settings
try:
    from database.database import DatabaseManager
    _saved_hf_token = DatabaseManager().get_setting("hf_token", "")
    if _saved_hf_token and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = _saved_hf_token
except Exception:
    pass

# Audio Settings Defaults
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 2
DEFAULT_BUFFER_SIZE = 1024

# GLOBAL AUDIO FORMAT REQUIREMENT: MP3 ONLY (Import and Export)
SUPPORTED_AUDIO_EXTENSIONS = {".mp3"}
AUDIO_FILE_FILTER = "MP3 Audio Files (*.mp3)"
DEFAULT_EXPORT_BITRATE = "320k"
AVAILABLE_MP3_BITRATES = ["128k", "192k", "256k", "320k"]

# Hardware Acceleration
CUDA_AVAILABLE = False
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    CUDA_DEVICE_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "None"
except Exception:
    CUDA_AVAILABLE = False
    CUDA_DEVICE_NAME = "None"

# UI Defaults & Layout Targets
DEFAULT_THEME = "obsidian_silver"
SIDEBAR_EXPANDED_WIDTH = 235
SIDEBAR_COLLAPSED_WIDTH = 68
TOP_BAR_HEIGHT = 64
BOTTOM_PLAYER_HEIGHT = 82
CARD_BORDER_RADIUS = 12
ANIMATION_DURATION_FAST = 180
ANIMATION_DURATION_NORMAL = 240
ANIMATION_DURATION_SLOW = 320

# AI Model Registry with sizes, verification metadata, and descriptions
AI_MODELS = {
    "separation": [
        {
            "id": "htdemucs",
            "name": "HTDemucs v4 (Vocals & Instrumental)",
            "type": "separation",
            "stems": ["vocals", "instrumental"],
            "size_mb": 80.1,
            "recommended": True,
            "vram_required_gb": 3.0,
            "installed": False,
            "description": "High-fidelity Hybrid Transformer Demucs model for vocal isolation and accompaniment.",
            "repo_id": "adefossez/htdemucs"
        },
        {
            "id": "mdx_kim_vocal_2",
            "name": "MDX-Net Kim Vocal 2",
            "type": "separation",
            "stems": ["vocals", "instrumental"],
            "size_mb": 65.4,
            "recommended": False,
            "vram_required_gb": 4.0,
            "installed": False,
            "description": "MDX architecture optimized for lead vocal separation and acapella extraction.",
            "repo_id": "kim/mdx_vocal_2"
        },
        {
            "id": "htdemucs_4stem",
            "name": "HTDemucs v4 4-Stem (Drums, Bass, Other, Vocals)",
            "type": "separation",
            "stems": ["drums", "bass", "other", "vocals"],
            "size_mb": 158.2,
            "recommended": False,
            "vram_required_gb": 4.5,
            "installed": False,
            "description": "Complete 4-stem band separation for drums, bassline, instrumental other, and vocals.",
            "repo_id": "adefossez/htdemucs_4s"
        }
    ],
    "denoising": [
        {
            "id": "deepfilternet3",
            "name": "DeepFilterNet 3 (Low Latency Denoising)",
            "type": "denoising",
            "size_mb": 24.5,
            "recommended": True,
            "installed": False,
            "description": "Full-band speech enhancement and background acoustic noise removal.",
            "repo_id": "deepfilter/dfn3"
        }
    ],
    "voice_conversion": [
        {
            "id": "rvc_v2_core",
            "name": "RVC v2 Inference Engine",
            "type": "voice_conversion",
            "size_mb": 112.0,
            "recommended": True,
            "installed": False,
            "description": "Retrieval-based Voice Conversion v2 local neural inference engine.",
            "repo_id": "rvc/v2_core"
        }
    ]
}
