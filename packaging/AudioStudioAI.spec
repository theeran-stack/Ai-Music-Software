# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller Specification File for Audio Studio AI.
Builds standalone Windows 11 Desktop Application in ONEDIR mode.
Packages PySide6, PyTorch, Demucs, FFmpeg, SoundFile, Mutagen, and Obsidian Silver UI.
"""
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

# Paths
SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
ROOT_DIR = os.path.dirname(SPEC_DIR)

block_cipher = None

# --- Data Files ---
datas = []

# 1. UI Themes (JSON files for Obsidian Silver, Graphite, Midnight, Platinum Light)
datas.append((os.path.join(ROOT_DIR, 'ui', 'themes', '*.json'), 'ui/themes'))

# 2. Application Icons and Visual Assets
datas.append((os.path.join(ROOT_DIR, 'assets', '*'), 'assets'))

# 3. SoundFile & libsndfile data
datas += collect_data_files('_soundfile_data')

# 4. Demucs model architectures and configuration definitions
datas += collect_data_files('demucs')

# 5. pyloudnorm broadcast loudness definitions
datas += collect_data_files('pyloudnorm')

# 6. PyTorch essential metadata
datas += collect_data_files('torch')

# --- Binaries ---
binaries = []

# 1. Bundled FFmpeg and FFprobe binaries for standalone audio decoding/encoding
binaries.append((os.path.join(ROOT_DIR, 'bin', 'ffmpeg.exe'), 'bin'))
binaries.append((os.path.join(ROOT_DIR, 'bin', 'ffprobe.exe'), 'bin'))

# 2. libsndfile C dynamic library
binaries += collect_dynamic_libs('_soundfile_data')

# 3. PyTorch C++ runtime and CPU SIMD kernels (c10, torch_cpu, libiomp5md, etc.)
binaries += collect_dynamic_libs('torch')

# --- Hidden Imports ---
hiddenimports = [
    # Qt6 / PySide6 core modules
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'PySide6.QtMultimedia',
    'PySide6.QtMultimediaWidgets',
    'PySide6.QtNetwork',
    'PySide6.QtSvg',
    'PySide6.QtSvgWidgets',
    
    # Audio engines & encoders
    'soundfile',
    '_soundfile',
    '_soundfile_data',
    'lameenc',
    'mutagen',
    'mutagen.mp3',
    'mutagen.wave',
    'mutagen.flac',
    'mutagen.oggvorbis',
    'mutagen.mp4',
    'pyloudnorm',
    'julius',
    
    # PyTorch and Demucs neural networks
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.hub',
    'torchvision',
    'demucs',
    'demucs.pretrained',
    'demucs.apply',
    'demucs.hdemucs',
    'demucs.htdemucs',
    'demucs.hf',
    'demucs.repo',
    'demucs.states',
    'demucs.separate',
    'safetensors',
    'safetensors.torch',
    'huggingface_hub',
    'dora',
    'einops',
    
    # Application internal modules
    'config',
    'core.audio_format',
    'core.clip_manager',
    'core.playback_engine',
    'core.waveform_engine',
    'database.database',
    'database.models',
    'ai.model_manager',
    'ai.separation.demucs_adapter',
    'ai.separation.audio_normalizer',
    'ai.enhancement.audio_enhancer',
    'ai.enhancement.neural_enhancer',
    'ai.voice.voice_converter',
    'ai.voice.rvc_engine',
    'ai.mixing.mix_plan',
    'ai.mixing.transition_analyzer',
    'ai.mixing.phrase_analyzer',
    'ai.mixing.vocal_overlap_detector',
    'ai.mixing.audio_analyzer',
    'ai.mixing.mix_planner',
    'ai.mixing.mix_renderer',
    'workers.separation_worker',
    'workers.enhancer_worker',
    'workers.voice_conversion_worker',
    'workers.mix_worker',
    'workers.analyzer_worker',
    'ui.main_window',
    'ui.sidebar',
    'ui.themes.theme_manager',
    'ui.animations.animation_manager',
    'ui.components.button',
    'ui.components.card',
    'ui.components.player_bar',
    'ui.components.stat_badge',
    'ui.components.visualizer_widget',
    'ui.components.waveform_view',
    'ui.components.mix_timeline',
    'ui.workspaces.dashboard',
    'ui.workspaces.library',
    'ui.workspaces.vocal_remover',
    'ui.workspaces.vocal_studio',
    'ui.workspaces.voice_lab',
    'ui.workspaces.audio_editor',
    'ui.workspaces.smart_mix',
    'ui.workspaces.ai_analyzer',
    'ui.workspaces.visualizer',
    'ui.workspaces.plugin_manager',
    'ui.workspaces.settings',
    
    # Utilities
    'watchdog',
    'watchdog.observers',
    'watchdog.events',
    'pyqtgraph',
    'scipy',
    'scipy.signal',
    'scipy.io',
    'scipy.ndimage',
    'sqlite3',
    'PIL',
    'PIL.Image',
    'shutil',
    'hashlib',
    'json',
    'multiprocessing'
]

# Exclude unnecessary test suites and heavy interactive packages to keep build clean
excludes = [
    'tkinter',
    'IPython',
    'jupyter',
    'notebook',
    'pytest',
    'unittest.test',
    'matplotlib.tests'
]

a = Analysis(
    [os.path.join(ROOT_DIR, 'main.py')],
    pathex=[ROOT_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AudioStudioAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # Windowed application mode (no console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(ROOT_DIR, 'assets', 'icon.ico'),
    version=os.path.join(ROOT_DIR, 'packaging', 'version_info.txt')
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='AudioStudioAI'
)

