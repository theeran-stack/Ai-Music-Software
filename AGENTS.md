# PROJECT INSTRUCTIONS: AUDIO STUDIO AI

## 1. Project Identity & Architecture
- **Application**: Audio Studio AI
- **Type**: Standalone Windows 11 Desktop Audio Application (64-bit x64).
- **Core Technologies**: Python 3.12, PySide6 (Qt6), PyTorch, HTDemucs v4, FFmpeg, SQLite.
- **Root Directory**: `C:\Users\Theeran\Documents\Music Software`

## 2. Strict Boundary Rules (DO NOT CROSS)
- **EXCLUSIVELY DESKTOP**: This codebase is strictly a local Windows desktop application.
- **NEVER MIX WITH SONORA**: Do NOT import, reference, modify, or apply any requirements, architecture, or code from **SONORA** (the Django web application located at `C:\Users\Theeran\Documents\Web App`).
- **NO WEB FRAMEWORKS**: Do NOT introduce Django, Django REST Framework, Webpack, Celery, or browser web server dependencies into this project.
- **DATA ISOLATION**:
  - Application binaries install to `C:\Program Files\Audio Studio AI` or `%LOCALAPPDATA%\Programs\Audio Studio AI`.
  - Mutable user databases, logs, settings, and downloaded AI models belong strictly in `%LOCALAPPDATA%\AudioStudioAI`.
