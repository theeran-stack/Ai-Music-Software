@echo off
setlocal enabledelayedexpansion

echo ===================================================================
echo AUDIO STUDIO AI - WINDOWS 11 64-BIT STANDALONE BUILD SCRIPT
echo ===================================================================

cd /d "%~dp0\.."

echo [1/4] Cleaning previous build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist release rmdir /s /q release
mkdir release

echo [2/4] Compiling Standalone ONEDIR Package via PyInstaller...
pyinstaller --clean --noconfirm packaging\AudioStudioAI.spec
if errorlevel 1 (
    echo [ERROR] PyInstaller compilation failed!
    exit /b 1
)

if not exist "dist\AudioStudioAI\AudioStudioAI.exe" (
    echo [ERROR] Executable dist\AudioStudioAI\AudioStudioAI.exe was not created!
    exit /b 1
)
echo [PASS] Standalone application built successfully in dist\AudioStudioAI\

echo [3/4] Compiling Windows Installer via Inno Setup 6...
set "ISCC_PATH=C:\Users\Theeran\AppData\Local\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC_PATH%" (
    set "ISCC_PATH=ISCC.exe"
)

"%ISCC_PATH%" packaging\AudioStudioAI.iss
if errorlevel 1 (
    echo [ERROR] Inno Setup compilation failed!
    exit /b 1
)

if not exist "release\AudioStudioAI_Setup.exe" (
    echo [ERROR] Installer release\AudioStudioAI_Setup.exe was not generated!
    exit /b 1
)

echo [4/4] Verifying Release Artifacts...
for %%A in ("release\AudioStudioAI_Setup.exe") do (
    echo [SUCCESS] Windows Installer: release\AudioStudioAI_Setup.exe [%%~zA bytes]
)

echo ===================================================================
echo BUILD COMPLETED SUCCESSFULLY!
echo ===================================================================

