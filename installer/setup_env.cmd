@echo off
rem Builds the Python environment of an installed Maki's CutOut. Setup runs it; double-click it to repair.
cd /d "%~dp0"
set "UV_PYTHON_INSTALL_DIR=%~dp0python"
set UV_PYTHON_INSTALL_BIN=0
set UV_PYTHON_INSTALL_REGISTRY=0
set UV_PROJECT_ENVIRONMENT=
uv.exe sync --frozen --no-cache --managed-python || goto fail

rem uv's .venv\Scripts\pythonw.exe opens a console. CPython's windowed venv launcher finds .venv\pyvenv.cfg one folder up.
for /f "tokens=1,* delims== " %%a in (.venv\pyvenv.cfg) do if "%%a"=="home" set "BASE=%%b"
if not exist .venv\gui mkdir .venv\gui
copy /y "%BASE%\Lib\venv\scripts\nt\pythonw.exe" .venv\gui\ >nul || goto fail
exit /b 0

:fail
echo.
echo Maki's CutOut: the Python environment could not be set up. Check your internet connection and run this file again.
if not "%~1"=="nopause" pause
exit /b 1
