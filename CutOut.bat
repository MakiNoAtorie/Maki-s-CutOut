@echo off
rem Double-click to run CutOut, or drop an image on this file to open it.
cd /d "%~dp0"
uv run main.py %*
if errorlevel 1 pause
