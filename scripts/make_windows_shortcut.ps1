# Creates a Start menu shortcut "CutOut" that runs this project without a console window.
# The shortcut points at this folder, so re-run this script if the project moves.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\make_windows_shortcut.ps1 [shortcut folder]
param([string]$Dir = [Environment]::GetFolderPath("Programs"))
$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $PSScriptRoot
$Data = Join-Path $env:LOCALAPPDATA "CutOut"
New-Item -ItemType Directory -Force $Data | Out-Null

uv sync --directory $Project
if ($LASTEXITCODE) { throw "uv sync failed" }
uv run --directory $Project python scripts/make_icon.py "$Data\CutOut.ico"
if ($LASTEXITCODE) { throw "icon build failed" }

# uv's .venv\Scripts\pythonw.exe is a console launcher (opens a terminal window).
# CPython's windowed venv launcher finds .venv\pyvenv.cfg one folder up.
$Home_ = (Get-Content (Join-Path $Project ".venv\pyvenv.cfg") | Where-Object { $_ -match "^home\s*=" }) -replace "^home\s*=\s*", ""
$Gui = Join-Path $Project ".venv\gui"
New-Item -ItemType Directory -Force $Gui | Out-Null
Copy-Item (Join-Path $Home_ "Lib\venv\scripts\nt\pythonw.exe") $Gui -Force

$Lnk = Join-Path $Dir "CutOut.lnk"
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($Lnk)
$s.TargetPath = Join-Path $Gui "pythonw.exe"
$s.Arguments = "`"$(Join-Path $Project 'main.py')`""
$s.WorkingDirectory = $Project
$s.IconLocation = "$Data\CutOut.ico"
$s.Description = "Cut subjects out of images"
$s.Save()
Write-Host "Created $Lnk (runs $Project, log: $Data\CutOut.log)"
