# Builds dist\MakisCutOut-v<version>-Setup.exe. Needs Inno Setup 7: winget install --id JRSoftware.InnoSetup.7 -e
# Usage: powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Build = Join-Path $Root "installer\build"
# Pinned uv release, shipped inside the installer. Update both lines together (hash from the release's .sha256 file).
$UvVersion = "0.12.3"
$UvSha256 = "b23350c79e8ad0192b8124af13a0f17e8d4e4549524785e1aef389ae5a06990e"
New-Item -ItemType Directory -Force $Build | Out-Null

$Zip = Join-Path $Build "uv-$UvVersion.zip"
if (-not (Test-Path $Zip)) {
    Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $Zip
}
if ((Get-FileHash $Zip -Algorithm SHA256).Hash -ne $UvSha256) { Remove-Item $Zip; throw "uv download does not match its checksum" }
Expand-Archive $Zip $Build -Force

uv run --directory $Root python scripts/make_icon.py (Join-Path $Build "CutOut.ico")
if ($LASTEXITCODE) { throw "icon build failed" }

$Iscc = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source
if (-not $Iscc) {
    $Iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 7\ISCC.exe", "$env:ProgramFiles\Inno Setup 7\ISCC.exe") |
        Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $Iscc) { throw "ISCC.exe not found. Install Inno Setup 7." }
$Version = (Select-String -Path (Join-Path $Root "app\__init__.py") -Pattern '^VERSION = "([^"]+)"').Matches[0].Groups[1].Value
$NumVersion = ($Version -replace '-.*$', '') + ".0"
& $Iscc "/DAppVersion=$Version" "/DAppNumVersion=$NumVersion" (Join-Path $Root "installer\MakisCutOut.iss")
if ($LASTEXITCODE) { throw "Inno Setup failed" }
