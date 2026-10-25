#!/bin/zsh
# Builds /Applications/CutOut.app: a thin launcher that runs this project with uv.
# The app points at this folder, so re-run this script if the project moves.
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
UV="$(command -v uv)"
APP="${1:-/Applications/CutOut.app}"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>CutOut</string>
  <key>CFBundleDisplayName</key><string>CutOut</string>
  <key>CFBundleIdentifier</key><string>local.cutout</string>
  <key>CFBundleExecutable</key><string>CutOut</string>
  <key>CFBundleIconFile</key><string>CutOut</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
EOF

# The Python GUI stub is copied into the bundle so the Dock and menu bar show CutOut, not "Python".
# It is re-copied on launch when it changes (e.g. after a Homebrew Python upgrade).
# __PYVENV_LAUNCHER__ makes that stub use the project's .venv.
# ponytail: launched from Finder there is no argv path, so "Open With" / drop-on-icon is ignored; use the in-app open.
cat > "$APP/Contents/MacOS/CutOut" <<EOF
#!/bin/zsh
exec >> "\$HOME/Library/Logs/CutOut.log" 2>&1
cd "$PROJECT"
[ -x .venv/bin/python ] || "$UV" sync
PY="\$(.venv/bin/python -c 'import sys, os; print(os.path.join(sys.base_prefix, "Resources/Python.app/Contents/MacOS/Python"))')"
STUB="\${0:h}/CutOut-python"
cmp -s "\$PY" "\$STUB" || cp "\$PY" "\$STUB"
__PYVENV_LAUNCHER__="$PROJECT/.venv/bin/python" exec "\$STUB" main.py "\$@"
EOF
chmod +x "$APP/Contents/MacOS/CutOut"

TMP="$(mktemp -d)"
"$UV" run --directory "$PROJECT" python scripts/make_icon.py "$TMP/CutOut.iconset"
iconutil -c icns "$TMP/CutOut.iconset" -o "$APP/Contents/Resources/CutOut.icns"
rm -rf "$TMP"

touch "$APP"
echo "Built $APP (runs $PROJECT, log: ~/Library/Logs/CutOut.log)"
