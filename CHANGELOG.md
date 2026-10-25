# Changelog

## Unreleased

### Added

- Auto subject: a **Reach** slider. Move it right to take in parts the model was unsure about, or left to keep only what it is sure of. It changes the last detection in place, without running the model again or adding an undo step.

### Changed

- Defringe starts at 100%. At 0%, background colour left in the soft edge pixels made a hazy halo that looked like a blurry edge.

### Fixed

- Auto subject, Click select and Refine: edges are no longer blurred where the subject and the background have similar brightness (for example a red hat on a dark grey wall).

## 0.1.0 (2026-10-25)

First release.

### Added

- Four selection modes, each with a live preview: Auto subject and Paint region (BiRefNet), Click select (SAM 2.1) and Color key.
- Combine modes (Replace, Add, Subtract, Intersect), undo and redo.
- Edge controls: Refine edges, Feather, Shrink/Grow and Defringe.
- Full-resolution PNG export, with an option to crop to the subject. Save the mask only, or copy the result to the clipboard.
- Copy result to clipboard keeps transparency when you paste into other apps. It also puts a PNG on the clipboard, which Windows apps need.
- macOS app launcher: `scripts/make_mac_app.sh`.
- Windows Start menu shortcut: `scripts/make_windows_shortcut.ps1`. It opens without a console window and logs to `%LOCALAPPDATA%\CutOut\CutOut.log`.
- App icon: a split circle, half solid and half transparent. It is used for the window, the Windows shortcut and the Mac app.
- MIT license. The README has usage instructions and credits.

### Changed

- Windows: if BiRefNet is already downloaded, CutOut loads it at launch. Before, the first detection took about 10 s.
- The Source and Preview panes now stay the same width when you resize or maximize the window.
- Windows: no more symlink warning when a model downloads.
