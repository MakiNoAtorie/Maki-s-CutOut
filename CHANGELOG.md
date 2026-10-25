# Changelog

## Unreleased

### Added

- Auto subject: a **Reach** slider. Move it right to take in parts the model was unsure about, or left to keep only what it is sure of. It changes the last detection in place, without running the model again or adding an undo step.
- Click select and Paint region: a **Fine edges** option, on by default. A matting model (ViTMatte) refines the border at full resolution, so hair and soft edges come out cleaner. It downloads ~105 MB the first time.

### Changed

- Defringe starts at 100%. At 0%, background colour left in the soft edge pixels made a hazy halo that looked like a blurry edge.

### Fixed

- Auto subject, Click select and Refine: edges are no longer blurred where the subject and the background have similar brightness (for example a red hat on a dark grey wall).
- Auto subject, Click select and Refine: edges follow the photo closely. There is no more soft band just inside the edge and no faint blurred copy of the subject just outside it.
- The subject is fully opaque. Before, it was very slightly transparent everywhere.
- Feather and Shrink no longer change the colours of the subject when Defringe is on. Defringe now only cleans pixels that are really part subject, part background.
- Refine edges no longer adds a ring of background around the subject with a soft halo at its outer edge.
- Grow and Feather no longer add a rim of background colour around the subject. Exported images also carry the subject's colour a few pixels past the edge, so apps that resize them do not pull the background into the edge.
- Paint region: the painted area defines what to keep. Background painted over by accident near the edge (sky above a hill, for example) is trimmed, and the model sees the whole surroundings for context. Before, it kept only the most visible object inside the paint, so a painted hill or wall came back nearly empty. Things that stand apart inside the paint, such as a castle on a painted hill, are kept whole, thin towers and walls included.
- Click select and Paint region: on large photos, the edge of a selection that does not fill most of the image follows the photo more closely. SAM looks again at a close-up of the selected area.
- Paint region: background painted over inside a rounded paint (valley or sky around a castle on a hill) is no longer kept in patches.
- Auto subject: a dragged rectangle is now given the same context padding as a painted area before detection, so a thin rectangle is no longer squashed into the model's square input. The result still stays inside the rectangle.
- Auto subject: pressing Detect subject while a rectangle selection is pending applies that selection first, like switching modes does. Before, it was silently dropped.
- Auto subject: Reach, used after Enter, now combines with the Combine choice that was active when you pressed Enter. Before, it used the choice from the moment of detection.
- Refine edges on an empty selection no longer adds a duplicate undo step.
- Refine edges on a large selection with no clear subject (a whole hill, for example) no longer eats away thin parts and gaps of the selection. It leaves the selection unchanged and says why in the status bar.
- Model download: answering Yes no longer loses the clicks or paint strokes that triggered it. Click select and Paint region continue from where you were.

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
