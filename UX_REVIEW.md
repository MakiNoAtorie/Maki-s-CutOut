# CutOut — UX review (2026-10-25)

Scope: every user-facing path in `main.py`, `app/window.py`, `app/viewer.py`, and the status/error surface of `engine/`.
Reviewed by reading the code end to end; no model was run. Each finding names the file and line, what the user hits,
and the smallest fix. P1 = breaks flow, P2 = friction, P3 = polish / design notes (not for the fix pass).

## Flow as it exists today

Open/paste/drop → BiRefNet auto-detect commits a mask → pick a mode → prompts build a live *candidate* in the preview →
**Apply** (Enter) commits, **Esc** discards → Finish sliders apply live → Export. The core loop is sound: proxy-first
rendering, stale-job dropping, one worker per class, and WYSIWYG export (what the preview shows is what gets saved).
The problems below are mostly at the seams: undo semantics, missing affordances, Mac input, and feedback when things
take long or go wrong.

## P1 — breaks the flow

### 1. Undo wipes every prompt instead of stepping back one
`window.py:508-513`. With three SAM points placed, Ctrl+Z (and Esc, and "Reset prompts") all do the same thing:
clear everything. There is no way to remove just the last click or the last box. Users expect Undo to retract the
last action.
**Fix:** in `undo()`, if `points` → pop the last point/label and re-run SAM; elif `box` → clear it and re-run;
elif `key_pick` → clear it; elif `paint` → reset (stroke history is not worth tracking, Esc covers it). Only when no
prompt state remains, pop `history`. Keep Esc as "discard all prompts".

### 2. Nothing tells the user a candidate is pending, or that Apply/Undo/Redo/Export can do anything
`window.py:191-193, 151-162`. Apply, Undo, Redo, Export, Detect subject, Copy are always enabled, with or without an
image, with or without a candidate. The preview silently blends the candidate into the committed mask, so the user
cannot tell what is committed and what still needs Enter. Switching modes silently commits the candidate
(`window.py:358-359`), which is correct but invisible.
**Fix:** one `_update_actions()` called from `refresh()`: Apply enabled iff `candidate is not None`; Undo iff
history or prompt state; Redo iff `redo_stack`; Export/Copy/Detect iff `rgb is not None`. In `_set_candidate`,
show a persistent status "Selection pending — Enter applies, Esc discards" and clear it on apply/reset.

### 3. Box-limited auto-detect commits immediately; every other prompt makes a candidate
`window.py:387-391, 443-448`. The Auto panel says "Drag a rectangle to limit detection", but the result is committed
straight away through the current Combine op, and the box overlay is wiped. Paint, Click and Color key all show a
candidate first. With Combine = Replace (default) a drag to "limit" the detection throws away the existing mask with no
preview step.
**Fix:** `run_subject(box)` with a box → route the result through `_set_candidate` (keep the box overlay until
apply/reset). The initial whole-image detection on load and the "Detect subject" button can keep committing directly.

### 4. Mac trackpad: two-finger scroll zooms, pinch does nothing
`viewer.py:158-165`. Every wheel event zooms. On a trackpad that means two-finger scrolling (the universal "pan")
zooms, and the pinch gesture is ignored. Middle-drag does not exist on a trackpad; Space+drag is the only pan and
is undocumented in the UI.
**Fix:** in `wheelEvent`, if `e.source() != Qt.MouseEventNotSynthesized` and no Ctrl/Cmd modifier → scroll the
scrollbars by `pixelDelta()` (pan); otherwise zoom as now. Override `event()` to handle `QEvent.NativeGesture` with
`ZoomNativeGesture` → `scale(1 + value)`. Mouse wheel keeps zooming.

### 5. Export opens the file dialog before checking there is anything to export
`window.py:584-612`. With no mask, the user picks a filename and only then gets "Error: ValueError: No selection yet"
in the status bar. `copy_result` with no image loaded raises the same error. Export → "Save mask" of an all-zero mask
(detection found nothing) silently writes a fully transparent PNG.
**Fix:** check `self._latest` (and `rgb`) before opening the dialog and say "Nothing to export yet" (4 s). Pair with
finding 2 (disable Export/Copy without an image).

### 6. Errors and long waits give the wrong feedback
- `window.py:128`: errors show as raw `TypeError: …` text for 10 s in a status bar that truncates long messages
  (Hugging Face connection errors are several lines). Easy to miss, impossible to read.
- `models.py:58`: "Loading BiRefNet (first run downloads it into models/)…" is shown on *every* model swap, so the
  user cannot tell a 2 s cache load from a 1.4 GB download. The download itself has no progress and no "this takes
  minutes" warning; it looks hung.
- Model swaps happen often by design (Click select + Refine = two swaps per Apply), so the loading line is the most
  frequent status the user sees.
**Fix:** errors: `showMessage("Error: …")` with no timeout, `statusBar().setToolTip(full text)`, and a `QMessageBox`
only for model-load failures (the one case the user must act on). Loading: say "Loading X…" by default and only
when `_load` falls through to the network branch call `on_status("Downloading X (~N MB, first run only)…")`.

### 7. Closing the window during a model job hangs the process
`window.py:37`, `main.py:17`. `ThreadPoolExecutor` threads are joined at interpreter exit, so quitting during a
download or a 50 MP render leaves a dead window on screen until the job finishes.
**Fix:** `closeEvent` → `pool.shutdown(wait=False, cancel_futures=True)` on both runners; in `main()` call
`os._exit(code)` after `app.exec()` so a job already running cannot block exit.

## P2 — friction

### 8. Shortcuts exist but are undiscoverable
`window.py:138-194`. Toolbar actions have no tooltips, so Ctrl+O/E/Z/Shift+Z, Enter and Ctrl+Shift+C are invisible.
Esc (Reset) is not shown anywhere: Auto and Color key panels have no reset button at all. Modes need a dropdown
trip every time. Numpad Enter does not trigger Apply (`Qt.Key_Return` only).
**Fix:** tooltips in `_act` (`f"{text} ({shortcut})"`), `1`–`4` switch modes, Apply accepts both
`Qt.Key_Return` and `Qt.Key_Enter`, add "(Esc)" to the hint labels, `Ctrl+0` fit / `Ctrl+1` 100% on the source view.

### 9. No cursor feedback for the active tool
`viewer.py:97-99`. Arrow cursor in every mode. Paint shows a dashed circle *plus* the arrow. Color key gives no
"pick" cue.
**Fix:** `set_tool` sets the viewport cursor: `Qt.CrossCursor` for box/click/key, `Qt.BlankCursor` for paint (the
circle is the cursor). Restore on leave/pan as already done for the hand cursor.

### 10. Empty state is two grey checkerboards
`window.py:237`. The only hint ("Open, paste or drop an image") is in the status bar.
**Fix:** in `ImageView.drawBackground`, when `self.full is None` draw a centred "Drop an image here · Ctrl+O to open
· Ctrl+V to paste" in the source view.

### 11. Refine edges toggle does nothing visible while a candidate is pending
`window.py:527`. Refinement only applies to the committed mask. Toggling it while a candidate is on screen changes
nothing and says nothing.
**Fix:** when toggled on with `candidate is not None`, status "Refine applies after you press Enter" (4 s).

### 12. Dead and wrong gestures
- `viewer.py:264-274`: the rubber band draws in Color key mode and emits a box that `on_box` ignores.
- `viewer.py:276-298`: a right-*drag* in Click select is not a box; on release it becomes an exclude click at the
  release point. A left-drag that ends under 5 px is a click — fine.
**Fix:** `on_box` in Color key → nothing changes, so suppress the band: `set_tool("none")` for Color key and skip
band creation when `tool == "none"`. For right-drag: if the press moved more than 5 px and it was the right button,
emit nothing.

### 13. Custom background cancel leaves the combo on "Custom…"
`window.py:281-288`. Cancelling the colour dialog keeps the old background but the combo shows Custom…
**Fix:** on cancel, `combo.setCurrentIndex(previous)` (block signals).

### 14. Click on the far edge pixel raises IndexError
`window.py:384, 430` and `viewer.py:297`. `QRectF.contains` includes the right/bottom edge, so x == W passes and
`self.rgb[y, x]` throws in a Qt slot.
**Fix:** clamp in `on_click`: `x = min(int(x), W - 1)`, same for y.

### 15. Combine = Replace silently drops the previous mask
`window.py:186-189`. After the auto-detected subject is committed, the first SAM click or colour pick shows a preview
without it. Correct for Replace, but the user asked to "add a second object" or "key the sky out of the subject"
and sees their subject vanish. Color key in particular almost always wants Intersect/Subtract against an existing mask.
**Fix:** when a candidate arrives, `history` is non-empty and op is Replace, append to the pending-status line:
"(Combine: Replace — pick Add/Subtract/Intersect to keep the current mask)". No behaviour change.

## P3 — polish and design notes (document only)

- **Refine + Click select thrashes models.** One model on device at a time means every Apply with Refine on costs
  two swaps (matting in, SAM back out). Inherent in the memory budget; the README should say so, and the status line
  from finding 6 makes the cost visible. Not a code change now.
- **No mask view.** A "Mask" entry in the preview background combo (render alpha as grey) would help judge edges.
  Worth doing later; it is a new render path.
- **Nothing persists across launches.** Window size, splitter, last folder, preview background. `QSettings`, a few
  lines, when someone asks.
- **Brush size from the keyboard** (`[` / `]`) — cheap, add with finding 8 if the fixer is already there.
- **Space held while leaving the window** leaves pan mode stuck until Space is pressed again (`viewer.py:216-228`).
  Clear `_space` in `leaveEvent`/`focusOutEvent`. One line; fold into finding 9.

## What is good and should stay

- Proxy-then-full rendering with stale-job dropping (`window.py:538-558`) keeps sliders fluid on 50 MP images.
- Export is WYSIWYG: `_latest` is whatever the preview last rendered, including a pending candidate.
- Mode switch auto-applies the candidate so the preview never changes under the user.
- Guarding results by `image_id` means a slow job for the previous image cannot land on the new one.
- Refusing to overwrite the source, EXIF transpose on load, Unicode-safe PNG write.

## Fix pass (same day)

All P1 and P2 findings plus the two folded-in P3 one-liners were applied by an Opus subagent and verified headless
(syntax, `engine/maskops.py` self-check, and an offscreen `MainWindow` smoke with a stubbed engine covering undo
stepping, action enable states, the pending status line, box-limited detect as a candidate, the far-edge clamp and the
export gate). Two deliberate deviations from the suggested fixes: the pinch handler lives in
`ImageView.viewportEvent` because native gestures are delivered to the viewport, and *any* drag over 5 px that draws
no box is ignored rather than only right-drags. Model-load failures are tagged at the source
(`Engine.use` raises "Could not load X: …") rather than by string-matching Hugging Face text.

Not verified: real trackpad pinch/scroll on macOS, and the Redo shortcut text on a real Mac (offscreen reports ⌘Y).
Known leftovers, not fixed: a failed model load re-opens its dialog on every job that retries the load; a box-limited
detect that finds nothing shows the pending line but not the "No clear subject found" hint; digit and bracket
shortcuts fire while a combo box has focus.
