# Recent Changes 2026-04-15

## Assistant Markup

- `Shift + Left Click` now uses `RF-DETR Medium` instead of template boxes.
- Detection runs on all 9 frames of the clip: `center_frame - 4` to `center_frame + 4`.
- Detection uses a local crop around the click / previous player position to preserve player and ball scale on resized frames.
- `player` and `ball` are written on all 9 frames.
- `action` is written only on `center_frame - 1` to `center_frame + 1`.
- `action` box is the union of `player` and `ball` boxes on those 3 center frames.
- `ball` is selected as the `sports ball` detection with the highest confidence on each frame.

## Assistant UX

- During `Shift + Left Click` processing, the UI shows a modal progress dialog for the 9-frame detection pass.
- Status bar progress updates are shown while assistant detection is running.

## Frame Editing

- `Shift + Delete` removes all boxes from the current frame only.
- After clearing a frame, the app refreshes the timelines/UI and saves the project state.
