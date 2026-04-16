# Assistant Auto-Markup

Trigger:

- `Shift + Left Click` on the frame.

Current behavior:

1. User selects action type in the right panel.
2. User uses `RF-DETR Medium` as the assistant auto-label model.
3. User selects `Assistant Crop` in the right panel:
   - `Far Plan: Crop`: use a local crop around the click / previous player position.
   - `Front Plan: Full Frame`: run detection on the whole frame without crop.
4. User clicks the player center on the central frame.
5. `assistant_annotator` builds a 9-frame clip: `center_frame - 4` to `center_frame + 4`.
6. `RF-DETR` runs on each of the 9 frames.
7. `player` is selected from `person` detections, keeping continuity from frame to frame.
8. `ball` is selected from `sports ball` detections using the highest confidence on each frame.
9. `player` and `ball` boxes are created on all 9 frames.
10. `action` box is created only on the center 3 frames: `center_frame - 1` to `center_frame + 1`.
11. `action` box is the union of `player` and `ball` boxes on those 3 frames.
12. A rally covering the clip is created or reused.
13. State is saved through `AnnotationManager` and project JSON.

Notes:

- Assistant processing shows a progress dialog in the UI during the 9-frame detection pass.
- The current assistant path does not use optical flow anymore.
- The current assistant path does not use `ball_data` CSV for ball box placement.
