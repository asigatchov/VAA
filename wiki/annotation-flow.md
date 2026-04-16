# Annotation Flow

## Manual Flow

1. Open a video or project.
2. `VideoProcessor` loads frames and metadata.
3. `MainWindow` refreshes current frame and annotation state.
4. User draws, edits, deletes, or reclassifies boxes in `ImageCanvas`.
5. `AnnotationManager` persists boxes and derives action segments inside rallies.
6. Project state is saved into `projects/...json`.
7. `Shift + Delete` clears all boxes from the current frame only.

## Rally Flow

1. User starts a rally.
2. Frames inside the rally accumulate action-class boxes.
3. `AnnotationManager` rebuilds derived action segments from frame boxes.
4. Timeline and action tree update from current state.

## Export Flow

1. User chooses export directory.
2. YOLO labels are written into `labels/*.txt`.
3. Rally/action metadata is written into `actions.json`.

## Assistant Flow

1. User selects action type in the right panel.
2. User selects `Assistant Crop`.
3. User presses `Shift + Left Click` on the target player.
4. UI opens a progress dialog for the 9-frame assistant pass.
5. `assistant_annotator` runs `RF-DETR` on frames `center_frame - 4` to `center_frame + 4`.
6. Detection uses the selected crop mode: local crop or full frame.
7. `player` and `ball` are written on all 9 frames.
8. `action` is written only on `center_frame - 1` to `center_frame + 1`.
9. `action` box is built as the union of `player` and `ball`.
10. Rally/timeline state is refreshed and the project is saved.
