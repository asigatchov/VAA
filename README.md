## Volleyball Action Annotator (VAA)

A desktop tool for annotating volleyball videos with:
- `rally` boundaries
- nested rally actions
- YOLO bounding boxes
- normal frame and `3-frame` superframe viewing
- manual ball position markup

## Features

- Load a video and save/load project state.
- Mark rally start and rally end.
- Derive nested actions inside a rally from box classes.
- Show a hierarchical panel on the right: `Rally -> Start / Actions / End`.
- Draw, move, delete, copy/paste, and resize bounding boxes.
- Export YOLO labels and project JSON.

## Project Structure

```text
core/
├── annotation_manager.py   # rallies, actions, yolo boxes, export
├── video_processor.py      # frame loading and superframe generation
└── yolo_tracker.py         # YOLO integration

ui/
├── image_canvas.py         # frame view, boxes, ball markup
├── main_window.py          # main window and shortcuts
└── widgets/
    ├── action_panel.py     # rally/action tree + playback step
    └── timeline.py         # timeline widgets

config/
└── config.py               # action types, colors, UI settings
```

## Typical Workflow

1. Open a video or project.
2. Start a `rally`.
3. Draw boxes with the needed classes such as `Serve`, `Receive`, `Set`, `Attack`, and others.
4. Switch to `superframe` when needed.
5. End the `rally`.
6. Review the rally tree on the right.
7. Export the annotations.

## Keyboard Shortcuts

### Navigation and Playback

- `Space` - Play / Pause
- `Left` - Previous frame
- `Right` - Next frame
- `A` - Previous frame
- `D` - Next frame
- `S` - Jump backward by `15` frames
- `W` - Jump forward by `15` frames
- `Q` - Go to previous annotated frame
- `E` - Go to next annotated frame
- `Home` - Go to the first frame
- `End` - Go to the last frame

### Rally Annotation

- `Return` - Start rally
- `Shift+Return` - End rally
- `[` - Start rally
- `]` - End rally
- `|` - Split the current rally at the current frame

### Display

- `F1` - Toggle `superframe`
- `F2` - Toggle bounding boxes

### Box Editing

- `Delete` - Delete selected box
- `Ctrl+C` - Copy boxes from the current frame
- `Ctrl+V` - Paste boxes into the current frame

## UI Notes

- The selected box has resize handles in the top-left and bottom-right corners.
- The right panel includes a `Step` field (`1..30`) that controls playback frame stepping during `Play`.
- Clicking any item in the rally/action tree seeks to the corresponding frame.

## Output Formats

- YOLO labels: one `.txt` file per frame
- Project JSON: `rallies`, nested `actions`, `yolo_boxes`, and saved app state
