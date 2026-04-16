# System Overview

VAA is a desktop annotation tool for volleyball video markup.

Main subsystems:

- `ui/`: PyQt6 interface, frame display, timelines, action panel, box editing.
- `core/video_processor.py`: frame loading, caching, superframe generation.
- `core/annotation_manager.py`: rallies, actions, YOLO boxes, export/state persistence.
- `core/assistant_annotator.py`: clip auto-markup around `Shift + Left Click`.
- `config/config.py`: action classes, colors, UI defaults.
- `projects/`: saved project state and annotation data.

See also:

- [Component Diagram](./component-diagram.md)
- [Annotation Flow](./annotation-flow.md)
- [Assistant Auto-Markup](./assistant-auto-markup.md)
