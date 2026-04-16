# Component Diagram

```text
User
  |
  v
PyQt UI
  |- ui/main_window.py
  |- ui/image_canvas.py
  |- ui/widgets/action_panel.py
  |- ui/widgets/timeline.py
  |
  v
Core Services
  |- core/video_processor.py
  |- core/annotation_manager.py
  |- core/assistant_annotator.py
  |- core/yolo_tracker.py
  |
  v
Project Data
  |- projects/<video_name>/<video_name>.json
  |- actions.json export
  |- labels/*.txt export
  |- *_predict_ball.csv input
  |- *_action4.json input
```

Interaction summary:

- `main_window` coordinates UI actions and persistence.
- `image_canvas` emits box edit and click events.
- `video_processor` provides frames for display and processing.
- `annotation_manager` stores boxes, rallies, actions, and exportable state.
- `assistant_annotator` builds 9-frame auto-markup clips around a clicked player.
