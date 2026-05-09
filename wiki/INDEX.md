# VAA Wiki

- [System Overview](./system-overview.md)
- [Component Diagram](./component-diagram.md)
- [Annotation Flow](./annotation-flow.md)
- [Assistant Auto-Markup](./assistant-auto-markup.md)
- [Recent Changes 2026-04-15](./recent-changes-2026-04-15.md)

## Current Shortcuts

- `Shift + Left Click`: assistant clip markup with the selected `Auto-Label Model`
- `Shift + Delete`: remove all boxes on the current frame

## Auto-Label Models

- UI select: `Auto-Label Model`
- Crop select: `Assistant Crop`
- Supported assistant backend: `RF-DETR Medium`
- `MixFormerV2 ONNX` assistant markup does not auto-restart `rally`
- `rally` starts from the button or from `Serve`
- after assistant markup of `Serve`, UI switches to `Receive`
- Supported assistant crop modes:
  - `Far Plan: Crop`
  - `Front Plan: Full Frame`

## Action Classifier Training

- dataset: `data/action_interaction_crop_markup_ball_224_20260509`
- model code: `src/models/action_detector_v1.py`, `src/models/action_detector.py`
- training entry point: `src/models/train_interaction_action.py`
- OpenVINO export entry point: `src/models/export_tiny_action_cpu_v2.py`
- inference entry point: `src/inference_openvino_seq_gray_track_action.py`
- action classes: `serve`, `receive`, `set`, `attack`
- `noaction` handling:
  - `--include-noaction-class` reserves class id `4` in the classifier config/export.
  - Current `action_interaction_crop_markup_ball_224_20260509` split has no negative/noaction samples, so disputed moments should be rejected in inference with `--action-noaction-threshold` and/or `--action-margin-threshold`.
  - Do not change YOLO txt helper ids: label id `4` is still `player`, label id `5` is still `ball`.

Training command:

```bash
uv run python -m src.models.train_interaction_action \
  --dataset-dir data/action_interaction_crop_markup_ball_224_20260509 \
  --output-dir model_tiny_action_cpu_v2_20260509_noaction \
  --model-size tiny-v2 \
  --epochs 80 \
  --batch-size 64 \
  --num-workers 2 \
  --dropout 0.1 \
  --player-box-weight 0.2 \
  --ball-box-weight 0.25 \
  --include-noaction-class
```

Export command:

```bash
uv run python -m src.models.export_tiny_action_cpu_v2 \
  --checkpoint model_tiny_action_cpu_v2_20260509_noaction/best.pt \
  --output-dir ov/tiny_action_cpu_v2_20260509_noaction
```

Inference command:

```bash
uv run python src/inference_openvino_seq_gray_track_action.py \
  --video-path data/test.mp4 \
  --ball-model-xml ov/VballNetGridV1b_seq15_grayscale_20260427_194144.xml \
  --action-model-xml ov/tiny_action_cpu_v2_20260509_noaction/tiny_action_cpu_v2.xml \
  --output-dir out/track_action_noaction \
  --window-overlap 14 \
  --action-noaction-threshold 0.55 \
  --action-margin-threshold 0.15
```
