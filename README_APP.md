# Volleyball Action Annotator (VAA)

A professional desktop application for annotating volleyball match videos with support for 3-frame RGB superframes and YOLO format export.

## Features

- **Video Playback**: Load and play volleyball match videos (MP4, AVI, MOV, MKV)
- **3-Frame Superframes**: Visualize motion across three consecutive frames by encoding temporal information spatially
- **Action Annotation**: Mark action boundaries with 7 predefined action types (Serve, Reception, Set, Attack, Block, Dig, Point)
- **Bounding Box Annotation**: Draw and edit YOLO-format bounding boxes for ball and player tracking
- **Timeline Visualization**: Visual timeline with action range markers
- **Export**: Export annotations in YOLO format (.txt files) and action metadata (JSON)
- **Keyboard Shortcuts**: Efficient workflow with comprehensive keyboard shortcuts

## Installation

### Prerequisites

- Python 3.13 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd /home/projects/www/vb-soft/VAA
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

3. **Run the application**:
   ```bash
   python main.py
   ```

## Usage

### Basic Workflow

1. **Load Video**: Click "📁 Load Video" or use Ctrl+O to open a video file
2. **Navigate**: Use playback controls or keyboard shortcuts:
   - `Space`: Play/Pause
   - `Left/Right Arrow`: Previous/Next frame
   - `Home/End`: Jump to first/last frame
3. **Annotate Actions**:
   - Select action type from dropdown
   - Click "▶ Start Action" at action beginning (or press Enter)
   - Click "⏹ End Action" at action conclusion (or press Shift+Enter)
4. **Draw Bounding Boxes**:
   - Click and drag on canvas to create box
   - Click existing box to select
   - Press Delete to remove selected box
5. **Toggle Views**:
   - Check "Show Superframe" to see 3-frame RGB composite (F1)
   - Check "Show Boxes" to display bounding boxes (F2)
6. **Export**: Click "💾 Export Annotations" or use Ctrl+S

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Open Video |
| Ctrl+S | Save/Export Annotations |
| Space | Play/Pause |
| Left Arrow | Previous Frame |
| Right Arrow | Next Frame |
| Home | First Frame |
| End | Last Frame |
| Enter | Start/End Action |
| F1 | Toggle Superframe Mode |
| F2 | Toggle Bounding Boxes |
| Delete | Delete Selected Box/Action |

### Superframe Visualization

The 3-frame superframe mode reveals motion patterns:
- **Cyan tints**: Movement from previous frame
- **Yellow tints**: Movement toward next frame
- **Grayscale regions**: Static elements

This is particularly useful for tracking ball trajectory and player movements.

## Export Format

### YOLO Labels

Exported to `labels/` directory with format:
```
labels/
  ├── 000000.txt
  ├── 000001.txt
  └── ...
```

Each file contains one line per bounding box:
```
class_id x_center y_center width height
```

Example:
```
0 0.512 0.384 0.045 0.062
```

### Action Metadata

Exported as `actions.json`:
```json
[
  {
    "start_time": 12.3,
    "end_time": 15.7,
    "type": "Serve",
    "start_frame": 369,
    "end_frame": 471
  }
]
```

## Configuration

Edit `config/config.py` to customize:
- Target resolution (default: 1920×1080)
- Cache limit (default: 100 frames)
- YOLO model path
- UI colors and appearance
- Action types

## Architecture

```
VAA/
├── core/
│   ├── video_processor.py      # Frame extraction & superframe generation
│   ├── annotation_manager.py   # Action & box management
│   └── yolo_tracker.py         # Optional YOLO integration
├── ui/
│   ├── main_window.py          # Main application window
│   ├── image_canvas.py         # Canvas with box editing
│   └── widgets/
│       ├── timeline.py         # Timeline with action markers
│       └── action_panel.py     # Action controls & list
├── config/
│   └── config.py               # Configuration classes
└── main.py                     # Application entry point
```

## Troubleshooting

### Video won't load
- Ensure the video file format is supported (MP4, AVI, MOV, MKV)
- Check that opencv-python is installed correctly
- Verify the video file is not corrupted

### Performance issues
- Reduce cache limit in configuration
- Use lower resolution videos
- Close other applications to free memory

### YOLO detection not working
- Install ultralytics: `pip install ultralytics`
- Download YOLO model (auto-downloaded on first use)
- Check YOLO model path in configuration

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document all public methods

## License

This project is provided as-is for educational and research purposes.

## Support

For issues and questions, please check the documentation or contact the development team.
