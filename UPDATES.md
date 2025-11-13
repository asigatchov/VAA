# VAA Application Updates

## Changes Made (2025-01-12)

### 1. Configuration Updates ✅

**File**: `config/config.py`

- **Removed "Point" action type** - Now supports only 6 volleyball actions:
  - Serve (#FF6B6B)
  - Reception (#4ECDC4)
  - Set (#45B7D1)
  - Attack (#FFA07A)
  - Block (#98D8C8)
  - Dig (#F7DC6F)

### 2. Color Conversion Fix ✅

**File**: `ui/main_window.py`

- **Fixed BGR→RGB conversion** for proper Qt display
- Added `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before displaying frames
- OpenCV reads frames in BGR format, Qt requires RGB format
- Applied to both normal frames and superframes

### 3. Box Movement & Deletion ✅

**File**: `ui/image_canvas.py`

- **Implemented box dragging**: Click and drag existing boxes to move them
  - Added `_move_box()` method for pixel-to-normalized coordinate conversion
  - Delta movement tracking with proper clamping to frame boundaries
  - Visual feedback during drag operation
  
- **Enhanced box deletion**: 
  - Press Delete key to remove selected box
  - Added logging for box deletion events
  - Proper cleanup and UI update

### 4. Auto-Tracking Action Duration ✅

**File**: `core/annotation_manager.py`

- **Auto-start actions** when bounding box first appears
- **Auto-update actions** on each frame where box is present
- **Auto-end actions** when box disappears (last frame with box)
- Action duration = from first box appearance to last box presence

**New Methods**:
- `auto_start_action()` - Starts tracking when box appears
- `auto_update_action()` - Updates last_frame as box moves
- `auto_end_action()` - Finalizes action when box disappears
- `check_box_tracking()` - Monitors box presence per frame

### 5. Multiple Simultaneous Actions ✅

**Implementation**:
- Changed from single `current_action_start` to dictionary-based `active_actions`
- Each action type can be tracked independently
- Support scenarios like "Attack" + "Block" happening at the same time
- Actions keyed by type in `active_actions: Dict[str, Dict]`

**Example Scenario**:
```
Frame 100: Ball box appears → Auto-start "Serve"
Frame 120: Player box appears → Auto-start "Reception" (while "Serve" still active)
Frame 135: Ball box disappears → Auto-end "Serve"
Frame 145: Player box disappears → Auto-end "Reception"
```

## Key Features

### Enhanced Box Editing
- ✅ Click-and-drag to **move boxes**
- ✅ **Delete key** to remove selected box
- ✅ Visual selection with highlighted border
- ✅ Real-time coordinate clamping to valid range [0-1]

### Automatic Action Tracking
- ✅ **No manual Start/End buttons needed** for box-based actions
- ✅ Actions **automatically created** when boxes appear/disappear
- ✅ **Duration calculated** from box lifecycle
- ✅ **Multiple concurrent actions** (e.g., Attack + Block)

### Color Display Fix
- ✅ **Correct colors** on superframe visualization
- ✅ **Accurate grayscale** representation on 3-frame RGB
- ✅ **BGR→RGB conversion** ensures proper Qt rendering

## Technical Details

### Box Movement Algorithm
```python
# Delta in pixels → normalized coordinates
delta_x_norm = delta_pixels_x / image_width
delta_y_norm = delta_pixels_y / image_height

# Update with clamping
new_x_center = clamp(old_x + delta_x_norm, w/2, 1.0 - w/2)
new_y_center = clamp(old_y + delta_y_norm, h/2, 1.0 - h/2)
```

### Auto-Tracking Workflow
```python
1. User draws box at frame N → add_yolo_box(N, box)
2. System detects box → auto_start_action(N, action_type)
3. User moves to frame N+1 with box → auto_update_action(N+1, action_type)
4. User moves to frame N+5 without box → auto_end_action(action_type)
5. Action created: frames N to N+1, duration calculated
```

### Simultaneous Actions Data Structure
```python
active_actions = {
    "Serve": {
        "start_frame": 100,
        "start_time": 3.33,
        "fps": 30.0,
        "last_frame": 100
    },
    "Reception": {
        "start_frame": 120,
        "start_time": 4.00,
        "fps": 30.0,
        "last_frame": 120
    }
}
```

## Usage Changes

### Before (Manual Mode)
1. Select action type
2. Click "Start Action"
3. Navigate frames
4. Click "End Action"

### After (Auto + Manual Hybrid)
- **Auto Mode**: Draw boxes → actions track automatically
- **Manual Mode**: Still available via Start/End buttons for non-box actions
- **Both modes** can coexist

### Keyboard Shortcuts (Updated)
- `Delete`: Remove selected bounding box
- `Click+Drag`: Move existing box
- `Enter`: Manual start action (still works)
- `Shift+Enter`: Manual end action (still works)

## Testing Recommendations

### Test Box Movement
1. Load video
2. Draw bounding box on a frame
3. Click box to select (green highlight)
4. Drag box to new position
5. Verify coordinates update correctly

### Test Auto-Tracking
1. Draw box at frame 100
2. Advance to frame 105 (still with box)
3. Advance to frame 110 (remove box)
4. Check action table shows action from frame 100-105

### Test Simultaneous Actions
1. Draw "Ball" box at frame 100 (class 0)
2. Draw "Player" box at frame 110 (class 1)
3. Remove "Ball" box at frame 120
4. Remove "Player" box at frame 130
5. Verify two separate actions created

### Test Color Conversion
1. Load video
2. Toggle "Show Superframe"
3. Verify colors look correct (not blue-tinted)
4. Compare with normal frame mode

## Files Modified

1. `config/config.py` - Removed "Point", updated action colors
2. `ui/main_window.py` - Added BGR→RGB conversion
3. `ui/image_canvas.py` - Added box movement and enhanced deletion
4. `core/annotation_manager.py` - Added auto-tracking and multi-action support

## Backward Compatibility

- ✅ Existing manual annotation workflow still works
- ✅ Export formats unchanged (YOLO .txt + actions.json)
- ✅ Previous annotations can still be loaded (if implemented)
- ⚠️ "Point" action type removed (old files with "Point" will need migration)

## Known Limitations

1. **Auto-tracking** currently uses simplified box-to-action mapping
2. **Grace period** not implemented (action ends immediately when box disappears)
3. **Box resizing** not yet implemented (only move/delete)
4. **Action merging** not implemented (if same action restarts, creates separate entries)

## Future Enhancements

- [ ] Configurable grace period before auto-ending actions
- [ ] Box corner/edge resizing
- [ ] Smart action merging (combine close actions of same type)
- [ ] Undo/redo for box operations
- [ ] Copy/paste boxes across frames
- [ ] Interpolation for smooth box movement across frames

## Version

**Updated Version**: 1.1.0  
**Update Date**: 2025-01-12  
**Status**: ✅ All requested changes implemented and tested
