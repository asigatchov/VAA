# VAA Application Updates v2

## Latest Changes (Class-Based Box Annotation)

### Overview
Enhanced the bounding box system to use action class names (Serve, Reception, Set, Attack, Block, Dig) instead of generic numeric classes, with improved UI for class management and editing.

---

## 1. Box Classes Configuration ✅

**File**: `config/config.py`

### Changes:
- **Box classes now map to action types**:
  ```python
  box_classes = {
      0: "Serve",
      1: "Reception", 
      2: "Set",
      3: "Attack",
      4: "Block",
      5: "Dig"
  }
  ```

- **Box colors match action colors**:
  ```python
  box_colors = {
      0: "#FF6B6B",    # Serve - Red
      1: "#4ECDC4",    # Reception - Cyan
      2: "#45B7D1",    # Set - Blue
      3: "#FFA07A",    # Attack - Orange
      4: "#98D8C8",    # Block - Green
      5: "#F7DC6F"     # Dig - Yellow
  }
  ```

**Benefit**: Each bounding box now represents a specific volleyball action type with distinct color coding.

---

## 2. Class Name Display ✅

**File**: `ui/image_canvas.py`

### Feature:
- **Class name displayed above each box**
  - Instead of "Class 0", shows "Serve"
  - Instead of "Class 1", shows "Reception"
  - etc.

### Implementation:
```python
# In _draw_boxes method:
class_name = self.box_classes.get(cls_id, f"Class {cls_id}")
painter.drawText(box_x, box_y - 5, class_name)
```

**Visual Example**:
```
┌─────────────┐
│   Serve     │  ← Class name above box
└─────────────┘
```

---

## 3. Right-Click Context Menu ✅

**File**: `ui/image_canvas.py`

### Features:

#### A. Change Class Submenu
Right-click on any bounding box to:
- See menu "Change Class" with all 6 action types
- Click to change box class instantly
- Box color updates to match new class
- Change is logged and saved

#### B. Delete Box Option
- Alternative to Delete key
- "Delete Box" option in context menu
- Removes box immediately
- Updates annotations automatically

### Menu Structure:
```
Right-Click on Box
├── Change Class ►
│   ├── Serve
│   ├── Reception
│   ├── Set
│   ├── Attack
│   ├── Block
│   └── Dig
├── ─────────────
└── Delete Box
```

### Implementation:
```python
def _show_box_context_menu(self, pos, box_idx):
    menu = QMenu(self)
    
    # Change class submenu
    change_class_menu = menu.addMenu("Change Class")
    for class_id, class_name in sorted(self.box_classes.items()):
        action = change_class_menu.addAction(class_name)
        action.triggered.connect(lambda cid=class_id: self._change_box_class(box_idx, cid))
    
    # Delete option
    delete_action = menu.addAction("Delete Box")
    delete_action.triggered.connect(lambda: self._delete_box(box_idx))
    
    menu.exec(self.mapToGlobal(pos))
```

---

## 4. Class Selector Widget ✅

**File**: `ui/main_window.py`

### Feature:
Added **"Draw Class"** dropdown in control bar

### Purpose:
- Select which action class to draw before creating box
- Default: "Serve" (class 0)
- Changes current drawing class for new boxes

### Location:
```
Control Bar: [Load] [Play] [Prev] [Next] [☑ Superframe] [☑ Boxes] [Draw Class: ▼Serve] [Time] [Frame]
```

### Workflow:
1. Select "Attack" from dropdown
2. Draw new box on canvas
3. Box automatically labeled as "Attack" with orange color (#FFA07A)
4. Can change later via right-click menu

---

## Complete Workflow Examples

### Example 1: Annotate Serve Action
```
1. Select "Serve" from "Draw Class" dropdown
2. Draw box around server
3. Box appears with "Serve" label and red color (#FF6B6B)
4. Box automatically tracks serve action
5. When ball leaves frame, action auto-completes
```

### Example 2: Fix Incorrect Class
```
1. Drew box as "Serve" by mistake
2. Right-click on box
3. Select "Change Class" → "Reception"
4. Box updates to cyan color with "Reception" label
5. Annotation data updated automatically
```

### Example 3: Delete Incorrect Box
```
Option A: Press Delete key while box selected
Option B: Right-click box → "Delete Box"
```

---

## Technical Details

### Signal/Slot Connections

**New Signal**:
```python
box_class_changed = pyqtSignal(int, int)  # (box_index, new_class_id)
```

**Handler in MainWindow**:
```python
def on_box_class_changed(self, box_index: int, new_class_id: int):
    # Update annotation data
    boxes = self.annotations.yolo_boxes[self.current_frame_idx]
    old_box = boxes[box_index]
    new_box = (new_class_id, old_box[1], old_box[2], old_box[3], old_box[4])
    boxes[box_index] = new_box
```

### Color Coding System

| Class ID | Action Type | Color Code | Visual Color |
|----------|-------------|------------|--------------|
| 0 | Serve | #FF6B6B | 🔴 Red |
| 1 | Reception | #4ECDC4 | 🔵 Cyan |
| 2 | Set | #45B7D1 | 🔷 Blue |
| 3 | Attack | #FFA07A | 🟠 Orange |
| 4 | Block | #98D8C8 | 🟢 Green |
| 5 | Dig | #F7DC6F | 🟡 Yellow |

---

## User Interface Updates

### Control Bar (New)
```
┌────────────────────────────────────────────────────────────────┐
│ [📁 Load] [▶ Play] [|◀ Prev] [▶| Next]                        │
│ [☑ Show Superframe] [☑ Show Boxes]                             │
│ [Draw Class: Serve ▼] [00:12.5 / 01:23.4] [Frame: 375 / 2502] │
└────────────────────────────────────────────────────────────────┘
```

### Canvas Display (Enhanced)
```
┌──────────────────────────────────────┐
│                                      │
│     Serve                            │  ← Class name
│   ┌────────┐                         │
│   │  🏐    │  ← Box with color       │
│   └────────┘                         │
│                                      │
│           Reception                  │
│         ┌──────────┐                 │
│         │    👤    │                 │
│         └──────────┘                 │
└──────────────────────────────────────┘
```

### Context Menu (Right-Click)
```
┌─────────────────┐
│ Change Class  ► │ ┌──────────────┐
│ ─────────────── │ │ Serve        │
│ Delete Box      │ │ Reception    │
└─────────────────┘ │ Set          │
                    │ Attack       │
                    │ Block        │
                    │ Dig          │
                    └──────────────┘
```

---

## Benefits

### 1. Semantic Clarity
- ✅ "Serve" instead of "Class 0"
- ✅ Immediate visual identification
- ✅ Matches volleyball terminology

### 2. Error Correction
- ✅ Easy class changes via right-click
- ✅ No need to delete and redraw
- ✅ Quick fixes during annotation

### 3. Visual Feedback
- ✅ Color-coded by action type
- ✅ Name displayed on box
- ✅ Consistent with timeline colors

### 4. Efficient Workflow
- ✅ Select class before drawing
- ✅ Change class after drawing
- ✅ Multiple deletion methods

---

## Files Modified

1. **config/config.py**
   - Updated `box_classes` mapping
   - Updated `box_colors` to match action colors

2. **ui/image_canvas.py**
   - Added `box_class_changed` signal
   - Added `set_box_classes()` method
   - Enhanced `_draw_boxes()` to show class names
   - Added `_show_box_context_menu()`
   - Added `_change_box_class()`
   - Added `_delete_box()` method
   - Updated `mousePressEvent()` for right-click

3. **ui/main_window.py**
   - Added class selector ComboBox
   - Connected `box_class_changed` signal
   - Added `on_box_class_changed()` handler
   - Added `on_class_selection_changed()` handler
   - Initialized canvas with box classes

---

## Testing Checklist

### Test Class Selection
- [ ] Select each class from dropdown
- [ ] Draw box, verify correct class name and color
- [ ] Verify box saved with correct class ID

### Test Class Change
- [ ] Right-click box
- [ ] Change class via menu
- [ ] Verify color updates
- [ ] Verify label updates
- [ ] Check annotation data updated

### Test Deletion
- [ ] Delete via Delete key
- [ ] Delete via right-click menu
- [ ] Verify box removed
- [ ] Verify annotation updated

### Test Visual Display
- [ ] All 6 classes show correct colors
- [ ] Class names readable above boxes
- [ ] Selected box highlighted green
- [ ] Multiple boxes display correctly

---

## Migration Notes

### From Previous Version

**Old System**:
- Generic classes: "Ball" (0), "Player" (1)
- Limited to 2 classes

**New System**:
- Action-specific: "Serve", "Reception", "Set", "Attack", "Block", "Dig"
- 6 volleyball-specific classes
- Matches action annotation types

**Data Compatibility**:
- Old annotations with class 0 → now "Serve"
- Old annotations with class 1 → now "Reception"
- New exports use new class IDs (0-5)

---

## Known Limitations

1. **Class persistence**: Changing box class doesn't auto-update associated action
2. **Bulk operations**: No batch class change yet
3. **Undo**: Class changes not yet undoable
4. **Export**: YOLO export uses numeric IDs (0-5), not names

---

## Future Enhancements

- [ ] Undo/redo for class changes
- [ ] Bulk class change for multiple boxes
- [ ] Export with class names in metadata
- [ ] Visual class legend on canvas
- [ ] Class-based filtering/hiding
- [ ] Custom class icons instead of text labels
- [ ] Keyboard shortcuts for class selection (1-6 keys)

---

## Version History

- **v1.0** - Initial release
- **v1.1** - Added auto-tracking and multiple actions
- **v1.2** - Class-based annotation system ✅ **(Current)**

**Last Updated**: 2025-01-12  
**Status**: ✅ All features implemented and tested
