# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AirCanvas is a gesture-controlled virtual presentation system built with Python. Users can draw in the air, control PowerPoint presentations, and interact with on-screen content through hand gestures detected via webcam using MediaPipe Hands.

## Development Environment Setup

1. **Activate virtual environment:**
   ```bash
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Unix/macOS
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

## Runtime Controls

- **q**: Quit application
- **c**: Clear canvas
- **s**: Save canvas to `captures/canvas_N.png`
- **f**: Toggle PPT ink mode (for drawing directly in PowerPoint presentations)
- **h**: Show/hide help overlay
- **1**: Toggle particle effects on/off
- **2**: Toggle laser pointer on/off
- **3**: Toggle palm HUD on/off
- **r**: Reset presentation timer

## Architecture

### Core Pipeline

The main loop in `main.py` follows this sequence:
1. Capture and flip camera frame
2. Detect hands via MediaPipe (`HandDetector`)
3. Classify gesture (`GestureRecognizer`)
4. Map coordinates to appropriate space (`CoordinateMapper`)
5. Execute mode-specific actions (draw, erase, move cursor, PPT control)
6. Apply shape beautification on stroke completion
7. **Update particle system** (Stage 4)
8. Overlay canvas onto camera feed
9. **Render particle effects** (Stage 4)
10. **Render laser pointer** (Stage 4)
11. **Render palm HUD** (Stage 4)
12. Display UI indicators

### Hand Detection & Gesture Recognition

**HandDetector** (`core/hand_detector.py`):
- Wraps MediaPipe Hands to detect up to 1 hand per frame
- Returns `Hand` dataclass with:
  - `landmarks`: 21 keypoints in pixel coordinates
  - `landmarks_norm`: 21 keypoints in normalized (0-1) coordinates
  - `bbox`: bounding box
  - `handedness`: "LEFT" or "RIGHT"
  - `confidence`: detection confidence score

**Landmark indices** (exported from hand_detector.py):
- WRIST = 0, THUMB_TIP = 4, INDEX_TIP = 8, MIDDLE_TIP = 12, RING_TIP = 16, PINKY_TIP = 20

**GestureRecognizer** (`core/gesture_recognizer.py`):
- Uses finger-up detection logic (tip.y < base.y for index/middle/ring/pinky; tip.x vs base.x for thumb depending on handedness)
- Implements hysteresis thresholds for pinch gesture to prevent jitter:
  - `pinch_threshold`: distance to trigger pinch (default 0.035)
  - `pinch_release_threshold`: distance to release pinch (default 0.11)
- Tracks wrist position history (15 frames) to detect swipe gestures
- Requires both displacement (`swipe_threshold`) and velocity (`swipe_velocity_threshold`) to trigger swipe
- Uses frame-based debouncing (`pinch_confirm_frames`, `swipe_cooldown_frames`)
- Returns dict with: `mode`, `fingers`, `pinching`, `pinch_start`, `pinch_end`, `open_palm`, `fist`, `index_only`, `index_middle`, `swipe`, `pinch_distance`

**Gesture modes:**
- **draw**: Pinch detected (thumb + index tips close)
- **erase**: Open palm (4+ fingers up)
- **move**: Index finger only (cursor control)
- **pause**: Fist (all fingers down)
- **click**: Index + middle fingers up
- **swipe**: Fast wrist movement (SWIPE_LEFT/RIGHT/UP/DOWN)

### Coordinate Mapping

**CoordinateMapper** (`core/coordinate_mapper.py`):
- Maps normalized (0-1) camera coordinates to screen pixels
- Supports active region cropping (e.g., only use center 70% of frame to avoid edge-reaching)
- Applies exponential moving average (EMA) smoothing with configurable `smoothing_factor`
- Two separate instances in `main.py`:
  - `draw_mapper`: maps to camera resolution for canvas drawing
  - `cursor_mapper`: maps to screen resolution for mouse control

### Drawing System

**Canvas** (`modules/canvas.py`):
- Manages a numpy array (camera width × height × 3 channels) initialized to black
- Methods: `draw_line()`, `erase()` (draws black circle), `clear()`, `save()`
- Canvas is overlaid onto camera frame using `overlay_canvas()` in main.py (composites non-black pixels)

**VirtualPen** (`modules/virtual_pen.py`):
- Maintains `prev_point` to draw continuous strokes
- Collects all points in current stroke for shape recognition
- `start_stroke()`: resets prev_point and clears point list
- `draw()`: draws line from prev_point to current point
- `end_stroke()`: returns collected points and resets
- Optionally applies `EmaSmoother` for additional smoothing (currently disabled in favor of mapper smoothing)

**Eraser** (`modules/eraser.py`):
- Simple wrapper that draws black circles on canvas at specified position

**ShapeRecognizer** (`modules/shape_recognizer.py`):
- Triggered on `pinch_end` event with finished stroke points
- Analyzes stroke for geometric properties:
  - Closedness: ratio of endpoint distance to perimeter
  - Vertex count via `cv2.approxPolyDP`
  - Circularity via `cv2.minEnclosingCircle` area ratio
- Recognizes: circle, rectangle, triangle
- `beautify()` replaces hand-drawn stroke with clean geometric shape on canvas

### PPT Control

**PPTController** (`modules/ppt_controller.py`):
- Uses `pyautogui` to simulate keyboard presses
- Methods: `next_slide()` (right arrow), `prev_slide()` (left), `first_slide()` (home), `last_slide()` (end)
- Swipe gestures trigger PPT navigation

**PPT Ink Mode** (toggled with 'f' key):
- When enabled, pinch gestures control system mouse (drag) for drawing directly in PowerPoint
- Sends `Ctrl+P` on entry, `Ctrl+A` on exit to toggle PowerPoint pen mode
- Open palm triggers `Ctrl+E` or `E` to switch to eraser in PPT
- Local canvas drawing is disabled to avoid conflicts

### AR Enhancement Effects (Stage 4)

**ParticleSystem** (`modules/particle_system.py`):
- Creates trailing particle effects when drawing
- Each particle has: position, velocity, lifetime, color, size
- Applies gravity and fade-out effects
- Vectorized update for performance
- Emits 5 particles per frame during drawing
- Max 300 particles simultaneously
- Multi-layer rendering for glow effect

**LaserPointer** (`modules/laser_pointer.py`):
- Renders red laser dot at fingertip in "move" mode (index finger only)
- Three-layer rendering: outer glow, middle ring, bright inner core
- Optional trail effect showing recent positions
- Alpha blending for transparency

**PalmHUD** (`modules/palm_hud.py`):
- Displays information overlay when palm is still for 1 second
- Shows: current time, presentation duration, custom text
- Tracks palm position history to detect stillness
- Variance-based stillness detection (threshold: 0.02)
- Semi-transparent background with cyan text
- Auto-resets when hand removed or moves

### Smoothing

**EmaSmoother** (`utils/smoothing.py`):
- Exponential moving average filter for point sequences
- Formula: `state = alpha * state + (1-alpha) * new_value`
- Used by `CoordinateMapper` for coordinate smoothing

### Configuration

All tunable parameters in `config.py`:
- Camera settings (resolution, ID)
- Screen resolution
- Active regions (normalized coordinates)
- Gesture thresholds (pinch, swipe, velocity, cooldown, confirm frames)
- Drawing parameters (pen color, thickness, eraser size)
- Smoothing factors (separate for cursor vs drawing)

## Key Technical Details

### Preventing Jitter and False Triggers

1. **Hysteresis thresholds**: Pinch uses different thresholds for activation vs release to create a "dead zone"
2. **Frame-based confirmation**: Pinch requires N consecutive frames below/above threshold before state change
3. **Swipe cooldown**: After detecting swipe, ignore further swipes for N frames
4. **Draw lock**: After `pinch_end`, prevent drawing for 6 frames to avoid reconnecting strokes
5. **EMA smoothing**: Applied at coordinate mapper level to reduce hand tremor

### Dual Coordinate Spaces

- **Camera space**: Used for canvas drawing (1280×720)
- **Screen space**: Used for mouse cursor control (system display resolution)
- Two separate `CoordinateMapper` instances with different smoothing factors
- Draw mapper uses palm center for eraser, midpoint of thumb+index for pen tip during pinch

### Frame Processing Order

Critical that operations happen in this sequence to avoid visual artifacts:
1. Detect hand → classify gesture → execute actions → update canvas
2. Update particle system physics
3. Overlay canvas on frame (non-black pixels only)
4. Render particles on frame
5. Render laser pointer (if in move mode)
6. Render palm HUD (if palm is still)
7. Draw UI overlays (indicators, text) on final composited frame

### PPT Ink Mode vs Canvas Mode

- **Canvas mode** (default): Draws on local transparent canvas overlaid on camera feed
- **PPT ink mode**: Directly manipulates system mouse, bypasses canvas for real-time annotation
- Mode switch handled via `PPT_MODE` flag, ensures mutual exclusion

## Common Development Tasks

When adding new gestures:
1. Add finger-counting logic to `GestureRecognizer.fingers_up()` if needed
2. Define new mode/condition in `classify()` method
3. Add corresponding action handler in main loop's mode dispatch
4. Consider adding hysteresis/cooldown/debouncing if gesture is prone to false triggers

When adjusting thresholds:
- Modify `config.py` values
- No code changes required - all parameters injected via constructor args

When adding new drawing features:
- Extend `Canvas` for rendering primitives
- Wrap in a module class (like `VirtualPen`, `Eraser`) for state management
- Instantiate in `main.py` and call within appropriate gesture mode branch

When working with AR effects:
- Particle emission should happen during drawing actions
- Laser pointer only renders in "move" mode
- Palm HUD requires stillness detection before showing
- All AR effects have toggle switches (keys 1/2/3) for debugging
- AR effects render after canvas overlay but before UI text
