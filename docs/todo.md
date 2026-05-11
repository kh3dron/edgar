# TODO

Roadmap, organized around the control flow in `docs/robot_arm.md`. Checkbox
state is the source of truth for what's built versus what's left.

## Setup / one-time

- [x] Connect to the arm over USB (`edgar.cli`)
- [x] Capture a frame from the overhead camera (`edgar.camera`)
- [x] Crop + corner-pick calibration (`board_vision pick`)
- [x] Perspective-warp to a top-down square (`board_vision.warp`)
- [ ] Calibrate arm ↔ board coordinates
  - [ ] Mount the board in a fixed pose relative to the arm
  - [ ] Teach-by-demo: jog arm to a few known grid points, fit a transform
  - [ ] Persist the calibration to disk
- [ ] Calibrate stone pickup positions
  - [ ] Black bowl pickup pose (XYZ + approach height)
  - [ ] White bowl pickup pose
  - [ ] Single-stone "flat" pickup as a fallback

## Vision pipeline (`read_board`)

- [x] Capture from configured camera
- [x] Rotate upright
- [x] Apply crop
- [x] Perspective-warp 4 corners to canonical square
- [x] CLAHE + saturation enhance (for visualization)
- [x] Per-cell median (L, S) features from un-enhanced warp
- [x] Multiplicative-threshold classifier (black/white/empty)
- [ ] Auto-refine grid alignment via Hough lines (handle small calibration drift)
- [ ] Sanity check: ≤ 1 new stone vs. previous board on human's turn
- [ ] Mask the lamp-shade vignette to avoid edge false positives

## Arm primitives

- [x] Connect / home / get state
- [x] Move to XYZ (interpolated + direct)
- [x] Move single joint (deg + rad), all joints, single axis
- [x] Gripper open / close, gripper-deg, gripper-rad
- [x] Torque lock, dynamic adaptation, continuous jog
- [ ] `pickup(color)` — pre-position, approach down, close, lift
- [ ] `place(x, y, z)` — pre-position above target, descend, open, lift
- [ ] Soft-collision safety: cap Z velocity near the board surface

## Game engine

- [ ] Spawn GnuGo as a subprocess in GTP mode
- [ ] Send `genmove`, parse response
- [ ] Track game state (turn, captures, ko, end conditions)
- [ ] Persist SGF of completed games

## Optional / later

- [ ] OGS bridge (play against online opponents)
- [ ] Capture trigger via voice or button instead of polling
- [ ] Failure recovery if vision and engine disagree
- [ ] Self-play mode for arm reliability testing

## Open challenges

- Calibration drift: arm + board + camera all need to stay put, or
  the calibration needs to be cheap enough to redo each session
- Stone pickup precision: clamp is not designed for thin flat objects;
  may need a custom 3D-printed tool or a felt pad on the gripper
- Removing captured stones without disturbing neighbors
- Lighting: the lamp casts a circular vignette in every frame; need
  to confirm color thresholds hold up across times of day
- Headless deploy: corner picker uses `cv2.imshow`; on a headless Pi
  we'll need either a web-based picker or auto board detection
