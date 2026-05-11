# Robot arm Go

Goal: play Go against a computer program on a physical board.

- Computer move: program picks a move, arm places a stone
- Human move: I place a stone, camera reads the board, GnuGo continues

## Hardware

- Waveshare RoArm-M2-S, USB serial @ 115200 (CP2102N bridge)
- Raspberry Pi 5 (8 GB) — eventual host; currently developing on a Mac
- HD Webcam eMeet C960, hung above the board

## Software stack

- Arm driver: JSON-over-serial, see `edgar/arm.py`
- Vision: OpenCV — board detection, grid extraction, stone classification
- Game engine: GnuGo (probably via GTP); optional hook to OGS later
- Coordinate map: board grid (A1..T19) → arm XYZ

## Roadmap (control flow)

The game loop, written as a TODO list. Each step is something that has to
exist before the loop closes. Sub-bullets are the concrete work.

### Setup / one-time

- [x] Connect to the arm over USB
- [x] Capture a frame from the overhead camera
- [ ] Detect the board in a captured frame
  - [ ] Find the four outer corners of the grid
  - [ ] Perspective-warp to a square top-down view
  - [ ] Cache the warp until the camera or board moves
- [ ] Calibrate arm ↔ board coordinates
  - [ ] Mount the board in a fixed pose relative to the arm
  - [ ] Teach-by-demo: jog arm to a few known grid points, fit a transform
  - [ ] Persist the calibration to disk
- [ ] Calibrate stone pickup positions
  - [ ] Black bowl pickup pose (XYZ + approach height)
  - [ ] White bowl pickup pose
  - [ ] Single-stone "flat" pickup as a fallback

### Per-turn loop

```
while game_in_progress:
    if computer_to_move:
        move = gnugo.genmove()                          # [ ] GnuGo GTP wrapper
        x, y, z = board_to_arm(move)                    # [ ] calibration map
        arm.pickup(stone_color)                         # [ ] pickup primitive
        arm.place(x, y, z)                              # [ ] place primitive
        arm.home()                                      # [x] arm.home()
    else:
        wait_for_human_move()                           # [ ] trigger: button or motion
        frame = camera.capture()                        # [x] camera.capture()
        board = read_board(frame)                       # [ ] vision pipeline
        move = diff(previous_board, board)              # [ ] state diff
        if not gnugo.play(move):
            handle_invalid_move()                       # [ ] error recovery
    previous_board = board
    if gnugo.game_over():
        break
```

### Vision pipeline (`read_board`)

- [ ] Crop/rotate to the warped board view
- [ ] For each of the 361 intersections, sample a patch
- [ ] Classify patch as empty / black / white
  - [ ] Color thresholds in HSV, calibrated per session under the lamp
  - [ ] Mask out the lamp-shade vignette
- [ ] Sanity check: ≤ 1 new stone vs. previous board on human's turn

### Arm primitives

- [x] Connect / home / get state
- [x] Move to XYZ (interpolated + direct)
- [x] Gripper open / close
- [ ] `pickup(color)` — pre-position, approach down, close, lift
- [ ] `place(x, y, z)` — pre-position above target, descend, open, lift
- [ ] Soft-collision safety: cap Z velocity near the board surface

### Game engine

- [ ] Spawn GnuGo as a subprocess in GTP mode
- [ ] Send `genmove`, parse response
- [ ] Track game state (turn, captures, ko, end conditions)
- [ ] Persist SGF of completed games

### Optional / later

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
