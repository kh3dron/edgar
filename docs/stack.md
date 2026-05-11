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
- Vision: OpenCV — crop, perspective warp, CLAHE+saturation enhance, per-cell
  feature thresholding (`edgar/board_vision.py`)
- Game engine: GnuGo (probably via GTP); optional hook to OGS later
- Coordinate map: board grid (A1..T19) → arm XYZ

## Per-turn control flow

Pseudocode for the game loop. Each commented line is a primitive that exists
or needs to be built — checkboxes below track which.

```python
while not game_over:
    if computer_to_move:
        move = gnugo.genmove()                          # [ ] GnuGo GTP wrapper
        x, y, z = board_to_arm(move)                    # [ ] calibration map
        arm.pickup(stone_color)                         # [ ] pickup primitive
        arm.place(x, y, z)                              # [ ] place primitive
        arm.home()                                      # [x] arm.home()
    else:
        wait_for_human_move()                           # [ ] trigger: button or motion
        frame = camera.capture()                        # [x] camera.capture()
        board = read_board(frame)                       # [x] board_vision.run_pipeline
        move = diff(previous_board, board)              # [ ] state diff
        if not gnugo.play(move):
            handle_invalid_move()                       # [ ] error recovery
    previous_board = board
```
