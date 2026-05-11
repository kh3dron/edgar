# edgar

Robot that plays Go against a computer program on a physical board.

## Layout

- `docs/robot_arm.md` — hardware notes, setup commands, control flow
- `docs/todo.md` — roadmap (checkboxes)
- `Taskfile.yml` — runnable tasks (`task`, `task calibrate-camera`, ...)
- `edgar/arm.py` — RoArm-M2-S serial driver
- `edgar/camera.py` — overhead webcam capture
- `edgar/board_vision.py` — capture → crop → warp → enhance → classify
- `edgar/cli.py` — interactive arm REPL
- `captures/` — saved frames (gitignored)
- `.camera_index`, `.board_config.json` — local calibration state (gitignored)

## Quick start

Uses [uv](https://docs.astral.sh/uv/) and [task](https://taskfile.dev).

```
task setup
task              # lists available tasks
```

See `docs/robot_arm.md` for the command flow.
