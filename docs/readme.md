# edgar

Robot that plays Go against a computer program on a physical board.

## Layout

- `docs/robot_arm.md` — hardware notes and roadmap (control flow + TODO)
- `edgar/arm.py` — RoArm-M2-S serial driver
- `edgar/camera.py` — overhead webcam capture
- `edgar/cli.py` — interactive arm REPL
- `captures/` — saved frames (gitignored)

## Quick start

```
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m edgar.camera --select   # pick the board camera
.venv/bin/python -m edgar.cli /dev/cu.usbserial-10   # drive the arm
```
