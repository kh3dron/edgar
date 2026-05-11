"""Teach-by-demo calibration: jog the arm to each board corner, save XYZ.

Run: python -m edgar.arm_calibration /dev/cu.usbserial-10

Keybindings during jog:
  w / s          +Y / -Y           (forward / back)
  a / d          -X / +X           (left / right)
  up / down      +Z / -Z           (raise / lower the gripper)
  left / right   wrist -/+         (twist the EoAT pitch)
  [ / ]          halve / double the linear step
  l              toggle the on-arm light
  enter          capture the current pose for the active corner
  r              redo the active corner (clears its capture)
  ?              print current arm state
  esc / q        quit

After all four corners are captured, an affine
  (col, row, 1) -> (x, y, z)
is fit by least-squares and written to .arm_calibration.json alongside the
raw corners and the wrist angle that was active at calibration time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import termios
import tty
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from edgar.arm import Arm

CONFIG_PATH = Path(".arm_calibration.json")

# Standard Go column letters (skips 'I' to avoid 1/I confusion).
GO_COLS = "ABCDEFGHJKLMNOPQRST"
GRID = 19

# Same TL/TR/BR/BL ordering used by the board_vision corner picker.
CORNER_ORDER = ["A19", "T19", "T1", "A1"]

DEFAULT_STEP_MM = 5.0
MIN_STEP_MM = 0.5
MAX_STEP_MM = 40.0
WRIST_STEP_RAD = 0.05

# Wrist angle the API treats as the default end-effector pose, which on
# this arm has the gripper pointing straight down at the workspace.
WRIST_DOWN_RAD = math.pi


def label_to_grid(label: str) -> tuple[int, int]:
    """'A1' -> (col=0, row=0), 'T19' -> (18, 18)."""
    col_letter = label[0].upper()
    row_num = int(label[1:])
    if col_letter not in GO_COLS:
        raise ValueError(f"bad column letter {col_letter!r}")
    if not 1 <= row_num <= GRID:
        raise ValueError(f"row {row_num} out of range")
    return GO_COLS.index(col_letter), row_num - 1


# -- calibration data --------------------------------------------------


@dataclass
class ArmCalibration:
    # label -> [x, y, z] arm coordinates, in mm.
    corners: dict[str, list[float]] = field(default_factory=dict)
    # Wrist angle (T:104 't' parameter, radians) held constant during teach.
    wrist_t: float = WRIST_DOWN_RAD

    @classmethod
    def load(cls) -> "ArmCalibration":
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"{CONFIG_PATH} not found. "
                f"Run `python -m edgar.arm_calibration <port>` first."
            )
        return cls(**json.loads(CONFIG_PATH.read_text()))

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))

    def fit(self) -> np.ndarray:
        """Fit a (col, row) -> (x, y, z) affine. Returns a 3x3 matrix M.

        For label L with grid (c, r): [x, y, z]^T = M @ [c, r, 1]^T.
        With 4 corners we have 4 equations per output and 3 unknowns;
        least-squares handles slight non-planarity in the taught points.
        """
        if len(self.corners) < 3:
            raise ValueError("need at least 3 corners to fit an affine")
        rows = []
        xs, ys, zs = [], [], []
        for label, (x, y, z) in self.corners.items():
            c, r = label_to_grid(label)
            rows.append([c, r, 1.0])
            xs.append(x); ys.append(y); zs.append(z)
        A = np.asarray(rows, dtype=float)
        M = np.zeros((3, 3), dtype=float)
        for i, B in enumerate((xs, ys, zs)):
            sol, *_ = np.linalg.lstsq(A, np.asarray(B, dtype=float), rcond=None)
            M[i] = sol
        return M

    def board_to_arm(self, label: str) -> tuple[float, float, float]:
        M = self.fit()
        c, r = label_to_grid(label)
        x, y, z = M @ np.array([c, r, 1.0])
        return float(x), float(y), float(z)


# -- key reader --------------------------------------------------------


class RawTerminal:
    """Put stdin into cbreak so we read one keystroke at a time, no echo."""

    def __enter__(self) -> "RawTerminal":
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc: object) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def read_key() -> str:
    """Read one logical keypress. Returns a single char, or a token like
    'UP'/'DOWN'/'LEFT'/'RIGHT'/'ENTER'/'ESC' for special keys.
    """
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        seq = sys.stdin.read(2)
        return {
            "[A": "UP",
            "[B": "DOWN",
            "[C": "RIGHT",
            "[D": "LEFT",
        }.get(seq, "ESC")
    if ch in ("\r", "\n"):
        return "ENTER"
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch


# -- interactive loop --------------------------------------------------


def status(
    x: float, y: float, z: float, t: float, step: float, target: str, light: bool
) -> None:
    sys.stdout.write(
        f"\r  [{target}]  x={x:7.1f}  y={y:7.1f}  z={z:7.1f}  "
        f"wrist={t:+.2f}  step={step:4.1f}mm  light={'ON ' if light else 'off'}    "
    )
    sys.stdout.flush()


def jog_to_corners(arm: Arm) -> ArmCalibration:
    """Walk the user through capturing each corner in CORNER_ORDER."""
    state = arm.get_state()
    x, y, z = state.x, state.y, state.z
    t = WRIST_DOWN_RAD
    step = DEFAULT_STEP_MM
    light_on = False
    cal = ArmCalibration(wrist_t=t)

    # Orient the gripper straight down before the user starts jogging.
    arm.move_xyz_direct(x, y, z, t=t)

    print(f"starting pose: x={x:.1f} y={y:.1f} z={z:.1f}")
    print(f"wrist set to {t:.3f} rad (straight down)")
    print()

    with RawTerminal():
        i = 0
        while i < len(CORNER_ORDER):
            target = CORNER_ORDER[i]
            print(f"--- corner {i + 1}/{len(CORNER_ORDER)}: {target} ---")
            print("  jog the gripper tip onto the intersection, then press ENTER")
            while True:
                status(x, y, z, t, step, target, light_on)
                key = read_key()
                dx = dy = dz = dt = 0.0
                if key == "w":
                    dy = +step
                elif key == "s":
                    dy = -step
                elif key == "a":
                    dx = -step
                elif key == "d":
                    dx = +step
                elif key == "UP":
                    dz = +step
                elif key == "DOWN":
                    dz = -step
                elif key == "LEFT":
                    dt = -WRIST_STEP_RAD
                elif key == "RIGHT":
                    dt = +WRIST_STEP_RAD
                elif key == "[":
                    step = max(step / 2, MIN_STEP_MM)
                elif key == "]":
                    step = min(step * 2, MAX_STEP_MM)
                elif key == "l":
                    light_on = not light_on
                    arm.light(255 if light_on else 0)
                elif key == "?":
                    s = arm.get_state()
                    sys.stdout.write(
                        f"\n  arm reports x={s.x:.1f} y={s.y:.1f} z={s.z:.1f} "
                        f"t={s.eoat:.3f}\n"
                    )
                elif key == "r":
                    cal.corners.pop(target, None)
                    sys.stdout.write(f"\n  cleared {target}\n")
                elif key in ("q", "ESC"):
                    raise KeyboardInterrupt
                elif key == "ENTER":
                    cal.corners[target] = [x, y, z]
                    cal.wrist_t = t
                    sys.stdout.write(
                        f"\n  captured {target} = "
                        f"({x:.1f}, {y:.1f}, {z:.1f})\n"
                    )
                    i += 1
                    break
                if dx or dy or dz or dt:
                    x += dx; y += dy; z += dz; t += dt
                    arm.move_xyz_direct(x, y, z, t=t)
    return cal


def report_fit(cal: ArmCalibration) -> None:
    """Print residuals and a few interpolated grid points as a sanity check."""
    M = cal.fit()
    print("\nfit residuals (corner - affine prediction, mm):")
    for label, (x, y, z) in cal.corners.items():
        c, r = label_to_grid(label)
        px, py, pz = M @ np.array([c, r, 1.0])
        print(
            f"  {label}: "
            f"dx={x - px:+5.2f}  dy={y - py:+5.2f}  dz={z - pz:+5.2f}"
        )
    print("\nspot checks (interpolated arm XYZ for a few intersections):")
    for label in ("K10", "A1", "T19", "D4", "Q16"):
        x, y, z = cal.board_to_arm(label)
        print(f"  {label}: x={x:7.2f}  y={y:7.2f}  z={z:7.2f}")


# -- entry point -------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="Arm <-> board calibration")
    p.add_argument("port", help="serial port, e.g. /dev/cu.usbserial-10")
    p.add_argument("--baud", type=int, default=115200)
    args = p.parse_args()

    with Arm(args.port, baud=args.baud) as arm:
        print(f"connected to {args.port} @ {args.baud}")
        print("the arm will rotate its wrist to straight-down on startup.\n")
        try:
            cal = jog_to_corners(arm)
        except KeyboardInterrupt:
            try:
                arm.light(0)
            except Exception:
                pass
            print("\naborted.")
            return 1
        try:
            arm.light(0)
        except Exception:
            pass

    cal.save()
    print(f"\nwrote {CONFIG_PATH}")
    report_fit(cal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
