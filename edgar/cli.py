"""Interactive REPL for the RoArm-M2-S.

Run: python -m edgar.cli /dev/cu.usbserial-10
Then type commands like:
    home
    state
    xyz 235 0 234
    joints 0 0 90 180
    open
    close
    torque off
    raw {"T":105}
    quit
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys

from edgar.arm import Arm


HELP = """\
Commands:
  home                       T:100, return to initial pose
  state                      query T:105 and print parsed feedback
  xyz X Y Z [T]              T:104 interpolated move (mm, t in rad, default 3.14)
  xyz! X Y Z [T]             T:1041 direct (non-blocking) move
  axis {x|y|z|t} POS         T:103 single-axis move
  joints B S E H             T:122 all joints in degrees
  joint {b|s|e|h} ANGLE      T:121 single joint in degrees
  open                       open gripper
  close                      close gripper
  grip ANGLE                 set gripper angle in degrees (180=closed, 45=open)
  torque {on|off}            T:210
  jog {b|s|e|h} {+|-|0}      T:123 angle-mode jog
  jogxy {x|y|z|t} {+|-|0}    T:123 coord-mode jog
  stop                       stop any jog
  raw <json>                 send arbitrary JSON command
  help                       show this help
  quit                       exit
"""

AXIS_MAP = {"x": 1, "y": 2, "z": 3, "t": 4}
JOINT_MAP = {"b": 1, "s": 2, "e": 3, "h": 4}
DIR_MAP = {"+": 1, "-": 2, "0": 0}


def run_command(arm: Arm, line: str) -> bool:
    """Execute one REPL line. Returns False to exit."""
    try:
        parts = shlex.split(line)
    except ValueError as e:
        print(f"parse error: {e}")
        return True
    if not parts:
        return True
    cmd, *args = parts

    if cmd in ("quit", "exit", "q"):
        return False
    if cmd in ("help", "?"):
        print(HELP)
    elif cmd == "home":
        arm.home()
    elif cmd == "state":
        state = arm.get_state()
        print(state)
    elif cmd == "xyz":
        x, y, z = float(args[0]), float(args[1]), float(args[2])
        t = float(args[3]) if len(args) > 3 else 3.14
        arm.move_xyz(x, y, z, t=t)
    elif cmd == "xyz!":
        x, y, z = float(args[0]), float(args[1]), float(args[2])
        t = float(args[3]) if len(args) > 3 else 3.14
        arm.move_xyz_direct(x, y, z, t=t)
    elif cmd == "axis":
        arm.move_axis(AXIS_MAP[args[0]], float(args[1]))
    elif cmd == "joints":
        b, s, e, h = (float(a) for a in args[:4])
        arm.move_joints_deg(b, s, e, h)
    elif cmd == "joint":
        arm.move_joint_deg(JOINT_MAP[args[0]], float(args[1]))
    elif cmd == "open":
        arm.open_gripper()
    elif cmd == "close":
        arm.close_gripper()
    elif cmd == "grip":
        arm.gripper_deg(float(args[0]))
    elif cmd == "torque":
        arm.torque(args[0] == "on")
    elif cmd == "jog":
        arm.jog_angle(JOINT_MAP[args[0]], DIR_MAP[args[1]])
    elif cmd == "jogxy":
        arm.jog_coord(AXIS_MAP[args[0]], DIR_MAP[args[1]])
    elif cmd == "stop":
        arm.stop_jog()
    elif cmd == "raw":
        arm.send(json.loads(" ".join(args)))
    else:
        print(f"unknown command: {cmd!r}  (type 'help')")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="RoArm-M2-S REPL")
    parser.add_argument("port", help="serial port, e.g. /dev/cu.usbserial-10")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    with Arm(args.port, baud=args.baud) as arm:
        print(f"connected to {args.port} @ {args.baud}")
        print("type 'help' for commands, 'quit' to exit")
        while True:
            try:
                line = input("arm> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            try:
                if not run_command(arm, line):
                    break
            except Exception as e:
                print(f"error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
