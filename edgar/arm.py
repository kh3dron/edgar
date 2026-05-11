"""Driver for the Waveshare RoArm-M2-S over USB serial.

Protocol reference:
- https://www.waveshare.com/wiki/RoArm-M2-S_JSON_Command_Meaning
- https://www.waveshare.com/wiki/RoArm-M2-S_Robotic_Arm_Control

Commands are line-delimited JSON. The arm replies with JSON objects keyed by
"T" (e.g. T:1051 for the feedback response to a T:105 query).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any

import serial

DEFAULT_BAUD = 115200

# Joint indices used by T:101 / T:121 (single-joint control).
JOINT_BASE = 1
JOINT_SHOULDER = 2
JOINT_ELBOW = 3
JOINT_EOAT = 4

# Cartesian axis indices used by T:103 (single-axis control).
AXIS_X = 1
AXIS_Y = 2
AXIS_Z = 3
AXIS_T = 4  # gripper/wrist angle in radians

# EoAT angle range in degrees. 180 = clamp closed, ~45 = clamp fully open.
GRIPPER_CLOSED_DEG = 180.0
GRIPPER_OPEN_DEG = 45.0


@dataclass
class ArmState:
    """Snapshot of arm state from a T:105 feedback response (T:1051)."""

    x: float
    y: float
    z: float
    # Joint angles in radians.
    base: float
    shoulder: float
    elbow: float
    eoat: float
    # Per-joint torque load.
    torque_base: int
    torque_shoulder: int
    torque_elbow: int
    torque_eoat: int
    # Torque-lock status (1 = on, 0 = off).
    torque_lock_base: int
    torque_lock_shoulder: int
    torque_lock_elbow: int
    torque_lock_eoat: int
    # Bus voltage in volts (raw field is 0.01V units).
    voltage: float

    @classmethod
    def from_feedback(cls, msg: dict[str, Any]) -> "ArmState":
        return cls(
            x=msg["x"],
            y=msg["y"],
            z=msg["z"],
            base=msg["b"],
            shoulder=msg["s"],
            elbow=msg["e"],
            eoat=msg["t"],
            torque_base=msg["torB"],
            torque_shoulder=msg["torS"],
            torque_elbow=msg["torE"],
            torque_eoat=msg["torH"],
            torque_lock_base=msg["torswitchB"],
            torque_lock_shoulder=msg["torswitchS"],
            torque_lock_elbow=msg["torswitchE"],
            torque_lock_eoat=msg["torswitchH"],
            voltage=msg["v"] / 100.0,
        )


class Arm:
    """Synchronous client for the RoArm-M2-S.

    Usage:
        with Arm("/dev/cu.usbserial-10") as arm:
            arm.home()
            arm.move_xyz(235, 0, 234, t=3.14)
            print(arm.get_state())
    """

    def __init__(
        self,
        port: str,
        baud: int = DEFAULT_BAUD,
        timeout: float = 1.0,
    ) -> None:
        self.port = port
        self.baud = baud
        self._timeout = timeout
        self._ser: serial.Serial | None = None
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._feedback_q: Queue[dict[str, Any]] = Queue()
        self._write_lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------

    def open(self) -> "Arm":
        # dsrdtr=None and explicit RTS/DTR=False keeps the ESP32 from rebooting
        # when the port is opened (see Waveshare's serial_simple_ctrl.py).
        # A short read timeout lets the reader thread return partial chunks
        # instead of blocking until a full buffer arrives.
        self._ser = serial.Serial(
            self.port, baudrate=self.baud, dsrdtr=None, timeout=0.1
        )
        self._ser.setRTS(False)
        self._ser.setDTR(False)
        self._stop.clear()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        if self._reader is not None:
            self._reader.join(timeout=1.0)
            self._reader = None

    def __enter__(self) -> "Arm":
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- low level --------------------------------------------------------

    def send(self, cmd: dict[str, Any]) -> None:
        """Write one JSON command terminated by newline."""
        if self._ser is None:
            raise RuntimeError("Arm is not open. Call open() or use a with-block.")
        payload = (json.dumps(cmd, separators=(",", ":")) + "\n").encode()
        with self._write_lock:
            self._ser.write(payload)

    def _read_loop(self) -> None:
        assert self._ser is not None
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(64)
            except Exception:
                break
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if isinstance(msg, dict):
                    self._feedback_q.put(msg)

    def _drain_until(self, predicate, timeout: float) -> dict[str, Any] | None:
        """Pull queued messages until one matches predicate, or timeout."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                msg = self._feedback_q.get(timeout=remaining)
            except Empty:
                return None
            if predicate(msg):
                return msg

    # -- high level: state ------------------------------------------------

    def get_state(self, timeout: float | None = None) -> ArmState:
        """Send T:105 and return the parsed T:1051 feedback."""
        timeout = self._timeout if timeout is None else timeout
        # Drain anything stale so we get this query's reply, not a previous one.
        while not self._feedback_q.empty():
            try:
                self._feedback_q.get_nowait()
            except Empty:
                break
        self.send({"T": 105})
        msg = self._drain_until(lambda m: m.get("T") == 1051, timeout=timeout)
        if msg is None:
            raise TimeoutError("No feedback (T:1051) received within timeout")
        return ArmState.from_feedback(msg)

    # -- high level: movement --------------------------------------------

    def home(self) -> None:
        """T:100 — move all joints to the initial position. Blocking on the arm."""
        self.send({"T": 100})

    def move_xyz(
        self,
        x: float,
        y: float,
        z: float,
        t: float = 3.14,
        spd: float = 0.25,
    ) -> None:
        """T:104 — interpolated EoAT move to (x, y, z) mm with wrist angle t (rad)."""
        self.send({"T": 104, "x": x, "y": y, "z": z, "t": t, "spd": spd})

    def move_xyz_direct(self, x: float, y: float, z: float, t: float = 3.14) -> None:
        """T:1041 — non-blocking, no interpolation. Use for streaming small deltas."""
        self.send({"T": 1041, "x": x, "y": y, "z": z, "t": t})

    def move_axis(self, axis: int, pos: float, spd: float = 0.25) -> None:
        """T:103 — move a single axis. axis is one of AXIS_X/Y/Z/T."""
        self.send({"T": 103, "axis": axis, "pos": pos, "spd": spd})

    def move_joints_rad(
        self,
        base: float,
        shoulder: float,
        elbow: float,
        hand: float,
        spd: int = 0,
        acc: int = 10,
    ) -> None:
        """T:102 — set all four joints in radians."""
        self.send({
            "T": 102,
            "base": base,
            "shoulder": shoulder,
            "elbow": elbow,
            "hand": hand,
            "spd": spd,
            "acc": acc,
        })

    def move_joint_rad(
        self,
        joint: int,
        rad: float,
        spd: int = 0,
        acc: int = 10,
    ) -> None:
        """T:101 — set one joint in radians. joint is one of JOINT_BASE/SHOULDER/ELBOW/EOAT."""
        self.send({"T": 101, "joint": joint, "rad": rad, "spd": spd, "acc": acc})

    def move_joints_deg(
        self,
        base: float,
        shoulder: float,
        elbow: float,
        hand: float,
        spd: int = 10,
        acc: int = 10,
    ) -> None:
        """T:122 — set all four joints in degrees."""
        self.send({
            "T": 122,
            "b": base,
            "s": shoulder,
            "e": elbow,
            "h": hand,
            "spd": spd,
            "acc": acc,
        })

    def move_joint_deg(
        self,
        joint: int,
        angle: float,
        spd: int = 10,
        acc: int = 10,
    ) -> None:
        """T:121 — set one joint in degrees."""
        self.send({"T": 121, "joint": joint, "angle": angle, "spd": spd, "acc": acc})

    # -- high level: gripper ---------------------------------------------

    def gripper_deg(self, angle: float, spd: int = 10, acc: int = 10) -> None:
        """Set the clamp angle in degrees (180 = closed, ~45 = open)."""
        self.move_joint_deg(JOINT_EOAT, angle, spd=spd, acc=acc)

    def gripper_rad(self, rad: float, spd: int = 0, acc: int = 0) -> None:
        """T:106 — set the clamp angle in radians (3.14 = closed, ~1.08 = open)."""
        self.send({"T": 106, "cmd": rad, "spd": spd, "acc": acc})

    def open_gripper(self) -> None:
        self.gripper_deg(GRIPPER_OPEN_DEG)

    def close_gripper(self) -> None:
        self.gripper_deg(GRIPPER_CLOSED_DEG)

    # -- high level: torque & dynamics ------------------------------------

    def torque(self, on: bool) -> None:
        """T:210 — lock or release torque. When off, joints can be moved by hand."""
        self.send({"T": 210, "cmd": 1 if on else 0})

    def dynamic_adaptation(
        self,
        on: bool,
        base: int = 1000,
        shoulder: int = 1000,
        elbow: int = 1000,
        hand: int = 1000,
    ) -> None:
        """T:112 — enable dynamic external force adaptation with per-joint torque caps."""
        self.send({
            "T": 112,
            "mode": 1 if on else 0,
            "b": base,
            "s": shoulder,
            "e": elbow,
            "h": hand,
        })

    # -- high level: peripherals -----------------------------------------

    def light(self, brightness: int) -> None:
        """T:114 — set the on-arm LED brightness (0-255). 0 = off."""
        self.send({"T": 114, "led": int(brightness)})

    # -- high level: continuous jog ---------------------------------------

    def jog_angle(self, joint: int, direction: int, spd: int = 10) -> None:
        """T:123 in angle mode. direction: 1=increase, 2=decrease, 0=stop."""
        self.send({"T": 123, "m": 0, "axis": joint, "cmd": direction, "spd": spd})

    def jog_coord(self, axis: int, direction: int, spd: int = 10) -> None:
        """T:123 in coordinate mode. direction: 1=increase, 2=decrease, 0=stop."""
        self.send({"T": 123, "m": 1, "axis": axis, "cmd": direction, "spd": spd})

    def stop_jog(self) -> None:
        self.send({"T": 123, "m": 0, "axis": 0, "cmd": 0, "spd": 0})
