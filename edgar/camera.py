"""USB webcam capture for the overhead Go-board camera.

The AVFoundation index for the USB camera can shift between the MacBook's
built-in FaceTime camera and the external webcam depending on plug order.
The selected index is persisted in .camera_index so future runs are stable;
use `--select` to re-run the chooser.
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

import cv2

CONFIG_PATH = Path(".camera_index")
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
PROBE_RANGE = 4
WARMUP_FRAMES = 10
WARMUP_DELAY_S = 0.1


def load_saved_index() -> int | None:
    if CONFIG_PATH.exists():
        try:
            return int(CONFIG_PATH.read_text().strip())
        except ValueError:
            return None
    return None


def save_index(index: int) -> None:
    CONFIG_PATH.write_text(f"{index}\n")


class Camera:
    """Thin wrapper around cv2.VideoCapture with warmup and context manager."""

    def __init__(
        self,
        index: int | None = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> None:
        if index is None:
            index = load_saved_index()
            if index is None:
                raise RuntimeError(
                    "No camera index configured. Run `python -m edgar.camera --select` first."
                )
        self.index = index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> "Camera":
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {self.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        for _ in range(WARMUP_FRAMES):
            ok, _ = cap.read()
            if ok:
                break
            time.sleep(WARMUP_DELAY_S)
        else:
            cap.release()
            raise RuntimeError("Camera opened but never produced a frame")
        self._cap = cap
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def capture(self):
        """Return one BGR frame (numpy array)."""
        if self._cap is None:
            raise RuntimeError("Camera is not open")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame")
        return frame

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = self.capture()
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"cv2.imwrite failed for {path}")
        return path


def probe_indices(width: int, height: int) -> list[int]:
    """Snap a thumbnail from each working index into captures/probe_idxN.jpg."""
    Path("captures").mkdir(exist_ok=True)
    available = []
    for idx in range(PROBE_RANGE):
        try:
            with Camera(idx, width, height) as cam:
                cam.save(f"captures/probe_idx{idx}.jpg")
            available.append(idx)
            print(f"  index {idx}: ok → captures/probe_idx{idx}.jpg")
        except Exception as e:
            print(f"  index {idx}: skipped ({e})")
    return available


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a frame from the webcam")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output path. Defaults to captures/cam_<timestamp>.jpg",
    )
    parser.add_argument(
        "--index", type=int, default=None, help="Override saved camera index"
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="Probe all indices, save thumbnails, and prompt for the board camera",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    args = parser.parse_args()

    if args.select:
        print("probing camera indices...")
        available = probe_indices(args.width, args.height)
        if not available:
            print("no working cameras found")
            return 1
        choice = input(f"which index is the board camera? {available}: ").strip()
        idx = int(choice)
        save_index(idx)
        print(f"saved index {idx} to {CONFIG_PATH}")
        return 0

    if args.output is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"captures/cam_{ts}.jpg"

    with Camera(args.index, args.width, args.height) as cam:
        path = cam.save(args.output)
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
