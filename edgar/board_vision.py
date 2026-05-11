"""Board recognition: capture -> rotate -> warp -> enhance -> classify.

Output is a 19x19 numpy array with values 0 (empty), 1 (black), 2 (white).

Pipeline:
  1. Rotate the raw frame upright (camera is mounted sideways).
  2. Perspective-warp the four user-picked board corners to a canonical square.
  3. CLAHE on the L channel of LAB to make black/white stones pop off the wood.
  4. Sample a small patch at each of 361 intersections and take mean luminance.
  5. 1-D k-means with k=3 on the per-cell luminances → cluster centers ordered
     dark/mid/bright map to BLACK/EMPTY/WHITE.

The corners and rotation are stored in `.board_config.json` so calibration
survives across runs. Run `python -m edgar.board_vision pick` once after the
camera or board moves.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

from edgar.camera import Camera

CONFIG_PATH = Path(".board_config.json")
GRID = 19
WARP_SIZE = 760  # 19 cells * 40 px
DEFAULT_ROTATION_CCW_90 = 1
DEFAULT_MARGIN = 0.04

EMPTY, BLACK, WHITE = 0, 1, 2
GLYPH = {EMPTY: ".", BLACK: "X", WHITE: "O"}


@dataclass
class BoardConfig:
    rotation_ccw_90: int = DEFAULT_ROTATION_CCW_90
    # [x, y, w, h] in the rotated frame. Empty = no crop.
    crop: list[int] = field(default_factory=list)
    # Corners in (x, y) in the cropped frame, ordered TL, TR, BR, BL.
    corners: list[list[float]] = field(default_factory=list)
    warp_size: int = WARP_SIZE
    margin: float = DEFAULT_MARGIN

    @classmethod
    def load(cls) -> "BoardConfig":
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"{CONFIG_PATH} not found. Run `python -m edgar.board_vision pick` first."
            )
        data = json.loads(CONFIG_PATH.read_text())
        return cls(**data)

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))


# -- pipeline stages -----------------------------------------------------


def rotate(img: np.ndarray, k: int) -> np.ndarray:
    """Rotate the image counter-clockwise by k * 90 degrees."""
    out = img
    for _ in range(k % 4):
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return out


def apply_crop(img: np.ndarray, crop: list[int]) -> np.ndarray:
    """Crop to [x, y, w, h]. No-op if crop is empty."""
    if not crop:
        return img
    x, y, w, h = crop
    return img[y:y + h, x:x + w]


def warp(img: np.ndarray, corners: list[list[float]], size: int) -> np.ndarray:
    """Perspective-warp the 4 corners (TL, TR, BR, BL) to a size x size square."""
    src = np.array(corners, dtype=np.float32)
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (size, size))


def enhance(img: np.ndarray) -> np.ndarray:
    """Make the image severe: stronger CLAHE plus saturation boost.

    Wood is colored, stones are achromatic. After saturation boost the
    wood pops in chroma while stones stay gray, which makes saturation
    a clean discriminator in classify().
    """
    # Slight blur to suppress grid line noise inside cells.
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # CLAHE on luminance for contrast.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6)).apply(L)
    bgr = cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)
    # Saturation boost in HSV.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def cell_centers(size: int, grid: int, margin: float) -> tuple[np.ndarray, int]:
    """Return integer (grid, grid, 2) intersection centers and a patch half-size.

    Patch is kept around 25% of cell pitch so it's small enough that grid
    lines don't dominate the median, but large enough to tolerate a few
    pixels of warp misalignment.
    """
    inset = int(size * margin)
    inner = size - 2 * inset
    step = inner / (grid - 1)
    half = max(3, int(step * 0.25))
    centers = np.zeros((grid, grid, 2), dtype=np.int32)
    for r in range(grid):
        for c in range(grid):
            cx = int(round(inset + c * step))
            cy = int(round(inset + r * step))
            centers[r, c] = [cx, cy]
    return centers, half


def cell_features(
    img: np.ndarray, grid: int = GRID, margin: float = DEFAULT_MARGIN
) -> tuple[np.ndarray, np.ndarray]:
    """Per-intersection median (L, S) from LAB + HSV.

    Median is robust to a grid line bisecting the patch. Returns two
    (grid, grid) arrays.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    L = lab[:, :, 0]
    S = hsv[:, :, 1]
    h, w = L.shape
    centers, half = cell_centers(min(h, w), grid, margin)
    L_out = np.zeros((grid, grid), dtype=np.float32)
    S_out = np.zeros((grid, grid), dtype=np.float32)
    for r in range(grid):
        for c in range(grid):
            cx, cy = centers[r, c]
            y0, y1 = max(cy - half, 0), min(cy + half, h)
            x0, x1 = max(cx - half, 0), min(cx + half, w)
            L_out[r, c] = np.median(L[y0:y1, x0:x1])
            S_out[r, c] = np.median(S[y0:y1, x0:x1])
    return L_out, S_out


def cell_luminance(
    img: np.ndarray, grid: int = GRID, margin: float = DEFAULT_MARGIN
) -> np.ndarray:
    """Backwards-compat: median L only."""
    L, _ = cell_features(img, grid, margin)
    return L


def kmeans_1d(x: np.ndarray, k: int = 3, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """1-D k-means. Returns (sorted centers, labels remapped to sorted order)."""
    centers = np.linspace(x.min(), x.max(), k)
    labels = np.zeros_like(x, dtype=np.int32)
    for _ in range(iters):
        labels = np.abs(x[:, None] - centers[None, :]).argmin(axis=1)
        new_centers = np.array(
            [x[labels == j].mean() if (labels == j).any() else centers[j]
             for j in range(k)]
        )
        if np.allclose(new_centers, centers, atol=1e-3):
            centers = new_centers
            break
        centers = new_centers
    order = np.argsort(centers)
    remap = {old: new for new, old in enumerate(order)}
    relabeled = np.vectorize(remap.get)(labels)
    return centers[order], relabeled


def classify(
    L: np.ndarray, S: np.ndarray | None = None, sigma: float = 3.0
) -> tuple[np.ndarray, dict]:
    """Detect stones as outliers from the dominant (empty) cell distribution.

    Why outlier-based: most cells are empty wood, which clusters tightly in
    (L, S) space. Stones are achromatic (low S) and either dark (BLACK) or
    bright (WHITE). k-means with k=3 would force splitting empties even
    when no stones exist; this avoids that.

    Returns (board, info). info contains the empty centroid and scale.
    """
    if S is None:
        # Fall back to 1-D classification on L alone (legacy path).
        flat = L.flatten().astype(np.float32)
        centers, labels = kmeans_1d(flat, k=3)
        mapping = np.array([BLACK, EMPTY, WHITE])
        board = mapping[labels].reshape(L.shape)
        return board, {"centers_1d": centers.tolist()}

    L_flat = L.flatten().astype(np.float32)
    S_flat = S.flatten().astype(np.float32)
    # Robust centroid of the dominant (empty) cluster.
    empty_L = float(np.median(L_flat))
    empty_S = float(np.median(S_flat))

    # Multiplicative thresholds anchored on empty centroid. Black stones are
    # ~5-10x darker than wood (well below empty_L / 2), so this rejects grid
    # lines and shadows which only drop L moderately. White stones are
    # significantly brighter and less saturated than wood.
    #
    # sigma scales the gap: higher sigma -> stricter (gap further from wood).
    black_frac = 1.0 - 0.12 * sigma   # at sigma=4: L < 0.52 * empty_L
    white_frac = 1.0 + 0.12 * sigma   # at sigma=4: L > 1.48 * empty_L
    sat_frac = 1.0 - 0.10 * sigma     # at sigma=4: S < 0.60 * empty_S

    is_black = L_flat < empty_L * black_frac
    is_white_bright = L_flat > empty_L * white_frac
    is_white_pale = (S_flat < empty_S * sat_frac) & (L_flat > empty_L)
    is_white = is_white_bright | is_white_pale

    out = np.full(L_flat.shape, EMPTY, dtype=np.int32)
    out[is_white] = WHITE
    out[is_black] = BLACK

    info = {
        "empty_L": empty_L,
        "empty_S": empty_S,
        "sigma": sigma,
        "black_threshold_L": empty_L * black_frac,
        "white_threshold_L": empty_L * white_frac,
        "white_threshold_S": empty_S * sat_frac,
        "n_black": int((out == BLACK).sum()),
        "n_white": int((out == WHITE).sum()),
    }
    return out.reshape(L.shape), info


def format_board(board: np.ndarray) -> str:
    rows = []
    for r in range(board.shape[0]):
        rows.append(" ".join(GLYPH[int(c)] for c in board[r]))
    return "\n".join(rows)


# -- visualization helpers ----------------------------------------------


def draw_grid_overlay(img: np.ndarray, board: np.ndarray, margin: float) -> np.ndarray:
    """Overlay classification result on the warped board for debugging."""
    out = img.copy()
    h, w = img.shape[:2]
    centers, half = cell_centers(min(h, w), board.shape[0], margin)
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            cx, cy = centers[r, c]
            cls = int(board[r, c])
            if cls == EMPTY:
                cv2.circle(out, (cx, cy), 2, (0, 200, 0), -1)
            elif cls == BLACK:
                cv2.circle(out, (cx, cy), half, (0, 0, 255), 2)
            else:
                cv2.circle(out, (cx, cy), half, (255, 0, 0), 2)
    return out


# -- interactive pickers ------------------------------------------------


def select_crop_interactive(img: np.ndarray) -> list[int] | None:
    """Drag a rectangle around the board. Returns [x, y, w, h] or None."""
    win = "drag a rectangle around the board, then ENTER/SPACE  (c=cancel)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # cv2.selectROI blocks until ENTER/SPACE or 'c'.
    rect = cv2.selectROI(win, img, fromCenter=False, showCrosshair=False)
    cv2.destroyAllWindows()
    x, y, w, h = (int(v) for v in rect)
    if w == 0 or h == 0:
        return None
    return [x, y, w, h]


def pick_corners_interactive(img: np.ndarray) -> list[list[float]]:
    pts: list[list[float]] = []
    clone = img.copy()
    labels = ["TL", "TR", "BR", "BL"]
    win = "pick corners: TL, TR, BR, BL  (r=reset, enter=accept)"

    def redraw() -> None:
        clone[:] = img
        for i, p in enumerate(pts):
            cv2.circle(clone, tuple(int(v) for v in p), 8, (0, 255, 0), 2)
            cv2.putText(
                clone, labels[i],
                (int(p[0]) + 12, int(p[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([float(x), float(y)])
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, clone)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            pts.clear()
            redraw()
        elif key in (13, 10):
            if len(pts) == 4:
                break
        elif key == 27:
            pts.clear()
            break
    cv2.destroyAllWindows()
    return pts


# -- entry point ---------------------------------------------------------


def capture_frame(image_path: str | None) -> np.ndarray:
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        return img
    with Camera() as cam:
        return cam.capture()


def run_pipeline(
    frame: np.ndarray, cfg: BoardConfig, sigma: float = 3.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Returns (rotated, cropped, warped, enhanced, board, info).

    Feature extraction is done from the warped (un-enhanced) image so that
    natural saturation is preserved: black stones stay near S=0, wood is
    moderately saturated. CLAHE+sat-boost is only for visualization.
    """
    rotated = rotate(frame, cfg.rotation_ccw_90)
    cropped = apply_crop(rotated, cfg.crop)
    warped = warp(cropped, cfg.corners, cfg.warp_size)
    enhanced = enhance(warped)
    L, S = cell_features(warped, grid=GRID, margin=cfg.margin)
    board, info = classify(L, S, sigma=sigma)
    return rotated, cropped, warped, enhanced, board, info


def main() -> int:
    p = argparse.ArgumentParser(description="Edgar board vision pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser(
        "pick",
        help="Two-step calibration: drag a crop rectangle, then click 4 corners",
    )
    sp.add_argument("--image", help="Use this image instead of capturing live")
    sp.add_argument(
        "--corners-only",
        action="store_true",
        help="Skip crop, only re-pick corners using the existing crop",
    )

    sw = sub.add_parser("warp", help="Capture + warp + enhance to captures/warp.jpg")
    sw.add_argument("--image")

    sc = sub.add_parser("classify", help="Full pipeline: prints 19x19 board")
    sc.add_argument("--image")
    sc.add_argument("--debug", action="store_true", help="Save intermediate frames")
    sc.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Outlier threshold (z-distance from empty centroid). Higher = stricter.",
    )

    args = p.parse_args()

    if args.cmd == "pick":
        frame = capture_frame(args.image)
        existing: BoardConfig | None = None
        if CONFIG_PATH.exists():
            try:
                existing = BoardConfig.load()
            except Exception:
                existing = None
        rot = existing.rotation_ccw_90 if existing else DEFAULT_ROTATION_CCW_90
        rotated = rotate(frame, rot)

        if args.corners_only:
            if not existing or not existing.crop:
                print(
                    "no saved crop to reuse; run without --corners-only",
                    file=sys.stderr,
                )
                return 1
            crop = existing.crop
        else:
            print("step 1/2: drag a tight rectangle around the board")
            crop = select_crop_interactive(rotated)
            if crop is None:
                print("aborted (no crop selected)", file=sys.stderr)
                return 1

        cropped = apply_crop(rotated, crop)
        print(
            f"step 2/2: click the 4 outermost grid intersections (TL, TR, BR, BL). "
            f"Working on a {cropped.shape[1]}x{cropped.shape[0]} crop."
        )
        pts = pick_corners_interactive(cropped)
        if len(pts) != 4:
            print("aborted (need 4 points)", file=sys.stderr)
            return 1

        cfg = BoardConfig(rotation_ccw_90=rot, crop=crop, corners=pts)
        cfg.save()
        print(f"saved {CONFIG_PATH}: crop={crop} corners={pts}")
        return 0

    cfg = BoardConfig.load()
    frame = capture_frame(args.image)

    if args.cmd == "warp":
        rotated = rotate(frame, cfg.rotation_ccw_90)
        cropped = apply_crop(rotated, cfg.crop)
        warped = warp(cropped, cfg.corners, cfg.warp_size)
        enhanced = enhance(warped)
        Path("captures").mkdir(exist_ok=True)
        cv2.imwrite("captures/warp.jpg", enhanced)
        print("saved captures/warp.jpg")
        return 0

    if args.cmd == "classify":
        rotated, cropped, warped, enhanced, board, info = run_pipeline(
            frame, cfg, sigma=args.sigma
        )
        print(format_board(board))
        print(f"info: {info}")
        if args.debug:
            Path("captures").mkdir(exist_ok=True)
            cv2.imwrite("captures/dbg_rotated.jpg", rotated)
            cv2.imwrite("captures/dbg_cropped.jpg", cropped)
            cv2.imwrite("captures/dbg_warp.jpg", warped)
            cv2.imwrite("captures/dbg_enhanced.jpg", enhanced)
            cv2.imwrite(
                "captures/dbg_overlay.jpg",
                draw_grid_overlay(enhanced, board, cfg.margin),
            )
            print("debug frames saved to captures/dbg_*.jpg")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
