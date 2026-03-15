# fisheye_multiview_dewarp.py
import cv2
import numpy as np

# ========== 1. CONFIG – TUNE THESE FOR YOUR CAMERA ==========
#
# name      : window name
# roll_deg  : rotate around the fisheye circle (which side of room)
# pitch_deg : tilt up/down (0 = horizontal, negative = look more down)
# fov_deg   : zoom (bigger = wider & more distorted, smaller = more zoom & flatter)

VIEW_CONFIGS = [
    # Try to make this see the entrance (bottom-right of fisheye)
    {"name": "entrance",    "roll_deg":  -105,  "pitch_deg": -70, "fov_deg": 40},

    # Corridor / walking path in front of entrance
    {"name": "corridor",    "roll_deg": -100, "pitch_deg": -55, "fov_deg": 70},

    # Left & right seating zones
    {"name": "left_seats",  "roll_deg": 180, "pitch_deg": -55, "fov_deg": 80},
    {"name": "right_seats", "roll_deg": 160, "pitch_deg": 45, "fov_deg": 80},
]

OUTPUT_SHAPE = (360, 640)   # (height, width) of each planar view
INPUT_FOV_DEG = 180         # approx fisheye lens FOV
CURRENT_VIEW_CONFIGS = [cfg.copy() for cfg in VIEW_CONFIGS]

_dewarper = None


def set_view_configs(view_configs):
    """
    Replace the current view configs and force remap rebuild on next get_views().
    view_configs: list of dicts with keys: name, roll_deg, pitch_deg, fov_deg
    """
    global CURRENT_VIEW_CONFIGS, _dewarper
    CURRENT_VIEW_CONFIGS = [cfg.copy() for cfg in view_configs]
    _dewarper = None


# ========== 2. FISHEYE PROJECTION MATH ==========
def build_fisheye_remap(input_shape, output_shape,
                        input_fov_deg=180, output_fov_deg=120,
                        yaw_deg=0, pitch_deg=0, roll_deg=0):
    in_h, in_w = input_shape
    out_h, out_w = output_shape

    in_fov = np.deg2rad(input_fov_deg)
    out_fov = np.deg2rad(output_fov_deg)

    v_range = np.tan(out_fov / 2)
    u_range = v_range * (out_w / out_h)

    xs, ys = np.meshgrid(
        np.linspace(-u_range,  u_range,  out_w, dtype=np.float32),
        np.linspace( v_range, -v_range, out_h, dtype=np.float32)
    )
    zs = np.ones_like(xs)

    dirs = np.stack([xs, ys, zs], axis=-1)

    r_yaw   = np.deg2rad(yaw_deg)
    r_pitch = np.deg2rad(pitch_deg)
    r_roll  = np.deg2rad(roll_deg)

    Ry = np.array([
        [ np.cos(r_yaw), 0, np.sin(r_yaw)],
        [ 0,             1, 0            ],
        [-np.sin(r_yaw), 0, np.cos(r_yaw)]
    ])

    Rx = np.array([
        [1, 0,                0               ],
        [0, np.cos(r_pitch), -np.sin(r_pitch)],
        [0, np.sin(r_pitch),  np.cos(r_pitch)]
    ])

    Rz = np.array([
        [ np.cos(r_roll), -np.sin(r_roll), 0],
        [ np.sin(r_roll),  np.cos(r_roll), 0],
        [ 0,               0,              1]
    ])

    R = Rz @ Rx @ Ry

    rotated = dirs @ R.T
    x, y, z = rotated[..., 0], rotated[..., 1], rotated[..., 2]

    theta = np.arctan2(y, x)
    phi   = np.arctan2(np.sqrt(x * x + y * y), z)

    radius = phi * min(in_w, in_h) / in_fov

    map_x = (in_w / 2 + radius * np.cos(theta)).astype(np.float32)
    map_y = (in_h / 2 - radius * np.sin(theta)).astype(np.float32)

    return map_x, map_y


def fisheye_to_planar(frame, map_x, map_y):
    return cv2.remap(frame, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


# ========== 3. MULTI-VIEW DEWARPER CLASS ==========
class FisheyeMultiViewDewarper:
    """
    Precomputes multiple fisheye→planar mappings for different directions
    (entrance, corridor, seats, etc.).
    """

    def __init__(self, fisheye_shape,
                 view_configs=VIEW_CONFIGS,
                 output_shape=OUTPUT_SHAPE,
                 input_fov=INPUT_FOV_DEG):

        h, w, _ = fisheye_shape

        # Crop to centre square to avoid black border sides
        self.side_length = min(h, w)
        self.crop_offset = (w - self.side_length) // 2
        self.cropped_shape = (self.side_length, self.side_length)

        self.views = []
        for cfg in view_configs:
            name      = cfg["name"]
            roll_deg  = cfg.get("roll_deg", 0.0)
            pitch_deg = cfg.get("pitch_deg", 0.0)
            fov_deg   = cfg.get("fov_deg", 90.0)

            map_x, map_y = build_fisheye_remap(
                input_shape=self.cropped_shape,
                output_shape=output_shape,
                input_fov_deg=input_fov,
                output_fov_deg=fov_deg,
                yaw_deg=0.0,
                pitch_deg=pitch_deg,
                roll_deg=roll_deg,
            )

            self.views.append({
                "name": name,
                "map_x": map_x,
                "map_y": map_y,
            })

    def generate_views(self, frame):
        """
        Takes one fisheye frame and returns a dict of dewarped views.
        """
        cropped = frame[:, self.crop_offset:self.crop_offset + self.side_length]

        outputs = {}
        for v in self.views:
            planar = fisheye_to_planar(cropped, v["map_x"], v["map_y"])

            # Fix orientation per view if needed
            if v["name"] == "right_seats":
                # rotate 180 degrees to make it upright
                planar = cv2.rotate(planar, cv2.ROTATE_180)

            outputs[v["name"]] = planar

        return outputs

    def generate_views_with_meta(self, frame):
        """
        Returns:
          outputs: dict view_name -> planar image (same as generate_views)
          meta: dict view_name -> dict containing map_x/map_y + crop_offset/side_length + rotation info
        """
        cropped = frame[:, self.crop_offset:self.crop_offset + self.side_length]

        outputs = {}
        meta = {}

        for v in self.views:
            planar = fisheye_to_planar(cropped, v["map_x"], v["map_y"])

            rotated_180 = False
            if v["name"] == "right_seats":
                planar = cv2.rotate(planar, cv2.ROTATE_180)
                rotated_180 = True

            outputs[v["name"]] = planar
            meta[v["name"]] = {
                "map_x": v["map_x"],
                "map_y": v["map_y"],
                "crop_offset": self.crop_offset,
                "side_length": self.side_length,
                "rot180": rotated_180,
                "out_w": planar.shape[1],
                "out_h": planar.shape[0],
            }

        return outputs, meta


# ========== 4. SIMPLE API FOR THE REST OF YOUR CODE ==========
def get_views(frame):
    global _dewarper
    if _dewarper is None:
        _dewarper = FisheyeMultiViewDewarper(
            frame.shape,
            view_configs=CURRENT_VIEW_CONFIGS,
            output_shape=OUTPUT_SHAPE,
            input_fov=INPUT_FOV_DEG
        )
    return _dewarper.generate_views(frame)

def get_views_with_meta(frame):
    """
    Like get_views(), but also returns per-view remap meta for ROI filtering.
    """
    global _dewarper
    if _dewarper is None:
        _dewarper = FisheyeMultiViewDewarper(
            frame.shape,
            view_configs=CURRENT_VIEW_CONFIGS,
            output_shape=OUTPUT_SHAPE,
            input_fov=INPUT_FOV_DEG
        )
    return _dewarper.generate_views_with_meta(frame)

# =============== Detect Fisheye or Normal =================
def is_fisheye(frame_bgr,
               border_pct=0.12,
               black_thresh=35,
               black_ratio_thresh=0.25):
    """
    Heuristic fisheye detector using black border ratio.
    Returns True if frame looks like fisheye.
    """
    h, w = frame_bgr.shape[:2]
    b = int(min(h, w) * border_pct)
    if b < 5:
        return False

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[:b, :] = 255
    mask[-b:, :] = 255
    mask[:, :b] = 255
    mask[:, -b:] = 255

    border_pixels = gray[mask == 255]
    if border_pixels.size == 0:
        return False

    dark_ratio = np.mean(border_pixels < black_thresh)
    return dark_ratio > black_ratio_thresh

