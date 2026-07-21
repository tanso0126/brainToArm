"""Central config. One machine, no router — everything is local.

Every value that depends on hardware we don't have in hand yet is a documented
best-guess with a comment on how to confirm it. If the real device/arm/camera
matches these guesses, the system runs unmodified. If not, you edit ONLY the
wrong constant here — no code changes.
"""

import re
from pathlib import Path


def _load_home_pose():
    """Read the same named constants compiled into the Uno firmware."""
    source = (Path(__file__).resolve().parents[1]
              / "firmware" / "arm_controller" / "home_pose.h")
    values = {
        int(index): int(angle)
        for index, angle in re.findall(
            r"^#define\s+ARM_HOME_SERVO_([1-6])\s+(\d+)\s*$",
            source.read_text(encoding="utf-8"),
            flags=re.MULTILINE)
    }
    if sorted(values) != list(range(1, 7)):
        raise RuntimeError(f"invalid six-servo home pose in {source}")
    pose = [values[index] for index in range(1, 7)]
    if any(not 0 <= angle <= 180 for angle in pose):
        raise RuntimeError(f"home angle outside 0..180 in {source}")
    return pose


# ======================================================================
# Arduino (robot arm)
# ======================================================================
ARM_PORT = "auto"           # Uno port changes with USB topology; resolve at runtime
# The ESP32 CP2102 has stable serial "0001" and must never be opened/reset by
# arm discovery. The Uno CH340 currently enumerates as usbserial-110.
ARM_PORT_EXCLUDE = ("/dev/cu.usbserial-0001",)
ARM_BAUD = 115200
ARM_MOCK = False            # physical Uno is connected; calibration gate remains below
ARM_CALIBRATED = False      # set True only after arm_jog confirms geometry/offset/direction/limits
N_JOINTS = 6
HOME_POSE = _load_home_pose()
SERVO_PINS = [13, 12, 11, 10, 9, 8]
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_pitch", "gripper", "wrist_roll"]

# Joint indices (readable names). Bottom -> top of the arm.
J_BASE, J_SHOULDER, J_ELBOW, J_WRIST, J_GRIP, J_ROLL = range(N_JOINTS)
J_TILT = J_ROLL            # compatibility alias; servo6 is physically wrist roll
GRIP_OPEN = 90             # physically verified: 90=open
GRIP_CLOSED = 180          # physically verified: 180=closed

# ---- Camera-calibrated planar arm mode (physically verified 2026-07-20) ----
# Servo1 is operational again, but this legacy side-camera calibration is valid
# only at base=90. Generic jog/IK may use servo1's full global 0..180 range;
# planar_pick.py deliberately keeps the base fixed while using this calibration.
PLANAR_CAM_INDEX = "auto"  # macOS indices change when an iPhone camera appears/disappears
PLANAR_FRAME_SIZE = (1920, 1080)
PLANAR_ARM_CALIBRATED = True  # successful physical pick/lift/place on 2026-07-20
PLANAR_BASE_ANGLE = 90
PLANAR_SERVO_MIN = [90, 0, 0, 10, 90, 0]
PLANAR_SERVO_MAX = [90, 150, 180, 180, 180, 180]

# Background-independent FastSAM perception and the calibrated local image
# Jacobian. At the verified observation pose, an object centered at x=1435 px
# is reached with elbow=90. The model file is downloaded once before competition
# and then runs offline; no empty-table image or venue-specific background is used.
PLANAR_VISION_MODEL = "FastSAM-s.pt"
PLANAR_VISION_DEVICE = "cpu"  # measured faster than MPS on this Mac/FastSAM-s
PLANAR_VISION_IMAGE_SIZE = 640
PLANAR_VISION_CONFIDENCE = 0.20  # target stays >0.9; avoids slow low-quality fragments
PLANAR_GEOMETRY_REFERENCE = "data/vision/planar_background.jpg"
PLANAR_MAX_CAMERA_SCALE_DRIFT = 0.025
PLANAR_MAX_CAMERA_ROTATION_DEG = 1.0
PLANAR_MAX_CAMERA_TRANSLATION_PX = 35.0
PLANAR_TARGET_MIN_BOX_RATIO = 0.0005
PLANAR_TARGET_MAX_BOX_RATIO = 0.04
PLANAR_TARGET_MIN_ASPECT = 0.35
PLANAR_TARGET_MAX_ASPECT = 5.0
PLANAR_TARGET_MIN_Y_RATIO = 0.52
PLANAR_MIN_LIFT_PX = 30

# At the grasp pose, reducing elbow by 1 degree moves the gripper about 7 px
# right; shoulder +1 degree moves it about 11 px down. Vision remeasures both
# the object and gripper between bounded corrections instead of replaying a pose.
# Near the grasp pose, reducing elbow by 1 degree moves the gripper about 7 px
# right; shoulder +1 degree moves it about 11 px down. Every run re-detects the
# object and corrects elbow before descending instead of replaying one pose.
PLANAR_REFERENCE_OBJECT_X = 1435.0
PLANAR_REFERENCE_ELBOW = 90
PLANAR_ELBOW_PX_PER_DEG = -7.0
PLANAR_REFERENCE_GRASP_SHOULDER = 140
PLANAR_GRASP_DEPTH_OFFSET_DEG = 2  # measured extra descent for full side contact
PLANAR_ELBOW_Y_PX_PER_DEG = 6.0
PLANAR_SHOULDER_PX_PER_DEG = 11.0
PLANAR_APPROACH_CLEARANCE_DEG = 16
PLANAR_LIFT_DEG = 12
PLANAR_PICK_ELBOW_RANGE = (78, 110)
PLANAR_WRIST_PITCH = 180
PLANAR_WRIST_ROLL = 180
PLANAR_PLACE_ELBOW = 80
PLANAR_PLACE_SHOULDER = 140
PLANAR_RETREAT_SHOULDER = 134
PLANAR_GRIP_CLOSE_STEP = 10
GRIP_FEEDBACK_ENABLED = False  # requires an FSR/current-sensor signal on Uno A0
GRIP_FEEDBACK_DELTA = 80       # ADC counts above the open-gripper baseline

# ---- Pick-and-place heights (cm above the table) and the place/delivery zone.
Z_APPROACH = 6.0           # hover height above an object before descending
Z_GRASP    = 1.0           # height at which the gripper closes on the object
Z_LIFT     = 10.0          # lift height for transporting
Z_PLACE    = 2.0           # release height over the place zone
PLACE_LOCATION = (0.0, 16.0)   # workspace (x,y) delivery point — e.g. near the human
# For a "sort into 2 bins" demo, list zones and pick per rule (or per second veto):
PLACE_ZONES = {"deliver": (0.0, 16.0), "reject_bin": (-14.0, 6.0)}
GRASP_VERIFY = True        # after lifting, confirm the object left its spot
GRASP_RETRIES = 2
GRASP_VERIFY_RADIUS_CM = 2.0  # non-bgsub methods: no detection may remain this close
POLICY_SPATIAL_LEARNING = False  # unsafe without an explicit persistent task identity

# ---- Arm geometry for inverse kinematics (cm). MEASURE THESE on your build ----
# Distances between joint axes along the arm. Defaults are rough values for the
# MakerWorld "Robotic Arm with Servo" model; replace with a ruler measurement.
L_BASE_HEIGHT = 6.0        # table -> shoulder axis height
L_UPPER = 10.5             # shoulder axis -> elbow axis
L_FORE  = 9.5              # elbow axis -> wrist axis
L_HAND  = 8.0              # wrist axis -> gripper contact point

# ---- Servo calibration: map an IK joint angle (deg, math convention) to the
# servo.write() value. servo_cmd = clamp(offset + direction * joint_deg).
# Tune offset so the arm's real neutral matches, and direction if a joint runs
# backwards. Defaults assume 90 = neutral, positive = "up/outward".
SERVO_OFFSET    = [90, 90, 90, 90, 90, 90]
SERVO_DIRECTION = [1, 1, 1, 1, 1, 1]
SERVO_MIN       = [0, 0, 0, 10, 90, 0]
SERVO_MAX       = [180, 150, 180, 180, 180, 180]

# ======================================================================
# EEG (LAXTHA PolyG-I)
# ======================================================================
EEG_SOURCE = "hid"         # "hid" = this PID 0x0010 device; "mock" | "serial" | "tcp"
EEG_PORT = "auto"          # or e.g. "/dev/cu.usbserial-XXXX"; run eeg_detect.py
EEG_BAUD = 115200          # confirm from LXSDF PDF; PolyG-I is high-rate, try 921600 too

# This physical unit is vendor-defined USB HID, not serial. The exact command
# sequence, marking-bit removal, offset-binary decoder, and voltage coefficient
# were recovered from LXSM-D1WD10.dll and checked on the connected device.
EEG_HID_VID = 0x0F1F
EEG_HID_PID = 0x0010
EEG_HID_CHANNELS = 8
EEG_HID_MAX_CHANNELS = 16
EEG_HID_GAIN_INDEX = 6
EEG_HID_SAMPLE_SELECTOR = 8  # D1WD10: 2**8 = 256 Hz
EEG_HID_STALL_TIMEOUT_S = 2.0
EEG_HID_ADC_UV_PER_COUNT = -38.14697265625  # ADC input only; not electrode input

# The device28 config describes 16 physical polygraph signals, but this vendor
# DLL exposes 16-channel acquisition blocks. Its first signal group is EEG
# inputs 1..8. EEG_CHANNEL_MAP selects/reorders those eight output slots.
EEG_TOTAL_CHANNELS = None  # LXSDF sources only: None = auto-detect; HID has 8 fixed slots
EEG_CHANNEL_MAP = [0, 1, 2, 3, 4, 5, 6, 7]   # packet slot -> EEG ch 0..7
EEG_CHANNELS = 8
# D1WD10 selector 8 is 256 Hz. Physical A/B capture measured 255.9–256.2 Hz;
# each 1024-byte report contains 32 rows across 16 physical channels.
EEG_FS = 256
EEG_MIN_EPOCH_FRACTION = 0.80  # abort a decision if too many onset-locked samples are missing
EEG_CONFIG_VERIFIED = False    # transport works; set True only after montage/rate signal validation

# Compatibility-source scaling: LXSDF ships a 12-bit ADC value.
# uV = (raw - ADC_ZERO) * ADC_UV_PER_LSB. Defaults center a 12-bit unipolar code
# and use a placeholder gain; set ADC_UV_PER_LSB from the device datasheet for
# absolute uV. Relative values already work for ErrP without exact scaling.
ADC_BITS = 12
ADC_ZERO = 2048            # midpoint of a 12-bit unipolar code (0..4095)
ADC_UV_PER_LSB = 1.0       # placeholder; datasheet gives the real Vref/gain

# Legacy/alternate-device compatibility: a Windows helper calling LXSMWD12.dll
# forwards a raw LXSDF stream here. PID 0x0010 uses the native HID path above.
EEG_TCP_HOST = "127.0.0.1"
EEG_TCP_PORT = 9000

# ======================================================================
# ErrP (error-related potential)
# ======================================================================
ERRP_BACKEND = "baseline"  # "baseline" (zero-training heuristic) | "model" (trained)
ERRP_MODEL_PATH = "errp_model.pkl"
ERRP_FRONTOCENTRAL = [0, 1, 2]   # EEG ch indices over Fz/FCz/Cz — set to YOUR montage
ERRP_WINDOW_S = 0.8        # epoch length after action onset
ERRP_BASELINE_S = 0.2      # pre-onset baseline for correction
ERRP_BAND = (1.0, 10.0)    # ErrP bandpass (Hz)
ERRP_THRESHOLD = 0.5       # decision threshold (tune with real data)

# ======================================================================
# Vision (overhead camera)
# ======================================================================
CAM_INDEX = 0              # cv2.VideoCapture index; 0 = default camera
CAM_MOCK = True            # True until a real camera is wired; orchestrator uses mock scene
CAM_CALIBRATED = False     # set True after pasting calibrate_workspace.py results
# Detection method (MARKERLESS by default — no stickers, no props):
#   "bgsub" : background subtraction. Snapshot the empty table once, then objects
#             and the arm are foreground. Zero physical setup. DEFAULT, best for
#             a clean 2-3 object demo.
#   "yolo"  : ultralytics YOLO. Semantic labels + robust in clutter, but needs
#             `pip install ultralytics`, is heavier, and COCO can't tell
#             "big vs small nail" (custom training needed). See note in vision.py.
#   "hsv"   : color blobs (no markers, but sensitive to lighting).
#   "aruco" : printed markers (most robust, but you have none).
OBJECT_METHOD = "bgsub"
BGSUB_THRESH = 30          # foreground pixel diff threshold (raise if noisy)
OBJECT_MIN_AREA = 200      # px^2; ignore blobs smaller than a real object
ARM_MIN_AREA = 800         # px^2; the arm blob is larger than the objects

# YOLO options (only if OBJECT_METHOD="yolo")
YOLO_WEIGHTS = "yolov8n.pt"    # COCO nano; or your custom-trained .pt
YOLO_CONF = 0.35
YOLO_CLASSES = None            # None = all; or a list of class ids to keep

# Marker path (only if you switch OBJECT_METHOD="aruco"):
ARM_TIP_ARUCO_ID = 0
ARUCO_DICT = "DICT_4X4_50"
OBJECT_ARUCO = {1: "nail_big", 2: "nail_small", 3: "screw"}

# ---- Visual servoing: close the loop so a cheap camera + rough IK still land
# accurately. After the veto decides WHICH object, drive the tip to it by
# repeatedly measuring the tip->target error and nudging, cancelling systematic
# bias. Precision comes from feedback, not optics.
SERVO_ENABLE = True
SERVO_GAIN = 0.6           # correction fraction applied per iteration (0..1)
SERVO_TOL_CM = 0.5         # stop when tip within this of target
SERVO_MAX_ITERS = 8

# Lens distortion: run camera_calibrate.py once with a printed checkerboard to
# fill these (fisheye correction for cheap/phone cameras). None = skip undistort.
CAM_MATRIX = None          # 3x3 intrinsic matrix
CAM_DIST = None            # distortion coefficients

# Pixel -> workspace(cm) homography calibration: 4+ point correspondences.
# Put markers/tape at known workspace coords, note their pixel coords, fill here.
# image points (px)                     world points (cm)
CAM_CALIB_IMAGE_PTS = [[100, 100], [540, 100], [540, 380], [100, 380]]
CAM_CALIB_WORLD_PTS = [[-15, 15], [15, 15], [15, -15], [-15, -15]]

# HSV color ranges for mock/real object detection (H:0-179, S:0-255, V:0-255).
# label -> (lower, upper). Tune under your lighting.
OBJECT_HSV = {
    "red":   ([0, 120, 70], [10, 255, 255]),
    "blue":  ([100, 120, 70], [130, 255, 255]),
    "green": ([40, 80, 70], [80, 255, 255]),
}
