"""Central config. One machine, no router — everything is local.

Every value that depends on hardware we don't have in hand yet is a documented
best-guess with a comment on how to confirm it. If the real device/arm/camera
matches these guesses, the system runs unmodified. If not, you edit ONLY the
wrong constant here — no code changes.
"""

# ======================================================================
# Arduino (robot arm)
# ======================================================================
ARM_PORT = "auto"          # "auto" = first usb-serial, or e.g. "/dev/cu.usbmodem1101"
ARM_BAUD = 115200
ARM_MOCK = True             # explicit safety switch; False requires a real, responsive board
N_JOINTS = 7
UNUSED_JOINT = 2           # servo3 index (0-based) — attached but not driven
HOME_POSE = [90, 90, 90, 90, 90, 170, 180]

# Joint indices (readable names). Bottom -> top of the arm.
J_BASE, J_SHOULDER, _UNUSED, J_ELBOW, J_WRIST, J_TILT, J_GRIP = range(7)
GRIP_OPEN = 180
GRIP_CLOSED = 90

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
SERVO_OFFSET    = [90, 90, 90, 90, 90, 90, 90]
SERVO_DIRECTION = [1, 1, 1, 1, 1, 1, 1]
SERVO_MIN       = [0, 0, 0, 0, 0, 0, 0]
SERVO_MAX       = [180, 180, 180, 180, 180, 180, 180]

# ======================================================================
# EEG (LAXTHA PolyG-I) — LXSDF T2A serial/USB stream
# ======================================================================
EEG_SOURCE = "mock"        # "mock" | "serial" (Path A) | "tcp" (Path B win bridge)
EEG_PORT = "auto"          # or e.g. "/dev/cu.usbserial-XXXX"; run eeg_detect.py
EEG_BAUD = 115200          # confirm from LXSDF PDF; PolyG-I is high-rate, try 921600 too

# PolyG-I streams up to 16 (14+AUX2) channels interleaved in each LXSDF packet:
# EEG x8, ECG, EMG x2, PPG, GSR, RESP, AUX x2. The parser AUTO-DETECTS how many
# 2-byte channel slots each packet carries. You then say which slots are the 8
# EEG channels. DEFAULT GUESS: EEG is the first 8 slots. Fix if the montage or
# device ordering differs (check against TeleScan's live channel display).
EEG_TOTAL_CHANNELS = None  # None = auto-detect from packet length; or force an int
EEG_CHANNEL_MAP = [0, 1, 2, 3, 4, 5, 6, 7]   # packet slot -> EEG ch 0..7
EEG_CHANNELS = 8
EEG_FS = 256               # sampling rate (Hz) — confirm in TeleScan; 256 typical
EEG_MIN_EPOCH_FRACTION = 0.80  # abort a decision if too many onset-locked samples are missing

# ADC scaling: raw 2-byte sample -> microvolts. LXSDF ships a 12-bit ADC value.
# uV = (raw - ADC_ZERO) * ADC_UV_PER_LSB. Defaults center a 12-bit unipolar code
# and use a placeholder gain; set ADC_UV_PER_LSB from the device datasheet for
# absolute uV. Relative values already work for ErrP without exact scaling.
ADC_BITS = 12
ADC_ZERO = 2048            # midpoint of a 12-bit unipolar code (0..4095)
ADC_UV_PER_LSB = 1.0       # placeholder; datasheet gives the real Vref/gain

# Path B (only if PolyG-I refuses to be a plain COM port): a Windows helper
# calling LXSMWD12.dll forwards the RAW LXSDF byte stream here over localhost /
# a direct ethernet cable. Same parser runs on both paths. No router.
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
