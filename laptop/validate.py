"""Static config sanity checks — catch contradictory settings BEFORE running,
so a typo in config.py fails with a clear message instead of a weird mid-task
crash. Called by orchestrator preflight; also runnable standalone:

    python validate.py
"""
import math
import config
import kinematics


def validate():
    errs, warns = [], []

    # --- servo arrays all length N_JOINTS ---
    servo_arrays_ok = True
    for name in ("SERVO_PINS", "JOINT_NAMES", "SERVO_OFFSET", "SERVO_DIRECTION",
                 "SERVO_MIN", "SERVO_MAX", "HOME_POSE"):
        arr = getattr(config, name)
        if len(arr) != config.N_JOINTS:
            errs.append(f"{name} has {len(arr)} entries, expected N_JOINTS={config.N_JOINTS}")
            servo_arrays_ok = False
    if len(set(config.SERVO_PINS)) != len(config.SERVO_PINS):
        errs.append("SERVO_PINS contains duplicate Arduino pins")
    if not isinstance(config.ARM_MOCK, bool):
        errs.append("ARM_MOCK must be True or False")
    if not config.ARM_MOCK and not config.ARM_PORT:
        errs.append("ARM_PORT must be set when ARM_MOCK=False")
    if not isinstance(config.ARM_CALIBRATED, bool):
        errs.append("ARM_CALIBRATED must be True or False")
    elif not config.ARM_MOCK and not config.ARM_CALIBRATED:
        errs.append("real arm requires ARM_CALIBRATED=True after arm_jog verification")

    # --- home pose within servo limits ---
    if servo_arrays_ok:
        for i, a in enumerate(config.HOME_POSE):
            if config.SERVO_MIN[i] > config.SERVO_MAX[i]:
                errs.append(f"SERVO_MIN[{i}] > SERVO_MAX[{i}]")
            if not (config.SERVO_MIN[i] <= a <= config.SERVO_MAX[i]):
                errs.append(f"HOME_POSE[{i}]={a} outside SERVO_MIN/MAX [{config.SERVO_MIN[i]},{config.SERVO_MAX[i]}]")
            if config.SERVO_DIRECTION[i] not in (-1, 1):
                errs.append(f"SERVO_DIRECTION[{i}] must be -1 or 1")
        for name, value in (("GRIP_OPEN", config.GRIP_OPEN),
                            ("GRIP_CLOSED", config.GRIP_CLOSED)):
            if not config.SERVO_MIN[config.J_GRIP] <= value <= config.SERVO_MAX[config.J_GRIP]:
                errs.append(f"{name}={value} outside gripper safe range")

    # --- physically verified fixed-base planar mode ---
    for name in ("PLANAR_SERVO_MIN", "PLANAR_SERVO_MAX"):
        values = getattr(config, name)
        if len(values) != config.N_JOINTS:
            errs.append(f"{name} has {len(values)} entries, expected {config.N_JOINTS}")
    if not isinstance(config.PLANAR_ARM_CALIBRATED, bool):
        errs.append("PLANAR_ARM_CALIBRATED must be True or False")
    if (len(config.PLANAR_SERVO_MIN) == config.N_JOINTS
            and len(config.PLANAR_SERVO_MAX) == config.N_JOINTS):
        for i, (lo, hi) in enumerate(zip(config.PLANAR_SERVO_MIN,
                                          config.PLANAR_SERVO_MAX)):
            if lo > hi:
                errs.append(f"PLANAR_SERVO_MIN[{i}] > PLANAR_SERVO_MAX[{i}]")
        if not (config.PLANAR_SERVO_MIN[config.J_BASE]
                == config.PLANAR_SERVO_MAX[config.J_BASE]
                == config.PLANAR_BASE_ANGLE == 90):
            errs.append("camera-calibrated planar mode must keep servo1 at 90")
    if config.GRIP_OPEN != 90 or config.GRIP_CLOSED != 180:
        errs.append("physical gripper calibration requires 90=open and 180=closed")
    if config.PLANAR_WRIST_ROLL != 180 or config.PLANAR_WRIST_PITCH != 180:
        errs.append("verified planar top grasp requires wrist pitch/roll 180")

    # --- EEG channel map sane ---
    total = config.EEG_TOTAL_CHANNELS
    if total is not None and (not isinstance(total, int) or isinstance(total, bool) or total <= 0):
        errs.append("EEG_TOTAL_CHANNELS must be a positive integer or None")
    for slot in config.EEG_CHANNEL_MAP:
        if not isinstance(slot, int) or isinstance(slot, bool) or slot < 0:
            errs.append(f"EEG_CHANNEL_MAP slot {slot!r} must be a non-negative integer")
            continue
        if total is not None and slot >= total:
            errs.append(f"EEG_CHANNEL_MAP slot {slot} >= EEG_TOTAL_CHANNELS {total}")
    if not isinstance(config.EEG_CHANNELS, int) or isinstance(config.EEG_CHANNELS, bool) or config.EEG_CHANNELS <= 0:
        errs.append("EEG_CHANNELS must be a positive integer")
    elif len(config.EEG_CHANNEL_MAP) != config.EEG_CHANNELS:
        errs.append(f"EEG_CHANNEL_MAP has {len(config.EEG_CHANNEL_MAP)} slots but EEG_CHANNELS={config.EEG_CHANNELS}")
    for ch in config.ERRP_FRONTOCENTRAL:
        if not isinstance(ch, int) or isinstance(ch, bool) or ch < 0:
            errs.append(f"ERRP_FRONTOCENTRAL ch {ch!r} must be a non-negative integer")
        elif ch >= config.EEG_CHANNELS:
            errs.append(f"ERRP_FRONTOCENTRAL ch {ch} >= EEG_CHANNELS {config.EEG_CHANNELS}")
    if len(set(config.EEG_CHANNEL_MAP)) != len(config.EEG_CHANNEL_MAP):
        errs.append("EEG_CHANNEL_MAP contains duplicate packet slots")
    if not config.ERRP_FRONTOCENTRAL:
        errs.append("ERRP_FRONTOCENTRAL must contain at least one channel")
    elif len(set(config.ERRP_FRONTOCENTRAL)) != len(config.ERRP_FRONTOCENTRAL):
        errs.append("ERRP_FRONTOCENTRAL contains duplicate channels")

    # --- ErrP band below Nyquist ---
    if not isinstance(config.EEG_FS, (int, float)) or config.EEG_FS <= 0:
        errs.append("EEG_FS must be > 0")
    if (not isinstance(config.ERRP_BAND, (tuple, list)) or len(config.ERRP_BAND) != 2
            or not all(isinstance(v, (int, float)) and math.isfinite(v)
                       for v in config.ERRP_BAND)):
        errs.append("ERRP_BAND must contain two finite numbers")
    elif config.EEG_FS > 0:
        lo, hi = config.ERRP_BAND
        if not 0 < lo < hi:
            errs.append("ERRP_BAND must satisfy 0 < low < high")
        elif hi >= config.EEG_FS / 2:
            errs.append(f"ERRP_BAND high {hi}Hz >= Nyquist {config.EEG_FS/2}Hz")
    if not (0 < config.EEG_MIN_EPOCH_FRACTION <= 1):
        errs.append("EEG_MIN_EPOCH_FRACTION must be in (0, 1]")
    if config.ERRP_BASELINE_S <= 0 or config.ERRP_WINDOW_S <= 0:
        errs.append("ERRP_BASELINE_S and ERRP_WINDOW_S must be > 0")
    if not 0 <= config.ERRP_THRESHOLD <= 1:
        errs.append("ERRP_THRESHOLD must be in [0, 1]")
    if config.ADC_BITS <= 0 or not 0 <= config.ADC_ZERO < 2 ** config.ADC_BITS:
        errs.append("ADC_BITS/ADC_ZERO are inconsistent")
    if config.ADC_UV_PER_LSB <= 0:
        errs.append("ADC_UV_PER_LSB must be > 0")

    # --- source selection valid ---
    if config.EEG_SOURCE not in ("mock", "serial", "tcp", "hid"):
        errs.append(f"EEG_SOURCE '{config.EEG_SOURCE}' invalid")
    if not isinstance(config.EEG_CONFIG_VERIFIED, bool):
        errs.append("EEG_CONFIG_VERIFIED must be True or False")
    elif config.EEG_SOURCE != "mock" and not config.EEG_CONFIG_VERIFIED:
        errs.append("real EEG requires EEG_CONFIG_VERIFIED=True after rate/channel checks")
    if config.EEG_SOURCE == "serial" and not config.EEG_PORT:
        errs.append("EEG_PORT must be set for serial EEG")
    if config.EEG_SOURCE == "hid":
        if config.EEG_HID_VID != 0x0F1F or config.EEG_HID_PID != 0x0010:
            errs.append("HID VID/PID must match the verified PolyG-I 0x0F1F/0x0010")
        if config.EEG_HID_CHANNELS != 8:
            errs.append("verified PolyG-I HID decoder requires EEG_HID_CHANNELS=8")
        if config.EEG_HID_MAX_CHANNELS != 16:
            errs.append("verified PolyG-I D1WD10 decoder requires EEG_HID_MAX_CHANNELS=16")
        if (isinstance(config.EEG_HID_SAMPLE_SELECTOR, bool)
                or not isinstance(config.EEG_HID_SAMPLE_SELECTOR, int)
                or config.EEG_HID_SAMPLE_SELECTOR != 8):
            errs.append("verified PolyG-I dashboard requires EEG_HID_SAMPLE_SELECTOR=8")
        elif config.EEG_FS != 2 ** config.EEG_HID_SAMPLE_SELECTOR:
            errs.append("EEG_FS must equal 2**EEG_HID_SAMPLE_SELECTOR (256 Hz)")
        if (isinstance(config.EEG_HID_GAIN_INDEX, bool)
                or not isinstance(config.EEG_HID_GAIN_INDEX, int)
                or not 0 <= config.EEG_HID_GAIN_INDEX <= 15):
            errs.append("EEG_HID_GAIN_INDEX must be in [0, 15]")
        if (not isinstance(config.EEG_HID_STALL_TIMEOUT_S, (int, float))
                or isinstance(config.EEG_HID_STALL_TIMEOUT_S, bool)
                or not math.isfinite(config.EEG_HID_STALL_TIMEOUT_S)
                or config.EEG_HID_STALL_TIMEOUT_S <= 0):
            errs.append("EEG_HID_STALL_TIMEOUT_S must be > 0")
        if (not isinstance(config.EEG_HID_ADC_UV_PER_COUNT, (int, float))
                or isinstance(config.EEG_HID_ADC_UV_PER_COUNT, bool)
                or not math.isfinite(config.EEG_HID_ADC_UV_PER_COUNT)
                or not math.isclose(config.EEG_HID_ADC_UV_PER_COUNT,
                                    -1.25 / 32768 * 1_000_000,
                                    rel_tol=0.0, abs_tol=1e-12)):
            errs.append("EEG_HID_ADC_UV_PER_COUNT must match the D1WD10 ADC coefficient")
    if (not config.ARM_MOCK and config.EEG_SOURCE == "serial"
            and config.ARM_PORT != "auto" and config.ARM_PORT == config.EEG_PORT):
        errs.append("ARM_PORT and EEG_PORT refer to the same serial device")
    if config.OBJECT_METHOD not in ("bgsub", "yolo", "hsv", "aruco"):
        errs.append(f"OBJECT_METHOD '{config.OBJECT_METHOD}' invalid")
    if not isinstance(config.CAM_MOCK, bool) or not isinstance(config.CAM_CALIBRATED, bool):
        errs.append("CAM_MOCK and CAM_CALIBRATED must be True or False")
    elif not config.CAM_MOCK and not config.CAM_CALIBRATED:
        errs.append("real camera requires CAM_CALIBRATED=True after workspace calibration")

    # --- heights ordered sensibly ---
    heights = (config.Z_GRASP, config.Z_APPROACH, config.Z_LIFT, config.Z_PLACE)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in heights):
        errs.append("pick/place heights must be finite numbers")
    elif not (config.Z_GRASP < config.Z_APPROACH <= config.Z_LIFT):
        warns.append(f"heights odd: expect Z_GRASP < Z_APPROACH <= Z_LIFT "
                     f"({config.Z_GRASP},{config.Z_APPROACH},{config.Z_LIFT})")
    if config.GRASP_RETRIES < 0:
        errs.append("GRASP_RETRIES must be >= 0")
    if config.GRASP_VERIFY_RADIUS_CM <= 0:
        errs.append("GRASP_VERIFY_RADIUS_CM must be > 0")
    if not isinstance(config.POLICY_SPATIAL_LEARNING, bool):
        errs.append("POLICY_SPATIAL_LEARNING must be True or False")
    if not (0 < config.SERVO_GAIN <= 1):
        errs.append("SERVO_GAIN must be in (0, 1]")
    if config.SERVO_TOL_CM <= 0 or config.SERVO_MAX_ITERS <= 0:
        errs.append("SERVO_TOL_CM and SERVO_MAX_ITERS must be > 0")

    # --- camera homography point sets ---
    if len(config.CAM_CALIB_IMAGE_PTS) < 4:
        errs.append("CAM_CALIB_IMAGE_PTS needs at least 4 points")
    if len(config.CAM_CALIB_IMAGE_PTS) != len(config.CAM_CALIB_WORLD_PTS):
        errs.append("camera image/world calibration point counts differ")
    for name, points in (("CAM_CALIB_IMAGE_PTS", config.CAM_CALIB_IMAGE_PTS),
                         ("CAM_CALIB_WORLD_PTS", config.CAM_CALIB_WORLD_PTS)):
        try:
            normalized = [tuple(float(v) for v in point) for point in points]
            if any(len(point) != 2 or not all(math.isfinite(v) for v in point)
                   for point in normalized):
                raise ValueError
            if len(set(normalized)) != len(normalized):
                errs.append(f"{name} contains duplicate points")
        except (TypeError, ValueError):
            errs.append(f"{name} must contain finite 2D points")

    # --- link lengths positive ---
    links_ok = True
    for name in ("L_BASE_HEIGHT", "L_UPPER", "L_FORE", "L_HAND"):
        value = getattr(config, name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            errs.append(f"{name} must be > 0")
            links_ok = False

    # --- place locations reachable by IK ---
    try:
        px, py = config.PLACE_LOCATION
        place_finite = all(math.isfinite(float(v)) for v in (px, py))
    except (TypeError, ValueError):
        place_finite = False
    if not place_finite:
        errs.append("PLACE_LOCATION must be a finite (x, y) pair")
    elif links_ok:
        for z in (config.Z_PLACE, config.Z_LIFT):
            if not kinematics.reachable(px, py, z):
                errs.append(
                    f"PLACE_LOCATION {config.PLACE_LOCATION} unreachable at z={z}")

    return errs, warns


def main():
    errs, warns = validate()
    for w in warns:
        print(f"  warn: {w}")
    for e in errs:
        print(f"  ERROR: {e}")
    if not errs and not warns:
        print("config OK")
    return not errs


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
