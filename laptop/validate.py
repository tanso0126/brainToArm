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
    for name in ("SERVO_OFFSET", "SERVO_DIRECTION", "SERVO_MIN", "SERVO_MAX", "HOME_POSE"):
        arr = getattr(config, name)
        if len(arr) != config.N_JOINTS:
            errs.append(f"{name} has {len(arr)} entries, expected N_JOINTS={config.N_JOINTS}")
    if not isinstance(config.ARM_MOCK, bool):
        errs.append("ARM_MOCK must be True or False")
    if not config.ARM_MOCK and not config.ARM_PORT:
        errs.append("ARM_PORT must be set when ARM_MOCK=False")

    # --- home pose within servo limits ---
    for i, a in enumerate(config.HOME_POSE):
        if config.SERVO_MIN[i] > config.SERVO_MAX[i]:
            errs.append(f"SERVO_MIN[{i}] > SERVO_MAX[{i}]")
        if not (config.SERVO_MIN[i] <= a <= config.SERVO_MAX[i]):
            errs.append(f"HOME_POSE[{i}]={a} outside SERVO_MIN/MAX [{config.SERVO_MIN[i]},{config.SERVO_MAX[i]}]")
        if config.SERVO_DIRECTION[i] not in (-1, 1):
            errs.append(f"SERVO_DIRECTION[{i}] must be -1 or 1")

    # --- EEG channel map sane ---
    total = config.EEG_TOTAL_CHANNELS
    for slot in config.EEG_CHANNEL_MAP:
        if total is not None and slot >= total:
            errs.append(f"EEG_CHANNEL_MAP slot {slot} >= EEG_TOTAL_CHANNELS {total}")
    if len(config.EEG_CHANNEL_MAP) != config.EEG_CHANNELS:
        warns.append(f"EEG_CHANNEL_MAP has {len(config.EEG_CHANNEL_MAP)} slots but EEG_CHANNELS={config.EEG_CHANNELS}")
    for ch in config.ERRP_FRONTOCENTRAL:
        if ch >= config.EEG_CHANNELS:
            errs.append(f"ERRP_FRONTOCENTRAL ch {ch} >= EEG_CHANNELS {config.EEG_CHANNELS}")

    # --- ErrP band below Nyquist ---
    if config.EEG_FS <= 0:
        errs.append("EEG_FS must be > 0")
    elif config.ERRP_BAND[1] >= config.EEG_FS / 2:
        errs.append(f"ERRP_BAND high {config.ERRP_BAND[1]}Hz >= Nyquist {config.EEG_FS/2}Hz")
    if not (0 < config.EEG_MIN_EPOCH_FRACTION <= 1):
        errs.append("EEG_MIN_EPOCH_FRACTION must be in (0, 1]")

    # --- source selection valid ---
    if config.EEG_SOURCE not in ("mock", "serial", "tcp"):
        errs.append(f"EEG_SOURCE '{config.EEG_SOURCE}' invalid")
    if config.OBJECT_METHOD not in ("bgsub", "yolo", "hsv", "aruco"):
        errs.append(f"OBJECT_METHOD '{config.OBJECT_METHOD}' invalid")

    # --- heights ordered sensibly ---
    if not (config.Z_GRASP < config.Z_APPROACH <= config.Z_LIFT):
        warns.append(f"heights odd: expect Z_GRASP < Z_APPROACH <= Z_LIFT "
                     f"({config.Z_GRASP},{config.Z_APPROACH},{config.Z_LIFT})")

    # --- place location reachable by IK ---
    px, py = config.PLACE_LOCATION
    if not kinematics.reachable(px, py, config.Z_PLACE):
        errs.append(f"PLACE_LOCATION {config.PLACE_LOCATION} unreachable with current link lengths")

    # --- link lengths positive ---
    for name in ("L_BASE_HEIGHT", "L_UPPER", "L_FORE", "L_HAND"):
        if getattr(config, name) <= 0:
            errs.append(f"{name} must be > 0")

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
