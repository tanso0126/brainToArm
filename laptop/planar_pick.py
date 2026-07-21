"""Background-independent, camera-closed-loop pick/place for the planar arm.

This side-camera calibration locks base yaw at 90 and leaves servo3 unused. It
uses FastSAM to locate the single portable object and the gripper in every
correction frame. No empty-table image, venue-specific ROI, or background
learning is required.

    python3 laptop/planar_pick.py --detect-only
    python3 laptop/planar_pick.py --run
"""
from pathlib import Path
import argparse
import sys
import time

import cv2
import numpy as np

import config
from arm_serial import ArmSerial
from vision_segment import FastSAMDetector, annotate_pair, estimate_camera_transform


DEBUG_DIR = Path(__file__).resolve().parents[1] / "data/vision"


def _clamp(value, limits):
    return max(limits[0], min(limits[1], value))


def stepped_values(start, target, step=2):
    """Integer waypoints excluding start and including target."""
    if step <= 0:
        raise ValueError("step must be positive")
    start, target = int(start), int(target)
    if start == target:
        return []
    direction = 1 if target > start else -1
    values = list(range(start + direction * step, target, direction * step))
    values.append(target)
    return values


def pick_pose_for_object_x(center_x):
    """Return a safe calibrated seed; live gripper vision refines it."""
    if not np.isfinite(center_x):
        raise ValueError("object x must be finite")
    elbow_float = (config.PLANAR_REFERENCE_ELBOW
                   + (float(center_x) - config.PLANAR_REFERENCE_OBJECT_X)
                   / config.PLANAR_ELBOW_PX_PER_DEG)
    elbow = int(round(_clamp(elbow_float, config.PLANAR_PICK_ELBOW_RANGE)))
    elbow_y = ((elbow - config.PLANAR_REFERENCE_ELBOW)
               * config.PLANAR_ELBOW_Y_PX_PER_DEG)
    shoulder = int(round(config.PLANAR_REFERENCE_GRASP_SHOULDER
                         - elbow_y / config.PLANAR_SHOULDER_PX_PER_DEG))
    shoulder = int(_clamp(
        shoulder,
        (config.PLANAR_SERVO_MIN[config.J_SHOULDER],
         config.PLANAR_SERVO_MAX[config.J_SHOULDER])))
    return shoulder, elbow


def open_planar_camera(detector=None):
    """Find the camera that actually sees both the target and the gripper."""
    configured = config.PLANAR_CAM_INDEX
    candidates = range(4) if configured == "auto" else [int(configured)]
    width, height = config.PLANAR_FRAME_SIZE
    failures = []
    ranked = []
    for index in candidates:
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            failures.append(f"{index}:open")
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame = None
        for _ in range(8):
            ok, candidate = cap.read()
            if ok:
                frame = candidate
        if (frame is None or frame.shape[:2] != (height, width)
                or float(frame.std()) < 5.0):
            failures.append(f"{index}:blank-or-wrong-size")
            cap.release()
            continue
        if detector is None:
            ranked.append((float(frame.std()), index))
        else:
            try:
                # The Uno auto-reset can leave the arm moving while cameras are
                # enumerated. A stationary target identifies the work view;
                # gripper visibility is enforced after the preparation pose.
                target, _gripper, _ = detector.locate(
                    frame, require_gripper=False)
                ranked.append((target.confidence, index))
            except RuntimeError:
                failures.append(f"{index}:no-target")
        cap.release()
    if not ranked:
        raise RuntimeError("no usable work camera found (" + ", ".join(failures) + ")")
    _score, selected = max(ranked)
    cap = cv2.VideoCapture(selected, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"selected camera {selected} could not be reopened")
    print(f"[planar] work camera index {selected} selected ({width}x{height})")
    return cap


class PlanarPicker:
    def __init__(self, arm=None, camera=None, detector=None):
        if not config.PLANAR_ARM_CALIBRATED:
            raise RuntimeError("planar physical calibration gate is not enabled")
        # Load perception before opening serial: a model/download failure must
        # not reset or move the physical arm.
        self.detector = detector or FastSAMDetector()
        reference_path = Path(config.PLANAR_GEOMETRY_REFERENCE)
        if not reference_path.is_absolute():
            reference_path = Path(__file__).resolve().parents[1] / reference_path
        self.geometry_reference = cv2.imread(str(reference_path))
        if self.geometry_reference is None:
            raise RuntimeError(f"missing camera geometry reference: {reference_path}")
        self.arm = arm or ArmSerial()
        self.cap = camera or open_planar_camera(self.detector)
        self._owns_camera = camera is None
        if not self.cap.isOpened():
            self.arm.close()
            raise RuntimeError("cannot open planar camera")
        self.pose = self.arm.status()
        self._validate_pose(self.pose)
        for _ in range(8):
            self._read()

    def _validate_pose(self, pose):
        if len(pose) != config.N_JOINTS:
            raise ValueError("planar pose must contain seven joints")
        for index, value in enumerate(pose):
            lo, hi = config.PLANAR_SERVO_MIN[index], config.PLANAR_SERVO_MAX[index]
            if not lo <= value <= hi:
                raise ValueError(
                    f"planar joint {index + 1}={value} outside [{lo},{hi}]")
        if pose[config.J_BASE] != config.PLANAR_BASE_ANGLE:
            raise ValueError("camera-calibrated planar base must stay at 90 degrees")
        if pose[config._UNUSED] != config.PLANAR_UNUSED_ANGLE:
            raise ValueError("unused servo3 must stay locked at 90 degrees")

    def _read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return frame

    def capture(self, count=4):
        # AVFoundation retains frames while servos move. Drop them and allow
        # auto-exposure to settle so a median never contains motion ghosts.
        time.sleep(0.20)
        for _ in range(8):
            self._read()
        return np.median(
            np.stack([self._read() for _ in range(count)]), axis=0).astype(np.uint8)

    def assert_camera_geometry(self, frame):
        motion = estimate_camera_transform(self.geometry_reference, frame)
        print("[planar] camera geometry "
              f"scale={motion['scale']:.4f}, rotation={motion['rotation_deg']:.2f}°, "
              f"translation={motion['translation_px']:.1f}px")
        if (abs(motion["scale"] - 1.0) > config.PLANAR_MAX_CAMERA_SCALE_DRIFT
                or abs(motion["rotation_deg"]) > config.PLANAR_MAX_CAMERA_ROTATION_DEG
                or motion["translation_px"] > config.PLANAR_MAX_CAMERA_TRANSLATION_PX):
            raise RuntimeError(
                "camera framing changed; turn off Center Stage and restore the fixed view")

    def move_joint(self, joint, target, stepped=False):
        values = stepped_values(self.pose[joint], target, 2) if stepped else [int(target)]
        for value in values:
            next_pose = list(self.pose)
            next_pose[joint] = value
            self._validate_pose(next_pose)
            self.arm.send_angles(next_pose)
            self.arm.wait_done()
            self.pose = next_pose

    def prepare_observation(self):
        # Rotation and full opening happen before any descent. One joint moves
        # at a time and firmware independently slew-limits the physical servos.
        self.move_joint(config.J_GRIP, config.GRIP_OPEN)
        self.move_joint(config.J_WRIST, config.PLANAR_WRIST_PITCH)
        self.move_joint(config.J_ROLL, config.PLANAR_WRIST_ROLL)
        self.move_joint(config.J_SHOULDER, 124)
        self.move_joint(config.J_ELBOW, config.PLANAR_REFERENCE_ELBOW)

    def observe(self, previous=None, label="target", require_gripper=True,
                verify_geometry=False):
        frame = self.capture()
        if verify_geometry:
            self.assert_camera_geometry(frame)
        target, gripper, _candidates = self.detector.locate(
            frame, previous=previous, require_gripper=require_gripper)
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(DEBUG_DIR / f"last_{label}.jpg"),
                    annotate_pair(frame, target, gripper, label))
        return target, gripper

    def close_gripper(self):
        """Close slowly; stop early only when a real feedback sensor says load."""
        baseline = self.arm.grip_feedback()
        last_feedback = baseline
        start = self.pose[config.J_GRIP]
        angles = list(range(
            start + config.PLANAR_GRIP_CLOSE_STEP,
            config.GRIP_CLOSED,
            config.PLANAR_GRIP_CLOSE_STEP)) + [config.GRIP_CLOSED]
        for angle in angles:
            self.move_joint(config.J_GRIP, angle)
            last_feedback = self.arm.grip_feedback()
            if (config.GRIP_FEEDBACK_ENABLED and baseline is not None
                    and last_feedback is not None
                    and last_feedback - baseline >= config.GRIP_FEEDBACK_DELTA):
                print(f"[planar] grip load detected: {last_feedback - baseline} ADC")
                break
        return last_feedback

    def _failed_grasp_recovery(self):
        self.move_joint(config.J_GRIP, config.GRIP_OPEN)
        safe_shoulder = max(
            config.PLANAR_SERVO_MIN[config.J_SHOULDER],
            self.pose[config.J_SHOULDER] - config.PLANAR_LIFT_DEG)
        self.move_joint(config.J_SHOULDER, safe_shoulder, stepped=True)

    def run(self):
        print("[planar] preparing rotated, fully open observation pose")
        self.prepare_observation()
        target, _gripper = self.observe(
            label="pick", require_gripper=False, verify_geometry=True)
        seed_shoulder, seed_elbow = pick_pose_for_object_x(target.center[0])
        approach_shoulder = seed_shoulder - config.PLANAR_APPROACH_CLEARANCE_DEG
        print(f"[planar] target={target.center}; seed shoulder={seed_shoulder}, "
              f"elbow={seed_elbow}")
        self.move_joint(config.J_SHOULDER, approach_shoulder)
        self.move_joint(config.J_ELBOW, seed_elbow, stepped=True)
        self.move_joint(config.J_GRIP, config.GRIP_OPEN)

        grasp_shoulder = int(_clamp(
            seed_shoulder + config.PLANAR_GRASP_DEPTH_OFFSET_DEG,
            (config.PLANAR_SERVO_MIN[config.J_SHOULDER],
             config.PLANAR_SERVO_MAX[config.J_SHOULDER])))
        self.move_joint(config.J_SHOULDER, grasp_shoulder, stepped=True)
        grasp_target, _ = self.observe(
            previous=target, label="aligned", require_gripper=False,
            verify_geometry=True)
        print(f"[planar] staged descent complete at shoulder={grasp_shoulder}")
        self.close_gripper()
        before_lift_y = grasp_target.center[1]
        self.move_joint(
            config.J_SHOULDER,
            self.pose[config.J_SHOULDER] - 6,
            stepped=True)
        lifted, _ = self.observe(
            previous=grasp_target, label="lifted", require_gripper=False)
        lift_px = before_lift_y - lifted.center[1]
        print(f"[planar] tracked object lift={lift_px:.1f}px")
        if lift_px < config.PLANAR_MIN_LIFT_PX:
            self._failed_grasp_recovery()
            raise RuntimeError("grasp verification failed: object did not rise with gripper")

        transport_shoulder = max(
            config.PLANAR_SERVO_MIN[config.J_SHOULDER],
            self.pose[config.J_SHOULDER] - 6)
        self.move_joint(config.J_SHOULDER, transport_shoulder, stepped=True)
        self.move_joint(config.J_ELBOW, config.PLANAR_PLACE_ELBOW, stepped=True)
        self.move_joint(config.J_SHOULDER, config.PLANAR_PLACE_SHOULDER, stepped=True)
        for angle in stepped_values(self.pose[config.J_GRIP], config.GRIP_OPEN, 10):
            self.move_joint(config.J_GRIP, angle)
        self.move_joint(config.J_SHOULDER,
                        config.PLANAR_RETREAT_SHOULDER, stepped=True)

        placed, _ = self.observe(
            previous=lifted, label="placed", require_gripper=False)
        down_px = placed.center[1] - lifted.center[1]
        if down_px < config.PLANAR_MIN_LIFT_PX / 2:
            raise RuntimeError(
                "place verification failed: object is not visibly back down")
        travel = float(np.linalg.norm(
            np.subtract(placed.center, grasp_target.center)))
        print(f"[planar] SUCCESS: vision verified lift, release, and separation; "
              f"net image travel={travel:.1f}px")
        return {"pick": grasp_target, "lifted": lifted, "placed": placed,
                "travel_px": travel}

    def close(self):
        if self._owns_camera:
            self.cap.release()
        self.arm.close()


def _open_camera(detector):
    cap = open_planar_camera(detector)
    frames = []
    for index in range(12):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("camera read failed")
        if index >= 8:
            frames.append(frame)
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true", help="execute one physical pick/place")
    action.add_argument("--detect-only", action="store_true",
                        help="background-free camera detection without motion")
    action.add_argument("--prefetch", action="store_true",
                        help="download/load the model now for later offline use")
    args = parser.parse_args(argv)

    detector = FastSAMDetector()
    if args.prefetch:
        print(f"[planar] model ready for offline use: {detector.model_path}")
        return 0
    if args.detect_only:
        frame = _open_camera(detector)
        target, _gripper, _ = detector.locate(frame, require_gripper=False)
        print(f"[planar] target center={target.center}, bbox={target.bbox}, "
              f"confidence={target.confidence:.3f}")
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(DEBUG_DIR / "last_detection.jpg"),
                    annotate_pair(frame, target))
        return 0

    picker = None
    try:
        picker = PlanarPicker(detector=detector)
        picker.run()
        return 0
    finally:
        if picker is not None:
            picker.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, TimeoutError) as exc:
        print(f"[planar] STOPPED: {exc}", file=sys.stderr)
        sys.exit(1)
