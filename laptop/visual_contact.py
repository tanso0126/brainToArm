"""Scene-independent gripper contact detection from the wrist-camera markers.

A hobby servo reports neither shaft position nor torque. The rigid wrist camera
does observe the two fingers, however. Calibrate the empty jaw once across its
commanded angle range, then compare the measured blue/red marker separation with
that free-motion curve. If the commanded jaws remain significantly wider than
the empty baseline, a physical object is blocking them.

This is deliberately a contact detector, not a monocular metric-depth claim.
Missing calibration, missing markers, or an angle outside the calibrated range
returns UNKNOWN and must prevent autonomous lift/grasp decisions.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional
import argparse
import json
import math
import time

import numpy as np


DEFAULT_BASELINE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "calibration"
    / "wrist_jaw_baseline.json")


@dataclass(frozen=True)
class JawSample:
    angle_deg: int
    opening_px: float
    opening_mad_px: float
    center_x_px: float
    center_y_px: float
    sample_count: int


@dataclass(frozen=True)
class ContactAssessment:
    state: str
    angle_deg: int
    observed_opening_px: Optional[float]
    expected_opening_px: Optional[float]
    residual_px: Optional[float]
    threshold_px: Optional[float]
    reason: str

    @property
    def contact(self):
        return self.state == "CONTACT"


class JawBaseline:
    VERSION = 1

    def __init__(self, samples: Iterable[JawSample], frame_size=(1280, 720),
                 camera_name="", created_at=None):
        self.samples = tuple(sorted(samples, key=lambda item: item.angle_deg))
        self.frame_size = tuple(int(value) for value in frame_size)
        self.camera_name = str(camera_name)
        self.created_at = created_at or time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.validate()

    def validate(self):
        if len(self.samples) < 3:
            raise ValueError("jaw baseline needs at least three commanded angles")
        angles = [item.angle_deg for item in self.samples]
        if len(set(angles)) != len(angles) or angles != sorted(angles):
            raise ValueError("jaw baseline angles must be unique and ordered")
        if any(not 0 <= item.angle_deg <= 180 for item in self.samples):
            raise ValueError("jaw baseline angle outside 0..180")
        if any(
            not math.isfinite(value) or value <= 0
            for item in self.samples
            for value in (item.opening_px, item.sample_count)
        ):
            raise ValueError("jaw baseline opening/count must be positive")
        if any(
            not math.isfinite(value) or value < 0
            for item in self.samples
            for value in (item.opening_mad_px,)
        ):
            raise ValueError("jaw baseline MAD must be finite and non-negative")
        if any(
            not math.isfinite(value)
            for item in self.samples
            for value in (item.center_x_px, item.center_y_px)
        ):
            raise ValueError("jaw baseline center must be finite")
        # Increasing the close command must not make an empty jaw substantially
        # wider. A 3 px allowance tolerates camera/segmentation noise.
        for first, second in zip(self.samples, self.samples[1:]):
            if second.opening_px > first.opening_px + 3.0:
                raise ValueError("empty-jaw opening curve is not monotonic")
        if len(self.frame_size) != 2 or any(value <= 0 for value in self.frame_size):
            raise ValueError("jaw baseline frame size must be positive width/height")
        return True

    def expected(self, angle_deg):
        angle = float(angle_deg)
        if angle < self.samples[0].angle_deg or angle > self.samples[-1].angle_deg:
            raise ValueError("gripper angle outside calibrated jaw range")
        for item in self.samples:
            if angle == item.angle_deg:
                return item
        for left, right in zip(self.samples, self.samples[1:]):
            if left.angle_deg <= angle <= right.angle_deg:
                fraction = ((angle - left.angle_deg)
                            / (right.angle_deg - left.angle_deg))
                return JawSample(
                    int(round(angle)),
                    left.opening_px + fraction * (
                        right.opening_px - left.opening_px),
                    left.opening_mad_px + fraction * (
                        right.opening_mad_px - left.opening_mad_px),
                    left.center_x_px + fraction * (
                        right.center_x_px - left.center_x_px),
                    left.center_y_px + fraction * (
                        right.center_y_px - left.center_y_px),
                    min(left.sample_count, right.sample_count),
                )
        raise RuntimeError("could not interpolate calibrated jaw angle")

    def assess(self, angle_deg, observation, minimum_close_angle=140,
               minimum_margin_px=7.0, mad_multiplier=5.0):
        if angle_deg < minimum_close_angle:
            return ContactAssessment(
                "UNKNOWN", int(angle_deg), None, None, None, None,
                f"contact is not evaluated below {minimum_close_angle} degrees")
        if observation is None or observation.gripper is None:
            return ContactAssessment(
                "UNKNOWN", int(angle_deg), None, None, None, None,
                "both finger markers are required")
        try:
            expected = self.expected(angle_deg)
        except ValueError as exc:
            return ContactAssessment(
                "UNKNOWN", int(angle_deg),
                float(observation.gripper.opening_px), None, None, None, str(exc))
        observed = float(observation.gripper.opening_px)
        residual = observed - expected.opening_px
        threshold = max(
            float(minimum_margin_px),
            float(mad_multiplier) * expected.opening_mad_px)
        if residual > threshold:
            state = "CONTACT"
            reason = (
                f"jaw remained {residual:.1f}px wider than empty baseline "
                f"(threshold {threshold:.1f}px)")
        elif abs(residual) <= threshold:
            state = "FREE"
            reason = "jaw matches empty free-motion baseline"
        else:
            state = "UNKNOWN"
            reason = (
                "observed jaw is narrower than the empty baseline; "
                "camera/profile geometry is inconsistent")
        return ContactAssessment(
            state, int(angle_deg), observed, expected.opening_px,
            residual, threshold, reason)

    def to_dict(self):
        return {
            "version": self.VERSION,
            "created_at": self.created_at,
            "camera_name": self.camera_name,
            "frame_size": list(self.frame_size),
            "samples": [asdict(item) for item in self.samples],
        }

    def save(self, path=DEFAULT_BASELINE_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path=DEFAULT_BASELINE_PATH):
        path = Path(path)
        if not path.exists():
            raise RuntimeError(
                f"empty-jaw visual baseline is missing: {path}. "
                "Autonomous contact/lift must remain disabled.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != cls.VERSION:
            raise ValueError("unsupported jaw baseline version")
        return cls(
            [JawSample(**item) for item in payload["samples"]],
            frame_size=payload["frame_size"],
            camera_name=payload.get("camera_name", ""),
            created_at=payload.get("created_at"),
        )


def _median_mad(values):
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad


def calibrate_empty_jaw(path=DEFAULT_BASELINE_PATH, angles=None,
                        frames_per_angle=7, frame_timeout=12.0,
                        mechanical_settle_s=1.8, discard_frames=3):
    """Physically measure the free jaw while one serial session stays open."""
    import cv2
    import config
    from arm_serial import ArmSerial
    from wrist_vision import LATEST_RAW_PATH, WristDetector

    angles = tuple(angles or (90, 110, 130, 150, 170, 180))
    if frames_per_angle < 5:
        raise ValueError("empty-jaw calibration needs at least five frames per angle")
    detector = WristDetector()
    arm = ArmSerial()
    samples = []
    pose = list(config.HOME_POSE)

    def wait_for_new_raw_frames(count):
        seen = 0
        previous_mtime = None
        deadline = time.monotonic() + frame_timeout
        while seen < count and time.monotonic() < deadline:
            try:
                mtime = LATEST_RAW_PATH.stat().st_mtime_ns
            except FileNotFoundError:
                time.sleep(0.05)
                continue
            if mtime != previous_mtime:
                previous_mtime = mtime
                seen += 1
            time.sleep(0.05)
        if seen < count:
            raise RuntimeError(
                f"camera published only {seen}/{count} settle frames")

    try:
        pose[config.J_GRIP] = config.GRIP_OPEN
        arm.send_angles(pose)
        arm.wait_done(timeout=10)
        time.sleep(mechanical_settle_s)
        wait_for_new_raw_frames(discard_frames)
        for angle in angles:
            pose[config.J_GRIP] = int(angle)
            arm.send_angles(pose)
            arm.wait_done(timeout=10)
            # Firmware DONE describes its software slew, not the real hobby
            # servo shaft. Let the linkage settle, then discard fresh frames so
            # no previous-angle silhouette contaminates this command.
            time.sleep(mechanical_settle_s)
            wait_for_new_raw_frames(discard_frames)
            measurements = []
            deadline = time.monotonic() + frame_timeout
            previous_mtime = None
            while (len(measurements) < frames_per_angle
                   and time.monotonic() < deadline):
                try:
                    stat = LATEST_RAW_PATH.stat()
                except FileNotFoundError:
                    time.sleep(0.05)
                    continue
                if stat.st_mtime_ns == previous_mtime:
                    time.sleep(0.05)
                    continue
                previous_mtime = stat.st_mtime_ns
                frame = cv2.imread(str(LATEST_RAW_PATH))
                if frame is None:
                    continue
                observation, _masks = detector.detect(frame)
                if observation.gripper is None:
                    continue
                gripper = observation.gripper
                measurements.append((
                    float(gripper.opening_px),
                    float(gripper.center[0]),
                    float(gripper.center[1]),
                ))
            if len(measurements) < frames_per_angle:
                raise RuntimeError(
                    f"only {len(measurements)}/{frames_per_angle} valid marker "
                    f"frames at empty-jaw angle {angle}")
            openings = [item[0] for item in measurements]
            centers_x = [item[1] for item in measurements]
            centers_y = [item[2] for item in measurements]
            opening, mad = _median_mad(openings)
            sample = JawSample(
                int(angle), opening, mad,
                float(np.median(centers_x)),
                float(np.median(centers_y)),
                len(measurements))
            samples.append(sample)
            print(
                f"[jaw-cal] angle={angle:3d} opening={opening:6.1f}px "
                f"MAD={mad:4.1f}px center=({sample.center_x_px:.1f},"
                f"{sample.center_y_px:.1f}) n={sample.sample_count}",
                flush=True)
        baseline = JawBaseline(
            samples, frame_size=config.WRIST_FRAME_SIZE,
            camera_name=config.WRIST_CAMERA_NAME)
        saved = baseline.save(path)
        print(f"[jaw-cal] saved {saved}", flush=True)
        return baseline
    finally:
        # Success, validation failure, camera failure, and Ctrl-C must all make
        # a best effort to leave an empty gripper open instead of stalled shut.
        try:
            pose[config.J_GRIP] = config.GRIP_OPEN
            arm.send_angles(pose)
            arm.wait_done(timeout=10)
            time.sleep(mechanical_settle_s)
            print("[jaw-cal] final jaw command: OPEN 90", flush=True)
        except Exception as exc:
            print(f"[jaw-cal] WARNING: could not reopen jaw: {exc}", flush=True)
        arm.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate-empty", action="store_true",
                        help="cycle the physical empty jaw and save its baseline")
    parser.add_argument("--path", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--frames", type=int, default=7)
    args = parser.parse_args()
    if args.calibrate_empty:
        calibrate_empty_jaw(args.path, frames_per_angle=args.frames)
        return
    baseline = JawBaseline.load(args.path)
    print(json.dumps(baseline.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
