"""Robust wrist-ultrasonic ranging and forearm-mount calibration.

An HC-SR04 does not identify an object.  It returns one acoustic reflector from
its broad cone and a hard tabletop often produces several specular/multipath
clusters.  This module consequently exposes a *profile* (raw echoes, dominant
cluster, MAD, support and validity) rather than pretending every integer is a
usable depth.

The sensor is physically attached to the motor-4 camera bracket.  Its extrinsic
transform is therefore expressed in :func:`arm_fk.sensor_pose`: it follows
motor-4 pitch but not motor-6 roll.  A floor calibration fits the local x/z
offset and beam pitch from several known robot poses and their stable
empty-table ranges.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
import argparse
import json
import math
import time

import numpy as np

import arm_fk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_PATH = (
    ROOT / "data" / "calibration" / "ultrasonic_mount.json")


@dataclass(frozen=True)
class RangeProfile:
    raw_mm: tuple[Optional[float], ...]
    cluster_mm: tuple[float, ...]
    distance_mm: Optional[float]
    mad_mm: Optional[float]
    valid_fraction: float
    support_fraction: float
    stable: bool
    reason: str


@dataclass(frozen=True)
class RepeatedRangeProfile:
    batches: tuple[RangeProfile, ...]
    distance_mm: Optional[float]
    batch_spread_mm: Optional[float]
    stable: bool
    reason: str


def robust_profile(values: Iterable[Optional[float]], cluster_width_mm=8.0,
                   min_valid_fraction=0.70, min_support_fraction=0.60,
                   max_mad_mm=4.0, min_cluster_samples=7) -> RangeProfile:
    """Select the densest bounded echo cluster and grade its reliability.

    ``support_fraction`` is relative to valid echoes, while ``valid_fraction``
    includes timeouts.  This rejects both the all-timeout case and a misleading
    median formed from two similarly sized reflectors.
    """
    raw = tuple(None if value is None else float(value) for value in values)
    if not raw:
        raise ValueError("range profile needs at least one requested sample")
    if cluster_width_mm <= 0:
        raise ValueError("cluster width must be positive")
    valid = sorted(value for value in raw
                   if value is not None and math.isfinite(value) and value > 0)
    valid_fraction = len(valid) / len(raw)
    if not valid:
        return RangeProfile(raw, (), None, None, valid_fraction, 0.0, False,
                            "no valid echoes")

    best = []
    right = 0
    for left, lower in enumerate(valid):
        right = max(right, left)
        while right < len(valid) and valid[right] - lower <= cluster_width_mm:
            right += 1
        candidate = valid[left:right]
        if (len(candidate) > len(best)
                or (len(candidate) == len(best) and candidate
                    and (candidate[-1] - candidate[0])
                    < (best[-1] - best[0]))):
            best = candidate

    distance = float(np.median(best))
    mad = float(np.median(np.abs(np.asarray(best) - distance)))
    support = len(best) / len(valid)
    reasons = []
    if valid_fraction < min_valid_fraction:
        reasons.append(
            f"valid echoes {valid_fraction:.0%} < {min_valid_fraction:.0%}")
    if support < min_support_fraction:
        reasons.append(
            f"dominant cluster {support:.0%} < {min_support_fraction:.0%}")
    if len(best) < min_cluster_samples:
        reasons.append(
            f"cluster samples {len(best)} < {min_cluster_samples}")
    if mad > max_mad_mm:
        reasons.append(f"MAD {mad:.1f} mm > {max_mad_mm:.1f} mm")
    stable = not reasons
    return RangeProfile(
        raw, tuple(best), distance, mad, valid_fraction, support, stable,
        "stable dominant echo" if stable else "; ".join(reasons))


def acquire_profile(client, samples=15, interval_s=0.060, **profile_options):
    """Acquire independent one-shot echoes through the persistent arm session."""
    if samples < 1:
        raise ValueError("samples must be positive")
    values = []
    for index in range(int(samples)):
        response = client.request({"command": "distance", "samples": 1})
        value = response.get("distanceMm") if response.get("valid") else None
        values.append(value)
        if index + 1 < samples and interval_s:
            time.sleep(float(interval_s))
    return robust_profile(values, **profile_options)


def repeated_profile(client, batches=3, samples_per_batch=12,
                     max_batch_spread_mm=5.0, pause_s=0.20,
                     **profile_options):
    """Require independent stable echo batches to agree over time."""
    if batches < 2:
        raise ValueError("repeated profile needs at least two batches")
    profiles = []
    for index in range(int(batches)):
        profiles.append(acquire_profile(
            client, samples=samples_per_batch, **profile_options))
        if index + 1 < batches and pause_s:
            time.sleep(float(pause_s))
    usable = [
        profile.distance_mm for profile in profiles
        if profile.stable and profile.distance_mm is not None
    ]
    if len(usable) != len(profiles):
        return RepeatedRangeProfile(
            tuple(profiles), None, None, False,
            f"only {len(usable)}/{len(profiles)} batches were stable")
    spread = float(max(usable) - min(usable))
    distance = float(np.median(usable))
    stable = spread <= float(max_batch_spread_mm)
    return RepeatedRangeProfile(
        tuple(profiles), distance, spread, stable,
        ("stable across batches" if stable else
         f"batch spread {spread:.1f} mm > {max_batch_spread_mm:.1f} mm"))


def wait_for_stable_profile(client, timeout_s=8.0, batches=2,
                            samples_per_batch=8, retry_pause_s=0.25,
                            **profile_options):
    """Wait for measured stability, not an assumed fixed servo settle delay."""
    deadline = time.monotonic() + float(timeout_s)
    attempts = []
    while time.monotonic() < deadline:
        result = repeated_profile(
            client, batches=batches, samples_per_batch=samples_per_batch,
            **profile_options)
        attempts.append(result)
        if result.stable:
            return result, tuple(attempts)
        if retry_pause_s:
            time.sleep(float(retry_pause_s))
    last_reason = attempts[-1].reason if attempts else "no measurement"
    raise TimeoutError(
        f"ultrasonic echo did not settle within {timeout_s:.1f}s: "
        f"{last_reason}")


@dataclass(frozen=True)
class SonarMount:
    """Planar sonar extrinsic relative to the motor-4 camera bracket frame."""

    origin_x_m: float
    origin_z_m: float
    beam_pitch_deg: float
    pitch_scale_deg_per_servo_deg: float = -1.5

    def ray(self, servo: Sequence[float]):
        """Return ``(origin, unit_direction)`` in the robot base frame."""
        rotation, pivot = arm_fk.sensor_pose(
            servo, self.pitch_scale_deg_per_servo_deg)
        pitch = math.radians(self.beam_pitch_deg)
        local_origin = np.array(
            [self.origin_x_m, 0.0, self.origin_z_m], dtype=float)
        # Positive beam pitch means downward relative to forearm +x.
        local_direction = np.array(
            [math.cos(pitch), 0.0, -math.sin(pitch)], dtype=float)
        origin = pivot + rotation @ local_origin
        direction = rotation @ local_direction
        direction /= np.linalg.norm(direction)
        return origin, direction

    def plane_range_mm(self, servo: Sequence[float], table_z_m=0.0):
        """Expected range to a horizontal plane, or ``None`` if ray misses it."""
        origin, direction = self.ray(servo)
        if direction[2] >= -1e-6:
            return None
        distance_m = (float(table_z_m) - origin[2]) / direction[2]
        if distance_m <= 0:
            return None
        return float(distance_m * 1000.0)

    def point(self, servo: Sequence[float], distance_mm: float):
        origin, direction = self.ray(servo)
        return origin + direction * (float(distance_mm) / 1000.0)


@dataclass(frozen=True)
class FloorCalibrationSample:
    pose: tuple[int, ...]
    distance_mm: float
    mad_mm: float = 0.0
    support_fraction: float = 1.0


def fit_floor_mount(samples: Sequence[FloorCalibrationSample],
                    table_z_m=0.0, initial=None):
    """Fit forearm-local ``x, z, beam_pitch`` using stable floor ranges."""
    if len(samples) < 5:
        raise ValueError("at least five stable floor poses are required")
    if len({tuple(sample.pose) for sample in samples}) != len(samples):
        raise ValueError("calibration poses must be distinct")
    sensor_angles = [
        arm_fk.shoulder_joint_deg(sample.pose[1])
        + arm_fk.elbow_joint_deg(sample.pose[2])
        + arm_fk.wrist_pitch_joint_deg(sample.pose[3])
        for sample in samples
    ]
    if max(sensor_angles) - min(sensor_angles) < 8.0:
        raise ValueError(
            "sensor calibration poses need at least 8 degrees of angle span")

    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - deployment dependency check
        raise RuntimeError("scipy is required for sonar calibration") from exc

    initial = np.asarray(
        initial if initial is not None else [0.015, 0.055, 25.0, -1.0],
        dtype=float)
    if initial.shape == (3,):
        initial = np.append(initial, -1.5)
    if initial.shape != (4,) or not np.isfinite(initial).all():
        raise ValueError("initial sonar mount must contain four finite values")

    def residual(parameters):
        mount = SonarMount(*parameters)
        errors = []
        for sample in samples:
            predicted = mount.plane_range_mm(sample.pose, table_z_m)
            errors.append(
                1000.0 if predicted is None
                else predicted - float(sample.distance_mm))
        return np.asarray(errors)

    bounds = ([-0.12, -0.20, -45.0, -2.5],
              [0.22, 0.20, 85.0, -0.20])
    result = least_squares(
        residual, initial, bounds=bounds, loss="soft_l1", f_scale=4.0,
        max_nfev=5000)
    first_residuals = residual(result.x)
    centre = float(np.median(first_residuals))
    residual_mad = float(np.median(np.abs(first_residuals - centre)))
    rejection_limit = max(8.0, 3.5 * 1.4826 * residual_mad)
    inlier_indices = [
        index for index, value in enumerate(first_residuals)
        if abs(float(value) - centre) <= rejection_limit
    ]
    # A broad-cone sensor can occasionally lock onto a different reflector even
    # when the echo cluster itself is tight.  One robust refit prevents that
    # pose from bending the physical mount, while retaining at least four poses.
    if len(inlier_indices) >= 5 and len(inlier_indices) < len(samples):
        def inlier_residual(parameters):
            return residual(parameters)[inlier_indices]

        result = least_squares(
            inlier_residual, result.x, bounds=bounds, loss="soft_l1",
            f_scale=4.0, max_nfev=5000)
    mount = SonarMount(*[float(value) for value in result.x])
    residuals = residual(result.x)
    inlier_residuals = residuals[inlier_indices]
    rms = float(np.sqrt(np.mean(np.square(inlier_residuals))))
    inlier_max = float(np.max(np.abs(inlier_residuals)))
    lower = np.asarray(bounds[0], dtype=float)
    upper = np.asarray(bounds[1], dtype=float)
    span = upper - lower
    boundary_fraction = np.minimum(
        (result.x - lower) / span, (upper - result.x) / span)
    boundary_clear = bool(np.all(boundary_fraction >= 0.02))
    quality_ok = (
        bool(result.success)
        and len(inlier_indices) >= 5
        and rms <= 5.0
        and inlier_max <= 10.0
        and boundary_clear
    )
    return {
        "mount": mount,
        "rms_mm": rms,
        "max_abs_error_mm": float(np.max(np.abs(residuals))),
        "max_inlier_error_mm": inlier_max,
        "parameter_boundary_clear": boundary_clear,
        "parameter_boundary_fraction": [
            float(value) for value in boundary_fraction
        ],
        "residuals_mm": [float(value) for value in residuals],
        "inlier_indices": inlier_indices,
        "excluded_indices": [
            index for index in range(len(samples))
            if index not in inlier_indices
        ],
        "sensor_angle_span_deg": float(max(sensor_angles)
                                       - min(sensor_angles)),
        "success": bool(result.success),
        "quality_ok": quality_ok,
        "message": str(result.message),
    }


def save_calibration(path, result, samples, table_z_m=0.0):
    if not result.get("quality_ok"):
        raise ValueError(
            "refusing to save low-quality ultrasonic calibration: "
            f"rms={result.get('rms_mm')} mm, "
            f"max_inlier={result.get('max_inlier_error_mm')} mm")
    path = Path(path)
    payload = {
        "version": 1,
        "coordinate_frame": "motor4_pivot_after_pitch_before_roll",
        "table_z_m": float(table_z_m),
        "mount": asdict(result["mount"]),
        "fit": {key: value for key, value in result.items() if key != "mount"},
        "samples": [asdict(sample) for sample in samples],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_calibration(path=DEFAULT_CALIBRATION_PATH):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("unsupported ultrasonic calibration version")
    return SonarMount(**payload["mount"]), payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    profile = subparsers.add_parser("profile", help="read only; never moves arm")
    profile.add_argument("--samples", type=int, default=15)
    profile.add_argument("--interval", type=float, default=0.060)
    repeat = subparsers.add_parser(
        "repeat-profile", help="read only; checks stability across batches")
    repeat.add_argument("--batches", type=int, default=3)
    repeat.add_argument("--samples", type=int, default=12)
    stable = subparsers.add_parser(
        "wait-stable", help="read only; waits out servo/backlash transients")
    stable.add_argument("--timeout", type=float, default=8.0)
    predict = subparsers.add_parser("predict-floor")
    predict.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH))
    args = parser.parse_args()

    from arm_session import ArmSessionClient
    client = ArmSessionClient()
    if args.action == "profile":
        result = acquire_profile(
            client, samples=args.samples, interval_s=args.interval)
        print(json.dumps(asdict(result), indent=2))
        return
    if args.action == "repeat-profile":
        result = repeated_profile(
            client, batches=args.batches, samples_per_batch=args.samples)
        print(json.dumps(asdict(result), indent=2))
        return
    if args.action == "wait-stable":
        result, attempts = wait_for_stable_profile(
            client, timeout_s=args.timeout)
        print(json.dumps({
            "result": asdict(result),
            "attempts": len(attempts),
        }, indent=2))
        return
    pose = client.request({"command": "status"})["pose"]
    mount, payload = load_calibration(args.calibration)
    print(json.dumps({
        "pose": pose,
        "predictedFloorMm": mount.plane_range_mm(
            pose, payload.get("table_z_m", 0.0)),
    }, indent=2))


if __name__ == "__main__":
    main()
