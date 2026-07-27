"""Authoritative real-arm collision checks and trajectory interlock.

This module intentionally uses :mod:`arm_fk`, the same calibrated geometry as
the real grasp controller.  It does not use the obsolete rough dimensions in
``config.L_*``.  The checks cover swept motion, not only target poses, and use
conservative volumes for the printed base, links, gripper, and mounted webcam.

The model is a safety interlock, not proof that arbitrary surroundings are
clear.  Unknown furniture, people, loose cables, and objects still require the
wrist camera/operator.  What it does guarantee is that a command which folds
the gripper into the robot's own housing is rejected before serial transmission.
"""

from dataclasses import dataclass
import math

import numpy as np

import arm_fk
import config


# Conservative envelopes derived from the original 3D asset extents and the
# photographed assembled arm.  The lower enclosure is 210 x 120 x 70 mm, but
# its transform relative to the shoulder axis is absent from the source 3MF.
# A 235 mm radial keep-out therefore bounds every possible planar placement of
# that enclosure plus its forward cover; this is intentionally conservative.
BASE_RADIUS_M = 0.235
BASE_TOP_M = 0.120
MAST_RADIUS_M = 0.075
MAST_BOTTOM_M = 0.085
MAST_TOP_M = 0.225

UPPER_RADIUS_M = 0.035
FORE_RADIUS_M = 0.035
WRIST_RADIUS_M = 0.040
HAND_RADIUS_M = 0.040
CAMERA_RADIUS_M = 0.060
MODEL_MARGIN_M = 0.012
MIN_TOOL_Z_M = -0.003
TRAJECTORY_STEP_DEG = 0.5


@dataclass(frozen=True)
class SafetyViolation:
    kind: str
    clearance_mm: float
    detail: str


@dataclass(frozen=True)
class SafetyReport:
    safe: bool
    minimum_clearance_mm: float
    violations: tuple[SafetyViolation, ...]

    def explain(self):
        if self.safe:
            return f"safe; minimum model clearance={self.minimum_clearance_mm:.1f} mm"
        return "; ".join(
            f"{item.kind} ({item.clearance_mm:.1f} mm): {item.detail}"
            for item in self.violations)


def _point_segment_distance(point, start, end):
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    scale = float(np.dot(delta, delta))
    if scale <= 1e-15:
        return float(np.linalg.norm(point - start))
    amount = float(np.dot(point - start, delta) / scale)
    amount = min(1.0, max(0.0, amount))
    return float(np.linalg.norm(point - (start + amount * delta)))


def _segment_distance(a, b, c, d):
    """Exact closest distance between two 3-D line segments."""
    a, b, c, d = (np.asarray(value, dtype=float) for value in (a, b, c, d))
    u, v, w = b - a, d - c, a - c
    aa, bb, cc = float(np.dot(u, u)), float(np.dot(u, v)), float(np.dot(v, v))
    dd, ee = float(np.dot(u, w)), float(np.dot(v, w))
    denominator = aa * cc - bb * bb
    small = 1e-15
    s_num, s_den = denominator, denominator
    t_num, t_den = denominator, denominator
    if denominator < small:
        s_num, s_den = 0.0, 1.0
        t_num, t_den = ee, cc
    else:
        s_num = bb * ee - cc * dd
        t_num = aa * ee - bb * dd
        if s_num < 0.0:
            s_num, t_num, t_den = 0.0, ee, cc
        elif s_num > s_den:
            s_num, t_num, t_den = s_den, ee + bb, cc
    if t_num < 0.0:
        t_num = 0.0
        if -dd < 0.0:
            s_num = 0.0
        elif -dd > aa:
            s_num = s_den
        else:
            s_num, s_den = -dd, aa
    elif t_num > t_den:
        t_num = t_den
        if -dd + bb < 0.0:
            s_num = 0.0
        elif -dd + bb > aa:
            s_num = s_den
        else:
            s_num, s_den = -dd + bb, aa
    sc = 0.0 if abs(s_num) < small else s_num / s_den
    tc = 0.0 if abs(t_num) < small else t_num / t_den
    return float(np.linalg.norm(w + sc * u - tc * v))


def _point_capped_cylinder_distance(point, radius, bottom, top):
    point = np.asarray(point, dtype=float)
    radial_gap = max(0.0, float(np.linalg.norm(point[:2])) - radius)
    vertical_gap = max(bottom - float(point[2]), float(point[2]) - top, 0.0)
    return math.hypot(radial_gap, vertical_gap)


def _segment_cylinder_distance(start, end, radius, bottom, top):
    # Sampling at <=2 mm has <=1 mm geometric miss.  MODEL_MARGIN_M is 12 mm,
    # so this numerical approximation remains safely on the conservative side.
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    count = max(2, int(math.ceil(np.linalg.norm(end - start) / 0.002)) + 1)
    return min(_point_capped_cylinder_distance(point, radius, bottom, top)
               for point in np.linspace(start, end, count))


class PhysicalArmSafety:
    """Collision report for a pose and every firmware slew state."""

    def __init__(self, margin_m=MODEL_MARGIN_M, slew_step_deg=TRAJECTORY_STEP_DEG,
                 table_z_m=0.0):
        self.margin_m = float(margin_m)
        self.slew_step_deg = float(slew_step_deg)
        self.table_z_m = float(table_z_m)
        if self.margin_m < 0 or self.slew_step_deg <= 0:
            raise ValueError("safety margin must be nonnegative and step positive")
        if not math.isfinite(self.table_z_m):
            raise ValueError("table z must be finite")

    @staticmethod
    def _validate_pose(pose):
        values = np.asarray(tuple(pose), dtype=float)
        if values.shape != (config.N_JOINTS,) or not np.isfinite(values).all():
            raise ValueError("pose must contain six finite servo values")
        for joint, value in enumerate(values):
            if not config.SERVO_MIN[joint] <= value <= config.SERVO_MAX[joint]:
                raise ValueError(
                    f"joint {joint + 1}={value:g} outside configured limits")
        return values

    def pose_report(self, pose):
        pose = self._validate_pose(pose)
        g = arm_fk.geometry(pose)
        violations = []
        clearances = []

        def check(kind, clearance_m, detail):
            clearances.append(clearance_m)
            if clearance_m < 0.0:
                violations.append(SafetyViolation(
                    kind, clearance_m * 1000.0, detail))

        distal = (
            ("wrist", g.wrist_pitch, g.wrist_roll, WRIST_RADIUS_M),
            ("hand", g.wrist_roll, g.tool, HAND_RADIUS_M),
        )
        for name, start, end, radius in distal:
            distance = _segment_cylinder_distance(
                start, end, BASE_RADIUS_M, 0.0, BASE_TOP_M)
            check("base-housing", distance - radius - self.margin_m,
                  f"{name} enters conservative lower-body envelope")

        # The webcam is larger than the wrist itself and must be checked as a
        # separate sphere against both housing and shoulder mast.
        camera_base = _point_capped_cylinder_distance(
            g.camera, BASE_RADIUS_M, 0.0, BASE_TOP_M)
        check("camera-base", camera_base - CAMERA_RADIUS_M - self.margin_m,
              "mounted webcam enters lower-body envelope")

        for name, start, end, radius in distal:
            distance = _segment_cylinder_distance(
                start, end, MAST_RADIUS_M, MAST_BOTTOM_M, MAST_TOP_M)
            check("shoulder-mast", distance - radius - self.margin_m,
                  f"{name} enters shoulder/servo envelope")
        camera_mast = _point_capped_cylinder_distance(
            g.camera, MAST_RADIUS_M, MAST_BOTTOM_M, MAST_TOP_M)
        check("camera-mast", camera_mast - CAMERA_RADIUS_M - self.margin_m,
              "mounted webcam enters shoulder/servo envelope")

        # All rigid bodies except the fingertip contact centre stay above the
        # shared table.  The tool is permitted to reach the known floor plane,
        # but never to command through it.
        for name, points, radius in (
            ("upper", (g.shoulder, g.elbow), UPPER_RADIUS_M),
            ("forearm", (g.elbow, g.wrist_pitch), FORE_RADIUS_M),
            ("wrist", (g.wrist_pitch, g.wrist_roll), WRIST_RADIUS_M),
        ):
            clearance = (min(float(point[2]) for point in points)
                         - self.table_z_m - radius - self.margin_m)
            check("table", clearance, f"{name} envelope reaches below table")
        minimum_tool_z_m = self.table_z_m + MIN_TOOL_Z_M
        check("tool-through-table", float(g.tool[2]) - minimum_tool_z_m,
              "fingertip centre is commanded below calibrated floor tolerance")
        check("camera-table",
              float(g.camera[2]) - self.table_z_m
              - CAMERA_RADIUS_M - self.margin_m,
              "mounted webcam reaches the table")

        # Non-adjacent self-collision.  Adjacent links share their joint by
        # design and are therefore intentionally excluded.
        upper_hand = _segment_distance(
            g.shoulder, g.elbow, g.wrist_roll, g.tool)
        check("self-upper-hand",
              upper_hand - UPPER_RADIUS_M - HAND_RADIUS_M - self.margin_m,
              "gripper/hand folds into upper arm")
        upper_camera = _point_segment_distance(g.camera, g.shoulder, g.elbow)
        check("self-upper-camera",
              upper_camera - UPPER_RADIUS_M - CAMERA_RADIUS_M - self.margin_m,
              "mounted webcam folds into upper arm")
        # Camera and forearm meet at the wrist-pitch assembly; their deliberately
        # inflated envelopes overlap in every valid pose, just like adjacent
        # link capsules at a hinge.  Only the non-adjacent upper-arm check is a
        # meaningful camera self-collision constraint.

        minimum = min(clearances) * 1000.0 if clearances else math.inf
        return SafetyReport(not violations, minimum, tuple(violations))

    def pose_is_safe(self, pose):
        try:
            return self.pose_report(pose).safe
        except (TypeError, ValueError):
            return False

    def slew_states(self, start, target):
        start, target = self._validate_pose(start), self._validate_pose(target)
        moving = range(config.N_JOINTS)
        ticks = int(math.ceil(max(abs(target[j] - start[j]) for j in moving)
                              / self.slew_step_deg))
        for tick in range(1, ticks + 1):
            elapsed = tick * self.slew_step_deg
            pose = start.copy()
            for joint in moving:
                difference = target[joint] - start[joint]
                if difference:
                    pose[joint] = start[joint] + math.copysign(
                        min(abs(difference), elapsed), difference)
            yield pose.tolist()

    def transition_report(self, start, target):
        start_report = self.pose_report(start)
        if not start_report.safe:
            return start_report
        minimum = start_report.minimum_clearance_mm
        for index, pose in enumerate(self.slew_states(start, target), start=1):
            report = self.pose_report(pose)
            minimum = min(minimum, report.minimum_clearance_mm)
            if not report.safe:
                details = tuple(SafetyViolation(
                    item.kind, item.clearance_mm,
                    f"trajectory step {index}: {item.detail}")
                    for item in report.violations)
                return SafetyReport(False, minimum, details)
        return SafetyReport(True, minimum, ())

    def transition_is_safe(self, start, target):
        try:
            return self.transition_report(start, target).safe
        except (TypeError, ValueError):
            return False

    def assert_transition_safe(self, start, target):
        report = self.transition_report(start, target)
        if not report.safe:
            raise RuntimeError(f"physical collision interlock: {report.explain()}")
        return report
