"""Analytic forward kinematics for the planar brainToArm (base fixed at 90).

Returns SE(3) poses of the wrist and tool in the robot base frame from the six
servo commands. The link lengths and the (nonlinear, geared) servo->joint maps
are copied from the calibrated MuJoCo model in ``simul/mujoco_robot.py`` and were
verified to reproduce its tool_center to <1 mm. This module is self-contained so
the laptop runtime never imports the simulation package.

Frames (base frame: x forward, y left, z up; the arm moves in the x-z plane):

    base -> shoulder(+z=0.210) -Ry(sh)-> upper(0.242) -Ry(el)-> fore(0.1725)
         -Ry(wp)-> wrist(0.045) -Rx(roll)-> tool(0.090, dz=-0.008)

All rotations for shoulder/elbow/wrist_pitch are about the base y axis; wrist
roll is about the local x axis (the reach direction).
"""

from dataclasses import dataclass
from typing import Iterable, Tuple
import math

import numpy as np


# Geometry (metres) and geared servo->joint slopes, from the calibrated sim model.
UPPER_M = 0.241767
FORE_M = 0.1725
HAND_M = 0.090
TOOL_DZ_M = -0.008
WRISTROLL_M = 0.045
SHOULDER_HEIGHT_M = 0.210

_SH_FLOOR_SCALE = 0.25
_SH_FLOOR_ANCHOR_SERVO = 113.0
_SH_FLOOR_ANCHOR_JOINT = -20.5529
_SH_HOME_JOINT = -60.0
_SH_HOME_SERVO = 70.0
_EL_SCALE = 0.30
_EL_REF_JOINT = 68.4919
_WP_SCALE = -1.5


def shoulder_joint_deg(servo: float) -> float:
    servo = float(servo)
    if servo >= _SH_FLOOR_ANCHOR_SERVO:
        return _SH_FLOOR_ANCHOR_JOINT + _SH_FLOOR_SCALE * (servo - _SH_FLOOR_ANCHOR_SERVO)
    slope = (_SH_FLOOR_ANCHOR_JOINT - _SH_HOME_JOINT) / (_SH_FLOOR_ANCHOR_SERVO - _SH_HOME_SERVO)
    return _SH_HOME_JOINT + slope * (servo - _SH_HOME_SERVO)


def elbow_joint_deg(servo: float) -> float:
    return _EL_REF_JOINT + _EL_SCALE * (float(servo) - 90.0)


def wrist_pitch_joint_deg(servo: float) -> float:
    return _WP_SCALE * (float(servo) - 180.0)


def _roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


@dataclass(frozen=True)
class ArmGeometry:
    """Calibrated joint centres and frame rotations in base coordinates."""

    shoulder: np.ndarray
    elbow: np.ndarray
    wrist_pitch: np.ndarray
    wrist_roll: np.ndarray
    tool: np.ndarray
    camera: np.ndarray
    wrist_rotation: np.ndarray
    forearm_rotation: np.ndarray


def geometry(servo: Iterable[float]) -> ArmGeometry:
    """Return the authoritative real-arm geometry for collision checking.

    Servo 1 was historically locked, which let older planar code ignore it.
    It is operational again, so the complete chain is rotated about base Z.
    The base=90 calibration remains numerically unchanged.
    """
    servo = [float(v) for v in servo]
    if len(servo) != 6 or not np.isfinite(servo).all():
        raise ValueError("servo pose must contain six finite values")
    base_yaw = math.radians(servo[0] - 90.0)
    R_base = _rotz(base_yaw)
    a1 = math.radians(shoulder_joint_deg(servo[1]))
    a2 = a1 + math.radians(elbow_joint_deg(servo[2]))
    a3 = a2 + math.radians(wrist_pitch_joint_deg(servo[3]))
    roll = math.radians(servo[5] - 170.0)
    # cumulative planar orientation is a3 (about y); wrist roll adds about x.
    shoulder = np.array([0.0, 0.0, SHOULDER_HEIGHT_M])
    elbow = shoulder + R_base @ _roty(a1) @ np.array([UPPER_M, 0, 0])
    R_forearm = R_base @ _roty(a2)
    wristp = elbow + R_forearm @ np.array([FORE_M, 0, 0])
    wristroll = wristp + R_base @ _roty(a3) @ np.array([WRISTROLL_M, 0, 0])
    R_wrist = R_base @ _roty(a3) @ _rotx(roll)
    tool = wristroll + R_wrist @ np.array([HAND_M, 0, TOOL_DZ_M])
    # Measured/CAD seed for the webcam centre.  The collision envelope around
    # this point is deliberately much larger than the uncertainty in the mount.
    camera = wristroll + R_wrist @ np.array([0.015, 0.0, 0.075])
    return ArmGeometry(
        shoulder, elbow, wristp, wristroll, tool, camera, R_wrist, R_forearm)


def _chain(servo: Iterable[float]):
    chain = geometry(servo)
    # Preserve the historical private return contract for existing callers.
    direction = chain.wrist_rotation[:, 0]
    planar_angle = math.atan2(-direction[2], math.hypot(direction[0], direction[1]))
    return chain.wrist_roll, chain.wrist_rotation, chain.tool, planar_angle


def wrist_pose(servo: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R 3x3, t 3) of the wrist-roll frame in the base frame."""
    wristroll, R_wrist, _tool, _a3 = _chain(servo)
    return R_wrist, wristroll


def tool_position(servo: Iterable[float]) -> np.ndarray:
    """Tool-center position in the base frame (metres)."""
    return _chain(servo)[2]


def wrist_pitch_position(servo: Iterable[float]) -> np.ndarray:
    """Position of the motor-4 pivot, used as the wrist-camera proxy.

    The camera is fixed beside this pivot in the real build.  Unlike the tool
    centre, this point does not translate when motor 4 merely changes viewing
    pitch, which is essential for separating LOOK from REACH.
    """
    return geometry(servo).wrist_pitch.copy()


def forearm_pose(servo: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return the frame carrying the wrist camera and ultrasonic sensor.

    On the physical build both sensors are attached above the motor-4 pivot.
    They therefore follow shoulder+elbow (the forearm), but do not follow
    motor-4 gripper pitch or motor-6 roll.  Treating them as part of the tool
    frame creates a false depth change whenever only the gripper is rotated.
    """
    chain = geometry(servo)
    return chain.forearm_rotation.copy(), chain.wrist_pitch.copy()


def wrist_matrix(servo: Iterable[float]) -> np.ndarray:
    """4x4 homogeneous wrist-roll pose in the base frame."""
    R, t = wrist_pose(servo)
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


if __name__ == "__main__":
    for pose, label in ([90, 124, 90, 180, 90, 170], "hover"), \
                       ([90, 142, 90, 180, 90, 170], "grasp"):
        print(label, "tool", np.round(tool_position(pose), 4))
