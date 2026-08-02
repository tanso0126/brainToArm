"""MuJoCo model for the current rigid-wrist, reduced-DOF physical arm.

This is intentionally separate from :mod:`simul.mujoco_robot`.  The old model
keeps motor 4 as an actuated wrist for possible hardware repair.  Here motor 4
and motor 6 are fixed bodies and motor 1 is absent; MuJoCo can actuate only
motor 2, motor 3, and the two coupled slides representing motor 5.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable, Mapping
import math
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "laptop") not in sys.path:
    sys.path.insert(0, str(ROOT / "laptop"))

from laptop import arm_fk, config  # noqa: E402
from laptop.reduced_dof import (  # noqa: E402
    ELBOW_RANGE, FIXED_ROLL_GEOMETRY_DEG, SHOULDER_RANGE,
    geometry_pose, reduced_home, validate_command_pose,
)

try:
    from .mujoco_robot import servo_to_joint_targets
    from .prepare_assets import extract_sources
except ImportError:
    from mujoco_robot import servo_to_joint_targets
    from prepare_assets import extract_sources


HERE = Path(__file__).resolve().parent
ACTIVE_ACTUATORS = (
    "shoulder_position", "elbow_position",
    "grip_left_position", "grip_right_position",
)


def _f(value: float) -> str:
    return f"{float(value):.9g}"


def _range(values) -> str:
    return f"{_f(min(values))} {_f(max(values))}"


def _joint_limits(joint: str) -> tuple[float, float]:
    values = []
    endpoints = SHOULDER_RANGE if joint == "shoulder" else ELBOW_RANGE
    for value in endpoints:
        pose = reduced_home()
        pose[config.J_SHOULDER if joint == "shoulder" else config.J_ELBOW] = value
        values.append(servo_to_joint_targets(geometry_pose(pose))[joint])
    return min(values), max(values)


def build_reduced_mjcf(parameters: Mapping[str, float] | None = None) -> str:
    """Build the independent reduced model with bounded geometry variation.

    ``upper_dx`` and ``upper_dz`` make the joint-centre offset explicit.  The
    current calibrated default is straight, while training randomizes this
    fixed transform so a policy cannot depend on that unverified simplification.
    """

    p = dict(parameters or {})
    upper_dx = float(p.get("upper_dx", arm_fk.UPPER_M))
    upper_dz = float(p.get("upper_dz", 0.0))
    fore = float(p.get("forearm", arm_fk.FORE_M))
    camera_x = float(p.get("camera_x", 0.015))
    camera_z = float(p.get("camera_z", 0.075))
    camera_pitch = float(p.get("camera_pitch", -42.0))
    camera_roll = float(p.get("camera_roll", -90.0))
    if not 0.225 <= upper_dx <= 0.258 or not -0.018 <= upper_dz <= 0.018:
        raise ValueError("upper joint-centre transform is outside its calibration envelope")
    if not 0.160 <= fore <= 0.185:
        raise ValueError("forearm is outside its calibration envelope")

    meshes = extract_sources()
    def mesh_path(name):
        return escape(str(meshes / name), quote=True)
    shoulder = _joint_limits("shoulder")
    elbow = _joint_limits("elbow")
    shoulder_deg = tuple(math.degrees(v) for v in shoulder)
    elbow_deg = tuple(math.degrees(v) for v in elbow)
    fixed_roll = FIXED_ROLL_GEOMETRY_DEG - 170.0

    return f"""<mujoco model="brain_to_arm_reduced">
  <compiler angle="degree" autolimits="true" inertiafromgeom="true"/>
  <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
  <visual><global offwidth="640" offheight="360"/><map znear="0.003" zfar="3"/></visual>
  <default>
    <joint damping="1.5" armature="0.015"/>
    <geom friction="0.9 0.01 0.001" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="2" mass="0" rgba="0.92 0.92 0.90 1"/></default>
    <default class="collision"><geom group="3" contype="2" conaffinity="1" rgba="0.25 0.27 0.31 1"/></default>
  </default>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.9 0.9 0.88" rgb2="0.62 0.64 0.66" width="512" height="512"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="4 4"/>
    <mesh name="base_case" file="{mesh_path('Alt_Kasa.stl')}" scale="0.001 0.001 0.001"/>
    <mesh name="base_cover" file="{mesh_path('Alt_Kapak.stl')}" scale="0.001 0.001 0.001"/>
    <mesh name="base_rotor" file="{mesh_path('Alt_Govde.stl')}" scale="0.001 0.001 0.001"/>
    <mesh name="upper_visual" file="{mesh_path('Alt_Kol.stl')}" scale="0.001 0.001 0.001"/>
    <mesh name="fore_visual" file="{mesh_path('On_Kol.stl')}" scale="0.001 0.001 0.001"/>
    <mesh name="wrist_visual" file="{mesh_path('Bilek.stl')}" scale="0.001 0.001 0.001"/>
    <mesh name="palm_visual" file="{mesh_path('El_Ust.stl')}" scale="0.001 0.001 0.001"/>
    <mesh name="finger_visual" file="{mesh_path('Parmak_2 X 2.stl')}" scale="0.001 0.001 0.001"/>
  </asset>
  <worldbody>
    <light pos="0 -0.4 1.1" dir="0 0.2 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="1.2 1.2 0.02" material="floor_mat"/>
    <camera name="overview" pos="0.72 -0.78 0.56" xyaxes="0.74 0.68 0 -0.27 0.29 0.92" fovy="48"/>
    <body name="base">
      <!-- Same +X housing envelope and 12 mm margin as the real interlock. -->
      <geom name="base_collision" class="collision" type="box" pos="0.1025 0 0.0835" size="0.1945 0.087 0.0835" contype="4" conaffinity="2"/>
      <geom class="visual" mesh="base_case" pos="0.1 0 0" euler="0 0 180"/>
      <geom class="visual" mesh="base_cover" pos="0.1 0 0.07"/>
      <geom class="visual" mesh="base_rotor" pos="0 0 0.08"/>
      <body name="shoulder_link" pos="0 0 {_f(arm_fk.SHOULDER_HEIGHT_M)}">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="{_range(shoulder_deg)}"/>
        <geom name="upper_collision" class="collision" type="capsule" fromto="0 0 0 {_f(upper_dx)} 0 {_f(upper_dz)}" size="0.035"/>
        <geom class="visual" mesh="upper_visual" pos="0.1275 0 0.02" euler="0 90 0"/>
        <body name="elbow_link" pos="{_f(upper_dx)} 0 {_f(upper_dz)}">
          <joint name="elbow" type="hinge" axis="0 1 0" range="{_range(elbow_deg)}"/>
          <geom name="fore_collision" class="collision" type="capsule" fromto="0 0 0 {_f(fore)} 0 0" size="0.035"/>
          <geom class="visual" mesh="fore_visual" pos="0.1525 0 0"/>
          <!-- Motor 4 has no joint or actuator: this complete subtree is rigid. -->
          <body name="fixed_wrist" pos="{_f(fore)} 0 0">
            <geom name="wrist_collision" class="collision" type="capsule" fromto="0 0 0 {_f(arm_fk.WRISTROLL_M)} 0 0" size="0.040"/>
            <geom class="visual" mesh="wrist_visual" pos="0.022 0 0" euler="0 90 0"/>
            <camera name="wrist" pos="{_f(arm_fk.WRISTROLL_M + camera_x)} 0 {_f(camera_z)}" euler="0 {_f(camera_pitch)} {_f(camera_roll)}" fovy="73"/>
            <site name="camera_mount" pos="{_f(arm_fk.WRISTROLL_M + camera_x)} 0 {_f(camera_z)}" size="0.0001" rgba="0 0 0 0"/>
            <!-- Motor 6 is also a fixed transform, not a simulated joint. -->
            <body name="fixed_roll" pos="{_f(arm_fk.WRISTROLL_M)} 0 0" euler="{_f(fixed_roll)} 0 0">
              <geom name="palm_collision" class="collision" type="box" pos="0.022 0 0" size="0.027 0.028 0.012"/>
              <geom class="visual" mesh="palm_visual" pos="0.022 0 -0.0075"/>
              <body name="left_finger"><joint name="grip_left" type="slide" axis="0 1 0" range="0.007 0.045" damping="3"/>
                <geom name="left_finger_collision" class="collision" type="box" pos="0.087 0 -0.004" size="0.057 0.007 0.011" friction="2.8 0.08 0.02"/>
                <geom class="visual" mesh="finger_visual" pos="0.034 -0.008 -0.006"/><geom type="box" pos="0.074 0 0.007" size="0.016 0.007 0.004" contype="0" conaffinity="0" rgba="0.02 0.20 1 1"/>
              </body>
              <body name="right_finger"><joint name="grip_right" type="slide" axis="0 -1 0" range="0.007 0.045" damping="3"/>
                <geom name="right_finger_collision" class="collision" type="box" pos="0.087 0 -0.004" size="0.057 0.007 0.011" friction="2.8 0.08 0.02"/>
                <geom class="visual" mesh="finger_visual" pos="0.034 -0.008 -0.006" euler="180 0 0"/><geom type="box" pos="0.074 0 0.007" size="0.016 0.007 0.004" contype="0" conaffinity="0" rgba="1 0.03 0.02 1"/>
              </body>
              <site name="tool_center" pos="{_f(arm_fk.HAND_M)} 0 {_f(arm_fk.TOOL_DZ_M)}" size="0.0001" rgba="0 0 0 0"/>
              <site name="grasp_center" pos="{_f(arm_fk.HAND_M + 0.040)} 0 {_f(arm_fk.TOOL_DZ_M)}" size="0.0001" rgba="0 0 0 0"/>
              <site name="finger_tip" pos="{_f(arm_fk.HAND_M + arm_fk.FINGER_EXTENSION_M)} 0 {_f(arm_fk.TOOL_DZ_M)}" size="0.0001" rgba="0 0 0 0"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="target" pos="0.40 0 0.018"><freejoint name="target_free"/><geom name="target_geom" type="box" size="0.018 0.018 0.016" mass="0.035" rgba="1 0.82 0.02 1" friction="1.1 0.01 0.001"/></body>
  </worldbody>
  <actuator>
    <position name="shoulder_position" joint="shoulder" kp="220" kv="18" ctrllimited="true" ctrlrange="{_range(shoulder)}"/>
    <position name="elbow_position" joint="elbow" kp="180" kv="14" ctrllimited="true" ctrlrange="{_range(elbow)}"/>
    <position name="grip_left_position" joint="grip_left" kp="520" kv="20" ctrllimited="true" ctrlrange="0.007 0.045"/>
    <position name="grip_right_position" joint="grip_right" kp="520" kv="20" ctrllimited="true" ctrlrange="0.007 0.045"/>
  </actuator>
</mujoco>"""


def reduced_targets(pose: Iterable[float]) -> dict[str, float]:
    servo = validate_command_pose(pose)
    targets = servo_to_joint_targets(geometry_pose(servo))
    return {name: targets[name] for name in ("shoulder", "elbow", "grip_left", "grip_right")}


def _qpos_address(model, name):
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
        raise KeyError(name)
    return int(model.jnt_qposadr[joint])


def set_reduced_pose(model, data, pose, *, reset_velocity=True):
    servo = validate_command_pose(pose)
    targets = reduced_targets(servo)
    for name, value in targets.items():
        data.qpos[_qpos_address(model, name)] = value
    if reset_velocity:
        data.qvel[:] = 0
    data.ctrl[:] = tuple(targets[name] for name in ("shoulder", "elbow", "grip_left", "grip_right"))
    mujoco.mj_forward(model, data)
    return np.asarray(servo, dtype=float)


def command_reduced_pose(model, data, pose):
    servo = validate_command_pose(pose)
    targets = reduced_targets(servo)
    data.ctrl[:] = tuple(targets[name] for name in ("shoulder", "elbow", "grip_left", "grip_right"))
    return np.asarray(servo, dtype=float)


def load_reduced_model(*, parameters=None):
    model = mujoco.MjModel.from_xml_string(build_reduced_mjcf(parameters))
    data = mujoco.MjData(model)
    set_reduced_pose(model, data, reduced_home())
    return model, data


def site_position(model, data, name):
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site < 0:
        raise KeyError(name)
    return data.site_xpos[site].copy()


def place_target(model, data, xyz):
    position = np.asarray(tuple(xyz), dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("target position must contain three finite values")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
    qadr = int(model.jnt_qposadr[joint])
    dadr = int(model.jnt_dofadr[joint])
    data.qpos[qadr:qadr + 3] = position
    data.qpos[qadr + 3:qadr + 7] = (1, 0, 0, 0)
    data.qvel[dadr:dadr + 6] = 0
    mujoco.mj_forward(model, data)
    return position.copy()


def model_summary(model):
    def names(kind, count):
        return [mujoco.mj_id2name(model, kind, i) for i in range(count)]
    return {
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        "joints": names(mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        "actuators": names(mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
    }


__all__ = [
    "ACTIVE_ACTUATORS", "build_reduced_mjcf", "command_reduced_pose",
    "load_reduced_model", "model_summary", "place_target", "reduced_targets",
    "set_reduced_pose", "site_position",
]
