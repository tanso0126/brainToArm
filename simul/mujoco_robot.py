"""MuJoCo model of the physical six-command brainToArm robot.

The printable STL files define appearance and measured envelopes, but do not
contain assembly transforms.  This module therefore keeps two deliberately
separate layers:

* original STL meshes are non-colliding visual geometry;
* explicit joint frames and convex primitives define reproducible kinematics
  and collision geometry.

All public poses use the same six servo degrees as the real Uno protocol:
base, shoulder, elbow, wrist pitch, gripper, wrist roll.  Motor 1 is a fixed
body in the current model and is always reported as 90 degrees.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping
import math

import mujoco
import numpy as np

try:
    from .prepare_assets import extract_sources, load_manifest
except ImportError:  # Direct execution/import from the simul directory.
    from prepare_assets import extract_sources, load_manifest


HERE = Path(__file__).resolve().parent

SERVO_NAMES = (
    "base",
    "shoulder",
    "elbow",
    "wrist_pitch",
    "gripper",
    "wrist_roll",
)

# The linkage does not rotate one physical joint degree per commanded servo
# degree.  These first-order slopes reproduce the measured floor-plane
# shoulder/elbow compensation (-6/11) and the two known floor reference poses.
_SHOULDER_FLOOR_SCALE = 0.250
# The calibrated floor curve spans shoulder commands 113..131 when elbow moves
# through 110..78, so the near-floor slope must cover that full interval.
_SHOULDER_FLOOR_ANCHOR_SERVO = 113.0
_SHOULDER_FLOOR_ANCHOR_JOINT = 57.25
_SHOULDER_HOME_JOINT = -30.0
_ELBOW_SCALE = 0.200
_ELBOW_REFERENCE_JOINT = -70.0
_WRIST_PITCH_SCALE = -1.500


@lru_cache(maxsize=1)
def _prepared_meshes() -> Path:
    """Validate immutable source geometry once per Python process."""

    return extract_sources()


@dataclass(frozen=True)
class RobotSpec:
    servo_min_deg: np.ndarray
    servo_max_deg: np.ndarray
    home_servo_deg: np.ndarray
    hover_servo_deg: np.ndarray
    grasp_servo_deg: np.ndarray
    base_locked_deg: float
    upper_arm_m: float = 0.241767
    forearm_m: float = 0.1725
    hand_m: float = 0.090
    # Fitted so the measured 142-degree grasp reference puts the tool center
    # just above the shared z=0 floor while 124 degrees remains a safe hover.
    shoulder_height_m: float = 0.205

    @classmethod
    def from_manifest(cls) -> "RobotSpec":
        control = load_manifest()["physical_control"]
        floor = control["floor_reference"]
        return cls(
            servo_min_deg=np.asarray(control["servo_min_deg"], dtype=np.float64),
            servo_max_deg=np.asarray(control["servo_max_deg"], dtype=np.float64),
            home_servo_deg=np.asarray(control["home_servo_deg"], dtype=np.float64),
            hover_servo_deg=np.asarray(floor["hover_pose_deg"], dtype=np.float64),
            grasp_servo_deg=np.asarray(floor["grasp_pose_deg"], dtype=np.float64),
            base_locked_deg=float(control["locked_base_deg"]),
        )

    def validate_servo_pose(self, pose: Iterable[float]) -> np.ndarray:
        values = np.asarray(tuple(pose), dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("servo pose must contain six finite degrees")
        if not math.isclose(values[0], self.base_locked_deg, abs_tol=1e-6):
            raise ValueError(
                f"base is locked at {self.base_locked_deg:g} degrees in simulation")
        if np.any(values < self.servo_min_deg) or np.any(values > self.servo_max_deg):
            raise ValueError(
                f"servo pose outside limits {self.servo_min_deg.tolist()}.."
                f"{self.servo_max_deg.tolist()}: {values.tolist()}")
        return values


def servo_to_joint_targets(pose: Iterable[float], spec: RobotSpec | None = None) -> dict[str, float]:
    """Convert the physical command convention to MuJoCo joint coordinates."""

    spec = spec or RobotSpec.from_manifest()
    servo = spec.validate_servo_pose(pose)
    # Gripper slide is the distance of each jaw from the centerline.  The
    # physical 90 degree command is open and 180 is closed.
    jaw = np.interp(servo[4], [90.0, 180.0], [0.042, 0.008])
    if servo[1] >= _SHOULDER_FLOOR_ANCHOR_SERVO:
        shoulder_deg = (
            _SHOULDER_FLOOR_ANCHOR_JOINT
            + _SHOULDER_FLOOR_SCALE
            * (servo[1] - _SHOULDER_FLOOR_ANCHOR_SERVO)
        )
    else:
        # The printed linkage is nonlinear.  The measured near-floor slope is
        # used only near the floor; below that region a monotonic segment joins
        # it to the observed raised HOME pose instead of extrapolating through
        # the floor.
        home_servo = 70.0
        slope = (
            (_SHOULDER_FLOOR_ANCHOR_JOINT - _SHOULDER_HOME_JOINT)
            / (_SHOULDER_FLOOR_ANCHOR_SERVO - home_servo)
        )
        shoulder_deg = _SHOULDER_HOME_JOINT + slope * (servo[1] - home_servo)
    return {
        "shoulder": math.radians(shoulder_deg),
        "elbow": math.radians(
            _ELBOW_REFERENCE_JOINT + _ELBOW_SCALE * (servo[2] - 90.0)),
        "wrist_pitch": math.radians(_WRIST_PITCH_SCALE * (servo[3] - 180.0)),
        "wrist_roll": math.radians(servo[5] - 170.0),
        "grip_left": float(jaw),
        "grip_right": float(jaw),
    }


def _joint_range(name: str, spec: RobotSpec) -> tuple[float, float]:
    samples = []
    for index, endpoint in enumerate(zip(spec.servo_min_deg, spec.servo_max_deg)):
        if index == 0:
            continue
        for value in endpoint:
            pose = spec.home_servo_deg.copy()
            pose[0] = spec.base_locked_deg
            pose[index] = value
            samples.append((index, servo_to_joint_targets(pose, spec)))
    logical = {"shoulder": 1, "elbow": 2, "wrist_pitch": 3, "wrist_roll": 5}
    values = [targets[name] for index, targets in samples if index == logical[name]]
    return min(values), max(values)


def _f(value: float) -> str:
    return f"{value:.9g}"


def _range_text(values: tuple[float, float]) -> str:
    return f"{_f(values[0])} {_f(values[1])}"


def _rgb_text(value: Iterable[float], name: str) -> str:
    rgb = tuple(float(channel) for channel in value)
    if len(rgb) != 3 or any(not 0.0 <= channel <= 1.0 for channel in rgb):
        raise ValueError(f"{name} must contain three values in [0, 1]")
    return " ".join(_f(channel) for channel in rgb)


def build_mjcf(
    spec: RobotSpec | None = None,
    *,
    parameters: Mapping[str, float] | None = None,
) -> str:
    """Return a complete MJCF model without accessing cameras or serial ports.

    ``parameters`` supports bounded geometric variation for later domain
    randomization.  It intentionally changes only simulator geometry.
    """

    spec = spec or RobotSpec.from_manifest()
    parameters = dict(parameters or {})
    link_scale = float(parameters.get("link_scale", 1.0))
    camera_fovy = float(parameters.get("camera_fovy", 73.0))
    camera_x = float(parameters.get("camera_x", 0.015))
    camera_z = float(parameters.get("camera_z", 0.075))
    # Camera is mounted above the palm and aimed through the jaw center rather
    # than straight down from its own mounting screw.
    camera_pitch = float(parameters.get("camera_pitch", -42.0))
    # Physical tape convention in the upright wrist frame: blue appears on the
    # left jaw and red on the right jaw.
    camera_roll = float(parameters.get("camera_roll", -90.0))
    floor_rgb1 = _rgb_text(parameters.get("floor_rgb1", (0.88, 0.88, 0.86)),
                           "floor_rgb1")
    floor_rgb2 = _rgb_text(parameters.get("floor_rgb2", (0.62, 0.64, 0.66)),
                           "floor_rgb2")
    target_rgb = _rgb_text(parameters.get("target_rgb", (1.0, 0.82, 0.02)),
                           "target_rgb")
    target_shape = str(parameters.get("target_shape", "box"))
    if target_shape not in {"box", "cylinder", "sphere"}:
        raise ValueError("target_shape must be box, cylinder, or sphere")
    target_size = float(parameters.get("target_size", 0.020))
    if not 0.012 <= target_size <= 0.030:
        raise ValueError("target_size must be in [0.012, 0.030] meters")
    if target_shape == "box":
        target_size_text = f"{_f(target_size)} {_f(target_size)} {_f(target_size * 0.9)}"
    elif target_shape == "cylinder":
        target_size_text = f"{_f(target_size)} {_f(target_size * 0.9)}"
    else:
        target_size_text = _f(target_size)
    if not 0.94 <= link_scale <= 1.06:
        raise ValueError("link_scale must remain inside the calibrated ±6% envelope")
    if not 55.0 <= camera_fovy <= 90.0:
        raise ValueError("camera_fovy must be in [55, 90] degrees")

    meshes = _prepared_meshes()
    mesh_path = lambda name: escape(str(meshes / name), quote=True)
    upper = spec.upper_arm_m * link_scale
    fore = spec.forearm_m * link_scale
    hand = spec.hand_m * link_scale
    shoulder_range = _range_text(_joint_range("shoulder", spec))
    elbow_range = _range_text(_joint_range("elbow", spec))
    wrist_range = _range_text(_joint_range("wrist_pitch", spec))
    roll_range = _range_text(_joint_range("wrist_roll", spec))

    # The camera's default optical axis is local -Z.  A randomized pitch is an
    # installation-tolerance perturbation, not privileged policy information.
    return f"""<mujoco model="brain_to_arm">
  <compiler angle="degree" autolimits="true" inertiafromgeom="true"/>
  <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
  <size nconmax="200" njmax="500"/>
  <visual>
    <global offwidth="640" offheight="360"/>
    <quality shadowsize="2048"/>
    <map znear="0.003" zfar="3"/>
  </visual>
  <default>
    <joint damping="1.5" armature="0.015"/>
    <geom friction="0.9 0.01 0.001" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2" rgba="0.12 0.14 0.17 1"/>
    </default>
    <default class="collision">
      <geom group="3" rgba="0.22 0.24 0.28 1"/>
    </default>
  </default>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="{floor_rgb1}" rgb2="{floor_rgb2}" width="512" height="512"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="4 4" reflectance="0.08"/>
    <mesh name="base_case" file="{mesh_path('Alt_Kasa.stl')}" scale="0.001 0.001 0.001"/>
    <mesh name="base_rotor" file="{mesh_path('Alt_Govde.stl')}" scale="0.001 0.001 0.001"/>
    <mesh name="upper_visual" file="{mesh_path('Alt_Kol.stl')}" scale="0.001 0.001 0.001"/>
    <mesh name="fore_visual" file="{mesh_path('On_Kol.stl')}" scale="0.001 0.001 0.001"/>
    <mesh name="wrist_visual" file="{mesh_path('Bilek.stl')}" scale="0.001 0.001 0.001"/>
    <mesh name="palm_visual" file="{mesh_path('El_Ust.stl')}" scale="0.001 0.001 0.001"/>
    <mesh name="finger_visual" file="{mesh_path('Parmak_2 X 2.stl')}" scale="0.001 0.001 0.001"/>
  </asset>
  <worldbody>
    <light name="key" pos="0 -0.4 1.1" dir="0 0.25 -1" diffuse="0.9 0.9 0.9" castshadow="true"/>
    <light name="fill" pos="0.4 0.6 0.7" dir="-0.2 -0.3 -1" diffuse="0.45 0.48 0.52"/>
    <geom name="floor" type="plane" size="1.2 1.2 0.02" material="floor_mat" friction="1.0 0.01 0.001"/>
    <camera name="overview" pos="0.72 -0.78 0.56" xyaxes="0.74 0.68 0 -0.27 0.29 0.92" fovy="48"/>

    <!-- Motor 1 is intentionally absent: the complete kinematic tree is fixed
         at the physical 90-degree base heading. -->
    <body name="base" pos="0 0 0">
      <geom name="base_collision" class="collision" type="box" pos="-0.045 0 0.035" size="0.105 0.06 0.035"/>
      <geom name="base_mesh" class="visual" mesh="base_case"/>
      <geom name="rotor_mesh" class="visual" mesh="base_rotor" pos="0 0 0.08"/>
      <body name="shoulder_link" pos="0 0 {_f(spec.shoulder_height_m)}">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="{shoulder_range}"/>
        <geom name="upper_collision" class="collision" type="capsule" fromto="0 0 0 {_f(upper)} 0 0" size="0.025"/>
        <geom name="upper_mesh" class="visual" mesh="upper_visual" pos="0.1275 0 0.02" euler="0 90 0"/>
        <body name="elbow_link" pos="{_f(upper)} 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" range="{elbow_range}"/>
          <geom name="fore_collision" class="collision" type="capsule" fromto="0 0 0 {_f(fore)} 0 0" size="0.024"/>
          <geom name="fore_mesh" class="visual" mesh="fore_visual" pos="0.1525 0 0"/>
          <body name="wrist_pitch_link" pos="{_f(fore)} 0 0">
            <joint name="wrist_pitch" type="hinge" axis="0 1 0" range="{wrist_range}"/>
            <geom name="wrist_collision" class="collision" type="capsule" fromto="0 0 0 0.045 0 0" size="0.022"/>
            <geom name="wrist_mesh" class="visual" mesh="wrist_visual" pos="0.022 0 0" euler="0 90 0"/>
            <body name="wrist_roll_link" pos="0.045 0 0">
              <joint name="wrist_roll" type="hinge" axis="1 0 0" range="{roll_range}"/>
              <geom name="palm_collision" class="collision" type="box" pos="0.022 0 0" size="0.027 0.028 0.012"/>
              <geom name="palm_mesh" class="visual" mesh="palm_visual" pos="0.022 0 -0.0075"/>
              <camera name="wrist" pos="{_f(camera_x)} 0 {_f(camera_z)}" euler="0 {_f(camera_pitch)} {_f(camera_roll)}" fovy="{_f(camera_fovy)}"/>
              <site name="camera_mount" pos="{_f(camera_x)} 0 {_f(camera_z)}" size="0.0001" rgba="0 0 0 0"/>

              <body name="left_finger" pos="0 0 0">
                <joint name="grip_left" type="slide" axis="0 1 0" range="0.007 0.045" damping="3"/>
                <geom name="left_finger_collision" class="collision" type="box" pos="0.064 0 -0.004" size="0.037 0.006 0.009"/>
                <geom name="left_finger_mesh" class="visual" mesh="finger_visual" pos="0.034 -0.008 -0.006"/>
                <geom name="blue_marker" type="box" pos="0.074 0 0.007" size="0.016 0.007 0.004" contype="0" conaffinity="0" rgba="0.02 0.20 1 1"/>
              </body>
              <body name="right_finger" pos="0 0 0">
                <joint name="grip_right" type="slide" axis="0 -1 0" range="0.007 0.045" damping="3"/>
                <geom name="right_finger_collision" class="collision" type="box" pos="0.064 0 -0.004" size="0.037 0.006 0.009"/>
                <geom name="right_finger_mesh" class="visual" mesh="finger_visual" pos="0.034 -0.008 -0.006" euler="180 0 0"/>
                <geom name="red_marker" type="box" pos="0.074 0 0.007" size="0.016 0.007 0.004" contype="0" conaffinity="0" rgba="1 0.03 0.02 1"/>
              </body>
              <site name="tool_center" pos="{_f(hand)} 0 -0.008" size="0.0001" rgba="0 0 0 0"/>
            </body>
          </body>
        </body>
      </body>
    </body>

    <body name="target" pos="0.46 0 0.018">
      <freejoint name="target_free"/>
      <geom name="target_geom" type="{target_shape}" size="{target_size_text}" mass="0.035" rgba="{target_rgb} 1" friction="1.1 0.01 0.001"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_position" joint="shoulder" kp="45" kv="6" ctrllimited="true" ctrlrange="{shoulder_range}"/>
    <position name="elbow_position" joint="elbow" kp="38" kv="5" ctrllimited="true" ctrlrange="{elbow_range}"/>
    <position name="wrist_pitch_position" joint="wrist_pitch" kp="18" kv="2.5" ctrllimited="true" ctrlrange="{wrist_range}"/>
    <position name="wrist_roll_position" joint="wrist_roll" kp="12" kv="1.5" ctrllimited="true" ctrlrange="{roll_range}"/>
    <position name="grip_left_position" joint="grip_left" kp="80" kv="5" ctrllimited="true" ctrlrange="0.007 0.045"/>
    <position name="grip_right_position" joint="grip_right" kp="80" kv="5" ctrllimited="true" ctrlrange="0.007 0.045"/>
  </actuator>
</mujoco>"""


def load_model(
    spec: RobotSpec | None = None,
    *,
    parameters: Mapping[str, float] | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, RobotSpec]:
    spec = spec or RobotSpec.from_manifest()
    model = mujoco.MjModel.from_xml_string(build_mjcf(spec, parameters=parameters))
    data = mujoco.MjData(model)
    set_servo_pose(model, data, spec.home_servo_deg, spec=spec)
    return model, data, spec


def _joint_qpos_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise KeyError(name)
    return int(model.jnt_qposadr[joint_id])


def set_servo_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: Iterable[float],
    *,
    spec: RobotSpec | None = None,
    reset_velocity: bool = True,
) -> np.ndarray:
    """Set a static pose and matching actuator targets without stepping time."""

    spec = spec or RobotSpec.from_manifest()
    servo = spec.validate_servo_pose(pose)
    targets = servo_to_joint_targets(servo, spec)
    for name, value in targets.items():
        data.qpos[_joint_qpos_address(model, name)] = value
    if reset_velocity:
        data.qvel[:] = 0
    actuator_targets = (
        targets["shoulder"], targets["elbow"], targets["wrist_pitch"],
        targets["wrist_roll"], targets["grip_left"], targets["grip_right"],
    )
    data.ctrl[:] = actuator_targets
    mujoco.mj_forward(model, data)
    return servo.copy()


def command_servo_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: Iterable[float],
    *,
    spec: RobotSpec | None = None,
) -> np.ndarray:
    """Update actuator targets for dynamic stepping, preserving current state."""

    spec = spec or RobotSpec.from_manifest()
    servo = spec.validate_servo_pose(pose)
    targets = servo_to_joint_targets(servo, spec)
    data.ctrl[:] = (
        targets["shoulder"], targets["elbow"], targets["wrist_pitch"],
        targets["wrist_roll"], targets["grip_left"], targets["grip_right"],
    )
    return servo.copy()


def site_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise KeyError(name)
    return data.site_xpos[site_id].copy()


def place_target_below_tool(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    z: float = 0.018,
    xy_offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Privileged reset helper; target pose is never an actor observation."""

    target = site_position(model, data, "tool_center")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
    address = int(model.jnt_qposadr[joint])
    data.qpos[address:address + 3] = (
        target[0] + xy_offset[0], target[1] + xy_offset[1], z)
    data.qpos[address + 3:address + 7] = (1.0, 0.0, 0.0, 0.0)
    velocity_address = int(model.jnt_dofadr[joint])
    data.qvel[velocity_address:velocity_address + 6] = 0
    mujoco.mj_forward(model, data)
    return data.qpos[address:address + 3].copy()


def place_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    xyz: Iterable[float],
) -> np.ndarray:
    """Place the free target for reset/reward code, never for actor input."""

    position = np.asarray(tuple(xyz), dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("target position must be three finite world coordinates")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
    address = int(model.jnt_qposadr[joint])
    data.qpos[address:address + 3] = position
    data.qpos[address + 3:address + 7] = (1.0, 0.0, 0.0, 0.0)
    velocity_address = int(model.jnt_dofadr[joint])
    data.qvel[velocity_address:velocity_address + 6] = 0
    mujoco.mj_forward(model, data)
    return position.copy()


def render_rgb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera: str = "wrist",
    width: int = 128,
    height: int = 72,
) -> np.ndarray:
    """Render RGB only; no depth or segmentation leaves this boundary."""

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()
    finally:
        renderer.close()


def model_summary(model: mujoco.MjModel) -> dict[str, object]:
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "joints": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(model.njnt)
        ],
        "cameras": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
            for index in range(model.ncam)
        ],
    }


__all__ = [
    "RobotSpec", "SERVO_NAMES", "build_mjcf", "command_servo_pose",
    "load_model", "model_summary", "place_target", "place_target_below_tool", "render_rgb",
    "servo_to_joint_targets", "set_servo_pose", "site_position",
]
