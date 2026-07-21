"""Collision-checked 2/3/4-joint search for a full-frame colored target.

The planner does not crop perception. It generates diverse camera viewpoints by
coordinating shoulder, elbow, and wrist pitch, evaluates each pose with planar
forward kinematics, and checks every intermediate state produced by the Uno's
equal-degrees-per-tick slew. Table, base-column, and non-adjacent-link clearances
must all pass. A failed search reverses the already verified route exactly.

Running without ``--run`` never opens hardware. Physical execution is blocked
until both ARM_CALIBRATED and WRIST_SEARCH_KINEMATICS_VERIFIED are true.
"""

from dataclasses import dataclass
import argparse
import itertools
import math

import numpy as np

import config
from arm_serial import ArmSerial
from wrist_vision import WristDetector, open_wrist_camera, observation_summary


def _point_segment_distance(point, start, end):
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    scale = float(np.dot(delta, delta))
    if scale <= 1e-12:
        return float(np.linalg.norm(point - start))
    amount = max(0.0, min(1.0, float(np.dot(point - start, delta)) / scale))
    return float(np.linalg.norm(point - (start + amount * delta)))


def _orientation(a, b, c):
    ab = np.subtract(b, a)
    ac = np.subtract(c, a)
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def _on_segment(a, b, point, epsilon=1e-9):
    return (min(a[0], b[0]) - epsilon <= point[0] <= max(a[0], b[0]) + epsilon
            and min(a[1], b[1]) - epsilon <= point[1] <= max(a[1], b[1]) + epsilon)


def _segments_intersect(a, b, c, d):
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    epsilon = 1e-9
    if ((o1 > epsilon and o2 < -epsilon or o1 < -epsilon and o2 > epsilon)
            and (o3 > epsilon and o4 < -epsilon or o3 < -epsilon and o4 > epsilon)):
        return True
    return ((abs(o1) <= epsilon and _on_segment(a, b, c))
            or (abs(o2) <= epsilon and _on_segment(a, b, d))
            or (abs(o3) <= epsilon and _on_segment(c, d, a))
            or (abs(o4) <= epsilon and _on_segment(c, d, b)))


def _segment_distance(a, b, c, d):
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


@dataclass(frozen=True)
class PlanarGeometry:
    base: tuple
    shoulder: tuple
    elbow: tuple
    wrist: tuple
    camera: tuple
    camera_angle_deg: float


class PlanarSearchSafety:
    """Forward-kinematics and collision checks for motors 2/3/4."""
    def __init__(self, offsets=None, directions=None, lengths=None,
                 base_height=None, table_clearance=None,
                 base_clearance=None, self_clearance=None,
                 camera_mount_angle=None, slew_step=None):
        self.offsets = tuple(config.SERVO_OFFSET if offsets is None else offsets)
        self.directions = tuple(
            config.SERVO_DIRECTION if directions is None else directions)
        self.lengths = tuple(
            (config.L_UPPER, config.L_FORE, config.L_HAND)
            if lengths is None else lengths)
        self.base_height = float(
            config.L_BASE_HEIGHT if base_height is None else base_height)
        self.table_clearance = float(
            config.WRIST_SEARCH_TABLE_CLEARANCE_CM
            if table_clearance is None else table_clearance)
        self.base_clearance = float(
            config.WRIST_SEARCH_BASE_CLEARANCE_CM
            if base_clearance is None else base_clearance)
        self.self_clearance = float(
            config.WRIST_SEARCH_SELF_CLEARANCE_CM
            if self_clearance is None else self_clearance)
        self.camera_mount_angle = float(
            config.WRIST_CAMERA_MOUNT_ANGLE_DEG
            if camera_mount_angle is None else camera_mount_angle)
        self.slew_step = float(
            config.WRIST_SEARCH_SLEW_STEP_DEG if slew_step is None else slew_step)
        if len(self.offsets) != config.N_JOINTS or len(self.directions) != config.N_JOINTS:
            raise ValueError("search model needs one offset/direction per joint")
        if len(self.lengths) != 3 or any(value <= 0 for value in self.lengths):
            raise ValueError("search model link lengths must be positive")
        if min(self.table_clearance, self.base_clearance,
               self.self_clearance, self.slew_step) <= 0:
            raise ValueError("search clearances and slew step must be positive")

    def _joint_angle(self, pose, joint):
        return math.radians(
            (float(pose[joint]) - self.offsets[joint]) / self.directions[joint])

    def forward(self, pose):
        if len(pose) != config.N_JOINTS:
            raise ValueError("search pose must contain all six joints")
        shoulder_angle = self._joint_angle(pose, config.J_SHOULDER)
        elbow_angle = self._joint_angle(pose, config.J_ELBOW)
        wrist_angle = self._joint_angle(pose, config.J_WRIST)
        angles = (
            shoulder_angle,
            shoulder_angle + elbow_angle,
            shoulder_angle + elbow_angle + wrist_angle,
        )
        base = np.asarray((0.0, 0.0))
        shoulder = np.asarray((0.0, self.base_height))
        points = [shoulder]
        current = shoulder
        for length, angle in zip(self.lengths, angles):
            current = current + length * np.asarray((math.cos(angle), math.sin(angle)))
            points.append(current)
        return PlanarGeometry(
            tuple(base), tuple(shoulder), tuple(points[1]), tuple(points[2]),
            tuple(points[3]),
            math.degrees(angles[-1]) + self.camera_mount_angle)

    def pose_is_safe(self, pose):
        for joint in config.WRIST_SEARCH_JOINTS:
            if not config.SERVO_MIN[joint] <= pose[joint] <= config.SERVO_MAX[joint]:
                return False
        geometry = self.forward(pose)
        shoulder = np.asarray(geometry.shoulder)
        elbow = np.asarray(geometry.elbow)
        wrist = np.asarray(geometry.wrist)
        camera = np.asarray(geometry.camera)
        # Straight links cannot dip below their endpoints in this planar model.
        if min(elbow[1], wrist[1], camera[1]) < self.table_clearance:
            return False
        base_bottom = np.asarray(geometry.base)
        # Forearm and hand must not sweep through the vertical base/mast.
        if _segment_distance(elbow, wrist, base_bottom, shoulder) < self.base_clearance:
            return False
        if _segment_distance(wrist, camera, base_bottom, shoulder) < self.base_clearance:
            return False
        # The non-adjacent upper-arm and hand capsules may not cross/touch.
        if _segment_distance(shoulder, elbow, wrist, camera) < self.self_clearance:
            return False
        return True

    def slew_states(self, start, target):
        """Match firmware: every moving servo advances equal degrees per tick."""
        start = [float(value) for value in start]
        target = [float(value) for value in target]
        ticks = int(math.ceil(max(
            abs(target[joint] - start[joint])
            for joint in config.WRIST_SEARCH_JOINTS) / self.slew_step))
        for tick in range(1, ticks + 1):
            elapsed = tick * self.slew_step
            pose = list(start)
            for joint in config.WRIST_SEARCH_JOINTS:
                difference = target[joint] - start[joint]
                pose[joint] = start[joint] + math.copysign(
                    min(abs(difference), elapsed), difference) if difference else start[joint]
            yield pose

    def transition_is_safe(self, start, target):
        return self.pose_is_safe(start) and all(
            self.pose_is_safe(pose) for pose in self.slew_states(start, target))

    def view_feature(self, pose):
        geometry = self.forward(pose)
        reach = max(sum(self.lengths), 1.0)
        angle = math.radians(geometry.camera_angle_deg)
        return np.asarray((
            geometry.camera[0] / reach,
            geometry.camera[1] / reach,
            math.cos(angle), math.sin(angle)), dtype=float)


def _grid_values(minimum, maximum, step):
    values = list(range(int(minimum), int(maximum) + 1, int(step)))
    if not values or values[-1] != int(maximum):
        values.append(int(maximum))
    return values


class CollisionFreeSearchPlanner:
    def __init__(self, safety=None):
        self.safety = safety or PlanarSearchSafety()

    def candidates(self, template_pose):
        ranges = []
        for joint, step in zip(config.WRIST_SEARCH_JOINTS,
                               config.WRIST_SEARCH_GRID_STEP):
            ranges.append(_grid_values(
                config.SERVO_MIN[joint], config.SERVO_MAX[joint], step))
        poses = []
        for values in itertools.product(*ranges):
            pose = list(template_pose)
            for joint, value in zip(config.WRIST_SEARCH_JOINTS, values):
                pose[joint] = value
            if self.safety.pose_is_safe(pose):
                poses.append(pose)
        return poses

    def plan(self, start, max_poses=None):
        max_poses = config.WRIST_SEARCH_MAX_POSES if max_poses is None else int(max_poses)
        if max_poses < 1:
            raise ValueError("max search poses must be positive")
        if not self.safety.pose_is_safe(start):
            raise RuntimeError(
                "current arm pose fails the calibrated collision model; do not search")
        remaining = self.candidates(start)
        current = list(start)
        visited_features = [self.safety.view_feature(current)]
        plan = []
        while remaining and len(plan) < max_poses:
            reachable = [pose for pose in remaining
                         if self.safety.transition_is_safe(current, pose)]
            if not reachable:
                break
            def coverage_score(pose):
                feature = self.safety.view_feature(pose)
                novelty = min(float(np.linalg.norm(feature - previous))
                              for previous in visited_features)
                travel = sum(abs(pose[joint] - current[joint])
                             for joint in config.WRIST_SEARCH_JOINTS) / 360.0
                return novelty - 0.08 * travel
            selected = max(reachable, key=coverage_score)
            plan.append(selected)
            visited_features.append(self.safety.view_feature(selected))
            current = selected
            remaining.remove(selected)
        return plan


class WristSearcher:
    def __init__(self, arm, camera, detector=None, planner=None):
        self.arm = arm
        self.camera = camera
        self.detector = detector or WristDetector()
        self.planner = planner or CollisionFreeSearchPlanner()

    def capture(self, discard=4, count=3):
        frames = []
        for index in range(discard + count):
            ok, frame = self.camera.read()
            if not ok:
                raise RuntimeError("wrist camera read failed during target search")
            if index >= discard:
                frames.append(frame)
        return np.median(np.stack(frames), axis=0).astype(np.uint8)

    def observe(self):
        observation, _masks = self.detector.detect(self.capture())
        return observation

    @staticmethod
    def _found(observation):
        return observation.quality.valid and observation.target is not None

    def find(self, max_poses=None):
        pose = list(self.arm.status())
        observation = self.observe()
        print(f"[wrist-search] initial {observation_summary(observation)}")
        if self._found(observation):
            return observation

        route = [list(pose)]
        try:
            for index, next_pose in enumerate(self.planner.plan(pose, max_poses), 1):
                # Recheck directly before commanding; planner/model config could
                # otherwise be mutated between plan construction and motion.
                if not self.planner.safety.transition_is_safe(route[-1], next_pose):
                    raise RuntimeError("planned wrist-search transition became unsafe")
                self.arm.send_angles(next_pose)
                self.arm.wait_done()
                route.append(list(next_pose))
                observation = self.observe()
                geometry = self.planner.safety.forward(next_pose)
                print(
                    f"[wrist-search] view={index} joints2/3/4="
                    f"{[next_pose[j] for j in config.WRIST_SEARCH_JOINTS]} "
                    f"camera=({geometry.camera[0]:.1f},{geometry.camera[1]:.1f})cm "
                    f"angle={geometry.camera_angle_deg:.1f}deg "
                    f"{observation_summary(observation)}")
                if self._found(observation):
                    return observation
            return None
        finally:
            # Failure/error retraces known-safe edges in reverse. On success keep
            # the target-visible viewpoint for the alignment controller.
            if len(route) > 1 and not self._found(observation):
                for previous in reversed(route[:-1]):
                    try:
                        self.arm.send_angles(previous)
                        self.arm.wait_done()
                    except Exception as exc:
                        print(f"[wrist-search] WARNING: route restore failed: {exc}")
                        break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="open the Uno and execute only verified-safe views")
    parser.add_argument("--target-hue", type=float,
                        help="OpenCV hue 0..179; omit only in a clean one-object scene")
    parser.add_argument("--max-poses", type=int, default=config.WRIST_SEARCH_MAX_POSES)
    parser.add_argument("--camera-name", default=config.WRIST_CAMERA_NAME)
    args = parser.parse_args()

    safety = PlanarSearchSafety()
    if not args.run:
        print("No hardware opened.")
        print("Search joints:", [joint + 1 for joint in config.WRIST_SEARCH_JOINTS])
        print("Kinematics verified:", config.WRIST_SEARCH_KINEMATICS_VERIFIED)
        print("Set ARM_CALIBRATED and WRIST_SEARCH_KINEMATICS_VERIFIED only after measurement.")
        return True
    if not config.ARM_CALIBRATED or not config.WRIST_SEARCH_KINEMATICS_VERIFIED:
        raise RuntimeError(
            "collision-checked search is locked: calibrate servo offsets/directions, "
            "link lengths, and then enable both kinematics gates")

    detector = WristDetector()
    if args.target_hue is not None:
        detector.set_target_hue(args.target_hue)
    camera = open_wrist_camera(name=args.camera_name)
    arm = None
    try:
        for _ in range(config.WRIST_CAMERA_WARMUP_FRAMES):
            ok, _frame = camera.read()
            if not ok:
                raise RuntimeError("wrist camera stopped during warmup")
        arm = ArmSerial()
        result = WristSearcher(
            arm, camera, detector,
            CollisionFreeSearchPlanner(safety)).find(args.max_poses)
        if result is None:
            raise RuntimeError("target color was not found in collision-free viewpoints")
        print("[wrist-search] TARGET FOUND", observation_summary(result))
        return True
    finally:
        camera.release()
        if arm is not None:
            arm.close()


if __name__ == "__main__":
    main()
