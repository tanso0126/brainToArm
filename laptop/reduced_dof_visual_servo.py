"""Autonomous camera/sonar approach for the reduced physical arm.

Only shoulder (2) and elbow (3) enter the Jacobian.  Base (1), the removed
wrist-pitch actuator (4), and wrist roll (6) remain fixed; servo 5 only opens or
closes the gripper.  The historical 2/3/4 controller remains in
``realtime_visual_servo.py`` for possible future motor repair.
"""

from pathlib import Path
import argparse
import time

import cv2

import config
from floor_grasp import WristSceneDetector
from realtime_visual_servo import (
    HistogramTargetTracker,
    LatestFrameStream,
    _gripper_geometry,
    dynamic_aim_y,
    floor_limited_grasp_readiness,
    grasp_readiness,
    select_realtime_seed,
)
from reduced_dof import (
    ReducedDofSafety,
    canonicalize_status,
    command_pose,
    find_lift_pose,
    fingertip_floor_clearance_mm,
    reduced_home,
    resolved_step,
    safe_route,
    search_poses,
)
from reduced_dof_session import client as reduced_client
from ultrasonic_target_reach import (
    FINGERTIP_FLOOR_STOP_MM,
    STOP_RANGE_MM,
    VividFallbackDetector,
)
from wrist_vision import WristDetector, _atomic_write_jpeg


ROOT = Path(__file__).resolve().parents[1]
PREVIEW_PATH = ROOT / "data" / "vision" / "reduced_dof_latest.jpg"
CONTROL_HZ = 8.0
SONAR_HZ = 5.0
PREVIEW_HZ = 8.0
TRACK_LOST_TIMEOUT_S = 0.9
MAX_HORIZONTAL_OPENING_FRACTION = 0.42


def _preview(frame, target, gripper_center, aim_y, distance_mm,
             clearance_mm, state):
    image = frame.copy()
    x, y, width, height = target.bbox
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 220, 255), 3)
    cv2.circle(
        image,
        tuple(int(round(value)) for value in target.center),
        7, (0, 220, 255), -1,
    )
    cv2.drawMarker(
        image,
        tuple(int(round(value)) for value in gripper_center),
        (255, 255, 0), cv2.MARKER_CROSS, 34, 2,
    )
    cv2.line(
        image, (0, int(round(aim_y))),
        (image.shape[1], int(round(aim_y))), (80, 255, 80), 2,
    )
    text = (
        f"REDUCED 2DOF {state} range="
        f"{'--' if distance_mm is None else f'{distance_mm:.0f}mm'} "
        f"floor={clearance_mm:.1f}mm")
    cv2.putText(
        image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
        0.70, (0, 0, 0), 4, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
        0.70, (255, 255, 255), 2, cv2.LINE_AA,
    )
    _atomic_write_jpeg(PREVIEW_PATH, image)


def _search_and_seed(connection, frames, detector, safety, execute,
                     stop_event=None, candidate_rank=0):
    pose = canonicalize_status(
        connection.request({"command": "status"})["pose"])
    opened = command_pose(
        pose[config.J_SHOULDER], pose[config.J_ELBOW], config.GRIP_OPEN)
    if execute and opened != pose:
        connection.request({
            "command": "move", "pose": opened, "require_camera": True})
        pose = opened

    frame, _stamp = frames.read(timeout_s=1.0)
    scene, _observation = detector.scene(frame)
    candidate = _ranked_seed(scene, frame, candidate_rank)
    if candidate is not None or not execute:
        return pose, frame, candidate

    for target in search_poses(config.GRIP_OPEN):
        if stop_event is not None and stop_event.is_set():
            return pose, frame, None
        report = safety.transition_report(pose, target)
        if not report.safe:
            continue
        connection.request({
            "command": "move", "pose": target, "require_camera": True})
        pose = target
        frame, _stamp = frames.read(timeout_s=1.0)
        scene, _observation = detector.scene(frame)
        candidate = _ranked_seed(scene, frame, candidate_rank)
        if candidate is not None:
            return pose, frame, candidate
    return pose, frame, None


def _ranked_seed(scene, frame, candidate_rank=0):
    """Select one stable ranked candidate while preserving the legacy default."""
    rank = max(0, int(candidate_rank))
    if rank == 0:
        return select_realtime_seed(scene, frame)
    # The multi-object GUI deliberately uses FastSAM's consolidated ranking.
    # The vivid fallback has no stable identity, so it remains rank-0 only.
    candidates = list(getattr(scene, "ranked", ()))
    return candidates[rank] if rank < len(candidates) else None


def close_lift_home(connection, safety, pose):
    """Close, lift, and return using only 2/3 while servo 5 holds the object."""
    closed = command_pose(
        pose[config.J_SHOULDER], pose[config.J_ELBOW], config.GRIP_CLOSED)
    connection.request({
        "command": "move", "pose": closed, "require_camera": True})
    lifted = find_lift_pose(closed, safety=safety)
    connection.request({
        "command": "move", "pose": lifted, "require_camera": True})

    # The external gripper supply was added specifically so the object can stay
    # clamped during transport.  Do not lower the command to the historical
    # 158-degree low-power workaround while returning HOME.
    home = reduced_home(config.GRIP_CLOSED)
    current = lifted
    for target in safe_route(current, home, safety=safety):
        connection.request({
            "command": "move", "pose": target, "require_camera": True})
        current = target
    return home


def run(execute=False, allow_grasp=False, max_seconds=60.0,
        learned_policy=False, stop_event=None, candidate_rank=0):
    connection = reduced_client()
    safety = ReducedDofSafety()
    frames = LatestFrameStream()
    scene_detector = VividFallbackDetector(WristSceneDetector())
    wrist_detector = WristDetector()
    pose, frame, candidate = _search_and_seed(
        connection, frames, scene_detector, safety, execute,
        stop_event=stop_event, candidate_rank=candidate_rank)
    if stop_event is not None and stop_event.is_set():
        return {"state": "stopped", "pose": pose, "mode": "reduced-2dof"}
    if candidate is None:
        return {"state": "no-target", "pose": pose, "mode": "reduced-2dof"}

    tracker = HistogramTargetTracker()
    policy = None
    if learned_policy:
        from reduced_policy_adapter import ReducedPolicyController
        policy = ReducedPolicyController()
    target = tracker.initialize(frame, candidate.bbox)
    distance = None
    last_sonar = 0.0
    last_control = 0.0
    last_preview = 0.0
    last_seen = time.monotonic()
    deadline = last_seen + float(max_seconds)

    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            try:
                connection.request({"command": "stop"})
            except Exception:
                pass
            return {
                "state": "stopped",
                "pose": canonicalize_status(
                    connection.request({"command": "status"})["pose"]),
                "preview": str(PREVIEW_PATH),
                "mode": "reduced-2dof",
            }
        frame, _stamp = frames.read(timeout_s=1.0)
        now = time.monotonic()
        tracked = tracker.update(frame)
        observation, _masks = wrist_detector.detect(frame)
        if tracked is not None:
            target = tracked
            last_seen = now
        elif now - last_seen > TRACK_LOST_TIMEOUT_S:
            return {
                "state": "target-lost",
                "pose": canonicalize_status(
                    connection.request({"command": "status"})["pose"]),
                "preview": str(PREVIEW_PATH),
                "mode": "reduced-2dof",
            }
        else:
            continue

        gripper_center, opening_px = _gripper_geometry(observation, frame)
        if now - last_sonar >= 1.0 / SONAR_HZ:
            response = connection.request({"command": "distance", "samples": 1})
            distance = (
                float(response["distanceMm"])
                if response.get("valid") else None)
            last_sonar = now
        if now - last_control < 1.0 / CONTROL_HZ:
            continue

        pose = canonicalize_status(
            connection.request({"command": "status"})["pose"])
        clearance = fingertip_floor_clearance_mm(pose)
        aim_y = dynamic_aim_y(frame.shape[0], distance, gripper_center[1])
        horizontal_error = abs(
            float(target.center[0]) - float(gripper_center[0]))
        within_plane = (
            horizontal_error
            <= MAX_HORIZONTAL_OPENING_FRACTION * float(opening_px))
        readiness = grasp_readiness(
            target, gripper_center, opening_px, distance, clearance)
        if readiness.ready and not within_plane:
            readiness = type(readiness)(
                False,
                "고정 베이스 작업선 밖에 물체가 있음",
                readiness.center_error_px,
            )
        policy_action = None
        if policy is not None:
            decision = policy.decide(
                pose=pose, target_center=target.center,
                gripper_center=gripper_center, opening_px=opening_px,
                frame_shape=frame.shape,
                quality_valid=bool(observation.quality.valid),
                target_locked=True, sonar_distance_mm=distance,
                phase="approach",
            )
            policy_action = decision.action
        if now - last_preview >= 1.0 / PREVIEW_HZ:
            _preview(
                frame, target, gripper_center, aim_y, distance, clearance,
                "READY" if readiness.ready else "TRACK",
            )
            last_preview = now

        if readiness.ready:
            if not allow_grasp:
                return {
                    "state": "grasp-ready", "pose": pose,
                    "distance_mm": distance,
                    "horizontal_error_px": horizontal_error,
                    "preview": str(PREVIEW_PATH),
                    "mode": "reduced-2dof",
                }
            if policy_action is not None:
                from simul.reduced_dof_task_env import ReducedTaskAction
                if policy_action != ReducedTaskAction.CLOSE:
                    last_control = now
                    continue
            home = close_lift_home(connection, safety, pose)
            return {
                "state": "home-after-grasp", "pose": home,
                "distance_mm": distance,
                "horizontal_error_px": horizontal_error,
                "preview": str(PREVIEW_PATH),
                "mode": "reduced-2dof",
            }

        step = resolved_step(
            pose,
            float(target.center[1]) - float(aim_y),
            distance,
            STOP_RANGE_MM,
            FINGERTIP_FLOOR_STOP_MM,
            safety=safety,
        )
        if step is None:
            floor_ready = floor_limited_grasp_readiness(
                target, gripper_center, opening_px, clearance)
            if floor_ready.ready and within_plane:
                if not allow_grasp:
                    return {
                        "state": "grasp-ready-floor", "pose": pose,
                        "distance_mm": distance,
                        "preview": str(PREVIEW_PATH),
                        "mode": "reduced-2dof",
                    }
                if policy_action is not None:
                    from simul.reduced_dof_task_env import ReducedTaskAction
                    if policy_action != ReducedTaskAction.CLOSE:
                        last_control = now
                        continue
                home = close_lift_home(connection, safety, pose)
                return {
                    "state": "home-after-grasp", "pose": home,
                    "distance_mm": distance,
                    "preview": str(PREVIEW_PATH),
                    "mode": "reduced-2dof",
                    "gate": floor_ready.reason,
                }
            return {
                "state": "safe-reach-exhausted", "pose": pose,
                "distance_mm": distance,
                "reason": readiness.reason,
                "preview": str(PREVIEW_PATH),
                "mode": "reduced-2dof",
            }
        if policy_action is not None:
            from simul.reduced_dof_task_env import ReducedTaskAction
            if policy_action not in (
                    ReducedTaskAction.APPROACH, ReducedTaskAction.CLOSE):
                last_control = now
                continue
        if not execute:
            return {
                "state": "planned", "pose": step.pose,
                "distance_mm": distance,
                "preview": str(PREVIEW_PATH),
                "mode": "reduced-2dof",
            }
        connection.request({
            "command": "stream", "pose": step.pose,
            "require_camera": True,
        })
        last_control = now

    return {
        "state": "time-limit",
        "pose": canonicalize_status(
            connection.request({"command": "status"})["pose"]),
        "preview": str(PREVIEW_PATH),
        "mode": "reduced-2dof",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="실물 로봇팔을 움직임")
    parser.add_argument(
        "--grasp", action="store_true", help="접근 후 집고 축소 HOME으로 복귀")
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument(
        "--learned-policy", action="store_true",
        help="축소 시뮬레이션에서 학습한 정책을 상위 행동 게이트로 사용")
    args = parser.parse_args()
    if args.grasp and not args.run:
        parser.error("--grasp를 사용하려면 --run도 함께 지정해야 합니다.")
    result = run(args.run, args.grasp, args.max_seconds, args.learned_policy)
    print(f"[축소 자유도 제어 결과] {result}", flush=True)


if __name__ == "__main__":
    main()
