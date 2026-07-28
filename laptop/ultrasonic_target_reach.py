"""Vision bearing + ultrasonic depth closed-loop approach.

The existing :mod:`look_reach` controller already locks a portable object and
keeps motors 2/3/4 pointing at it.  Its missing observation was distance along
that bearing.  This controller adds exactly that scalar; it does not estimate a
table plane or require an absolute sonar/world transform.

Every physical step is:

1. reacquire the same visual instance and keep it near the camera/sonar axis;
2. require a temporally stable ultrasonic echo;
3. plan one adaptive 15/10/5 mm motor-2/3/4 step with the existing controller;
4. collision-check the complete swept transition;
5. move, reacquire, and require the echo distance to decrease.

The sonar face is physically behind the finger contact plane.  With an object
inserted as deeply as the fingers permit, 180 live echoes formed a dominant
stable cluster at 36 mm.  The operational stop therefore retains 10 mm margin
and stops at 46 mm.  The only normal approach stops are that measured sonar
threshold and 10 mm fingertip-to-floor clearance.  Vision keeps steering the
same object but never ends an otherwise valid approach.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import time

import numpy as np

import arm_fk
import config
from arm_safety import PhysicalArmSafety
from arm_session import ArmSessionClient
from floor_grasp import WristSceneDetector
from floor_servo import FloorServo, _fresh_frame
from look_reach import (
    AIM_ONLY_THRESHOLD_PX,
    LookReachTargetSelector,
    acquire_initial_target,
    plan_aim_step,
    plan_resolved_step,
)
from ultrasonic_depth import wait_for_stable_profile


ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "data" / "vision" / "ultrasonic_target_reach_latest.jpg"

# Camera and HC-SR04 are mounted as one bracket.  The optical/acoustic boresight
# is near image centre after the operator's physical alignment.
SONAR_AIM_X_RATIO = 0.50
SONAR_AIM_Y_RATIO = 0.50
MAX_AIM_X_ERROR_PX = 90.0

# The deepest physically inserted object produced a stable 36 mm dominant echo
# cluster (180/180 valid samples, 1 mm MAD). Stop 10 mm before that plane.
MEASURED_DEEPEST_OBJECT_RANGE_MM = 36.0
SONAR_STOP_MARGIN_MM = 10.0
STOP_RANGE_MM = MEASURED_DEEPEST_OBJECT_RANGE_MM + SONAR_STOP_MARGIN_MM
FINGERTIP_FLOOR_STOP_MM = 10.0
FAR_ROW_GAP_PX = 180.0
FAR_ADVANCE_MM = 15.0
MID_ADVANCE_MM = 10.0
APPROACH_MAX_JOINT_STEP_DEG = 8
# The explicit fingertip-floor gate below is authoritative. This lower planner
# bound prevents an unrelated tool-centre clamp from ending the approach early.
MIN_TOOL_CENTER_Z_M = 0.010
MAX_STEPS = 24


@dataclass(frozen=True)
class RangeDecision:
    action: str
    reason: str


def approach_stop_decision(distance_mm, fingertip_floor_clearance_mm,
                           stop_range_mm=STOP_RANGE_MM,
                           floor_stop_mm=FINGERTIP_FLOOR_STOP_MM):
    """Return one of the two normal approach stops, otherwise continue."""
    clearance = float(fingertip_floor_clearance_mm)
    distance = float(distance_mm)
    if clearance <= float(floor_stop_mm):
        return RangeDecision(
            "floor",
            f"fingertip-floor clearance {clearance:.1f} mm <= "
            f"{float(floor_stop_mm):.1f} mm")
    if distance <= float(stop_range_mm):
        return RangeDecision(
            "sonar",
            f"sonar {distance:.1f} mm <= {float(stop_range_mm):.1f} mm")
    return RangeDecision(
        "continue",
        f"sonar {distance:.1f} mm; floor clearance {clearance:.1f} mm")


def adaptive_advance_mm(row_gap_px):
    """Make servo-scale moves; never request the ineffective old 5 mm step."""
    gap = float(row_gap_px)
    if gap > FAR_ROW_GAP_PX:
        return FAR_ADVANCE_MM
    return MID_ADVANCE_MM


def fingertip_floor_clearance_mm(pose, table_z_m=0.0):
    """Physical distal endpoint height over the shared table plane."""
    return (
        float(arm_fk.geometry(pose).finger_tip[2]) - float(table_z_m)
    ) * 1000.0


def transition_fingertip_floor_clearance_mm(
        start_pose, end_pose, table_z_m=0.0, step_deg=0.5):
    """Minimum fingertip clearance over the complete interpolated servo slew."""
    start = np.asarray(start_pose, dtype=float)
    end = np.asarray(end_pose, dtype=float)
    span = float(np.max(np.abs(end - start)))
    samples = max(2, int(np.ceil(span / float(step_deg))) + 1)
    return min(
        fingertip_floor_clearance_mm(pose, table_z_m)
        for pose in np.linspace(start, end, samples)
    )


def _reacquire(detector, selector, attempts=3):
    frame = scene = candidate = None
    for _ in range(attempts):
        frame = _fresh_frame(discard=1)
        scene, _observation = detector.scene(frame)
        candidate = selector.match(scene)
        if candidate is not None:
            return frame, scene, candidate
    return frame, scene, None


def _draw_preview(frame, scene, candidate, pose, distance_mm, step, decision):
    import cv2

    image = frame.copy()
    x, y, width, height = candidate.bbox
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 220, 255), 3)
    aim = (int(round(SONAR_AIM_X_RATIO * image.shape[1])),
           int(round(SONAR_AIM_Y_RATIO * image.shape[0])))
    cv2.drawMarker(image, aim, (255, 255, 255),
                   cv2.MARKER_CROSS, 34, 2)
    cv2.line(image, tuple(int(round(value)) for value in candidate.center),
             aim, (0, 220, 255), 2)
    gripper = getattr(scene, "gripper", None)
    if gripper is not None:
        cv2.drawMarker(
            image, tuple(int(round(value)) for value in gripper.center),
            (255, 255, 0), cv2.MARKER_CROSS, 34, 2)
    text = (
        f"step {step}  sonar {distance_mm:.1f} mm  "
        f"pose {pose[1]}/{pose[2]}/{pose[3]}  {decision.action}")
    cv2.putText(image, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(image, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(PREVIEW), image)


def _jaw_metrics(scene, candidate):
    gripper = getattr(scene, "gripper", None)
    if gripper is None:
        return False, "finger markers unavailable", None
    horizontal = abs(float(candidate.center[0]) - float(gripper.center[0]))
    gap = float(gripper.center[1]) - float(
        candidate.bbox[1] + candidate.bbox[3])
    if horizontal > 0.42 * float(gripper.opening_px):
        return (
            False,
            f"object outside open-jaw corridor ({horizontal:.0f}px)",
            gap,
        )
    if gap > 100.0:
        return False, f"object has not reached finger row (gap {gap:.0f}px)", gap
    return True, f"inside jaws; row gap {gap:.0f}px", gap


def _jaw_gate(scene, candidate):
    ready, reason, _gap = _jaw_metrics(scene, candidate)
    return ready, reason


def run(client=None, execute=False, allow_grasp=False, max_steps=MAX_STEPS,
        detector=None, selector=None):
    client = client or ArmSessionClient()
    detector = detector or WristSceneDetector()
    selector = selector or LookReachTargetSelector()
    safety = PhysicalArmSafety()
    mover = FloorServo(client, calib=None)
    pose = list(client.request({"command": "status"})["pose"])

    frame, scene, candidate = acquire_initial_target(
        detector, target_selector=selector, pose=pose)
    if candidate is None:
        return {"state": "no-target", "moved": False}

    for step in range(int(max_steps)):
        frame, scene, candidate = _reacquire(detector, selector)
        if candidate is None:
            raise RuntimeError("locked target lost; no blind approach")
        pose = list(client.request({"command": "status"})["pose"])
        aim_x = SONAR_AIM_X_RATIO * frame.shape[1]
        aim_y = SONAR_AIM_Y_RATIO * frame.shape[0]
        x_error = float(candidate.center[0]) - aim_x
        if abs(x_error) > MAX_AIM_X_ERROR_PX:
            raise RuntimeError(
                f"target {x_error:+.0f}px from sonar x axis; "
                "base/lateral alignment required")

        profile, attempts = wait_for_stable_profile(client, timeout_s=8.0)
        distance = float(profile.distance_mm)
        jaw_ready, jaw_reason, row_gap = _jaw_metrics(scene, candidate)
        if row_gap is None:
            raise RuntimeError(jaw_reason)
        floor_clearance = fingertip_floor_clearance_mm(pose)
        decision = approach_stop_decision(distance, floor_clearance)
        _draw_preview(
            frame, scene, candidate, pose, distance, step, decision)
        print(
            f"[sonar-reach] step={step:02d} target="
            f"({candidate.center[0]:.0f},{candidate.center[1]:.0f}) "
            f"range={distance:.1f}mm spread={profile.batch_spread_mm:.1f}mm "
            f"attempts={len(attempts)} decision={decision.action} "
            f"jaw={jaw_reason}", flush=True)

        if decision.action == "floor":
            return {
                "state": "floor-clearance-stop", "pose": pose,
                "distance_mm": distance,
                "fingertip_floor_clearance_mm": floor_clearance,
                "preview": str(PREVIEW),
            }
        if decision.action == "sonar":
            if not execute or not allow_grasp or not jaw_ready:
                return {
                    "state": (
                        "sonar-stop-ready" if jaw_ready
                        else "sonar-stop-image-disagrees"),
                    "pose": pose, "distance_mm": distance,
                    "fingertip_floor_clearance_mm": floor_clearance,
                    "preview": str(PREVIEW),
                }
            closed = list(pose)
            closed[config.J_GRIP] = config.GRIP_CLOSED
            report = safety.transition_report(pose, closed)
            if not report.safe:
                raise RuntimeError(
                    "close rejected: " + report.explain())
            mover.slow_move(closed, final_settle=0.8)
            return {
                "state": "closed", "pose": closed,
                "distance_mm": distance, "preview": str(PREVIEW)}

        vertical_error = float(candidate.center[1]) - aim_y
        plan = None
        if abs(vertical_error) > AIM_ONLY_THRESHOLD_PX:
            plan = plan_aim_step(
                pose, vertical_error, MIN_TOOL_CENTER_Z_M)
        if plan is None:
            advance_mm = adaptive_advance_mm(row_gap)
            plan = plan_resolved_step(
                pose, vertical_error, frame.shape[0],
                advance_mm=advance_mm,
                min_tool_z_m=MIN_TOOL_CENTER_Z_M,
                max_joint_step=APPROACH_MAX_JOINT_STEP_DEG)
        if plan is None:
            raise RuntimeError("no bounded 2/3/4 step remains")

        next_pose = list(plan["pose"])
        swept_floor_clearance = transition_fingertip_floor_clearance_mm(
            pose, next_pose)
        if swept_floor_clearance <= FINGERTIP_FLOOR_STOP_MM:
            return {
                "state": "floor-clearance-stop", "pose": pose,
                "distance_mm": distance,
                "fingertip_floor_clearance_mm": swept_floor_clearance,
                "preview": str(PREVIEW),
            }
        report = safety.transition_report(pose, next_pose)
        if not report.safe:
            raise RuntimeError(
                "approach rejected: " + report.explain())
        print(
            f"[sonar-reach] pose234 {pose[1:4]} -> {next_pose[1:4]} "
            f"commanded-advance={adaptive_advance_mm(row_gap):.0f}mm "
            f"visual-progress={plan['progress_mm']:.2f}mm "
            f"clearance={report.minimum_clearance_mm:.1f}mm", flush=True)
        if not execute:
            return {
                "state": "planned", "pose": next_pose,
                "distance_mm": distance, "preview": str(PREVIEW)}
        mover.slow_move(next_pose, final_settle=0.35)
        time.sleep(0.10)

    return {
        "state": "step-limit",
        "pose": client.request({"command": "status"})["pose"],
        "preview": str(PREVIEW),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--grasp", action="store_true")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()
    if args.grasp and not args.run:
        parser.error("--grasp requires --run")
    result = run(
        execute=args.run, allow_grasp=args.grasp, max_steps=args.max_steps)
    print(f"[sonar-reach] RESULT {result}")


if __name__ == "__main__":
    main()
