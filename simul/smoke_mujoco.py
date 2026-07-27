"""Headless visual/kinematic smoke test for the brainToArm MuJoCo model."""

from pathlib import Path
import argparse

import cv2
import numpy as np

try:
    from .mujoco_robot import (
        load_model,
        model_summary,
        place_target_below_tool,
        render_rgb,
        set_servo_pose,
        site_position,
    )
except ImportError:
    from mujoco_robot import (
        load_model,
        model_summary,
        place_target_below_tool,
        render_rgb,
        set_servo_pose,
        site_position,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "generated" / "mujoco_smoke.png"


def run(output=DEFAULT_OUTPUT):
    model, data, spec = load_model()
    poses = (
        ("HOME / wrist", spec.home_servo_deg, "wrist"),
        ("FLOOR HOVER / wrist", spec.hover_servo_deg, "wrist"),
        ("FLOOR GRASP / wrist", spec.grasp_servo_deg, "wrist"),
        ("FLOOR HOVER / overview", spec.hover_servo_deg, "overview"),
    )
    panels = []
    heights = []
    for label, pose, camera in poses:
        set_servo_pose(model, data, pose, spec=spec)
        place_target_below_tool(model, data)
        image = render_rgb(model, data, camera=camera, width=384, height=216)
        if image.std() < 5:
            raise RuntimeError(f"near-uniform render from {camera}")
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, 0), (383, 27), (10, 14, 20), -1)
        cv2.putText(bgr, label, (10, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (240, 245, 250), 1, cv2.LINE_AA)
        panels.append(bgr)
        heights.append((label, float(site_position(model, data, "tool_center")[2])))
    sheet = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"could not write {output}")
    print("MODEL", model_summary(model))
    print("TOOL_Z_M", ", ".join(f"{name}={height:.4f}" for name, height in heights))
    print(f"RENDER_OK {sheet.shape[1]}x{sheet.shape[0]} -> {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
