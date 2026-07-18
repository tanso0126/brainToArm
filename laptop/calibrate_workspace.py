"""Interactive camera->workspace homography calibration.

The homography maps camera pixels to real workspace centimeters. Without it the
arm can't turn "object is at pixel (x,y)" into "move to (cm,cm)". This is the one
camera step you can't skip — but it takes ~1 minute, no props beyond a ruler.

How:
  1. Put the arm's base at the workspace origin (0,0). Decide your axes (e.g.
     +x to the right, +y away from you), matching config's IK frame.
  2. Mark 4+ points on the table at KNOWN cm coordinates (tape + ruler). The
     defaults ask for a simple rectangle; edit WORLD_PTS for your own layout.
  3. Run this; click each point in the live image in the same order. Press 'u'
     to undo, ENTER when done.
  4. Paste the printed CAM_CALIB_IMAGE_PTS into config.py.

Usage:  python calibrate_workspace.py
"""
import numpy as np
import cv2

import config

# The known real-world coordinates (cm) you will click, in order. Edit to match
# the points you actually marked on your table.
WORLD_PTS = [(-15, 15), (15, 15), (15, -15), (-15, -15)]

clicks = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(WORLD_PTS):
        clicks.append((x, y))
        print(f"  point {len(clicks)}: pixel=({x},{y}) -> world={WORLD_PTS[len(clicks)-1]}")


def main():
    cap = cv2.VideoCapture(config.CAM_INDEX)
    if not cap.isOpened():
        print(f"cannot open camera {config.CAM_INDEX}")
        return
    cv2.namedWindow("calib")
    cv2.setMouseCallback("calib", on_mouse)
    print(f"Click these {len(WORLD_PTS)} world points IN ORDER: {WORLD_PTS}")
    print("keys: u=undo, ENTER=finish, ESC=cancel")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if config.CAM_MATRIX is not None and config.CAM_DIST is not None:
            frame = cv2.undistort(frame, np.array(config.CAM_MATRIX),
                                  np.array(config.CAM_DIST))
        for i, (px, py) in enumerate(clicks):
            cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(frame, str(WORLD_PTS[i]), (px + 8, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"{len(clicks)}/{len(WORLD_PTS)} clicked",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("calib", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:                       # ESC
            print("cancelled"); cap.release(); cv2.destroyAllWindows(); return
        if k in (13, 10):                 # ENTER
            break
        if k == ord("u") and clicks:      # undo
            clicks.pop()
    cap.release(); cv2.destroyAllWindows()

    if len(clicks) < len(WORLD_PTS):
        print("not enough points clicked"); return

    img = np.array(clicks, dtype=np.float32)
    wld = np.array(WORLD_PTS, dtype=np.float32)
    H, _ = cv2.findHomography(img, wld)
    # report reprojection error so you know the calibration is good
    proj = cv2.perspectiveTransform(img.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.mean(np.linalg.norm(proj - wld, axis=1)))

    print("\n# paste into config.py:")
    print(f"CAM_CALIB_IMAGE_PTS = {[[int(x), int(y)] for x, y in clicks]}")
    print(f"CAM_CALIB_WORLD_PTS = {[list(p) for p in WORLD_PTS]}")
    print("CAM_CALIBRATED = True")
    print(f"# mean reprojection error: {err:.2f} cm "
          f"({'good' if err < 1.0 else 'high — re-click more carefully'})")


if __name__ == "__main__":
    main()
