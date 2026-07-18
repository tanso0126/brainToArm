"""One-time lens calibration for a cheap/phone/laptop camera (fisheye removal).

Print a checkerboard (default 9x6 inner corners), tape it flat, and show it to
the overhead camera from several angles/distances. Press SPACE to capture each
good view (~15 views), ESC to finish. It prints CAM_MATRIX and CAM_DIST to paste
into config.py. After that, vision.py undistorts every frame automatically.

Cheap cameras bend straight lines near the edges; this removes that so the
pixel->workspace homography stays accurate across the whole frame.

Usage:
    python camera_calibrate.py            # 9x6 board, camera index from config
    python camera_calibrate.py 7 5        # custom inner-corner count
"""
import sys
import numpy as np
import cv2

import config


def main():
    cols = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    objpoints, imgpoints = [], []
    cap = cv2.VideoCapture(config.CAM_INDEX)
    if not cap.isOpened():
        print(f"cannot open camera {config.CAM_INDEX}")
        return
    print("SPACE = capture a view, ESC = finish (aim for ~15 views)")
    shape = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, (cols, rows))
        disp = frame.copy()
        if found:
            cv2.drawChessboardCorners(disp, (cols, rows), corners, found)
        cv2.putText(disp, f"views: {len(objpoints)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("calib", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:                        # ESC
            break
        if k == 32 and found:              # SPACE
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"captured view {len(objpoints)}")
    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 5:
        print("need at least ~5 views; try again")
        return
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    print("\n# paste into config.py:")
    print("CAM_MATRIX =", mtx.tolist())
    print("CAM_DIST =", dist.tolist())
    print(f"# reprojection error: {ret:.3f}px")


if __name__ == "__main__":
    main()
