"""Overhead camera -> candidate objects + arm tip, in workspace (cm) coords.

MARKERLESS by default — no stickers, no printed markers, no checkerboard, no
special backdrop. Setup burden = one keypress on an empty table at startup.

Real pipeline (config.CAM_MOCK = False), method "bgsub" (default):
  1. Once, with the arm parked out of the way, snapshot the empty scene as the
     background (learn_background()).
  2. Objects = foreground blobs vs that background (difference-based, so it
     tolerates a cheap sensor's color drift). Detected while the arm is parked.
  3. Arm tip = the foreground blob that appears when the arm enters, taking the
     point farthest from the arm's base — no marker on the gripper.
  4. Pixel -> workspace cm via the 4-point homography.

Other methods if you prefer: OBJECT_METHOD "aruco" (markers) or "hsv" (color).
Lens undistort is optional (only if CAM_MATRIX is set).

Mock pipeline (config.CAM_MOCK = True, default): fixed ambiguous scene from
sim.py so the whole stack runs with no camera. Return shapes are identical, so
orchestrator/policy never change.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import config
import sim

try:
    import cv2
    import numpy as np
    _HAVE_CV = True
except ImportError:
    _HAVE_CV = False


@dataclass
class Detection:
    obj_id: int
    label: str
    x: float
    y: float
    meta: dict = field(default_factory=dict)


class Vision:
    def __init__(self, mock=None):
        self.mock = config.CAM_MOCK if mock is None else mock
        self.cap = None
        self.H = None            # pixel->world homography
        self.Hinv = None         # world->pixel
        self._aruco = None
        self.background = None    # reference frame for markerless bgsub
        self._object_boxes = []   # pixel bounding boxes of detected objects
        self._base_px = None      # arm base location in pixels
        self._yolo = None
        if not self.mock:
            if not _HAVE_CV:
                raise RuntimeError("opencv-python not installed; pip install opencv-python")
            self._init_camera()
            if config.OBJECT_METHOD == "yolo":
                self._init_yolo()

    def _init_yolo(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("OBJECT_METHOD='yolo' needs: pip install ultralytics")
        self._yolo = YOLO(config.YOLO_WEIGHTS)

    def _init_camera(self):
        self.cap = cv2.VideoCapture(config.CAM_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera index {config.CAM_INDEX}")
        img = np.array(config.CAM_CALIB_IMAGE_PTS, dtype=np.float32)
        wld = np.array(config.CAM_CALIB_WORLD_PTS, dtype=np.float32)
        self.H, _ = cv2.findHomography(img, wld)
        self.Hinv, _ = cv2.findHomography(wld, img)
        self._base_px = self._world_to_px(0.0, 0.0)   # arm base at workspace origin
        if config.OBJECT_METHOD == "aruco":
            adict = getattr(cv2.aruco, config.ARUCO_DICT)
            self._aruco = cv2.aruco.getPredefinedDictionary(adict)

    # ---- coordinate transforms ----
    def _px_to_world(self, px, py) -> Tuple[float, float]:
        pt = np.array([[[px, py]]], dtype=np.float32)
        w = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(w[0]), float(w[1])

    def _world_to_px(self, x, y) -> Tuple[float, float]:
        pt = np.array([[[x, y]]], dtype=np.float32)
        p = cv2.perspectiveTransform(pt, self.Hinv)[0][0]
        return float(p[0]), float(p[1])

    def _grab(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        if config.CAM_MATRIX is not None and config.CAM_DIST is not None:
            frame = cv2.undistort(frame, np.array(config.CAM_MATRIX),
                                  np.array(config.CAM_DIST))
        return frame

    def _grab_stable(self, n=5):
        # average a few frames to beat sensor noise on a cheap camera
        acc = None
        for _ in range(n):
            f = self._grab().astype("float32")
            acc = f if acc is None else acc + f
        return (acc / n).astype("uint8")

    # ---- markerless: background subtraction ----
    def learn_background(self):
        """Call once with the arm parked and the empty table in view."""
        if self.mock:
            return
        self.background = self._grab_stable()
        print("[vision] background learned")

    def _foreground_mask(self, frame):
        if self.background is None:
            self.learn_background()
        diff = cv2.absdiff(frame, self.background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, config.BGSUB_THRESH, 255, cv2.THRESH_BINARY)
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    # ---- public API ----
    def detect(self) -> List[Detection]:
        if self.mock:
            return [Detection(i, lbl, x, y, m)
                    for i, (lbl, x, y, m) in enumerate(sim.WORLD.objects)]
        if config.OBJECT_METHOD == "aruco":
            return self._detect_aruco()
        if config.OBJECT_METHOD == "hsv":
            return self._detect_hsv()
        if config.OBJECT_METHOD == "yolo":
            return self._detect_yolo()
        return self._detect_bgsub()

    def _detect_yolo(self) -> List[Detection]:
        # NOTE: generic COCO YOLO gives class labels but CANNOT distinguish
        # "big vs small nail" (not COCO classes). For your task, detection should
        # give POSITIONS and leave the CHOICE ambiguous (that's what the EEG veto
        # resolves) — so bgsub is usually the better fit. Train custom weights
        # only if you truly need semantic classes.
        frame = self._grab()
        res = self._yolo.predict(frame, conf=config.YOLO_CONF,
                                 classes=config.YOLO_CLASSES, verbose=False)[0]
        names = res.names
        dets = []
        self._object_boxes = []
        for i, box in enumerate(res.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            wx, wy = self._px_to_world(cx, cy)
            self._object_boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            label = names[int(box.cls[0])]
            dets.append(Detection(i, label, wx, wy,
                                  {"conf": float(box.conf[0]), "px": (cx, cy)}))
        return dets

    def _detect_bgsub(self) -> List[Detection]:
        """Objects = foreground blobs vs the empty-table background. Run with the
        arm parked so only the objects are foreground."""
        frame = self._grab_stable()
        mask = self._foreground_mask(frame)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        self._object_boxes = []
        oid = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < config.OBJECT_MIN_AREA:
                continue
            M = cv2.moments(c)
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            wx, wy = self._px_to_world(cx, cy)
            self._object_boxes.append(cv2.boundingRect(c))
            dets.append(Detection(oid, f"obj{oid}", wx, wy,
                                  {"area_px": area, "px": (cx, cy)}))
            oid += 1
        return dets

    def arm_tip(self) -> Optional[Tuple[float, float]]:
        if self.mock:
            return sim.WORLD.tip()
        if config.OBJECT_METHOD == "aruco":
            return self._tip_aruco()
        return self._tip_bgsub()

    def _tip_bgsub(self) -> Optional[Tuple[float, float]]:
        """Arm tip without any marker: the arm is the large foreground blob that
        wasn't there in the empty background and isn't one of the known object
        boxes; its tip is the blob point farthest from the arm base."""
        frame = self._grab()
        mask = self._foreground_mask(frame)
        # erase known object regions so only the arm remains
        for (x, y, w, h) in self._object_boxes:
            mask[y:y + h, x:x + w] = 0
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > config.ARM_MIN_AREA]
        if not cnts:
            return None
        arm = max(cnts, key=cv2.contourArea)
        bx, by = self._base_px
        pts = arm.reshape(-1, 2)
        d2 = (pts[:, 0] - bx) ** 2 + (pts[:, 1] - by) ** 2
        tip_px = pts[int(d2.argmax())]                 # farthest from base = tip
        return self._px_to_world(float(tip_px[0]), float(tip_px[1]))

    # ---- optional marker path (kept for those who want it) ----
    def _detect_aruco(self) -> List[Detection]:
        frame = self._grab()
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self._aruco)
        dets = []
        if ids is None:
            return dets
        for c, i in zip(corners, ids.flatten()):
            i = int(i)
            if i == config.ARM_TIP_ARUCO_ID or i not in config.OBJECT_ARUCO:
                continue
            cx, cy = float(c[0][:, 0].mean()), float(c[0][:, 1].mean())
            wx, wy = self._px_to_world(cx, cy)
            dets.append(Detection(i, config.OBJECT_ARUCO[i], wx, wy, {"aruco": i}))
        return dets

    def _tip_aruco(self) -> Optional[Tuple[float, float]]:
        frame = self._grab()
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self._aruco)
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                if int(i) == config.ARM_TIP_ARUCO_ID:
                    cx, cy = float(c[0][:, 0].mean()), float(c[0][:, 1].mean())
                    return self._px_to_world(cx, cy)
        return None

    def _detect_hsv(self) -> List[Detection]:
        frame = self._grab()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dets, oid = [], 0
        for label, (lo, hi) in config.OBJECT_HSV.items():
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < config.OBJECT_MIN_AREA:
                    continue
                M = cv2.moments(c)
                cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                wx, wy = self._px_to_world(cx, cy)
                dets.append(Detection(oid, label, wx, wy))
                oid += 1
        return dets

    # ---- grasp verification (markerless): did the object leave its spot? ----
    def location_clear(self, det) -> bool:
        """True if the object's original location is now empty — i.e. the arm
        actually picked it up. Call after lifting, with the arm retracted away
        from that spot. Uses background subtraction at the object's box.
        In mock, reads the sim world (object removed on a successful lift)."""
        if self.mock:
            return sim.WORLD.index_of(det.label, det.x, det.y) is None
        px = det.meta.get("px")
        if px is None:
            return True
        cx, cy = px
        frame = self._grab_stable(3)
        mask = self._foreground_mask(frame)
        r = 25   # px half-window around the object's old centroid
        y0, y1 = max(0, int(cy - r)), int(cy + r)
        x0, x1 = max(0, int(cx - r)), int(cx + r)
        patch = mask[y0:y1, x0:x1]
        occupied_frac = (patch > 0).mean() if patch.size else 0.0
        return occupied_frac < 0.15   # spot mostly matches empty background

    def close(self):
        if self.cap is not None:
            self.cap.release()


if __name__ == "__main__":
    v = Vision()
    print("arm tip:", v.arm_tip())
    for d in v.detect():
        print(d)
    v.close()
