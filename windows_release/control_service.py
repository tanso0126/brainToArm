"""Windows 통합 GUI가 사용하는 장치·실물 작업 관리 계층입니다.

이 모듈을 import하는 것만으로 카메라나 Uno가 열리지 않습니다. 사용자가
GUI에서 명시적으로 연결 버튼을 눌렀을 때만 장치를 열며, 모든 실물 이동은
1·6번 고정, 2·3·4번 관절 + 5번 집게 구성을 사용합니다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from threading import Event, RLock, Thread
import json
import os
import platform
import subprocess
import sys
import time

import cv2


RELEASE = Path(__file__).resolve().parent
ROOT = RELEASE.parent
LAPTOP = ROOT / "laptop"
RUNTIME = ROOT / "data" / "runtime"
VISION = ROOT / "data" / "vision"
SETTINGS_PATH = ROOT / "data" / "windows_control_center.json"
CAMERA_READY = RUNTIME / "windows_camera.json"
RAW_FRAME = VISION / "wrist_camera_latest_raw.jpg"
ANNOTATED_FRAME = VISION / "windows_control_center_latest.jpg"
FASTSAM_ASSET = RELEASE / "assets" / "FastSAM-s.pt"

for entry in (str(LAPTOP), str(RELEASE), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import config  # noqa: E402
from windows_support import (  # noqa: E402
    Wrist3DofDirectArmClient,
    find_arm_port,
    open_arm,
    port_description,
)


DEFAULT_SETTINGS = {
    "schemaVersion": 1,
    "camera": "auto",
    "armPort": "auto",
    "armMode": "wrist-3dof",
    "candidateIndex": 0,
    "candidateReviewSeconds": 2.5,
    "maxTaskSeconds": 90,
    "errpEnabled": True,
    "tarEnabled": True,
    "autoRejectSimulation": True,
    "autoRejectPhysical": True,
}


@dataclass
class ControlEvent:
    id: int
    kind: str
    text: str
    at: str


def _frame_age(path: Path):
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except FileNotFoundError:
        return None


class ControlCenterService:
    """Own one camera process, one Uno handle, and at most one real task."""

    def __init__(self, settings_path=SETTINGS_PATH):
        self._lock = RLock()
        self._settings_path = Path(settings_path)
        self._settings = self._load_settings()
        self._camera_process = None
        self._camera_index = None
        self._arm = None
        self._arm_client = None
        self._arm_port = None
        self._distance_mm = None
        self._task_thread = None
        self._task_stop = Event()
        self._task_reject = Event()
        self._task_phase = "idle"
        self._task_result = None
        self._active_candidate = None
        self._rejected_candidates = []
        self._detections = []
        self._scene_detector = None
        self._event_id = 0
        self._events = deque(maxlen=120)
        self._last_error = None
        self._add_event(
            "Windows 통합 운영실이 준비되었습니다. 장치는 아직 열지 않았습니다.",
            "info")

    def _load_settings(self):
        values = dict(DEFAULT_SETTINGS)
        try:
            loaded = json.loads(self._settings_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                values.update({key: loaded[key] for key in DEFAULT_SETTINGS
                               if key in loaded})
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return self._validate_settings(values)

    @staticmethod
    def _validate_settings(values):
        result = dict(DEFAULT_SETTINGS)
        result.update(values)
        if result["armMode"] != "wrist-3dof":
            raise ValueError(
                "Windows 통합판은 1·6번을 고정하고 2·3·4번과 집게를 사용합니다.")
        result["candidateIndex"] = max(0, min(9, int(result["candidateIndex"])))
        result["candidateReviewSeconds"] = max(
            0.5, min(15.0, float(result["candidateReviewSeconds"])))
        result["maxTaskSeconds"] = max(
            10, min(300, int(result["maxTaskSeconds"])))
        result["camera"] = str(result["camera"] or "auto")
        result["armPort"] = str(result["armPort"] or "auto").upper()
        for key in (
                "errpEnabled", "tarEnabled",
                "autoRejectSimulation", "autoRejectPhysical"):
            result[key] = bool(result[key])
        return result

    def update_settings(self, values):
        if not isinstance(values, dict):
            raise ValueError("설정은 이름과 값의 묶음이어야 합니다.")
        with self._lock:
            merged = dict(self._settings)
            merged.update(values)
            self._settings = self._validate_settings(merged)
            self._settings_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._settings_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self._settings, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            temporary.replace(self._settings_path)
            self._add_event("설정을 저장했습니다.", "success")
            return dict(self._settings)

    def _add_event(self, text, kind="info"):
        self._event_id += 1
        self._events.appendleft(ControlEvent(
            self._event_id, kind, str(text),
            datetime.now().strftime("%H:%M:%S")))

    def _set_error(self, exc):
        self._last_error = f"{type(exc).__name__}: {exc}"
        self._add_event(self._last_error, "error")

    def _camera_running(self):
        return (self._camera_process is not None
                and self._camera_process.poll() is None)

    def start_camera(self, camera=None):
        with self._lock:
            if self._camera_running():
                return self.status()
            camera = str(camera if camera is not None else self._settings["camera"])
            try:
                CAMERA_READY.unlink()
            except FileNotFoundError:
                pass
            if getattr(sys, "frozen", False):
                command = [
                    sys.executable, "--camera-worker", "--camera", camera,
                ]
            else:
                command = [
                    sys.executable, "-u", str(RELEASE / "windows_camera.py"),
                    "--camera", camera, "--headless",
                ]
            self._camera_process = subprocess.Popen(
                command, cwd=str(ROOT),
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if os.name == "nt" else 0))

        deadline = time.monotonic() + 35.0
        while time.monotonic() < deadline:
            with self._lock:
                process = self._camera_process
            if process is None or process.poll() is not None:
                raise RuntimeError(
                    "카메라 백그라운드 서비스가 시작되지 않았습니다. "
                    "Windows 카메라 권한과 다른 카메라 앱을 확인하세요.")
            try:
                ready = json.loads(CAMERA_READY.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                time.sleep(0.1)
                continue
            if ready.get("pid") == process.pid and (_frame_age(RAW_FRAME) or 99) <= 1:
                with self._lock:
                    self._camera_index = ready.get("cameraIndex")
                    self._settings["camera"] = str(self._camera_index)
                    self._add_event(
                        f"손목 카메라 {self._camera_index}번을 연결했습니다.",
                        "success")
                return self.status()
            time.sleep(0.1)
        self.stop_camera()
        raise TimeoutError(
            "35초 안에 카메라 영상을 받지 못했습니다. USB 연결과 개인 정보 "
            "보호 > 카메라 권한을 확인하세요.")

    def stop_camera(self):
        with self._lock:
            process, self._camera_process = self._camera_process, None
            self._camera_index = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        with self._lock:
            self._add_event("손목 카메라 연결을 해제했습니다.", "info")
        return self.status()

    def connect_arm(self, port=None):
        with self._lock:
            if self._arm_client is not None:
                return self.status()
            if not self._camera_running() or (_frame_age(RAW_FRAME) or 99) > 1:
                raise RuntimeError(
                    "실물 로봇팔보다 손목 카메라를 먼저 연결하세요. "
                    "살아 있는 영상 없이 이동 명령을 허용하지 않습니다.")
        chosen = find_arm_port(port or self._settings["armPort"])
        arm = open_arm(chosen)
        client = Wrist3DofDirectArmClient(arm)
        try:
            pose = client.request({"command": "status"})["pose"]
            distance = client.request({"command": "distance", "samples": 1})
        except Exception:
            client.close()
            raise
        with self._lock:
            self._arm = arm
            self._arm_client = client
            self._arm_port = chosen
            self._distance_mm = distance.get("distanceMm")
            self._settings["armPort"] = chosen
            self._add_event(
                f"Arduino {chosen} 연결 · 현재 자세 {pose} · 초음파 "
                f"{distance.get('distanceMm') or '응답 없음'} mm", "success")
        return self.status()

    def disconnect_arm(self):
        self.stop_task(wait=True)
        with self._lock:
            client, self._arm_client = self._arm_client, None
            self._arm = None
            self._arm_port = None
            self._distance_mm = None
        if client is not None:
            try:
                client.request({"command": "stop"})
            except Exception:
                pass
            client.close()
        with self._lock:
            self._add_event("Arduino 연결을 해제했습니다.", "info")
        return self.status()

    def emergency_stop(self):
        self._task_stop.set()
        with self._lock:
            client = self._arm_client
            self._task_phase = "emergency-stop"
            self._add_event(
                "긴급정지: 남은 서보 이동을 취소했습니다. 실제 위험 시 "
                "외부 서보 전원도 분리하세요.", "error")
        if client is not None:
            client.request({"command": "stop"})
        return self.status()

    def measure_distance(self):
        with self._lock:
            client = self._arm_client
            task_running = (
                self._task_thread is not None and self._task_thread.is_alive())
        if client is None:
            raise RuntimeError("초음파 측정 전에 Arduino를 연결하세요.")
        if task_running:
            raise RuntimeError("자동 접근 중에는 제어기가 초음파를 사용 중입니다.")
        response = client.request({"command": "distance", "samples": 3})
        with self._lock:
            self._distance_mm = response.get("distanceMm")
            self._add_event(
                "초음파 거리: " + (
                    f"{self._distance_mm:.1f} mm" if self._distance_mm is not None
                    else "유효한 반사 없음"),
                "success" if self._distance_mm is not None else "error")
        return response

    def jog(self, shoulder, elbow, wrist, gripper):
        with self._lock:
            if self._task_thread is not None and self._task_thread.is_alive():
                raise RuntimeError("자동 작업 중에는 수동 관절 이동을 할 수 없습니다.")
            client = self._arm_client
        if client is None:
            raise RuntimeError("먼저 Arduino 연결 버튼을 누르세요.")
        pose = client.request({"command": "status"})["pose"]
        pose[config.J_SHOULDER] = int(shoulder)
        pose[config.J_ELBOW] = int(elbow)
        pose[config.J_WRIST] = int(wrist)
        pose[config.J_GRIP] = int(gripper)
        response = client.request({
            "command": "move", "pose": pose, "require_camera": True,
        })
        with self._lock:
            self._add_event(
                f"수동 이동: 2번 {shoulder}° · 3번 {elbow}° · "
                f"4번 {wrist}° · 집게 {gripper}°", "move")
        return response

    def home(self, gripper=None):
        with self._lock:
            client = self._arm_client
        if client is None:
            raise RuntimeError("먼저 Arduino를 연결하세요.")
        current = client.request({"command": "status"})["pose"]
        target = list(config.HOME_POSE)
        target[config.J_GRIP] = (
            current[config.J_GRIP] if gripper is None else int(gripper))
        response = client.request({
            "command": "move", "pose": target,
            "require_camera": True,
        })
        with self._lock:
            self._add_event("검증된 2·3·4축 구성으로 HOME에 복귀했습니다.", "success")
        return response

    def detect_objects(self):
        frame = cv2.imread(str(RAW_FRAME))
        if frame is None or (_frame_age(RAW_FRAME) or 99) > 1:
            raise RuntimeError("최신 손목 카메라 프레임이 없습니다.")
        config.PLANAR_VISION_MODEL = str(FASTSAM_ASSET)
        config.PLANAR_VISION_DEVICE = "cpu"
        from floor_grasp import WristSceneDetector
        with self._lock:
            if self._scene_detector is None:
                self._scene_detector = WristSceneDetector()
            detector = self._scene_detector
        scene, _observation = detector.scene(frame)
        annotated = frame.copy()
        detections = []
        for index, item in enumerate(scene.ranked):
            x, y, width, height = item.bbox
            cv2.rectangle(
                annotated, (x, y), (x + width, y + height), (0, 196, 255), 3)
            cv2.putText(
                annotated, f"#{index + 1}", (x, max(28, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 196, 255), 2,
                cv2.LINE_AA)
            detections.append({
                "index": index,
                "center": [round(float(value), 1) for value in item.center],
                "bbox": list(item.bbox),
                "area": round(float(item.area), 1),
                "confidence": round(float(item.confidence), 3),
            })
        ANNOTATED_FRAME.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ANNOTATED_FRAME), annotated)
        with self._lock:
            self._detections = detections
            self._add_event(
                f"카메라에서 파지 후보 {len(detections)}개를 분리했습니다.",
                "success" if detections else "error")
        return {"detections": detections}

    def start_task(self, candidate_index=None, autonomy=None):
        with self._lock:
            if self._arm_client is None:
                raise RuntimeError("실물 자동 작업 전에 Arduino를 연결하세요.")
            if not self._camera_running():
                raise RuntimeError("실물 자동 작업 전에 손목 카메라를 연결하세요.")
            if self._task_thread is not None and self._task_thread.is_alive():
                raise RuntimeError("이미 실물 자동 작업이 실행 중입니다.")
            self._task_stop.clear()
            self._task_reject.clear()
            self._task_result = None
            self._rejected_candidates = []
            self._active_candidate = max(
                0, int(self._settings["candidateIndex"]
                       if candidate_index is None else candidate_index))
            self._task_phase = "candidate-review"
            self._task_thread = Thread(
                target=self._task_worker,
                args=(float(autonomy) if autonomy is not None else None,),
                name="windows-physical-task", daemon=True)
            self._task_thread.start()
            self._add_event("실물 다중 후보 자동 작업을 시작했습니다.", "move")
        return self.status()

    def _task_worker(self, autonomy):
        try:
            detected = self.detect_objects()["detections"]
            if not detected:
                raise RuntimeError("파지 가능한 물체 후보가 없습니다.")
            with self._lock:
                selected = min(self._active_candidate or 0, len(detected) - 1)
            # TAR가 높을수록 로봇 주도권이 크므로 후보 검토 시간을 줄이고,
            # 낮을수록 사람이 ErrP를 낼 시간을 늘립니다.
            review = float(self._settings["candidateReviewSeconds"])
            if self._settings["tarEnabled"] and autonomy is not None:
                review *= 1.35 - 0.7 * max(0.0, min(1.0, autonomy))
            while True:
                with self._lock:
                    self._active_candidate = selected
                    self._task_phase = "candidate-review"
                    self._add_event(
                        f"후보 #{selected + 1} 제시 · {review:.1f}초 ErrP 검토",
                        "info")
                deadline = time.monotonic() + review
                while time.monotonic() < deadline and not self._task_stop.is_set():
                    if self._task_reject.wait(0.04):
                        self._task_reject.clear()
                        with self._lock:
                            self._rejected_candidates.append(selected)
                        remaining = [item["index"] for item in detected
                                     if item["index"] not in self._rejected_candidates]
                        if not remaining:
                            with self._lock:
                                self._rejected_candidates = []
                                self._add_event(
                                    "모든 후보가 거부되어 거부 기억을 초기화했습니다.",
                                    "info")
                            remaining = [item["index"] for item in detected]
                        selected = remaining[0]
                        break
                else:
                    break
                if self._task_stop.is_set():
                    break

            if self._task_stop.is_set():
                result = {"state": "stopped", "mode": "wrist-3dof"}
            else:
                with self._lock:
                    self._task_phase = "approaching"
                    client = self._arm_client
                import realtime_visual_servo as controller
                original_client = controller.ArmSessionClient
                controller.ArmSessionClient = lambda: client
                try:
                    result = controller.run(
                        execute=True,
                        allow_grasp=True,
                        max_seconds=float(self._settings["maxTaskSeconds"]),
                        stop_event=self._task_stop,
                        candidate_rank=selected,
                    )
                finally:
                    controller.ArmSessionClient = original_client
            with self._lock:
                self._task_result = result
                self._task_phase = (
                    "completed" if result.get("state") == "home-after-grasp"
                    else "stopped" if result.get("state") == "stopped"
                    else "failed")
                self._add_event(
                    f"실물 작업 종료: {result.get('state', 'unknown')}",
                    "success" if self._task_phase == "completed" else "error")
        except Exception as exc:
            with self._lock:
                self._task_phase = "failed"
                self._task_result = {"state": "error", "error": str(exc)}
                self._set_error(exc)

    def reject_task(self, source="manual"):
        with self._lock:
            running = self._task_thread is not None and self._task_thread.is_alive()
            phase = self._task_phase
            if not running:
                raise RuntimeError("현재 거부할 실물 작업 후보가 없습니다.")
            self._add_event(f"{source} 거부 신호를 받았습니다.", "error")
            if phase == "candidate-review":
                self._task_reject.set()
            else:
                # 이미 이동 중인 다른 물체를 추측해 계속 가는 것보다 현재
                # 목표를 취소하고 정지하는 것이 실물 팔의 안전한 의미다.
                self._task_stop.set()
                client = self._arm_client
                if client is not None:
                    client.request({"command": "stop"})
        return self.status()

    def stop_task(self, wait=False):
        self._task_stop.set()
        with self._lock:
            thread = self._task_thread
            client = self._arm_client
        if client is not None and thread is not None and thread.is_alive():
            try:
                client.request({"command": "stop"})
            except Exception:
                pass
        if wait and thread is not None and thread.is_alive():
            thread.join(timeout=3)
        with self._lock:
            if self._task_phase not in {"idle", "completed", "failed"}:
                self._task_phase = "stopped"
                self._add_event("실물 자동 작업을 중지했습니다.", "info")
        return self.status()

    def handle_errp(self, simulation=None):
        """Route one confirmed asynchronous ErrP to enabled targets."""
        routed = []
        if self._settings["autoRejectSimulation"] and simulation is not None:
            try:
                status = simulation.status()
                if status.get("activeId") or status.get("lastDeliveredId"):
                    simulation.reject()
                    routed.append("simulation")
            except RuntimeError:
                pass
        if self._settings["autoRejectPhysical"]:
            try:
                self.reject_task(source="PolyG-I ErrP")
                routed.append("physical")
            except RuntimeError:
                pass
        if routed:
            with self._lock:
                self._add_event(
                    "ErrP 자동 반영: " + ", ".join(routed), "error")
        return routed

    def frame_bytes(self):
        source = ANNOTATED_FRAME if (
            ANNOTATED_FRAME.exists()
            and (_frame_age(ANNOTATED_FRAME) or 99) <= 5) else RAW_FRAME
        body = source.read_bytes()
        if not body:
            raise RuntimeError("카메라 프레임 파일이 비어 있습니다.")
        return body

    def diagnostic(self):
        from serial.tools import list_ports
        ports = [{
            "device": item.device,
            "description": port_description(item),
        } for item in list_ports.comports()]
        checks = {
            "windows": os.name == "nt",
            "python": platform.python_version(),
            "fastsam": FASTSAM_ASSET.is_file(),
            "cameraFrame": (_frame_age(RAW_FRAME) or 99) <= 1,
            "armPorts": ports,
            "firmware": (ROOT / "firmware" / "arm_controller"
                         / "arm_controller.ino").is_file(),
        }
        self._add_event("움직임 없는 장치 진단을 완료했습니다.", "success")
        return checks

    def open_firmware(self):
        """Open the shipped Uno sketch without requiring a terminal."""
        sketch = ROOT / "firmware" / "arm_controller" / "arm_controller.ino"
        if not sketch.is_file():
            raise FileNotFoundError(
                "앱에 Arduino 펌웨어가 없습니다. 정식 설치 파일로 다시 "
                "설치하세요.")
        try:
            if os.name == "nt":
                os.startfile(str(sketch))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(sketch)])
            else:
                subprocess.Popen(["xdg-open", str(sketch)])
        except OSError as exc:
            raise RuntimeError(
                "Arduino IDE에서 펌웨어를 열지 못했습니다. Arduino IDE를 "
                "설치한 뒤 다시 누르세요.") from exc
        with self._lock:
            self._add_event(
                "Arduino IDE로 현재 Uno 펌웨어를 열었습니다. 보드와 COM "
                "포트를 선택한 뒤 업로드하세요.", "success")
        return {"opened": True, "path": str(sketch)}

    def status(self):
        with self._lock:
            camera_running = self._camera_running()
            task_running = (
                self._task_thread is not None and self._task_thread.is_alive())
            client = self._arm_client
            payload = {
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "python": platform.python_version(),
                    "windows": os.name == "nt",
                },
                "settings": dict(self._settings),
                "camera": {
                    "running": camera_running,
                    "index": self._camera_index,
                    "frameAgeSeconds": _frame_age(RAW_FRAME),
                    "previewUrl": "/api/control/camera/frame",
                },
                "arm": {
                    "connected": client is not None,
                    "port": self._arm_port,
                    "mode": "wrist-3dof",
                    "activeServos": [2, 3, 4, 5],
                    "fixedServos": [1, 6],
                    "pose": None,
                    "distanceMm": self._distance_mm,
                },
                "task": {
                    "running": task_running,
                    "phase": self._task_phase,
                    "activeCandidate": self._active_candidate,
                    "rejectedCandidates": list(self._rejected_candidates),
                    "result": self._task_result,
                },
                "detections": list(self._detections),
                "events": [asdict(item) for item in self._events],
                "lastError": self._last_error,
            }
        if client is not None and not task_running:
            try:
                payload["arm"]["pose"] = client.request(
                    {"command": "status"})["pose"]
            except Exception as exc:
                payload["arm"]["error"] = str(exc)
        return payload

    def close(self):
        try:
            self.disconnect_arm()
        finally:
            self.stop_camera()
