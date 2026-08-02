"""brainToArm Windows 통합 운영실 실행기.

한 프로세스가 PolyG-I, MuJoCo, 손목 카메라, Arduino를 소유하고 React
GUI에는 localhost API만 노출합니다. Windows에서는 가능하면 WebView2
창으로 열고, WebView를 사용할 수 없을 때만 기본 브라우저로 대체합니다.
"""

from __future__ import annotations

from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from urllib.parse import urlparse
import argparse
import ctypes
import json
import mimetypes
import os
import socket
import subprocess
import sys
import time
import webbrowser


RELEASE = Path(__file__).resolve().parent
ROOT = RELEASE.parent
LAPTOP = ROOT / "laptop"
DASHBOARD = ROOT / "dashboard"
UI_ROOT = RELEASE / "assets" / "ui"
APP_VERSION = "2.0.0"
for entry in (str(LAPTOP), str(RELEASE), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from control_service import ControlCenterService  # noqa: E402
from eeg_dashboard import (  # noqa: E402
    DashboardHTTPServer,
    DashboardHandler,
    EEGDashboardService,
)


def port_is_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


class UnifiedService:
    def __init__(self):
        self.eeg = EEGDashboardService()
        self.control = ControlCenterService()
        self._monitor_stop = Event()
        self._monitor = Thread(
            target=self._monitor_errp,
            name="control-center-errp-router", daemon=True)
        self._monitor.start()

    def __getattr__(self, name):
        return getattr(self.eeg, name)

    def _monitor_errp(self):
        last_sequence = 0
        while not self._monitor_stop.wait(0.04):
            try:
                status = self.eeg.asynchronous_errp_status()
                sequence = int(status.get("detectionSequence", 0))
                if sequence > last_sequence:
                    last_sequence = sequence
                    if self.control.status()["settings"]["errpEnabled"]:
                        self.control.handle_errp(
                            simulation=(self.eeg.simulation
                                        if self.eeg._simulation is not None
                                        else None))
            except Exception:
                # The dashboard exposes acquisition errors.  A routing poll must
                # never terminate the UI or create a second error source.
                pass

    def close(self):
        self._monitor_stop.set()
        if self._monitor.is_alive():
            self._monitor.join(timeout=1)
        self.control.close()
        self.eeg.close()


class ControlCenterHandler(DashboardHandler):
    def _static(self, request_path):
        root = UI_ROOT.resolve()
        relative = request_path.lstrip("/") or "index.html"
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            self._json({"error": "잘못된 GUI 파일 경로입니다."},
                       HTTPStatus.BAD_REQUEST)
            return
        if not candidate.is_file():
            candidate = root / "index.html"
        if not candidate.is_file():
            self._json(
                {"error": "내장 GUI 파일을 찾지 못했습니다. 앱을 다시 설치하세요."},
                HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        payload = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0]
        if content_type is None:
            content_type = "application/octet-stream"
        if content_type.startswith("text/") or content_type in {
                "application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header(
            "Cache-Control",
            "public, max-age=31536000, immutable"
            if "/assets/" in request_path else "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/control/status":
                self._json(self.service.control.status())
                return
            if parsed.path == "/api/control/camera/frame":
                self._binary(
                    self.service.control.frame_bytes(), "image/jpeg")
                return
            if parsed.path == "/api/control/diagnose":
                self._json(self.service.control.diagnostic())
                return
            if not parsed.path.startswith("/api/"):
                self._static(parsed.path)
                return
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            return
        except Exception as exc:
            self._json(
                {"error": f"{type(exc).__name__}: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/control/"):
            super().do_POST()
            return
        try:
            body = self._body()
            control = self.service.control
            actions = {
                "/api/control/settings": lambda: control.update_settings(body),
                "/api/control/camera/start": lambda: control.start_camera(
                    body.get("camera")),
                "/api/control/camera/stop": control.stop_camera,
                "/api/control/arm/connect": lambda: control.connect_arm(
                    body.get("port")),
                "/api/control/arm/disconnect": control.disconnect_arm,
                "/api/control/arm/stop": control.emergency_stop,
                "/api/control/arm/distance": control.measure_distance,
                "/api/control/arm/home": lambda: control.home(
                    body.get("gripper")),
                "/api/control/arm/jog": lambda: control.jog(
                    body.get("shoulder"), body.get("elbow"),
                    body.get("wrist"), body.get("gripper")),
                "/api/control/objects/detect": control.detect_objects,
                "/api/control/task/start": lambda: control.start_task(
                    body.get("candidateIndex"),
                    body.get("robotWeight")),
                "/api/control/task/stop": control.stop_task,
                "/api/control/task/reject": lambda: control.reject_task(
                    source="GUI 수동"),
                "/api/control/firmware/open": control.open_firmware,
            }
            action = actions.get(parsed.path)
            if action is None:
                self._json(
                    {"error": "통합 운영실 API 경로를 찾을 수 없습니다."},
                    HTTPStatus.NOT_FOUND)
                return
            self._json(action())
        except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except Exception as exc:
            self._json(
                {"error": f"{type(exc).__name__}: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR)


class UnifiedHTTPServer(DashboardHTTPServer):
    def __init__(self, address, service):
        ThreadingHTTPServer.__init__(self, address, ControlCenterHandler)
        self.daemon_threads = True
        self.service = service


def show_window(url):
    if os.name == "nt":
        try:
            import webview
            webview.create_window(
                "brainToArm 통합 운영실", url,
                width=1500, height=940, min_size=(1120, 720))
            webview.start(debug=False)
            return
        except Exception as exc:
            print(
                f"[안내] 전용 창을 열지 못해 기본 브라우저를 사용합니다: {exc}",
                flush=True)
    webbrowser.open(url)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        return


def frozen_self_test():
    modules = {}
    for name in (
            "cv2", "numpy", "scipy", "serial", "hid", "mujoco",
            "ultralytics", "webview"):
        module = __import__(name)
        modules[name] = getattr(module, "__version__", "포함됨")
    if not (UI_ROOT / "index.html").is_file():
        raise RuntimeError("내장 GUI index.html이 없습니다.")
    if not (RELEASE / "assets" / "FastSAM-s.pt").is_file():
        raise RuntimeError("내장 FastSAM 모델이 없습니다.")
    service = ControlCenterService()
    status = service.status()
    service.close()
    print(json.dumps({
        "ok": True,
        "version": APP_VERSION,
        "modules": modules,
        "armMode": status["arm"]["mode"],
        "activeServos": status["arm"]["activeServos"],
        "ui": str(UI_ROOT / "index.html"),
    }, ensure_ascii=False, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port", type=int, default=8765)
    parser.add_argument(
        "--no-window", action="store_true",
        help="GUI 창을 열지 않고 서비스만 실행")
    parser.add_argument("--self-test", action="store_true",
                        help="내장 구성 요소를 검사하고 종료")
    parser.add_argument("--camera-worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--camera", default="auto", help=argparse.SUPPRESS)
    parser.add_argument("--version", action="version",
                        version=f"brainToArm {APP_VERSION}")
    args = parser.parse_args()
    if args.self_test:
        return frozen_self_test()
    if args.camera_worker:
        from windows_camera import publish
        return publish(args.camera, headless=True)
    if not (UI_ROOT / "index.html").is_file():
        raise RuntimeError(
            "내장 GUI가 없습니다. 정식 Windows 설치 파일로 다시 설치하세요.")
    if port_is_open(args.api_port):
        raise RuntimeError(
            f"GUI 포트 {args.api_port}이 이미 사용 중입니다. 기존 "
            "brainToArm 창을 닫고 다시 실행하세요.")
    service = UnifiedService()
    server = UnifiedHTTPServer(("127.0.0.1", args.api_port), service)
    server_thread = Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.2},
        name="control-center-api", daemon=True)
    try:
        server_thread.start()
        url = f"http://127.0.0.1:{args.api_port}/"
        print(f"[통합 운영실] {url}", flush=True)
        if args.no_window:
            while True:
                time.sleep(0.5)
        else:
            show_window(url)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        service.close()
    return 0


def _fatal_startup_error(exc):
    message = (
        f"brainToArm 통합 운영실을 시작하지 못했습니다.\n\n"
        f"{type(exc).__name__}: {exc}\n\n"
        "앱을 다시 설치한 뒤에도 반복되면 아래 로그를 전달하세요.")
    local = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    log_path = local / "brainToArm" / "logs" / "startup-error.txt"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(message + "\n", encoding="utf-8")
        message += f"\n{log_path}"
    except OSError:
        pass
    if os.name == "nt" and "--camera-worker" not in sys.argv:
        ctypes.windll.user32.MessageBoxW(
            0, message, "brainToArm 시작 오류", 0x10)
    else:
        print(message, file=sys.stderr, flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        _fatal_startup_error(exc)
        raise SystemExit(1)
