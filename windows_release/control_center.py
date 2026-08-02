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


def start_ui(port):
    if port_is_open(port):
        raise RuntimeError(
            f"GUI 포트 {port}이 이미 사용 중입니다. 기존 brainToArm 창을 "
            "닫고 다시 실행하세요.")
    if not (DASHBOARD / "node_modules").is_dir():
        raise RuntimeError(
            "GUI 구성 요소가 설치되지 않았습니다. SETUP_WINDOWS.bat을 "
            "먼저 한 번 실행하세요.")
    command = ["npm.cmd" if os.name == "nt" else "npm", "run", "start",
               "--", "--port", str(port)]
    process = subprocess.Popen(
        command, cwd=str(DASHBOARD),
        creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"GUI 서비스가 오류 코드 {process.returncode}로 종료되었습니다. "
                "SETUP_WINDOWS.bat을 다시 실행하세요.")
        if port_is_open(port):
            return process
        time.sleep(0.1)
    process.terminate()
    raise TimeoutError("30초 안에 GUI가 준비되지 않았습니다.")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port", type=int, default=8765)
    parser.add_argument("--ui-port", type=int, default=3000)
    parser.add_argument(
        "--no-window", action="store_true",
        help="GUI 창을 열지 않고 서비스만 실행")
    args = parser.parse_args()
    service = UnifiedService()
    server = UnifiedHTTPServer(("127.0.0.1", args.api_port), service)
    server_thread = Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.2},
        name="control-center-api", daemon=True)
    ui_process = None
    try:
        server_thread.start()
        ui_process = start_ui(args.ui_port)
        url = f"http://127.0.0.1:{args.ui_port}/"
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
        if ui_process is not None and ui_process.poll() is None:
            ui_process.terminate()
            try:
                ui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ui_process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
