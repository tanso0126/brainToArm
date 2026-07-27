"""Persistent Uno owner: one USB open, many arm commands, no repeated reset.

Opening the Uno serial port toggles DTR and resets the board to HOME.  Short
one-shot scripts therefore create the inefficient HOME -> requested-pose motion
on every command.  This process opens the port exactly once and exposes a local
Unix-domain socket.  All later tools talk to the socket instead of reopening
the board.

Examples::

    python3 laptop/arm_session.py serve --floor hover
    python3 laptop/arm_session.py status
    python3 laptop/arm_session.py check 90 124 90 180 90 170
    python3 laptop/arm_session.py floor grasp 90
    python3 laptop/arm_session.py move 90 124 90 180 90 170
    python3 laptop/arm_session.py shutdown
"""

from pathlib import Path
import argparse
import json
import os
import signal
import socket
import sys
import time

import config
from arm_safety import PhysicalArmSafety
from arm_serial import ArmSerial
from floor_motion import floor_pose, floor_waypoints


ROOT = Path(__file__).resolve().parents[1]
WRIST_RAW_FRAME = ROOT / "data" / "vision" / "wrist_camera_latest_raw.jpg"
DEEPEST_TABLE_TOUCH_Z_M = -0.020


def session_socket_path(value=None):
    path = Path(value or config.ARM_SESSION_SOCKET)
    return path if path.is_absolute() else ROOT / path


def _json_line(payload):
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode()


class ArmSessionServer:
    def __init__(self, path=None, arm=None, safety=None):
        self.path = session_socket_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.arm = arm or ArmSerial()
        self.safety = safety or PhysicalArmSafety()
        self.running = True
        self.listener = None

    def _validated_sequence(self, poses):
        checked = []
        for pose in poses:
            if not isinstance(pose, list) or len(pose) != config.N_JOINTS:
                raise ValueError("each pose needs six joint values")
            values = []
            for joint, value in enumerate(pose):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"joint {joint + 1} must be numeric")
                if not float(value).is_integer():
                    raise ValueError(f"joint {joint + 1} must be an integer")
                value = int(value)
                if not config.SERVO_MIN[joint] <= value <= config.SERVO_MAX[joint]:
                    raise ValueError(
                        f"joint {joint + 1}={value} outside configured limits")
                values.append(value)
            checked.append(values)
        if not checked:
            raise ValueError("move sequence cannot be empty")
        return checked

    @staticmethod
    def _assert_camera_live(path=WRIST_RAW_FRAME):
        try:
            age = time.time() - path.stat().st_mtime
        except FileNotFoundError as exc:
            raise RuntimeError(
                "autonomous motion requires the wrist-camera publisher") from exc
        if age < -1.0 or age > config.WRIST_CAMERA_MAX_FRAME_AGE_S:
            raise RuntimeError(
                f"autonomous motion rejected: wrist frame is stale ({age:.1f}s old)")

    def _move_sequence(self, poses, timeout=15.0, settle_s=0.0,
                       require_camera=False, safety=None):
        poses = self._validated_sequence(poses)
        current = self.arm.status()
        safety = self.safety if safety is None else safety
        for pose in poses:
            if require_camera:
                self._assert_camera_live()
            report = safety.transition_report(current, pose)
            if not report.safe:
                raise RuntimeError(
                    "motion rejected before serial write: " + report.explain())
            self.arm.send_angles(pose)
            self.arm.wait_done(timeout=float(timeout))
            if settle_s:
                time.sleep(float(settle_s))
            current = pose
        return self.arm.status()

    def handle(self, request):
        command = request.get("command")
        if command == "ping":
            return {"ok": bool(self.arm.ping()), "pid": os.getpid()}
        if command == "status":
            return {"ok": True, "pose": self.arm.status(), "pid": os.getpid()}
        if command == "check":
            target = self._validated_sequence([request.get("pose")])[0]
            current = self.arm.status()
            report = self.safety.transition_report(current, target)
            return {
                "ok": True,
                "safe": report.safe,
                "current": current,
                "target": target,
                "minimum_clearance_mm": report.minimum_clearance_mm,
                "explanation": report.explain(),
            }
        if command == "move":
            pose = self._move_sequence(
                [request.get("pose")],
                timeout=request.get("timeout", 15.0),
                settle_s=request.get("settle_s", 0.0),
                require_camera=bool(request.get("require_camera", False)))
            return {"ok": True, "pose": pose}
        if command == "table_touch_move":
            table_z_m = float(request.get("table_z_m"))
            if not DEEPEST_TABLE_TOUCH_Z_M <= table_z_m <= 0.0:
                raise ValueError(
                    "table-touch z floor must be between "
                    f"{DEEPEST_TABLE_TOUCH_Z_M * 1000:.0f} and 0 mm")
            # Calibration may probe below the old z=0 estimate, but it does not
            # bypass collision checking: only the table plane moves. Base,
            # mast, self, camera, and swept-trajectory checks all remain active.
            calibration_safety = PhysicalArmSafety(table_z_m=table_z_m)
            pose = self._move_sequence(
                [request.get("pose")],
                timeout=request.get("timeout", 15.0),
                settle_s=request.get("settle_s", 0.0),
                require_camera=bool(request.get("require_camera", False)),
                safety=calibration_safety)
            return {"ok": True, "pose": pose, "table_z_m": table_z_m}
        if command == "sequence":
            pose = self._move_sequence(
                request.get("poses"),
                timeout=request.get("timeout", 15.0),
                settle_s=request.get("settle_s", 0.0),
                require_camera=bool(request.get("require_camera", False)))
            return {"ok": True, "pose": pose}
        if command == "floor":
            level = request.get("level", "hover")
            target_elbow = int(request.get(
                "elbow", config.FLOOR_REFERENCE_ELBOW))
            current = self.arm.status()
            start_elbow = current[config.J_ELBOW]
            if not (config.FLOOR_ELBOW_RANGE[0] <= start_elbow
                    <= config.FLOOR_ELBOW_RANGE[1]):
                poses = [floor_pose(target_elbow, level)]
            else:
                poses = floor_waypoints(
                    start_elbow, target_elbow, level,
                    request.get("step"), request.get("gripper"))
            pose = self._move_sequence(
                poses, timeout=request.get("timeout", 15.0),
                settle_s=request.get("settle_s", 0.0),
                require_camera=bool(request.get("require_camera", False)))
            return {"ok": True, "pose": pose, "level": level}
        if command == "shutdown":
            self.running = False
            return {"ok": True, "pose": self.arm.status(), "shutdown": True}
        raise ValueError(f"unknown arm-session command {command!r}")

    def serve(self, startup_poses=None):
        if self.path.exists():
            try:
                response = ArmSessionClient(self.path).request(
                    {"command": "ping"}, timeout=0.5)
            except Exception:
                self.path.unlink()
            else:
                raise RuntimeError(
                    f"arm session already running at {self.path}: {response}")
        if startup_poses:
            self._move_sequence(startup_poses, settle_s=1.0)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener = listener
        listener.bind(str(self.path))
        os.chmod(self.path, 0o600)
        listener.listen(4)
        listener.settimeout(0.5)
        print(
            f"[arm-session] READY pid={os.getpid()} socket={self.path} "
            f"pose={self.arm.status()}", flush=True)
        try:
            while self.running:
                try:
                    connection, _address = listener.accept()
                except socket.timeout:
                    continue
                with connection:
                    file = connection.makefile("rwb")
                    raw = file.readline(1024 * 1024)
                    try:
                        request = json.loads(raw.decode())
                        response = self.handle(request)
                    except Exception as exc:
                        response = {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    # A client that disconnected mid-request (e.g. an interrupted
                    # command) makes write/flush raise. That must never take down
                    # the persistent owner -- otherwise every interruption forces
                    # a reconnect and an extra Uno reset. Swallow it and keep
                    # serving; the arm already executed the completed command.
                    try:
                        file.write(_json_line(response))
                        file.flush()
                    except OSError as exc:
                        print(f"[arm-session] client write failed, ignoring: {exc}",
                              flush=True)
        finally:
            listener.close()
            self.listener = None
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self.arm.close()


class ArmSessionClient:
    def __init__(self, path=None):
        self.path = session_socket_path(path)

    def request(self, payload, timeout=30.0):
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(float(timeout))
        try:
            connection.connect(str(self.path))
            file = connection.makefile("rwb")
            file.write(_json_line(payload))
            file.flush()
            raw = file.readline(1024 * 1024)
        finally:
            connection.close()
        if not raw:
            raise RuntimeError("arm session closed without a response")
        response = json.loads(raw.decode())
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "arm session command failed"))
        return response


def _print_response(response):
    print(json.dumps(response, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", help="override local Unix socket path")
    subparsers = parser.add_subparsers(dest="action", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--floor", choices=("hover", "grasp"))
    serve.add_argument("--elbow", type=int, default=config.FLOOR_REFERENCE_ELBOW)
    subparsers.add_parser("status")
    check = subparsers.add_parser("check")
    check.add_argument("angles", type=int, nargs=config.N_JOINTS)
    move = subparsers.add_parser("move")
    move.add_argument("angles", type=int, nargs=config.N_JOINTS)
    floor = subparsers.add_parser("floor")
    floor.add_argument("level", choices=("hover", "grasp"))
    floor.add_argument("elbow", type=int, nargs="?",
                       default=config.FLOOR_REFERENCE_ELBOW)
    floor.add_argument("--step", type=int, default=config.FLOOR_VECTOR_STEP_DEG)
    subparsers.add_parser("shutdown")
    args = parser.parse_args()

    if args.action == "serve":
        startup = None
        if args.floor:
            # The single unavoidable reset is followed by a staged, level wrist
            # recovery before entering the long-lived command loop.
            level_pose = floor_pose(args.elbow, args.floor)
            ready = list(config.HOME_POSE)
            ready[config.J_WRIST] = config.FLOOR_WRIST_PITCH
            ready[config.J_GRIP] = config.GRIP_OPEN
            ready[config.J_ROLL] = config.FLOOR_WRIST_ROLL
            startup = [ready, level_pose]
        server = ArmSessionServer(args.socket)

        def stop(_signum, _frame):
            server.running = False

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        server.serve(startup)
        return True

    client = ArmSessionClient(args.socket)
    if args.action == "status":
        response = client.request({"command": "status"})
    elif args.action == "check":
        response = client.request({"command": "check", "pose": args.angles})
    elif args.action == "move":
        response = client.request({"command": "move", "pose": args.angles})
    elif args.action == "floor":
        response = client.request({
            "command": "floor", "level": args.level,
            "elbow": args.elbow, "step": args.step,
        })
    elif args.action == "shutdown":
        response = client.request({"command": "shutdown"})
    else:
        parser.error("unsupported action")
    _print_response(response)
    return True


if __name__ == "__main__":
    main()
