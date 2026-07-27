"""Persistent Uno owner: one USB open, many arm commands, no repeated reset.

Opening the Uno serial port toggles DTR and resets the board to HOME.  Short
one-shot scripts therefore create the inefficient HOME -> requested-pose motion
on every command.  This process opens the port exactly once and exposes a local
Unix-domain socket.  All later tools talk to the socket instead of reopening
the board.

Examples::

    python3 laptop/arm_session.py serve --floor hover
    python3 laptop/arm_session.py status
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
from arm_serial import ArmSerial
from floor_motion import floor_pose, floor_waypoints


ROOT = Path(__file__).resolve().parents[1]


def session_socket_path(value=None):
    path = Path(value or config.ARM_SESSION_SOCKET)
    return path if path.is_absolute() else ROOT / path


def _json_line(payload):
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode()


class ArmSessionServer:
    def __init__(self, path=None, arm=None):
        self.path = session_socket_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.arm = arm or ArmSerial()
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

    def _move_sequence(self, poses, timeout=15.0, settle_s=0.0):
        poses = self._validated_sequence(poses)
        for pose in poses:
            self.arm.send_angles(pose)
            self.arm.wait_done(timeout=float(timeout))
            if settle_s:
                time.sleep(float(settle_s))
        return self.arm.status()

    def handle(self, request):
        command = request.get("command")
        if command == "ping":
            return {"ok": bool(self.arm.ping()), "pid": os.getpid()}
        if command == "status":
            return {"ok": True, "pose": self.arm.status(), "pid": os.getpid()}
        if command == "move":
            pose = self._move_sequence(
                [request.get("pose")],
                timeout=request.get("timeout", 15.0),
                settle_s=request.get("settle_s", 0.0))
            return {"ok": True, "pose": pose}
        if command == "sequence":
            pose = self._move_sequence(
                request.get("poses"),
                timeout=request.get("timeout", 15.0),
                settle_s=request.get("settle_s", 0.0))
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
                settle_s=request.get("settle_s", 0.0))
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
