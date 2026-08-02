"""Persistent serial owner for the reduced 2-DOF-plus-gripper arm.

This uses a separate Unix socket and a reduced-geometry safety adapter, leaving
the historical arm_session.py service and six-servo workflow unchanged.
"""

from pathlib import Path
import argparse
import json
import signal

from arm_session import ArmSessionClient, ArmSessionServer
import config
from reduced_dof import (
    ReducedDofSafety,
    canonicalize_status,
    command_pose,
    reduced_home,
    validate_command_pose,
)


ROOT = Path(__file__).resolve().parents[1]
REDUCED_SOCKET = ROOT / "data" / "runtime" / "arm_reduced.sock"


class ReducedArmSessionServer(ArmSessionServer):
    def __init__(self, path=REDUCED_SOCKET, arm=None):
        super().__init__(path=path, arm=arm, safety=ReducedDofSafety())

    def _validated_sequence(self, poses):
        checked = super()._validated_sequence(poses)
        return [validate_command_pose(pose) for pose in checked]

    def handle(self, request):
        response = super().handle(request)
        if isinstance(response, dict) and isinstance(response.get("pose"), list):
            response["pose"] = canonicalize_status(response["pose"])
        response["mode"] = "reduced-2dof"
        response["activeServos"] = [2, 3, 5]
        return response


def client(path=REDUCED_SOCKET):
    return ArmSessionClient(path)


def _print(payload):
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=str(REDUCED_SOCKET))
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("serve")
    subparsers.add_parser("status")
    distance = subparsers.add_parser("distance")
    distance.add_argument("--samples", type=int, default=config.ULTRASONIC_SAMPLES)
    move = subparsers.add_parser("move")
    move.add_argument("shoulder", type=int)
    move.add_argument("elbow", type=int)
    move.add_argument("gripper", type=int, nargs="?", default=config.GRIP_OPEN)
    subparsers.add_parser("home")
    subparsers.add_parser("shutdown")
    args = parser.parse_args()

    if args.action == "serve":
        server = ReducedArmSessionServer(args.socket)

        def stop(_signum, _frame):
            server.running = False

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        server.serve()
        return 0

    connection = client(args.socket)
    if args.action == "status":
        result = connection.request({"command": "status"})
    elif args.action == "distance":
        result = connection.request({
            "command": "distance", "samples": args.samples})
    elif args.action == "move":
        result = connection.request({
            "command": "move",
            "pose": command_pose(args.shoulder, args.elbow, args.gripper),
        })
    elif args.action == "home":
        result = connection.request({
            "command": "move", "pose": reduced_home(config.GRIP_OPEN)})
    elif args.action == "shutdown":
        result = connection.request({"command": "shutdown"})
    else:
        parser.error("지원하지 않는 명령입니다.")
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
