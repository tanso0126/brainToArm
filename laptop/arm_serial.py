"""Laptop -> Arduino serial link for the robot arm.

Sends target joint angles, reads ACK/DONE. No router, no UDP — just USB serial.
Mock operation is explicit through config.ARM_MOCK. When a real arm is requested,
missing dependencies, a missing port, bad acknowledgements, and motion timeouts
raise instead of silently pretending that the command succeeded.
"""
import time
import glob
import config

try:
    import serial  # pyserial
    _HAVE_SERIAL = True
except ImportError:
    _HAVE_SERIAL = False


def _serial_candidates():
    # macOS Arduino boards enumerate as /dev/cu.usbmodem* or /dev/cu.usbserial*
    cands = glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/cu.usbserial*") \
        + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    excluded = set(getattr(config, "ARM_PORT_EXCLUDE", ()))
    return sorted(set(cands) - excluded)


def parse_status_line(line):
    """Parse firmware ``C a1..a6`` status without accepting partial data."""
    parts = str(line).strip().split()
    if len(parts) != config.N_JOINTS + 1 or parts[0] != "C":
        raise ValueError(f"malformed arm status: {line!r}")
    try:
        values = [int(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"non-integer arm status: {line!r}") from exc
    for index, value in enumerate(values):
        if not config.SERVO_MIN[index] <= value <= config.SERVO_MAX[index]:
            raise ValueError(
                f"arm status joint {index + 1}={value} outside configured range")
    return values


def parse_home_pose_line(line):
    """Parse the ``H a1..a6`` pose compiled into the connected Uno."""
    parts = str(line).strip().split()
    if len(parts) != config.N_JOINTS + 1 or parts[0] != "H":
        raise ValueError(f"malformed firmware home pose: {line!r}")
    try:
        values = [int(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"non-integer firmware home pose: {line!r}") from exc
    if any(not 0 <= value <= 180 for value in values):
        raise ValueError(f"firmware home pose outside 0..180: {line!r}")
    return values


def assert_home_pose_match(compiled_pose, local_pose=None):
    """Fail closed when source and connected firmware describe different homes."""
    local_pose = list(config.HOME_POSE if local_pose is None else local_pose)
    compiled_pose = list(compiled_pose)
    if compiled_pose != local_pose:
        raise RuntimeError(
            "Uno firmware HOME_POSE does not match local home_pose.h: "
            f"firmware={compiled_pose}, local={local_pose}. "
            "Run `python3 laptop/arm_jog.py --upload` before moving the arm.")
    return True


class ArmSerial:
    def __init__(self, port=None, baud=None, mock=None):
        self.port = port or config.ARM_PORT
        self.baud = baud or config.ARM_BAUD
        self.ser = None
        self.mock = config.ARM_MOCK if mock is None else bool(mock)
        if self.mock:
            print("[arm] MOCK mode (ARM_MOCK=True). Commands will be printed only.")
        else:
            if not _HAVE_SERIAL:
                raise RuntimeError("real arm requested but pyserial is missing")
            if self.port == "auto":
                candidates = _serial_candidates()
                if len(candidates) > 1:
                    raise RuntimeError(
                        "multiple non-excluded serial devices found; "
                        "set ARM_PORT or ARM_PORT_EXCLUDE explicitly: "
                        + ", ".join(candidates))
                self.port = candidates[0] if candidates else None
            if not self.port:
                raise RuntimeError("real arm requested but no serial board was found")
            try:
                # TIOCEXCL prevents a second process from reopening this Uno and
                # toggling DTR while a persistent arm session owns the board.
                self.ser = serial.Serial(
                    self.port, self.baud, timeout=0.2, exclusive=True)
            except Exception as exc:
                raise RuntimeError(f"cannot open arm serial port {self.port}: {exc}") from exc
            time.sleep(2.0)  # wait out the auto-reset on connect
            self._drain()
            print(f"[arm] connected {self.port} @ {self.baud}")
            try:
                self.verify_firmware_home_pose()
            except Exception:
                self.close()
                raise

    def _drain(self):
        if self.ser:
            while self.ser.in_waiting:
                self.ser.readline()

    def send_angles(self, angles):
        """angles: list of 6 ints (0..180), or -1 to hold a joint."""
        if len(angles) != config.N_JOINTS:
            raise ValueError(f"expected {config.N_JOINTS} joint values, got {len(angles)}")
        values = []
        for i, value in enumerate(angles):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"joint {i + 1} angle must be numeric")
            if not float(value).is_integer():
                raise ValueError(f"joint {i + 1} angle must be an integer, got {value}")
            value = int(value)
            if value != -1 and not (config.SERVO_MIN[i] <= value <= config.SERVO_MAX[i]):
                raise ValueError(
                    f"joint {i + 1} angle {value} outside configured safe range "
                    f"[{config.SERVO_MIN[i]}, {config.SERVO_MAX[i]}]")
            values.append(value)
        line = "A " + " ".join(str(a) for a in values) + "\n"
        if self.mock:
            print(f"[arm] -> {line.strip()}")
            return "OK"
        self.ser.write(line.encode())
        reply = self._wait_for({"OK"}, timeout=2.0)
        return reply

    def _wait_for(self, expected, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="replace").strip()
            if line.startswith("ERR"):
                raise RuntimeError(f"arm firmware rejected command: {line}")
            if line in expected:
                return line
        raise TimeoutError(f"arm serial timeout waiting for {sorted(expected)}")

    def wait_done(self, timeout=8.0):
        """Block until firmware reports DONE (all joints reached) or timeout."""
        if self.mock:
            time.sleep(0.3)
            return True
        self._wait_for({"DONE"}, timeout=timeout)
        return True

    def ping(self):
        if self.mock:
            return True
        self.ser.write(b"P\n")
        return self._wait_for({"PONG"}, timeout=2.0) == "PONG"

    def status(self):
        """Read the firmware's current commanded servo angles without motion."""
        if self.mock:
            return list(config.HOME_POSE)
        self.ser.write(b"S\n")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="replace").strip()
            if line.startswith("ERR"):
                raise RuntimeError(f"arm firmware rejected status request: {line}")
            if line.startswith("C"):
                return parse_status_line(line)
        raise TimeoutError("arm serial timeout waiting for status")

    def firmware_home_pose(self):
        """Return the HOME pose compiled into the connected firmware."""
        if self.mock:
            return list(config.HOME_POSE)
        self.ser.write(b"H\n")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="replace").strip()
            if line.startswith("ERR"):
                raise RuntimeError(
                    "connected Uno firmware cannot report its HOME_POSE; "
                    "upload the current sketch with "
                    "`python3 laptop/arm_jog.py --upload`")
            if line.startswith("H"):
                try:
                    return parse_home_pose_line(line)
                except ValueError as exc:
                    raise RuntimeError(
                        "connected Uno uses a different arm layout/protocol; "
                        "upload the current six-servo sketch with "
                        "`python3 laptop/arm_jog.py --upload`") from exc
        raise TimeoutError(
            "arm firmware did not report HOME_POSE; upload the current sketch "
            "with `python3 laptop/arm_jog.py --upload`")

    def verify_firmware_home_pose(self):
        """Require the connected firmware to match local ``home_pose.h``."""
        return assert_home_pose_match(self.firmware_home_pose())

    def grip_feedback(self):
        """Return optional A0 pressure/current feedback, or None if uninstalled."""
        if self.mock:
            return None
        self.ser.write(b"F\n")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="replace").strip()
            if line.startswith("ERR"):
                raise RuntimeError(f"arm firmware rejected feedback request: {line}")
            if line.startswith("F "):
                try:
                    value = int(line.split()[1])
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"malformed grip feedback: {line!r}") from exc
                if value == -1:
                    return None
                if not 0 <= value <= 1023:
                    raise ValueError(f"grip feedback outside ADC range: {value}")
                return value
        raise TimeoutError("arm serial timeout waiting for grip feedback")

    def gripper(self, open_=True):
        a = [-1] * config.N_JOINTS
        a[config.J_GRIP] = config.GRIP_OPEN if open_ else config.GRIP_CLOSED
        return self.send_angles(a)

    def home(self):
        return self.send_angles(config.HOME_POSE)

    def close(self):
        if self.ser:
            self.ser.close()


if __name__ == "__main__":
    arm = ArmSerial()
    arm.home(); arm.wait_done()
    arm.gripper(open_=True); arm.wait_done()
    arm.gripper(open_=False); arm.wait_done()
    arm.close()
