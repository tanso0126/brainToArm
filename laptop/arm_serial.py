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
    return sorted(set(cands))


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
                        "multiple serial devices found; set ARM_PORT explicitly: "
                        + ", ".join(candidates))
                self.port = candidates[0] if candidates else None
            if not self.port:
                raise RuntimeError("real arm requested but no serial board was found")
            try:
                self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
            except Exception as exc:
                raise RuntimeError(f"cannot open arm serial port {self.port}: {exc}") from exc
            time.sleep(2.0)  # wait out the auto-reset on connect
            self._drain()
            print(f"[arm] connected {self.port} @ {self.baud}")

    def _drain(self):
        if self.ser:
            while self.ser.in_waiting:
                self.ser.readline()

    def send_angles(self, angles):
        """angles: list of 7 ints (0..180), or -1 to hold a joint."""
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
