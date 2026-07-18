"""Laptop -> Arduino serial link for the robot arm.

Sends target joint angles, reads ACK/DONE. No router, no UDP — just USB serial.
Falls back to a print-only mock when pyserial or the board is absent, so the
rest of the stack runs on a laptop with nothing plugged in.
"""
import time
import glob
import config

try:
    import serial  # pyserial
    _HAVE_SERIAL = True
except ImportError:
    _HAVE_SERIAL = False


def _autodetect(pattern_hint="usbmodem"):
    # macOS Arduino boards enumerate as /dev/cu.usbmodem* or /dev/cu.usbserial*
    cands = glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/cu.usbserial*") \
        + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    return cands[0] if cands else None


class ArmSerial:
    def __init__(self, port=None, baud=None):
        self.port = port or config.ARM_PORT
        self.baud = baud or config.ARM_BAUD
        self.ser = None
        if self.port == "auto":
            self.port = _autodetect()
        self.mock = not (_HAVE_SERIAL and self.port)
        if self.mock:
            why = "pyserial missing" if not _HAVE_SERIAL else "no board found"
            print(f"[arm] MOCK mode ({why}). Commands will be printed only.")
        else:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2.0)  # wait out the auto-reset on connect
            self._drain()
            print(f"[arm] connected {self.port} @ {self.baud}")

    def _drain(self):
        if self.ser:
            while self.ser.in_waiting:
                self.ser.readline()

    def send_angles(self, angles):
        """angles: list of 7 ints (0..180), or -1 to hold a joint."""
        assert len(angles) == config.N_JOINTS
        line = "A " + " ".join(str(int(a)) for a in angles) + "\n"
        if self.mock:
            print(f"[arm] -> {line.strip()}")
            return "OK"
        self.ser.write(line.encode())
        return self.ser.readline().decode().strip()

    def wait_done(self, timeout=8.0):
        """Block until firmware reports DONE (all joints reached) or timeout."""
        if self.mock:
            time.sleep(0.3)
            return True
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self.ser.readline().decode().strip()
            if line == "DONE":
                return True
        return False

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
