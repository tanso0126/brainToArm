"""Native macOS/Linux HID transport for the LAXTHA PolyG-I (PID 0x0010).

The protocol below was recovered from TeleScan's installed LXSM-D1WD6.dll and
verified against the physical device on macOS.  This HID variant does *not* send
LXSDF frames: each 1024-byte INPUT report contains 512 encoded signed 16-bit
values, interleaved across eight acquisition channels.
"""
import time

try:
    import hid
except ImportError:
    hid = None


VID = 0x0F1F
PID = 0x0010
REPORT_BYTES = 1024

# LXSM-D1WD6 Set_SampleFreq selector -> (command byte 1, command byte 2).
# Selector 9 is the only setting used here: it produced a stable continuous
# stream on the attached PolyG-I and corresponds to the installed PolyG-I
# calibration file's nominal 256 Hz acquisition setting.
SAMPLE_RATE_COMMANDS = {
    7: (0x07, 0x48),
    8: (0x03, 0xA4),
    9: (0x01, 0xD3),
}


def command_report(command, arg1=0, arg2=0):
    """Build one HID OUTPUT report, including hidapi's report-ID byte."""
    values = (command, arg1, arg2)
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 0xFF
           for v in values):
        raise ValueError("PolyG-I command bytes must be integers in [0, 255]")
    return bytes([0, command, arg1, arg2, 0, 0, 0, 0, 0])


def decode_report(report, channels=8):
    """Decode one INPUT report to time-major rows of signed ADC counts.

    Windows ReadFile includes a leading report-ID byte; hidapi on macOS normally
    removes it.  Accept both representations so captures and tests are portable.
    LXSM-D1WD6 reconstructs each word as ``((high - 8) << 8) | low`` and then
    interprets that word as signed int16.
    """
    if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
        raise ValueError("channels must be a positive integer")
    payload = bytes(report)
    if len(payload) == REPORT_BYTES + 1 and payload[0] == 0:
        payload = payload[1:]
    if len(payload) != REPORT_BYTES:
        raise ValueError(
            f"PolyG-I INPUT report is {len(payload)} bytes, expected {REPORT_BYTES}")

    values = []
    for offset in range(0, REPORT_BYTES, 2):
        word = (((payload[offset] - 8) & 0xFF) << 8) | payload[offset + 1]
        values.append(word - 0x10000 if word & 0x8000 else word)
    if len(values) % channels:
        raise ValueError(
            f"{len(values)} values cannot be divided into {channels} channels")
    return [values[i:i + channels] for i in range(0, len(values), channels)]


def enumerate_devices(hid_module=None):
    backend = hid if hid_module is None else hid_module
    if backend is None:
        raise RuntimeError("hidapi missing; pip install hidapi")
    return backend.enumerate(VID, PID)


class PolyGIHID:
    """Exact-one-device session with deterministic start/stop lifecycle."""

    def __init__(self, channels=8, physical_pid=PID, gain_index=6,
                 mode=0, sample_selector=9, hid_module=None):
        self.channels = channels
        self.physical_pid = physical_pid
        self.gain_index = gain_index
        self.mode = mode
        self.sample_selector = sample_selector
        self._hid = hid if hid_module is None else hid_module
        self.device = None
        self.streaming = False

    def open(self):
        if self._hid is None:
            raise RuntimeError("hidapi missing; pip install hidapi")
        matches = self._hid.enumerate(VID, PID)
        if not matches:
            raise RuntimeError(
                f"PolyG-I HID not found (VID=0x{VID:04X}, PID=0x{PID:04X})")
        if len(matches) != 1:
            paths = ", ".join(repr(item.get("path")) for item in matches)
            raise RuntimeError(f"multiple PolyG-I HID devices found: {paths}")
        self.device = self._hid.device()
        self.device.open_path(matches[0]["path"])
        self.device.set_nonblocking(1)
        return self

    def _write(self, command, arg1=0, arg2=0):
        if self.device is None:
            raise RuntimeError("PolyG-I HID is not open")
        report = command_report(command, arg1, arg2)
        written = self.device.write(report)
        if written != len(report):
            raise OSError(f"short PolyG-I HID write: {written}/{len(report)} bytes")

    def start(self):
        if self.sample_selector not in SAMPLE_RATE_COMMANDS:
            raise ValueError(
                f"unsupported PolyG-I sample selector {self.sample_selector}; "
                f"expected one of {sorted(SAMPLE_RATE_COMMANDS)}")
        if (isinstance(self.gain_index, bool)
                or not isinstance(self.gain_index, int)
                or not 0 <= self.gain_index <= 7):
            raise ValueError("PolyG-I gain_index must be in [0, 7]")

        # Exact LXSM-D1WD6 initialization order.  Stop first so a process crash or
        # a prior acquisition cannot leave stale reports racing this new session.
        self._write(0x01, 0x00, 0x00)
        self._write(0x0A, self.mode, 0x00)
        self._write(0x02, self.physical_pid, 15 - self.gain_index)
        timer_hi, timer_lo = SAMPLE_RATE_COMMANDS[self.sample_selector]
        self._write(0x04, timer_hi, timer_lo)
        self._write(0x01, 0x01, 0x00)
        self.streaming = True

    def read_report(self):
        if self.device is None:
            raise RuntimeError("PolyG-I HID is not open")
        data = self.device.read(REPORT_BYTES)
        return bytes(data) if data else b""

    def read_rows(self):
        report = self.read_report()
        return decode_report(report, self.channels) if report else []

    def close(self):
        device, self.device = self.device, None
        if device is None:
            return
        try:
            # STOP is harmless when idle and fail-safe if START reached the
            # device but the host saw a short/failed transfer.
            report = command_report(0x01, 0x00, 0x00)
            written = device.write(report)
            if written != len(report):
                raise OSError(
                    f"short PolyG-I stop write: {written}/{len(report)} bytes")
        finally:
            self.streaming = False
            device.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()


def capture(seconds, **kwargs):
    """Small programmatic diagnostic: return ``(rows, report_arrival_times)``."""
    if seconds <= 0:
        raise ValueError("seconds must be > 0")
    rows, arrivals = [], []
    with PolyGIHID(**kwargs) as device:
        device.start()
        started = time.monotonic()
        deadline = started + seconds
        while time.monotonic() < deadline:
            block = device.read_rows()
            if block:
                rows.extend(block)
                arrivals.append(time.monotonic() - started)
            else:
                time.sleep(0.001)
    return rows, arrivals
