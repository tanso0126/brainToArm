"""Native macOS/Linux HID transport for the LAXTHA PolyG-I (PID 0x0010).

The protocol below was recovered from TeleScan's installed LXSM-D1WD10.dll,
checked against LAXTHA's D1WD10 manual, and verified on the physical device.
Each 1024-byte INPUT report contains 512 offset-binary ADC words.  PolyG-I has
16 physical channels, so one report contains 32 time rows; channels 1..8 are
the EEG group.  The least-significant bit is a digital marking bit, not ADC
data, and must be cleared exactly as the vendor DLL does.
"""
import time

try:
    import hid
except ImportError:
    hid = None


VID = 0x0F1F
PID = 0x0010
REPORT_BYTES = 1024
MAX_CHANNELS = 16
EEG_CHANNELS = 8
ROWS_PER_REPORT = REPORT_BYTES // 2 // MAX_CHANNELS

# Exact constant embedded at LXSM-D1WD10.dll .rdata:0x1000A180.  The vendor
# float stream is ADC-input volts, not electrode-input volts; the unknown fixed
# analogue front-end gain prevents an honest electrode-uV conversion.
ADC_VOLTS_PER_COUNT = -1.25 / 32768.0
PGA_GAINS = (
    0.1, 0.2, 0.4, 0.7, 1.0, 1.36, 1.70, 2.55,
    3.40, 4.25, 5.67, 6.80, 8.50, 10.20, 11.90, 17.00,
)
COMMAND_SETTLE_SECONDS = 0.12
STARTUP_DISCARD_SECONDS = 1.0

# D1WD10 defines sampling frequency as 2**selector Hz.  PolyG-I uses 16
# physical channels, whose documented maximum is 512 Hz (selector 9).
SAMPLE_SELECTORS = tuple(range(10))


def command_report(command, arg1=0, arg2=0):
    """Build one HID OUTPUT report, including hidapi's report-ID byte."""
    values = (command, arg1, arg2)
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 0xFF
           for v in values):
        raise ValueError("PolyG-I command bytes must be integers in [0, 255]")
    return bytes([0, command, arg1, arg2, 0, 0, 0, 0, 0])


def decode_report(report, max_channels=MAX_CHANNELS, output_channels=EEG_CHANNELS):
    """Decode one INPUT report to time-major EEG rows of signed ADC counts.

    Windows ReadFile includes a leading report-ID byte; hidapi on macOS normally
    removes it.  Accept both representations so captures and tests are portable.
    LXSM-D1WD10 reconstructs each word as
    ``(high - 0x80) * 256 + (low & 0xFE)``.  This removes the marking bit and
    converts the device's offset-binary word to a signed count.
    """
    for name, value in (("max_channels", max_channels),
                        ("output_channels", output_channels)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if output_channels > max_channels:
        raise ValueError("output_channels cannot exceed max_channels")
    payload = bytes(report)
    if len(payload) == REPORT_BYTES + 1 and payload[0] == 0:
        payload = payload[1:]
    if len(payload) != REPORT_BYTES:
        raise ValueError(
            f"PolyG-I INPUT report is {len(payload)} bytes, expected {REPORT_BYTES}")

    values = []
    for offset in range(0, REPORT_BYTES, 2):
        high, low = payload[offset], payload[offset + 1]
        values.append((high - 0x80) * 256 + (low & 0xFE))
    if len(values) % max_channels:
        raise ValueError(
            f"{len(values)} values cannot be divided into {max_channels} channels")
    return [values[i:i + output_channels]
            for i in range(0, len(values), max_channels)]


def counts_to_adc_mv(values):
    """Convert D1WD10 counts to the vendor DLL's ADC-input millivolts."""
    return [float(value) * ADC_VOLTS_PER_COUNT * 1000.0 for value in values]


def enumerate_devices(hid_module=None):
    backend = hid if hid_module is None else hid_module
    if backend is None:
        raise RuntimeError("hidapi missing; pip install hidapi")
    return backend.enumerate(VID, PID)


class PolyGIHID:
    """Exact-one-device session with deterministic start/stop lifecycle."""

    def __init__(self, channels=EEG_CHANNELS, max_channels=MAX_CHANNELS,
                 gain_index=2, sample_selector=8, hid_module=None,
                 command_settle_seconds=COMMAND_SETTLE_SECONDS,
                 startup_discard_seconds=STARTUP_DISCARD_SECONDS):
        self.channels = channels
        self.max_channels = max_channels
        self.gain_index = gain_index
        self.sample_selector = sample_selector
        self._hid = hid if hid_module is None else hid_module
        self.command_settle_seconds = float(command_settle_seconds)
        self.startup_discard_seconds = float(startup_discard_seconds)
        if self.command_settle_seconds < 0 or self.startup_discard_seconds < 0:
            raise ValueError("PolyG-I settle/discard durations must be non-negative")
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
        if self.sample_selector not in SAMPLE_SELECTORS:
            raise ValueError(
                f"unsupported PolyG-I sample selector {self.sample_selector}; "
                f"expected one of {list(SAMPLE_SELECTORS)}")
        if self.max_channels != MAX_CHANNELS:
            raise ValueError("PolyG-I must be configured for 16 physical channels")
        if (isinstance(self.gain_index, bool)
                or not isinstance(self.gain_index, int)
                or not 0 <= self.gain_index < len(PGA_GAINS)):
            raise ValueError("PolyG-I gain_index must be in [0, 15]")

        # Exact LXSM-D1WD10 initialization order. Stop first so a process crash or
        # a prior acquisition cannot leave stale reports racing this new session.
        self._write(0x01, 0x00, 0x00)
        time.sleep(self.command_settle_seconds)
        self._write(0x05, self.max_channels, 0x00)
        time.sleep(self.command_settle_seconds)
        self._write(0x04, self.sample_selector, 0x00)
        time.sleep(self.command_settle_seconds)
        # Command 0x0B sets one source group: arg1=gain, arg2=group. PolyG-I
        # source group 0 is EEG channels 1..8, leaving ECG/EMG/etc untouched.
        self._write(0x0B, self.gain_index, 0x00)
        time.sleep(self.command_settle_seconds)
        self._write(0x01, 0x01, 0x00)
        self.streaming = True
        # The physical unit emits a short transition while its analogue path
        # settles after START. Those exact-rail rows are neither brain signal nor
        # a valid rest baseline, so drain them before exposing the session.
        deadline = time.monotonic() + self.startup_discard_seconds
        while time.monotonic() < deadline:
            if not self.read_report():
                time.sleep(0.001)

    def read_report(self):
        if self.device is None:
            raise RuntimeError("PolyG-I HID is not open")
        data = self.device.read(REPORT_BYTES)
        return bytes(data) if data else b""

    def read_rows(self):
        report = self.read_report()
        return decode_report(report, self.max_channels, self.channels) if report else []

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
