"""Real-time EEG acquisition -> a rolling sample buffer the rest of the app reads.

Sources (config.EEG_SOURCE):
  "mock"   : synthetic 8ch signal, no hardware. mark_error() injects an ErrP-like
             burst so the full loop is testable. Goes through the SAME LXSDF
             encode+parse path as real data, so nothing downstream is special-cased.
  "serial" : Path A. Read raw LXSDF bytes off the PolyG-I USB COM port (macOS
             native, pyserial) -> LXSDFParser -> samples.
  "tcp"    : Path B. Windows helper (windows_eeg_server.py, LXSMWD12.dll) forwards
             the raw LXSDF byte stream over localhost/ethernet -> same parser.

Output: a thread-safe ring buffer of EEG samples in microvolts, shape
(n, EEG_CHANNELS). Consumers call snapshot(seconds).
"""
import time
import glob
import socket
import struct
import threading
import collections
import math
import random

import config
from lxsdf import LXSDFParser, build_packet

try:
    import serial
    _HAVE_SERIAL = True
except ImportError:
    _HAVE_SERIAL = False


def _raw_to_uv(raw):
    return (raw - config.ADC_ZERO) * config.ADC_UV_PER_LSB


def _select_eeg(all_channels):
    """Pick the 8 EEG slots out of the full interleaved channel list."""
    out = []
    for slot in config.EEG_CHANNEL_MAP:
        out.append(_raw_to_uv(all_channels[slot]) if slot < len(all_channels) else 0.0)
    return out


class RingBuffer:
    """Timestamped sample ring. Each entry is (t_monotonic, sample). Time-based
    epoching (not sample-count) keeps ErrP windows correctly aligned to the
    action onset on real hardware, regardless of thread jitter or exact fs."""
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
        self.lock = threading.Lock()

    def push(self, sample):
        with self.lock:
            self.buf.append((time.monotonic(), sample))

    def snapshot(self, n):
        with self.lock:
            return [s for _, s in list(self.buf)[-n:]]

    def epoch(self, onset_t, pre, post):
        """Samples with timestamp in [onset_t - pre, onset_t + post]."""
        lo, hi = onset_t - pre, onset_t + post
        with self.lock:
            return [s for t, s in self.buf if lo <= t <= hi]

    def latest_t(self):
        with self.lock:
            return self.buf[-1][0] if self.buf else None


class EEGBridge:
    def __init__(self):
        self.fs = config.EEG_FS
        self.channels = config.EEG_CHANNELS
        self.source = config.EEG_SOURCE
        self.ring = RingBuffer(capacity=self.fs * 12)   # 12s history
        self.parser = LXSDFParser(total_channels=config.EEG_TOTAL_CHANNELS)
        self._stop = threading.Event()
        self._error_burst_until = 0.0
        self._error_burst_start = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        print(f"[eeg] source={self.source} fs={self.fs} ch={self.channels}")
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()

    def snapshot(self, seconds):
        return self.ring.snapshot(int(seconds * self.fs))

    def mark_onset(self):
        """Timestamp the moment the arm commits to an action. Pair with
        wait_and_epoch() to read the brain's response aligned to THIS instant."""
        return time.monotonic()

    def wait_and_epoch(self, onset_t, pre=None, post=None):
        """Block until `post` seconds after onset, then return the epoch
        [onset-pre, onset+post] — time-aligned, the way real ErrP needs."""
        pre = config.ERRP_BASELINE_S if pre is None else pre
        post = config.ERRP_WINDOW_S if post is None else post
        target = onset_t + post
        while time.monotonic() < target and not self._stop.is_set():
            time.sleep(0.005)
        return self.ring.epoch(onset_t, pre, post)

    def mark_error(self, duration=0.8):
        """Mock only: simulate the human brain's ErrP to a wrong action.
        Time-locked: a monophasic negative deflection peaking ~300ms later."""
        now = time.time()
        self._error_burst_start = now
        self._error_burst_until = now + duration

    def _emit_from_bytes(self, data):
        for all_channels, _pc in self.parser.feed(data):
            self.ring.push(_select_eeg(all_channels))

    # ---- sources ----
    def _run(self):
        try:
            {"mock": self._run_mock,
             "serial": self._run_serial,
             "tcp": self._run_tcp}[self.source]()
        except Exception as e:
            print(f"[eeg] source thread died: {e}")

    def _run_mock(self):
        dt = 1.0 / self.fs
        t = 0.0
        pc = 0
        # emit the same number of slots a real PolyG-I would, so auto-detect and
        # channel mapping are exercised. 16 total slots, EEG = first 8.
        total = config.EEG_TOTAL_CHANNELS or 16
        while not self._stop.is_set():
            now = time.time()
            erroring = now < self._error_burst_until
            te = now - self._error_burst_start          # time since action onset
            # monophasic ErrP: negative Gaussian bump peaking ~300ms post-onset
            errp_uv = -40.0 * math.exp(-((te - 0.30) / 0.07) ** 2) if erroring else 0.0
            # widespread alpha (present on all electrodes, stronger occipital) —
            # this is a COMMON-MODE component, so CAR removes it cleanly, just
            # like real EEG. A shared 10Hz term + a per-channel gain.
            alpha = 10 * math.sin(2 * math.pi * 10 * t)
            raw_slots = []
            for slot in range(total):
                if slot < 8:  # EEG-like
                    gain = 1.6 if slot >= 6 else 1.0       # occipital a bit stronger
                    uv = gain * alpha + random.gauss(0, 4)
                    if slot in config.ERRP_FRONTOCENTRAL:
                        uv += errp_uv                      # ErrP only fronto-central
                    raw = int(config.ADC_ZERO + uv / max(config.ADC_UV_PER_LSB, 1e-6))
                else:
                    raw = config.ADC_ZERO
                raw_slots.append(max(0, min(4095, raw)))
            self._emit_from_bytes(build_packet(raw_slots, pc=pc))
            pc = (pc + 1) & 0xFF
            t += dt
            time.sleep(dt)

    def _resolve_serial_port(self):
        port = config.EEG_PORT
        if port == "auto":
            cands = glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.usbmodem*") \
                + glob.glob("/dev/tty.usbserial*")
            port = cands[0] if cands else None
        return port

    def _run_serial(self):
        if not _HAVE_SERIAL:
            raise RuntimeError("pyserial missing; pip install pyserial")
        port = self._resolve_serial_port()
        if not port:
            raise RuntimeError("no EEG serial port; run eeg_detect.py")
        s = serial.Serial(port, config.EEG_BAUD, timeout=0.1)
        print(f"[eeg] serial {port} @ {config.EEG_BAUD}")
        while not self._stop.is_set():
            data = s.read(512)
            if data:
                self._emit_from_bytes(data)
        s.close()

    def _run_tcp(self):
        while not self._stop.is_set():
            try:
                sock = socket.create_connection(
                    (config.EEG_TCP_HOST, config.EEG_TCP_PORT), timeout=3)
                print(f"[eeg] tcp {config.EEG_TCP_HOST}:{config.EEG_TCP_PORT}")
                while not self._stop.is_set():
                    data = sock.recv(4096)
                    if not data:
                        break
                    self._emit_from_bytes(data)
            except OSError as e:
                print(f"[eeg] tcp retry ({e})")
                time.sleep(1.0)


if __name__ == "__main__":
    eeg = EEGBridge().start()
    for _ in range(3):
        time.sleep(1)
        print(f"buffered {len(eeg.snapshot(1.0))} samples/1s, "
              f"parser ch={eeg.parser.total_channels} dropped={eeg.parser.dropped}")
    eeg.stop()
