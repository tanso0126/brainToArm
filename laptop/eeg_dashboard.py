"""Local real-time EEG dashboard API and one-command launcher.

Run from the repository root:

    python3 laptop/eeg_dashboard.py

The Python process owns the PolyG-I HID handle.  The browser only talks to this
localhost API, so UI refreshes cannot leave the device streaming or let two
clients race the hardware.  ``--api-only`` starts only the JSON API on port 8765.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from collections import deque
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi, sosfilt, sosfilt_zi

import config
from polyg_hid import (
    ADC_VOLTS_PER_COUNT,
    MAX_CHANNELS,
    PGA_GAINS,
    PID,
    REPORT_BYTES,
    ROWS_PER_REPORT,
    VID,
    PolyGIHID,
    enumerate_devices,
)
from errp import ErrPDetector


REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = REPO_ROOT / "dashboard"
RECORDING_DIR = REPO_ROOT / "recordings"
DEFAULT_API_PORT = 8765
DEFAULT_UI_PORT = 3000
CHANNELS = 8
FILTER_LOW_HZ = 0.5
FILTER_HIGH_HZ = 45.0
NOTCH_HZ = 60.0
NOTCH_Q = 30.0


def sanitize_recording_name(value):
    """Return a safe optional filename stem without changing its meaning."""
    value = str(value or "").strip()
    value = re.sub(r"[^0-9A-Za-z가-힣._-]+", "-", value).strip("-._")
    return value[:60]


def analyze_signal_quality(values, raw_counts=None, raw_adc_mv=None):
    """Return exact two-second-window metrics, never electrode impedance."""
    if not values:
        return {"state": "waiting", "rmsMv": 0.0, "peakToPeakMv": 0.0,
                "clippingPercent": 0.0, "dcOffsetMv": 0.0}
    rms = (sum(value * value for value in values) / len(values)) ** 0.5
    peak_to_peak = max(values) - min(values)
    counts = raw_counts or []
    clipping = (100.0 * sum(value <= -32768 or value >= 32766 for value in counts)
                / len(counts) if counts else 0.0)
    dc_offset = statistics.fmean(raw_adc_mv) if raw_adc_mv else 0.0
    effective_lsb_mv = abs(ADC_VOLTS_PER_COUNT * 2 * 1000.0)
    if clipping > 0.0:
        state = "saturated"
    elif peak_to_peak <= effective_lsb_mv * 2:
        state = "flat"
    else:
        state = "present"
    return {
        "state": state,
        "rmsMv": round(rms, 6),
        "peakToPeakMv": round(peak_to_peak, 6),
        "clippingPercent": round(clipping, 3),
        "dcOffsetMv": round(dc_offset, 6),
    }


class EEGSignalProcessor:
    """Stateful real-time EEG display filter in verified ADC millivolts.

    A second-order-per-edge Butterworth band-pass produces a fourth-order
    band-pass overall. The 60 Hz notch is a separate second-order IIR. States
    persist across HID reports, so block boundaries do not create artefacts.
    """

    def __init__(self, fs=config.EEG_FS, channels=CHANNELS):
        self.fs = float(fs)
        self.channels = channels
        self._notch_b, self._notch_a = iirnotch(NOTCH_HZ, NOTCH_Q, fs=self.fs)
        self._band_sos = butter(
            2, (FILTER_LOW_HZ, FILTER_HIGH_HZ), btype="bandpass",
            fs=self.fs, output="sos")
        self._notch_state = None
        self._band_state = None

    def process(self, count_rows):
        counts = np.asarray(count_rows, dtype=np.float64)
        if counts.ndim != 2 or counts.shape[1] != self.channels:
            raise ValueError(f"expected rows with {self.channels} EEG channels")
        adc_mv = counts * (ADC_VOLTS_PER_COUNT * 1000.0)
        if self._notch_state is None:
            self._notch_state = (lfilter_zi(self._notch_b, self._notch_a)[:, None]
                                 * adc_mv[0][None, :])
        notched, self._notch_state = lfilter(
            self._notch_b, self._notch_a, adc_mv, axis=0, zi=self._notch_state)
        if self._band_state is None:
            self._band_state = (sosfilt_zi(self._band_sos)[:, :, None]
                                * notched[0][None, None, :])
        filtered, self._band_state = sosfilt(
            self._band_sos, notched, axis=0, zi=self._band_state)
        return adc_mv.tolist(), filtered.tolist()


class EEGDashboardService:
    def __init__(self, device_factory=PolyGIHID, enumerate_fn=enumerate_devices):
        self._device_factory = device_factory
        self._enumerate = enumerate_fn
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread = None
        self._device = None
        self._rows = deque(maxlen=int(config.EEG_FS * 30))
        self._report_arrivals = deque(maxlen=256)
        self._sequence = 0
        self._running = False
        self._error = None
        self._session_id = None
        self._started_mono = None
        self._started_wall = None
        self._last_report_mono = None
        self._reports = 0
        self._samples = 0
        self._delayed_report_periods = 0
        self._record_file = None
        self._record_writer = None
        self._record_path = None
        self._record_started_mono = None
        self._record_rows = 0
        self._pending_marker = ""
        self._processor = None
        self._gain_index = config.EEG_HID_GAIN_INDEX
        self._sample_selector = config.EEG_HID_SAMPLE_SELECTOR
        self._errp_detector = ErrPDetector(backend=config.ERRP_BACKEND)
        self._errp_lock = threading.Lock()
        self._errp_baseline_ready = False
        self._errp_last = None
        self._simulation = None

    @property
    def simulation(self):
        """Lazily start MuJoCo without making EEG-only startup depend on OpenGL."""
        with self._lock:
            if self._simulation is None:
                root = str(REPO_ROOT)
                if root not in sys.path:
                    sys.path.insert(0, root)
                from simul.studio import MuJoCoStudio
                self._simulation = MuJoCoStudio()
            return self._simulation

    def device_available(self):
        with self._lock:
            if self._running or self._device is not None:
                return True
        try:
            return len(self._enumerate()) == 1
        except (RuntimeError, OSError):
            return False

    def start(self, gain_index=None, sample_selector=None):
        gain_index = config.EEG_HID_GAIN_INDEX if gain_index is None else gain_index
        sample_selector = (config.EEG_HID_SAMPLE_SELECTOR if sample_selector is None
                           else sample_selector)
        if sample_selector != config.EEG_HID_SAMPLE_SELECTOR:
            raise ValueError(
                f"정확한 필터 시간축을 위해 sampleSelector는 "
                f"{config.EEG_HID_SAMPLE_SELECTOR}(256 Hz)만 지원합니다")
        with self._lock:
            if self._running:
                return self.status()
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("이전 측정 세션이 아직 종료 중입니다")

        device = self._device_factory(
            channels=config.EEG_HID_CHANNELS,
            max_channels=config.EEG_HID_MAX_CHANNELS,
            gain_index=gain_index,
            sample_selector=sample_selector,
        )
        device.open()
        try:
            device.start()
        except Exception:
            device.close()
            raise

        now_mono = time.monotonic()
        with self._lock:
            self._device = device
            self._stop_event.clear()
            self._rows.clear()
            self._report_arrivals.clear()
            self._sequence = 0
            self._running = True
            self._error = None
            self._session_id = uuid.uuid4().hex
            self._started_mono = now_mono
            self._started_wall = datetime.now(timezone.utc)
            self._last_report_mono = now_mono
            self._reports = 0
            self._samples = 0
            self._delayed_report_periods = 0
            self._processor = EEGSignalProcessor(config.EEG_FS, CHANNELS)
            self._gain_index = gain_index
            self._sample_selector = sample_selector
            self._thread = threading.Thread(
                target=self._capture_loop, name="polyg-dashboard-capture", daemon=True)
            self._thread.start()
        return self.status()

    def stop(self):
        with self._lock:
            thread = self._thread
            self._stop_event.set()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        with self._lock:
            if thread is not None and thread.is_alive():
                raise TimeoutError("PolyG-I 측정 스레드가 3초 안에 종료되지 않았습니다")
            self._finish_recording_locked()
        return self.status()

    def close(self):
        try:
            self.stop()
        except (RuntimeError, TimeoutError, OSError):
            pass
        simulation, self._simulation = self._simulation, None
        if simulation is not None:
            simulation.close()

    def _capture_loop(self):
        try:
            while not self._stop_event.is_set():
                rows = self._device.read_rows()
                if not rows:
                    with self._lock:
                        last = self._last_report_mono or time.monotonic()
                    if time.monotonic() - last > config.EEG_HID_STALL_TIMEOUT_S:
                        raise TimeoutError(
                            f"{config.EEG_HID_STALL_TIMEOUT_S:.1f}초 동안 HID 보고서가 없습니다")
                    self._stop_event.wait(0.001)
                    continue
                arrival = time.monotonic()
                self._ingest_report(rows, arrival)
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            device, self._device = self._device, None
            try:
                if device is not None:
                    device.close()
            except Exception as exc:
                with self._lock:
                    self._error = self._error or f"종료 실패: {exc}"
            with self._lock:
                self._running = False
                self._stop_event.set()
                self._finish_recording_locked()

    def _ingest_report(self, rows, arrival):
        dt = 1.0 / config.EEG_FS
        first_sample_t = arrival - (len(rows) - 1) * dt
        raw_adc_mv_rows, filtered_rows = self._processor.process(rows)
        with self._lock:
            if self._reports == 0:
                # The first HID block was already accumulating inside the device
                # while START completed.  Anchor the session to its first sample
                # instead of exposing a small negative elapsed time in the UI/CSV.
                self._started_mono = first_sample_t
                self._started_wall = datetime.now(timezone.utc) - timedelta(
                    seconds=arrival - first_sample_t)
            if self._report_arrivals:
                interval = arrival - self._report_arrivals[-1]
                expected = ROWS_PER_REPORT / config.EEG_FS
                if interval > expected * 1.8:
                    # D1WD10 input blocks expose no report sequence counter. A
                    # long host-arrival interval can be USB loss or scheduler
                    # delay, so report the observable delay periods without
                    # claiming that samples were definitely lost.
                    self._delayed_report_periods += max(
                        0, round(interval / expected) - 1)
            self._report_arrivals.append(arrival)
            self._last_report_mono = arrival
            self._reports += 1

            for index, (raw_counts, raw_adc_mv, filtered_mv) in enumerate(zip(
                    rows, raw_adc_mv_rows, filtered_rows)):
                sample_t = first_sample_t + index * dt
                self._sequence += 1
                clean_counts = [int(value) for value in raw_counts]
                clean_raw_mv = [float(value) for value in raw_adc_mv]
                clean_filtered_mv = [float(value) for value in filtered_mv]
                self._rows.append((
                    self._sequence, sample_t, clean_counts,
                    clean_raw_mv, clean_filtered_mv))
                self._samples += 1
                self._write_record_row_locked(
                    sample_t, clean_counts, clean_raw_mv, clean_filtered_mv)
            if self._record_file is not None:
                self._record_file.flush()

    def start_recording(self, label=""):
        with self._lock:
            if not self._running:
                raise RuntimeError("측정을 먼저 시작해야 기록할 수 있습니다")
            if self._record_file is not None:
                return self.recording_status_locked()
            RECORDING_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = sanitize_recording_name(label)
            stem = f"eeg_{stamp}" + (f"_{suffix}" if suffix else "")
            path = RECORDING_DIR / f"{stem}.csv"
            counter = 1
            while path.exists():
                path = RECORDING_DIR / f"{stem}_{counter}.csv"
                counter += 1
            record_file = path.open("x", newline="", encoding="utf-8")
            writer = csv.writer(record_file)
            writer.writerow([
                "timestamp_utc", "elapsed_s", "sequence",
                *[f"ch{channel}_count" for channel in range(1, CHANNELS + 1)],
                *[f"ch{channel}_adc_mv_raw" for channel in range(1, CHANNELS + 1)],
                *[f"ch{channel}_eeg_mv_0p5_45_notch60"
                  for channel in range(1, CHANNELS + 1)],
                "marker",
            ])
            self._record_file = record_file
            self._record_writer = writer
            self._record_path = path
            self._record_started_mono = time.monotonic()
            self._record_rows = 0
            self._pending_marker = ""
            return self.recording_status_locked()

    def stop_recording(self):
        with self._lock:
            previous = self.recording_status_locked()
            self._finish_recording_locked()
            return previous

    def add_marker(self, label):
        label = str(label or "").strip()[:80]
        if not label:
            raise ValueError("마커 이름을 입력하세요")
        with self._lock:
            if self._record_file is None:
                raise RuntimeError("기록 중에만 마커를 추가할 수 있습니다")
            self._pending_marker = label
        return {"marker": label, "accepted": True}

    def calibrate_errp(self, seconds=None):
        """Learn session noise from recent resting EEG without moving the arm."""
        seconds = float(config.COG_REST_S if seconds is None else seconds)
        seconds = max(2.0, min(seconds, 30.0))
        with self._lock:
            if not self._running:
                raise RuntimeError("ErrP 휴식 보정 전에 EEG 측정을 시작하세요")
            now = time.monotonic()
            window = [item[4] for item in self._rows if item[1] >= now - seconds]
        required = int(seconds * config.EEG_FS * config.EEG_MIN_EPOCH_FRACTION)
        if len(window) < required:
            raise RuntimeError(
                f"휴식 데이터가 부족합니다 ({len(window)}/{required} samples). "
                f"눈을 편하게 두고 {seconds:.0f}초 후 다시 누르세요")
        self._errp_detector.update_baseline(window)
        with self._lock:
            self._errp_baseline_ready = True
        return {"ready": True, "samples": len(window), "seconds": seconds}

    def check_errp(self, marker="SIM_DECISION"):
        """Open one onset-locked decision window for the simulator.

        The request itself is the visual action onset.  It blocks for the
        configured post-onset interval, then evaluates exactly that timestamped
        epoch. Concurrent browser requests are serialized so epochs cannot
        silently overlap.
        """
        if not self._errp_lock.acquire(blocking=False):
            raise RuntimeError("다른 ErrP 판정 창이 이미 진행 중입니다")
        try:
            with self._lock:
                if not self._running:
                    raise RuntimeError("PolyG-I 측정이 실행 중이 아닙니다")
                if not self._errp_baseline_ready:
                    raise RuntimeError("먼저 ErrP 휴식 보정을 완료하세요")
                onset = time.monotonic()
                if self._record_file is not None:
                    self._pending_marker = str(marker or "SIM_DECISION")[:80]
            deadline = onset + config.ERRP_WINDOW_S
            while time.monotonic() < deadline:
                if self._stop_event.wait(min(0.01, deadline - time.monotonic())):
                    raise RuntimeError("EEG 측정이 판정 도중 중지되었습니다")
            with self._lock:
                lo = onset - config.ERRP_BASELINE_S
                epoch = [item[4] for item in self._rows if lo <= item[1] <= deadline]
            expected = int(
                (config.ERRP_BASELINE_S + config.ERRP_WINDOW_S) * config.EEG_FS)
            required = int(expected * config.EEG_MIN_EPOCH_FRACTION)
            if len(epoch) < required:
                raise RuntimeError(
                    f"ErrP epoch samples 부족 ({len(epoch)}/{required}); 판정을 적용하지 않습니다")
            probability = float(self._errp_detector.p_error(epoch))
            result = {
                "isError": probability >= config.ERRP_THRESHOLD,
                "probability": round(probability, 4),
                "threshold": config.ERRP_THRESHOLD,
                "samples": len(epoch),
                "marker": str(marker or "SIM_DECISION"),
            }
            with self._lock:
                self._errp_last = result
            return result
        finally:
            self._errp_lock.release()

    def _write_record_row_locked(self, sample_t, counts, raw_adc_mv, filtered_mv):
        if self._record_writer is None or self._started_mono is None:
            return
        elapsed = sample_t - self._started_mono
        timestamp = self._started_wall + timedelta(seconds=elapsed)
        marker, self._pending_marker = self._pending_marker, ""
        self._record_writer.writerow([
            timestamp.isoformat(timespec="milliseconds"), f"{elapsed:.6f}",
            self._sequence, *counts,
            *[f"{value:.9f}" for value in raw_adc_mv],
            *[f"{value:.9f}" for value in filtered_mv], marker,
        ])
        self._record_rows += 1

    def _finish_recording_locked(self):
        record_file, self._record_file = self._record_file, None
        self._record_writer = None
        self._record_path = None
        self._record_started_mono = None
        self._pending_marker = ""
        if record_file is not None:
            record_file.flush()
            os.fsync(record_file.fileno())
            record_file.close()

    def recording_status_locked(self):
        active = self._record_file is not None
        duration = (time.monotonic() - self._record_started_mono
                    if active and self._record_started_mono else 0.0)
        return {
            "active": active,
            "filename": self._record_path.name if self._record_path else None,
            "rows": self._record_rows,
            "durationSeconds": round(duration, 1),
        }

    def status(self):
        with self._lock:
            now = time.monotonic()
            running = self._running
            duration = now - self._started_mono if self._started_mono else 0.0
            recent_cutoff = now - 2.0
            recent = [item for item in self._rows if item[1] >= recent_cutoff]
            arrivals = list(self._report_arrivals)
            if len(arrivals) >= 2:
                intervals = [b - a for a, b in zip(arrivals, arrivals[1:])]
                measured_fs = ROWS_PER_REPORT / statistics.median(intervals)
            else:
                measured_fs = 0.0
            qualities = [
                analyze_signal_quality(
                    [item[4][channel] for item in recent],
                    [item[2][channel] for item in recent],
                    [item[3][channel] for item in recent],
                )
                for channel in range(CHANNELS)
            ]
            recording = self.recording_status_locked()
            last_age = now - self._last_report_mono if self._last_report_mono else None
            return {
                "device": {
                    "available": True if running else self.device_available(),
                    "name": "PolyG-I LAXTHA Inc.",
                    "vendorId": f"0x{VID:04X}",
                    "productId": f"0x{PID:04X}",
                    "transport": "USB HID",
                    "channels": CHANNELS,
                    "physicalChannels": MAX_CHANNELS,
                    "reportBytes": REPORT_BYTES,
                },
                "acquisition": {
                    "running": running,
                    "sessionId": self._session_id,
                    "startedAt": self._started_wall.isoformat() if self._started_wall else None,
                    "durationSeconds": round(duration, 1) if running else 0.0,
                    "nominalFs": config.EEG_FS,
                    "measuredFs": round(measured_fs, 1),
                    "reports": self._reports,
                    "samples": self._samples,
                    "delayedReportPeriodsEstimate": self._delayed_report_periods,
                    "lastReportAgeSeconds": round(last_age, 3) if last_age is not None else None,
                    "error": self._error,
                    "units": "mV_ADC_filtered",
                },
                "signal": {
                    "displayUnit": "mV (ADC input)",
                    "conversionVoltsPerCount": ADC_VOLTS_PER_COUNT,
                    "rawEncoding": "D1WD10 offset-binary; marker bit removed",
                    "pipeline": "real-time causal IIR",
                    "bandpassHz": [FILTER_LOW_HZ, FILTER_HIGH_HZ],
                    "bandpassOrder": 4,
                    "notchHz": NOTCH_HZ,
                    "notchQ": NOTCH_Q,
                    "metricWindowSeconds": 2.0,
                    "pgaGainIndex": self._gain_index,
                    "pgaGain": PGA_GAINS[self._gain_index],
                    "electrodeUvCalibrated": False,
                },
                "recording": recording,
                "errp": {
                    "baselineReady": self._errp_baseline_ready,
                    "backend": self._errp_detector.backend,
                    "threshold": config.ERRP_THRESHOLD,
                    "windowSeconds": config.ERRP_WINDOW_S,
                    "lastDecision": self._errp_last,
                },
                "quality": qualities,
            }

    def data_after(self, sequence, limit=2048):
        limit = max(1, min(int(limit), 4096))
        with self._lock:
            rows = list(self._rows)
            oldest = rows[0][0] if rows else self._sequence
            reset = bool(rows and sequence < oldest - 1)
            selected = [item for item in rows if item[0] > sequence][:limit]
            start_mono = self._started_mono
            return {
                "sessionId": self._session_id,
                "reset": reset,
                "oldestSequence": oldest,
                "latestSequence": self._sequence,
                "rows": [
                    {
                        "sequence": seq,
                        "elapsed": round(sample_t - start_mono, 6) if start_mono else 0,
                        "values": [round(value, 6) for value in filtered_mv],
                    }
                    for seq, sample_t, _counts, _raw_mv, filtered_mv in selected
                ],
            }

    def recordings(self):
        if not RECORDING_DIR.exists():
            return []
        items = []
        for path in sorted(RECORDING_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime,
                           reverse=True)[:50]:
            stat = path.stat()
            items.append({
                "filename": path.name,
                "bytes": stat.st_size,
                "modifiedAt": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
                "downloadUrl": f"/api/recordings/{path.name}",
            })
        return items


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "brainToArm-EEG/1.0"

    @property
    def service(self):
        return self.server.service

    def log_message(self, format, *args):
        if args and str(args[1]).startswith("5"):
            super().log_message(format, *args)

    def _cors(self):
        origin = self.headers.get("Origin", "")
        allowed = {
            "http://localhost:3000", "http://127.0.0.1:3000",
            "http://localhost:5173", "http://127.0.0.1:5173",
        }
        self.send_header("Access-Control-Allow-Origin", origin if origin in allowed else "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _binary(self, body, content_type):
        self.send_response(HTTPStatus.OK)
        self._cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length > 64 * 1024:
            raise ValueError("요청 본문이 너무 큽니다")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/status":
                self._json(self.service.status())
                return
            if parsed.path == "/api/simulation/status":
                self._json(self.service.simulation.status())
                return
            if parsed.path == "/api/simulation/frame":
                query = parse_qs(parsed.query)
                camera = query.get("camera", ["overview"])[0]
                width = int(query.get("width", ["960"])[0])
                height = int(query.get("height", ["540"])[0])
                self._binary(
                    self.service.simulation.render_jpeg(
                        camera=camera, width=width, height=height),
                    "image/jpeg",
                )
                return
            if parsed.path == "/api/data":
                query = parse_qs(parsed.query)
                sequence = int(query.get("after", ["0"])[0])
                limit = int(query.get("limit", ["2048"])[0])
                self._json(self.service.data_after(sequence, limit))
                return
            if parsed.path == "/api/recordings":
                self._json({"recordings": self.service.recordings()})
                return
            prefix = "/api/recordings/"
            if parsed.path.startswith(prefix):
                filename = Path(unquote(parsed.path[len(prefix):])).name
                path = RECORDING_DIR / filename
                if not filename.endswith(".csv") or not path.is_file():
                    self._json({"error": "기록 파일을 찾을 수 없습니다"}, HTTPStatus.NOT_FOUND)
                    return
                body = path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self._cors()
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                ascii_name = filename.encode("ascii", "ignore").decode() or "eeg.csv"
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(filename)}',
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._json({"error": "API 경로를 찾을 수 없습니다"}, HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"},
                       HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            body = self._body()
            if parsed.path == "/api/acquisition/start":
                payload = self.service.start(
                    gain_index=body.get("gainIndex"),
                    sample_selector=body.get("sampleSelector"),
                )
            elif parsed.path == "/api/acquisition/stop":
                payload = self.service.stop()
            elif parsed.path == "/api/recording/start":
                payload = self.service.start_recording(body.get("label", ""))
            elif parsed.path == "/api/recording/stop":
                payload = self.service.stop_recording()
            elif parsed.path == "/api/marker":
                payload = self.service.add_marker(body.get("label", ""))
            elif parsed.path == "/api/errp/calibrate":
                payload = self.service.calibrate_errp(body.get("seconds"))
            elif parsed.path == "/api/errp/check":
                payload = self.service.check_errp(body.get("marker", "SIM_DECISION"))
            elif parsed.path == "/api/simulation/start":
                payload = self.service.simulation.start()
            elif parsed.path == "/api/simulation/stop":
                payload = self.service.simulation.stop()
            elif parsed.path == "/api/simulation/reset":
                payload = self.service.simulation.reset()
            elif parsed.path == "/api/simulation/reject":
                payload = self.service.simulation.reject()
            elif parsed.path == "/api/simulation/objects/add":
                payload = self.service.simulation.add_object(
                    shape=body.get("shape", "box"),
                    color=body.get("color"),
                    label=body.get("label"),
                )
            elif parsed.path == "/api/simulation/objects/update":
                payload = self.service.simulation.update_object(
                    body.get("id", ""), body)
            elif parsed.path == "/api/simulation/objects/delete":
                payload = self.service.simulation.delete_object(body.get("id", ""))
            elif parsed.path == "/api/simulation/basket/update":
                payload = self.service.simulation.update_basket(
                    body.get("xMm"), body.get("yMm", 0))
            else:
                self._json({"error": "API 경로를 찾을 수 없습니다"}, HTTPStatus.NOT_FOUND)
                return
            self._json(payload)
        except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"},
                       HTTPStatus.INTERNAL_SERVER_ERROR)


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, service):
        super().__init__(address, DashboardHandler)
        self.service = service


def port_is_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def start_ui_process(port):
    if port_is_open("localhost", port):
        return None
    if not (DASHBOARD_DIR / "package.json").exists():
        raise RuntimeError("dashboard/package.json이 없습니다")
    return subprocess.Popen(
        ["npm", "run", "dev", "--", "--port", str(port)],
        cwd=DASHBOARD_DIR,
    )


def main():
    parser = argparse.ArgumentParser(description="PolyG-I 실시간 EEG 대시보드")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--ui-port", type=int, default=DEFAULT_UI_PORT)
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    service = EEGDashboardService()
    server = DashboardHTTPServer((args.host, args.port), service)
    ui_process = None
    try:
        if not args.api_only:
            ui_process = start_ui_process(args.ui_port)
            ui_url = f"http://localhost:{args.ui_port}/"
            deadline = time.monotonic() + 15.0
            while not port_is_open("localhost", args.ui_port):
                if ui_process is not None and ui_process.poll() is not None:
                    raise RuntimeError("대시보드 UI 서버가 시작되지 않았습니다")
                if time.monotonic() >= deadline:
                    raise TimeoutError("대시보드 UI 시작 시간이 초과됐습니다")
                time.sleep(0.1)
            if not args.no_browser:
                webbrowser.open(ui_url)
            print(f"[dashboard] UI  {ui_url}")
        print(f"[dashboard] API http://{args.host}:{args.port}/api/status")
        print("[dashboard] Ctrl-C로 종료합니다")
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\n[dashboard] 종료 중...")
    finally:
        service.close()
        server.server_close()
        if ui_process is not None:
            ui_process.terminate()
            try:
                ui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ui_process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
