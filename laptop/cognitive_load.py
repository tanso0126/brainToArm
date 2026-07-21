"""Continuous theta/alpha cognitive-load estimation and autonomy allocation.

User-facing channel numbers are one-based. The configured zero-based defaults
implement theta power from CH1-CH4 and alpha power from CH8. A per-session rest
baseline turns TAR into ``(current - rest) / rest`` before an EMA prevents a
single noisy PSD window from abruptly changing robot authority.
"""

from dataclasses import dataclass
import math
import threading

import config

try:
    import numpy as np
    from scipy.signal import welch
    _HAVE_SPECTRAL = True
except ImportError:
    _HAVE_SPECTRAL = False


@dataclass(frozen=True)
class CognitiveLoadState:
    theta_powers: tuple
    alpha_powers: tuple
    theta_power: float
    alpha_power: float
    tar: float
    rest_tar: float
    relative_tar: float
    smoothed_relative_tar: float
    valid: bool = True
    reason: str = ""


@dataclass(frozen=True)
class AutonomyAllocation:
    robot_weight: float
    human_weight: float
    errp_threshold: float
    errp_apply_stride: int
    load: CognitiveLoadState


@dataclass(frozen=True)
class ErrPDecision:
    veto: bool
    p_error: float
    threshold: float
    applied: bool
    override: bool
    allocation: AutonomyAllocation


def _band_power(signal, fs, band):
    """Welch PSD band power using a half-open frequency interval [lo, hi)."""
    if not _HAVE_SPECTRAL:
        raise RuntimeError("TAR calculation requires numpy and scipy")
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or len(values) < max(16, int(fs)):
        raise ValueError("PSD window must contain at least one second of samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("PSD window contains a non-finite EEG value")
    values = values - values.mean()
    if float(values.std()) <= config.COG_MIN_SIGNAL_STD:
        raise ValueError("PSD channel is flat")
    nperseg = min(len(values), max(int(fs), int(config.COG_PSD_SEGMENT_S * fs)))
    frequencies, density = welch(
        values, fs=fs, window="hann", nperseg=nperseg,
        noverlap=nperseg // 2, detrend=False, scaling="density")
    lo, hi = band
    keep = (frequencies >= lo) & (frequencies < hi)
    if not np.any(keep):
        raise ValueError(f"PSD has no bins in band {band}")
    if len(frequencies) < 2:
        raise ValueError("PSD frequency resolution is insufficient")
    df = float(frequencies[1] - frequencies[0])
    return float(np.sum(density[keep]) * df)


class CognitiveLoadEstimator:
    def __init__(self):
        self.fs = config.EEG_FS
        self.theta_channels = tuple(config.COG_THETA_CHANNELS)
        self.alpha_channels = tuple(config.COG_ALPHA_CHANNELS)
        self.theta_band = tuple(config.COG_THETA_BAND)
        self.alpha_band = tuple(config.COG_ALPHA_BAND)
        self.rest_tar = None
        self._smoothed_relative = None
        self.last_state = None

    def _powers(self, window):
        minimum = int(config.COG_WINDOW_S * self.fs
                      * config.COG_MIN_WINDOW_FRACTION)
        if len(window) < minimum:
            raise ValueError(
                f"cognitive-load window has {len(window)}/{minimum} required samples")
        required = max(self.theta_channels + self.alpha_channels)
        if any(len(row) <= required for row in window):
            raise ValueError(f"cognitive-load analysis requires EEG CH{required + 1}")
        theta = tuple(_band_power(
            [row[channel] for row in window], self.fs, self.theta_band)
            for channel in self.theta_channels)
        alpha = tuple(_band_power(
            [row[channel] for row in window], self.fs, self.alpha_band)
            for channel in self.alpha_channels)
        theta_mean = sum(theta) / len(theta)
        alpha_mean = sum(alpha) / len(alpha)
        if alpha_mean <= config.COG_MIN_ALPHA_POWER:
            raise ValueError("alpha reference power is too small for a stable TAR")
        tar = theta_mean / alpha_mean
        if not math.isfinite(tar) or tar <= 0:
            raise ValueError("TAR is not finite and positive")
        return theta, alpha, theta_mean, alpha_mean, tar

    def calibrate(self, resting_window):
        theta, alpha, theta_mean, alpha_mean, tar = self._powers(resting_window)
        self.rest_tar = tar
        self._smoothed_relative = 0.0
        self.last_state = CognitiveLoadState(
            theta, alpha, theta_mean, alpha_mean, tar, tar, 0.0, 0.0)
        return self.last_state

    def update(self, window):
        if self.rest_tar is None:
            raise RuntimeError("calibrate resting TAR before continuous updates")
        try:
            theta, alpha, theta_mean, alpha_mean, tar = self._powers(window)
            relative = (tar - self.rest_tar) / self.rest_tar
            coefficient = config.COG_EMA_ALPHA
            self._smoothed_relative = (
                coefficient * relative
                + (1.0 - coefficient) * self._smoothed_relative)
            self.last_state = CognitiveLoadState(
                theta, alpha, theta_mean, alpha_mean, tar, self.rest_tar,
                relative, self._smoothed_relative)
        except (RuntimeError, ValueError) as exc:
            previous = self.last_state
            if previous is None:
                raise
            self.last_state = CognitiveLoadState(
                previous.theta_powers, previous.alpha_powers,
                previous.theta_power, previous.alpha_power,
                previous.tar, previous.rest_tar, previous.relative_tar,
                previous.smoothed_relative_tar, valid=False, reason=str(exc))
        return self.last_state


class CognitiveLoadMonitor:
    """Update TAR in the background while EEG acquisition continues."""
    def __init__(self, eeg, estimator=None):
        self.eeg = eeg
        self.estimator = estimator or CognitiveLoadEstimator()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._state = None

    def calibrate(self, resting_window):
        state = self.estimator.calibrate(resting_window)
        with self._lock:
            self._state = state
        return state

    def start(self):
        if self._state is None:
            raise RuntimeError("calibrate cognitive-load rest TAR before monitoring")
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def current(self):
        with self._lock:
            return self._state

    def _run(self):
        while not self._stop.wait(config.COG_UPDATE_S):
            state = self.estimator.update(self.eeg.snapshot(config.COG_WINDOW_S))
            with self._lock:
                self._state = state


class AutonomyAllocator:
    """Convert normalized TAR into robot/human authority and an ErrP decision."""
    def __init__(self):
        self._decision_count = 0

    def allocate(self, load):
        if load is None or not load.valid:
            robot_weight = config.AUTONOMY_ROBOT_MIN
            relative = config.AUTONOMY_RELATIVE_MIN
        else:
            relative = max(
                config.AUTONOMY_RELATIVE_MIN,
                min(config.AUTONOMY_RELATIVE_MAX,
                    load.smoothed_relative_tar))
            if abs(relative) <= config.AUTONOMY_TAR_DEADBAND:
                relative = 0.0
            else:
                relative = math.copysign(
                    abs(relative) - config.AUTONOMY_TAR_DEADBAND, relative)
            robot_weight = (
                config.AUTONOMY_ROBOT_BASE
                + config.AUTONOMY_TAR_GAIN * relative)
            robot_weight = max(
                config.AUTONOMY_ROBOT_MIN,
                min(config.AUTONOMY_ROBOT_MAX, robot_weight))
        human_weight = 1.0 - robot_weight
        threshold = config.ERRP_THRESHOLD + config.AUTONOMY_ERRP_THRESHOLD_GAIN * (
            robot_weight - config.AUTONOMY_ROBOT_BASE)
        threshold = max(0.0, min(1.0, threshold))
        if relative <= 0:
            stride = 1
        else:
            stride = 1 + min(
                config.AUTONOMY_MAX_ERRP_STRIDE - 1,
                int(math.ceil(relative / config.AUTONOMY_STRIDE_TAR_STEP)))
        return AutonomyAllocation(
            robot_weight, human_weight, threshold, stride, load)

    def decide(self, p_error, load):
        if not isinstance(p_error, (int, float)) or not math.isfinite(p_error):
            raise ValueError("ErrP probability must be finite")
        if not 0.0 <= p_error <= 1.0:
            raise ValueError("ErrP probability must be in [0, 1]")
        allocation = self.allocate(load)
        self._decision_count += 1
        scheduled = ((self._decision_count - 1)
                     % allocation.errp_apply_stride == 0)
        override = p_error >= config.AUTONOMY_ERRP_OVERRIDE_THRESHOLD
        veto = override or (scheduled and p_error >= allocation.errp_threshold)
        return ErrPDecision(
            veto, p_error, allocation.errp_threshold,
            scheduled or override, override, allocation)
