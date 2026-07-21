"""Error-related potential (ErrP) detection.

The human never steers the arm. When the arm commits to the WRONG object, the
human's brain involuntarily fires an ErrP — a fronto-central negative deflection
~250-450 ms after the action. We read that as a veto. No motor imagery, no user
training to push a mental button.

Pipeline (given an epoch of EEG right after action onset):
  1. band-pass 1-10 Hz (scipy) over all configured EEG channels
  2. baseline-correct against the pre-onset segment
  3. extract features (negative peak amplitude + latency, peak-to-peak, mean)
  4a. baseline backend: threshold the negative-peak amplitude (zero training)
  4b. model backend : trained classifier probability (needs collected data)

numpy/scipy/sklearn are used when present; a pure-python fallback keeps the mock
demo runnable on a bare interpreter. Real deployment: pip install -r requirements.
"""
import math
import config

try:
    import numpy as np
    from scipy.signal import butter, filtfilt
    _HAVE_SP = True
except ImportError:
    _HAVE_SP = False


def _col(window, ch):
    return [row[ch] for row in window if ch < len(row)]


def _bandpass(sig, fs, band):
    if not _HAVE_SP:
        return sig
    lo, hi = band
    ny = 0.5 * fs
    b, a = butter(4, [lo / ny, min(hi / ny, 0.99)], btype="band")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(sig) <= padlen:
        return sig
    return filtfilt(b, a, sig).tolist()


def _sigmoid(x):
    """Numerically stable logistic function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class ErrPDetector:
    MODEL_FORMAT = 2

    def __init__(self, backend=None, model_path=None):
        self.backend = backend or config.ERRP_BACKEND
        self.fs = config.EEG_FS
        self.band = config.ERRP_BAND
        self.chans = config.ERRP_CHANNELS
        self.use_car = config.ERRP_USE_CAR
        self.threshold = config.ERRP_THRESHOLD
        self.model = None
        self._baseline_std = None
        path = model_path or config.ERRP_MODEL_PATH
        if self.backend == "model":
            try:
                import pickle
                with open(path, "rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict) and "model" in payload:
                    self._validate_model_metadata(payload.get("metadata", {}), path)
                    self.model = payload["model"]
                else:
                    raise ValueError(
                        f"legacy model at {path} has no acquisition metadata; retrain it")
                print(f"[errp] loaded model {path}")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"ERRP_BACKEND='model' but no model exists at {path}") from exc

    def _model_metadata(self):
        return {
            "format": self.MODEL_FORMAT,
            "fs": self.fs,
            "band": list(self.band),
            "channels": list(self.chans),
            "use_car": self.use_car,
            "baseline_s": config.ERRP_BASELINE_S,
            "window_s": config.ERRP_WINDOW_S,
        }

    def _validate_model_metadata(self, metadata, path):
        expected = self._model_metadata()
        mismatches = [
            key for key, value in expected.items()
            if metadata.get(key) != value
        ]
        if mismatches:
            details = ", ".join(
                f"{key}: model={metadata.get(key)!r} current={expected[key]!r}"
                for key in mismatches)
            raise ValueError(f"ErrP model/config mismatch at {path}: {details}")

    # --- calibration: learn resting variability so the threshold is adaptive ---
    def update_baseline(self, resting_window):
        # Measure resting configured-channel std after the same reference and
        # band-pass used at decision time, so the z-score below is in matching
        # units. This makes the detector scale-invariant: it works without
        # knowing the ADC's uV/LSB,
        # because everything is expressed in units of the person's own EEG noise.
        rw = self._reference(resting_window)
        signals = []
        for ch in self.chans:
            sig = _bandpass(_col(rw, ch), self.fs, self.band)
            if sig:
                signals.append(sig)
        mean_sig = self._mean_signal(signals)
        m = sum(mean_sig) / len(mean_sig)
        var = sum((v - m) ** 2 for v in mean_sig) / len(mean_sig)
        self._baseline_std = math.sqrt(var)

    # --- feature extraction (shared by both backends) ---
    def _reference(self, window):
        """Common Average Reference: subtract, at each timepoint, the mean across
        ALL electrodes. Cancels noise shared by every channel (mains hum, motion,
        reference drift) — standard, near-mandatory on real EEG. Cheap sensors
        need it most. It is deliberately disabled for the configured all-channel
        ErrP average because averaging every CAR channel would be exactly zero."""
        if not window:
            return window
        if not self.use_car:
            return [list(row) for row in window]
        nch = len(window[0])
        if nch < 2:
            return window
        out = []
        for row in window:
            m = sum(row) / nch
            out.append([v - m for v in row])
        return out

    def _preprocess(self, window):
        # Apply the configured spatial reference, then per-channel band-pass
        # (which also removes DC drift and mains, since the ErrP band is 1-10Hz),
        # then baseline-correct each configured channel against its pre-onset
        # segment.
        window = self._reference(window)
        n_base = int(config.ERRP_BASELINE_S * self.fs)
        chans = []
        for ch in self.chans:
            sig = _bandpass(_col(window, ch), self.fs, self.band)
            if not sig:
                sig = [0.0]
            base = sig[:n_base] if len(sig) > n_base else sig[:1]
            b = sum(base) / len(base)
            chans.append([v - b for v in sig])       # baseline-corrected
        return chans

    def _deflection(self, sig):
        """Strongest sustained NEGATIVE deflection in the post-onset search
        region. Slides a ~150ms averaging window (ErrP is a sustained dip, not a
        spike) over [onset, onset+0.6s] and returns the most-negative mean, in uV.
        Searching a range makes it robust to ErrP latency jitter across people
        and to timing slop, unlike locking to one exact 250-450ms window."""
        n = len(sig)
        onset = int(config.ERRP_BASELINE_S * self.fs)
        win = max(3, int(0.15 * self.fs))          # 150 ms averaging window
        end = min(n - win, onset + int(0.6 * self.fs))
        if end <= onset:
            seg = sig[onset:] or sig
            return sum(seg) / len(seg)
        best = 0.0
        first = True
        for s in range(onset, end + 1):
            m = sum(sig[s:s + win]) / win
            if first or m < best:
                best = m
                first = False
        return best

    def _features(self, window):
        chans = self._preprocess(window)
        feats = []
        for sig in chans:
            neg_peak = min(sig)
            pos_peak = max(sig)
            p2p = pos_peak - neg_peak
            lat = sig.index(neg_peak) / max(1, len(sig))
            defl = self._deflection(sig)       # strongest sustained negative dip
            feats += [neg_peak, pos_peak, p2p, lat, defl]
        return feats

    # --- decision ---
    def p_error(self, window):
        if self.backend == "model" and self.model is not None:
            f = self._features(window)
            if hasattr(self.model, "predict_proba"):
                return float(self.model.predict_proba([f])[0][1])
            return float(self.model.predict([f])[0])
        return self._baseline_prob(window)

    def _baseline_prob(self, window):
        # ErrP = a sustained NEGATIVE deflection ~250-450ms after
        # the action. Score the strongest sustained dip, expressed as a Z-SCORE
        # against the person's resting noise (from update_baseline). Z-scoring
        # makes this independent of ADC scaling / uV calibration — a ~2.5-sigma
        # dip reads ~0.5, ~4-sigma ~0.9. Falls back to a fixed uV scale only if
        # no baseline was captured.
        chans = self._preprocess(window)
        # Spatially average all configured channels first, then search for the
        # dip. Independent per-channel noise cancels. With all eight channels the
        # configuration disables CAR because the average of every CAR channel is
        # identically zero; a trained multichannel model remains the deployment
        # target after participant-specific ErrP collection.
        mean_sig = self._mean_signal(chans)
        neg = -self._deflection(mean_sig)             # positive when deflected down
        std = self._baseline_std
        if std and std > 1e-6:
            # Baseline std is measured on the same spatially averaged signal.
            z = neg / std
            # center at 3.3 sigma: real ErrP is many-sigma (saturates ~1.0), so
            # this keeps full sensitivity while rejecting noise-search false alarms.
            return _sigmoid(1.2 * (z - 3.3))
        return _sigmoid(0.25 * (neg - 12.0))   # uncalibrated fallback

    @staticmethod
    def _mean_signal(chans):
        if not chans:
            return [0.0]
        n = min(len(c) for c in chans)
        return [sum(c[i] for c in chans) / len(chans) for i in range(n)]

    def is_error(self, window):
        return self.p_error(window) >= self.threshold

    # --- training (model backend) ---
    def fit(self, windows, labels):
        if not windows or len(windows) != len(labels):
            raise ValueError("windows and labels must be non-empty and the same length")
        if set(labels) != {0, 1}:
            raise ValueError("ErrP training needs both labels 0 (correct) and 1 (error)")
        X = [self._features(w) for w in windows]
        self.model = self.make_model().fit(X, labels)
        self.backend = "model"
        return self

    @staticmethod
    def make_model():
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )

    def save(self, path=None):
        import pickle
        if self.model is None:
            raise RuntimeError("cannot save an unfitted ErrP model")
        with open(path or config.ERRP_MODEL_PATH, "wb") as f:
            pickle.dump({
                "model": self.model,
                "metadata": self._model_metadata(),
            }, f)
