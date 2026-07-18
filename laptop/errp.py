"""Error-related potential (ErrP) detection.

The human never steers the arm. When the arm commits to the WRONG object, the
human's brain involuntarily fires an ErrP — a fronto-central negative deflection
~250-450 ms after the action. We read that as a veto. No motor imagery, no user
training to push a mental button.

Pipeline (given an epoch of EEG right after action onset):
  1. band-pass 1-10 Hz (scipy) over fronto-central channels
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
    if not _HAVE_SP or len(sig) < 15:
        return sig
    lo, hi = band
    ny = 0.5 * fs
    b, a = butter(4, [lo / ny, min(hi / ny, 0.99)], btype="band")
    return filtfilt(b, a, sig).tolist()


class ErrPDetector:
    def __init__(self, backend=None, model_path=None):
        self.backend = backend or config.ERRP_BACKEND
        self.fs = config.EEG_FS
        self.band = config.ERRP_BAND
        self.chans = config.ERRP_FRONTOCENTRAL
        self.threshold = config.ERRP_THRESHOLD
        self.model = None
        self._baseline_std = None
        path = model_path or config.ERRP_MODEL_PATH
        if self.backend == "model":
            try:
                import pickle
                with open(path, "rb") as f:
                    self.model = pickle.load(f)
                print(f"[errp] loaded model {path}")
            except FileNotFoundError:
                print(f"[errp] no model at {path}; falling back to baseline")
                self.backend = "baseline"

    # --- calibration: learn resting variability so the threshold is adaptive ---
    def update_baseline(self, resting_window):
        # Measure resting fronto-central std AFTER the same CAR+bandpass used at
        # decision time, so the z-score below is in matching units. This makes the
        # detector scale-invariant: it works without knowing the ADC's uV/LSB,
        # because everything is expressed in units of the person's own EEG noise.
        rw = self._car(resting_window)
        amps = []
        for ch in self.chans:
            sig = _bandpass(_col(rw, ch), self.fs, self.band)
            if sig:
                m = sum(sig) / len(sig)
                var = sum((v - m) ** 2 for v in sig) / len(sig)
                amps.append(math.sqrt(var))
        self._baseline_std = (sum(amps) / len(amps)) if amps else 10.0

    # --- feature extraction (shared by both backends) ---
    def _car(self, window):
        """Common Average Reference: subtract, at each timepoint, the mean across
        ALL electrodes. Cancels noise shared by every channel (mains hum, motion,
        reference drift) — standard, near-mandatory on real EEG. Cheap sensors
        need it most. No-op if only one channel is present."""
        if not window:
            return window
        nch = len(window[0])
        if nch < 2:
            return window
        out = []
        for row in window:
            m = sum(row) / nch
            out.append([v - m for v in row])
        return out

    def _preprocess(self, window):
        # CAR across all channels first, then per-channel band-pass (which also
        # removes DC drift and mains, since ErrP band is 1-10Hz), then baseline-
        # correct each fronto-central channel against its pre-onset segment.
        window = self._car(window)
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
            try:
                return float(self.model.predict_proba([f])[0][1])
            except Exception:
                return float(self.model.predict([f])[0])
        return self._baseline_prob(window)

    def _baseline_prob(self, window):
        # ErrP = a sustained fronto-central NEGATIVE deflection ~250-450ms after
        # the action. Score the strongest sustained dip, expressed as a Z-SCORE
        # against the person's resting noise (from update_baseline). Z-scoring
        # makes this independent of ADC scaling / uV calibration — a ~2.5-sigma
        # dip reads ~0.5, ~4-sigma ~0.9. Falls back to a fixed uV scale only if
        # no baseline was captured.
        chans = self._preprocess(window)
        # Spatially AVERAGE the fronto-central channels first, THEN search for the
        # dip. A real ErrP is coherent across Fz/FCz/Cz so it survives averaging;
        # independent per-channel noise cancels. (Averaging each channel's own
        # worst dip instead would compound noise + search bias -> false alarms.)
        mean_sig = self._mean_signal(chans)
        neg = -self._deflection(mean_sig)             # positive when deflected down
        std = self._baseline_std
        if std and std > 1e-6:
            # spatial averaging cut the noise by ~sqrt(n_ch); the search over ~0.6s
            # still biases upward, so center the sigmoid at 3.0 sigma.
            z = neg / (std / math.sqrt(max(1, len(chans))))
            # center at 3.3 sigma: real ErrP is many-sigma (saturates ~1.0), so
            # this keeps full sensitivity while rejecting noise-search false alarms.
            return 1.0 / (1.0 + math.exp(-1.2 * (z - 3.3)))
        return 1.0 / (1.0 + math.exp(-0.25 * (neg - 12.0)))   # uncalibrated fallback

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
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        X = [self._features(w) for w in windows]
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ).fit(X, labels)
        self.backend = "model"
        return self

    def save(self, path=None):
        import pickle
        with open(path or config.ERRP_MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
