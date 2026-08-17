"""
27 — Live breathing demo from the two-ESP32 rig.

Architecture (the right way this time):
  Stage 1: WARMUP — collect WARMUP_S seconds of CSI. Run full paper_mode
           preprocessing (DWT db4 D6+D7 + 40 Hz Fourier resample + PCA on
           all 52 subcarriers) ONCE. Pick the PC with the best in-band
           prominence and SAVE its right-singular-vector v (52-dim).
  Stage 2: LIVE — for each new CSI packet, project its 52-amp vector
           onto v to get one scalar "fused" sample. Append to a ring
           buffer of the last DISPLAY_S seconds.
  Stage 3: BPM update — every UPDATE_PERIOD seconds, Welch the last
           BPM_WIN_S seconds of fused-trace, find the in-band peak.

Why this is better than the previous version:
  - PC basis is fixed; trace is reproducible second-to-second.
  - Holding your breath visibly flattens the trace (we plot the actual
    projected signal, not a per-refresh re-normalised PC).
  - Smooth animation (we only append new points, not redraw everything).

Run:
    .venv/bin/python notebooks/27_live_demo.py
"""
from pathlib import Path
import os, sys, time, threading, collections
import numpy as np
import serial
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from csi_pipeline import parse_csi_bytes, peak_bpm, dwt_band_filter  # noqa: E402
from csi_serial import resolve_port  # noqa: E402

# ---- config ----
# PORT/BAUD are resolved at runtime in main(): CSI_PORT / CSI_BAUD if set,
# otherwise auto-detected by listening for the board that emits CSI.
FS_TARGET     = 40.0
BAND          = (0.15625, 0.625)      # 9.4–37.5 BPM
WARMUP_S      = 30.0                  # how long to gather before locking PC
DISPLAY_S     = 30.0                  # rolling window shown on screen
BPM_WIN_S     = 20.0                  # Welch window for live BPM
UPDATE_PERIOD = 0.5                   # plot refresh rate (s)
PLOT_MEDIAN_K = 5                     # moving-median window for display only
PRECHARGE_KEEP_S = 8.0                # warmup history drawn when the plot opens

DATA_IDX = list(np.r_[6:32, 33:59])   # 52 active HT20 subcarriers


# -------------------- raw-packet ring buffer (for warmup) -----------------
class RawBuffer:
    def __init__(self, max_seconds):
        self.max_seconds = max_seconds
        self.t = collections.deque()
        self.amp = collections.deque()
        self.lock = threading.Lock()

    def push(self, t, amp52):
        with self.lock:
            self.t.append(t); self.amp.append(amp52)
            cutoff = t - self.max_seconds
            while self.t and self.t[0] < cutoff:
                self.t.popleft(); self.amp.popleft()

    def snapshot(self):
        with self.lock:
            return np.array(self.t), np.array(self.amp)

    def __len__(self):
        with self.lock:
            return len(self.t)


# -------------------- fused 1-D ring buffer (for live phase) --------------
class FusedBuffer:
    def __init__(self, max_seconds, fs):
        self.fs = fs
        self.max_samples = int(max_seconds * fs)
        self.t = collections.deque()
        self.y = collections.deque()
        self.lock = threading.Lock()

    def push(self, t, y):
        with self.lock:
            self.t.append(t); self.y.append(y)
            while len(self.t) > self.max_samples:
                self.t.popleft(); self.y.popleft()

    def snapshot(self):
        with self.lock:
            return np.array(self.t), np.array(self.y)


# -------------------- reader thread (always running) ----------------------
def reader_thread(port, baud, raw_buf: RawBuffer,
                  fused_state: dict, fused_buf: FusedBuffer,
                  stop_evt: threading.Event, display=None):
    s = serial.Serial()
    s.port = port; s.baudrate = baud; s.timeout = 0.05
    s.dtr = False; s.rts = False
    s.open()
    print(f'[reader] opened {port} @ {baud}')
    n = 0
    while not stop_evt.is_set():
        line = s.readline().decode('utf-8', errors='replace')
        if not line or not line.startswith('CSI_DATA'):
            continue
        try:
            parts = line.strip().split(',', 25)
            if int(parts[5]) != 1 or int(parts[7]) != 1 or int(parts[24]) != 384:
                continue
            bytes128 = parse_csi_bytes(parts[25])
            imag = bytes128[0::2].astype(np.float32)
            real = bytes128[1::2].astype(np.float32)
            amp52 = np.sqrt(real * real + imag * imag)[DATA_IDX]
        except Exception:
            continue

        now = time.time()
        raw_buf.push(now, amp52)

        # Once PC is locked, project every new packet straight onto it.
        v = fused_state.get('v')
        if v is not None:
            mean = fused_state['mean']
            std = fused_state['std']
            sample = float(np.dot((amp52 - mean) / (std + 1e-9), v))
            fused_buf.push(now, sample)     # raw, for the BPM estimate
            if display is not None:
                display.push(now, sample)   # filtered once, for the plot

        n += 1
    s.close()
    print(f'[reader] stopped after {n} packets')


# -------------------- warmup: pick PC basis from first window ------------
def warmup_pick_pc(t_arr, amp_arr):
    """Returns (v, mean, std, bpm0, prom0) — the basis to project onto."""
    t = t_arr - t_arr[0]
    span = t[-1]
    t_grid = np.arange(0, span, 1.0 / FS_TARGET)
    feats = np.stack(
        [np.interp(t_grid, t, amp_arr[:, k]) for k in range(amp_arr.shape[1])],
        axis=1).astype(np.float32)

    feats_b = dwt_band_filter(feats, FS_TARGET).astype(np.float32)
    mean = feats_b.mean(0)
    std = feats_b.std(0)
    X = (feats_b - mean) / (std + 1e-9)
    # SVD: X = U S Vt   -> PC scores = U[:,i]*S[i] = X @ v_i  where v_i = Vt[i].T
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    nperseg = min(int(FS_TARGET * BPM_WIN_S), feats_b.shape[0])

    def prom(i):
        pc = U[:, i] * S[i]
        f, P = signal.welch(pc, fs=FS_TARGET, nperseg=nperseg)
        m = (f >= BAND[0]) & (f <= BAND[1])
        if not m.any(): return 0.0, pc
        return float(P[m].max() / (np.median(P[m]) + 1e-12)), pc

    n_pcs = min(4, U.shape[1])
    best_i, best_prom, best_pc = 0, 0.0, None
    for i in range(n_pcs):
        p, pc = prom(i)
        print(f'  warmup PC{i+1} prominence = {p:.1f}')
        if p > best_prom:
            best_prom, best_i, best_pc = p, i, pc

    v = Vt[best_i]                    # 52-dim basis vector
    # Match sign of warmup PC for visual consistency: if the trace
    # ended up mostly-negative, flip v so the live trace looks positive.
    if abs(best_pc.min()) > abs(best_pc.max()):
        v = -v
        best_pc = -best_pc
    bpm0, _ = peak_bpm(best_pc, FS_TARGET, lo=BAND[0], hi=BAND[1],
                       nperseg=nperseg, harmonic_guard=False)
    print(f'  -> chose PC{best_i+1}, initial BPM = {bpm0:.2f}, prom = {best_prom:.1f}')
    return v, mean, std, bpm0, best_prom, best_i + 1


# -------------------- live BPM helper on fused 1-D ring ------------------
def bpm_from_fused(t, y, fs_target=FS_TARGET):
    """Resample non-uniform fused samples to fs_target, then Welch."""
    if len(t) < 200:
        return None
    span = t[-1] - t[0]
    if span < 5.0:
        return None
    grid = np.arange(0, span, 1.0 / fs_target)
    yg = np.interp(grid, t - t[0], y)
    nperseg = min(int(fs_target * BPM_WIN_S), len(yg))
    bpm, _ = peak_bpm(yg, fs_target, lo=BAND[0], hi=BAND[1],
                      nperseg=nperseg, harmonic_guard=False)
    return bpm


# -------------------- streaming display filter ----------------------------
# A bandpass is the wrong shape for a live position display, even though it is
# the right shape for measuring a rate. Two symptoms give it away:
#
#   "I exhale and hold, and the trace sinks then drifts back to the middle."
#       A held breath is a constant offset. The bandpass's lower edge is a
#       high-pass, and a high-pass cannot represent a constant - so it decays
#       whatever you hold back to zero, with a time constant of about
#       1/(2*pi*0.156) = 1 second. Nothing to do with noise.
#
#   "A quick inhale dips down before it goes up."
#       Butterworth filters overshoot on a sharp edge. That pre-dip is the
#       filter ringing, not your chest moving the wrong way.
#
# So the display uses a different decomposition: smooth with a low-pass whose
# step response barely overshoots, and remove slow drift by subtracting a
# gently-tracking baseline instead of high-passing.
#
#   value = lowpass(x) - baseline(x)
#
# Bessel rather than Butterworth for the low-pass: Bessel is maximally flat in
# group delay, so edges arrive intact rather than ringing. The baseline is an
# exponential moving average with a long time constant, which acts as a very
# soft high-pass at 1/(2*pi*tau) with no ringing at all.
# 0.65 Hz = 39 BPM, just above the breathing band's upper edge so the second
# harmonic of a breath still gets through and the waveform keeps its shape.
# Measured on the paced capture: this is both smoother than the old bandpass
# (spikiness 0.033 vs 0.037) and slightly quicker (0.25 s lag vs 0.28 s).
# Raising it to 0.9 lets visible packet noise back in; lowering it to 0.5
# buys little smoothness for noticeably more delay.
DISPLAY_LP_HZ = 0.65
DISPLAY_LP_ORDER = 2
# The baseline's time constant is what removes slow wander. Measured on the
# paced capture, the drawn trace was 74% breathing, 13.5% slow drift under
# 0.1 Hz, 7% in-band interference and under 2% above the low-pass - so the
# visible roughness was mostly that drift, not high-frequency hash. Lowering
# the low-pass corner therefore does almost nothing for it (0.65 -> 0.30 Hz
# only moves smoothness 0.033 -> 0.026 while wrecking waveform fidelity,
# 0.76 -> 0.55); shortening this does. At 5 s the drift share falls to 9.6%
# and the breathing share rises to 78%.
#
# It can be this short only because of the hold gate below. An always-on 5 s
# average would erase a held breath in about five seconds; gated, it stops
# when you stop moving. The two changes are a pair.
BASELINE_TAU_S = 5.0         # soft high-pass corner ~0.03 Hz = 1.9 BPM

# The baseline still has to stop chasing a held breath. Any always-on average,
# however slow, eventually equals whatever you hold, and the trace it is
# subtracted from returns to zero - so the display says "middle" while the
# chest has not moved. But the baseline cannot simply be frozen either: real
# captures drift as the AGC and the room change, and a frozen baseline walks
# the trace off the top of the axis in a minute.
#
# So the baseline's speed is gated on whether the chest is actually moving.
# Measure the recent peak-to-peak swing of the smoothed signal; compare it to
# a reference of how big the swing is when the subject is breathing. Breathing
# -> full-speed baseline. Held breath -> near-frozen.
# Savitzky-Golay polishing, taken from SA-WiSense (arXiv 2507.17623), which
# is the closest published system to this rig: single-antenna ESP32, same 52
# HT20 subcarriers. Its pipeline ends with Hampel for impulse noise then
# Savitzky-Golay for smoothing, and the choice is deliberate - SG fits a
# low-order polynomial across a sliding window instead of averaging it, so it
# removes jitter without flattening the peaks and troughs of a breath. A
# moving average or a lower low-pass corner does the opposite: our own sweep
# showed the corner from 0.65 Hz down to 0.30 Hz bought almost no smoothness
# (0.033 -> 0.026) while waveform fidelity collapsed (0.76 -> 0.55).
#
# SG is centred, so it needs half a window of FUTURE samples. We refuse to
# rewrite drawn history, so instead each point is held back until its future
# half-window has arrived and is then drawn final. SG's smoothing becomes a
# fixed, honest 0.75 s delay rather than a retroactive edit.
#
# Measured on the paced capture with the basis fixed on the warmup: SSNR
# (in-band energy over out-of-band) rises 34.9 -> 117.8, a 3.4x cleaner trace,
# with BPM unchanged at 14.8 and no step overshoot reintroduced.
SG_WIN     = 61    # samples at 40 Hz = 1.5 s span, 0.75 s of it in the future
SG_POLY    = 3     # cubic: follows a breath's curvature, ignores jitter

# The trace is in arbitrary units - PC projections of standardised subcarrier
# amplitudes - so its size depends on the room, the placement and which PC won
# the warmup. An autoscaled axis is the usual answer, but it is a moving
# reference: the same breath draws at a different height depending on what
# else happened in the last thirty seconds, and it takes time to settle.
#
# Instead, divide by the size of a breath and fix the axis. The hold gate
# already tracks amp_ref, the peak-to-peak swing of normal breathing; half of
# that is the amplitude, so value/(amp_ref/2) puts an ordinary breath at about
# +/-1.0 and the y-axis can carry real units: "breaths, relative to yours".
#
# This also fixes the startup problem directly. amp_ref is established during
# the warmup replay, before the window ever opens, so the very first frame is
# already correctly scaled - there is nothing to settle.
# The gate's own 3 s window is too short to measure breathing DEPTH: at 15 BPM
# a breath lasts 4 s, so a 3 s window never sees a full peak-to-peak and
# reports a swing smaller than the real one, which made a normal breath draw
# at 1.4 rather than 1.0. Depth therefore gets its own, longer window - long
# enough for two breaths at 15 BPM and one at 6 BPM, the slow end of the band.
NORM_WIN_S = 12.0    # window used to measure how deep a breath is
NORM_ALPHA = 0.002   # how fast the divisor follows the breathing size
# Axis half-height, in breaths. 1.0 is a normal breath by construction - a
# clean sine normalises to exactly 1.00 - but real breathing varies in depth,
# and the deepest breaths on our paced captures peak at 1.67. 2.0 covers that
# with margin, so in ordinary use the axis never moves at all.
Y_FIXED    = 2.0

# Gross motion - shifting in the chair, an arm across the link - moves the CSI
# far more than a chest does, and in one direction. Two things then go wrong.
# The trace spikes off the axis, and worse, the lurch gets counted as a breath
# when the depth reference is measured, so "one breath" is redefined as
# something huge and every real breath afterwards is drawn tiny. That second
# effect is why motion seemed to break the sense of scale rather than just
# produce a spike.
#
# Both SA-WiSense and Katabi's Vital-Radio handle this the same way: detect
# gross-movement frames and exclude them, rather than trying to filter them.
# We do that here with a slope test, which is what separates the two cases.
# Breathing is bounded in speed. At the top of our band, 37.5 BPM, a breath of
# normal depth sweeps at most 2*pi*0.625 = 3.9 breaths per second. The
# threshold sits at 6.0, comfortably above that, because someone breathing
# both fast AND unusually deeply is still breathing - measured, a 2.5x-depth
# breath at 37.5 BPM tripped a 4.0 threshold on 87% of its samples.
#
# When motion is detected, for MOTION_HOLDOFF_S afterwards:
#   - the depth reference and the breathing reference stop updating, so the
#     lurch cannot redefine what a breath is,
#   - the drawn trace is slew-limited, so it bends rather than spiking.
# Nothing is discarded and no history is rewritten; the trace just refuses to
# move faster than a chest can.
MOTION_SLOPE    = 6.0   # breaths/second; above this it is not breathing
MOTION_HOLDOFF_S = 1.5  # keep references frozen this long after the last spike
MOTION_DEPTH_MULT = 3.0 # also ignore anything this many times a breath deep
MOTION_RECOVER_S = 20.0 # how long that extra test stays armed after motion
                        # (roughly four baseline time constants, long enough
                        #  for the baseline to absorb a displaced link)

HOLD_WIN_S     = 3.0   # window whose swing decides "moving or held"
HOLD_LO        = 0.20  # swing <= 20% of normal breathing: certainly held
HOLD_HI        = 0.50  # swing >= 50%: certainly breathing
HOLD_FLOOR     = 0.02  # residual baseline speed while held (tau ~ 8/0.02 min)
AMP_REF_TAU_S  = 45.0  # how fast the "normal breathing swing" reference fades


class StreamingDisplay:
    """Filters each sample once, as it arrives, and never revisits it.

    The obvious way to smooth a live trace is to bandpass the whole visible
    window on every redraw with sosfiltfilt. It looks beautiful and it is
    wrong: filtfilt runs forwards and backwards over whatever data exists at
    that moment, so every refresh recomputes the entire curve and the drawn
    past visibly rewrites itself. A breath you saw two seconds ago changes
    shape. That is fine for offline analysis and misleading in a live demo.

    This keeps persistent filter state instead, so a point, once drawn, is
    final. The cost is a small fixed delay rather than retroactive edits.

    Pipeline per sample:
      1. resample onto a uniform grid (arrival times are jittery; an IIR
         filter assumes even spacing)
      2. running median, to kill single-sample AGC spikes BEFORE anything
         else - a lone spike through an IIR rings into a smooth bump that
         looks exactly like a breath
      3. Bessel low-pass, for smoothing without step overshoot
      4. subtract a slow exponential baseline, to remove drift while letting
         a held breath stay held for several seconds
    """

    def __init__(self, fs, band, max_seconds, median_k=PLOT_MEDIAN_K,
                 tau_s=BASELINE_TAU_S):
        self.fs = fs
        self.dt = 1.0 / fs
        # `band` is accepted for interface compatibility with the BPM path;
        # only the low-pass corner is used here, see the note above.
        self.sos = signal.bessel(DISPLAY_LP_ORDER, DISPLAY_LP_HZ, btype='low',
                                 fs=fs, norm='delay', output='sos')
        self.zi = np.zeros((self.sos.shape[0], 2))
        self.med = collections.deque(maxlen=median_k)
        self.alpha = self.dt / max(tau_s, self.dt)   # EMA coefficient
        self.baseline = None
        # Hold detection: recent smoothed values, and a slow reference for how
        # wide the swing is when breathing normally.
        self.swing = collections.deque(maxlen=max(2, int(HOLD_WIN_S * fs)))
        self.amp_ref = 0.0
        self.amp_decay = np.exp(-self.dt / AMP_REF_TAU_S)
        self.norm = 0.0             # divisor that puts a breath at +/-1.0
        self.norm_swing = collections.deque(maxlen=max(2, int(NORM_WIN_S * fs)))
        self.depth_ref = 0.0        # peak-to-peak depth of a normal breath
        self.motion_left = 0.0      # seconds of motion hold-off remaining
        self.recover_left = 0.0     # seconds of post-motion recovery remaining
        self.prev_out = None        # last drawn value, for the slew limit
        self.prev_smoothed = None   # previous filtered value, for the slope
        self.n_motion = 0           # how many samples were slew-limited
        # Savitzky-Golay polish, applied last and emitted with a fixed delay.
        win = SG_WIN + 1 if SG_WIN % 2 == 0 else SG_WIN
        self.sg_half = win // 2
        self.sg_coef = signal.savgol_coeffs(win, SG_POLY, deriv=0, use='dot')
        self.sg_buf = collections.deque(maxlen=win)
        self.next_t = None          # next uniform-grid timestamp to emit
        self.prev = None            # previous raw (t, y), for interpolation
        self.max_samples = int(max_seconds * fs)
        self.t = collections.deque()
        self.y = collections.deque()
        self.lock = threading.Lock()

    def push(self, t, y):
        """Feed one raw projected sample at its true arrival time."""
        if self.prev is None:
            self.prev = (t, y)
            self.next_t = t
            return
        t0, y0 = self.prev
        # Emit every grid point that falls in (t0, t], interpolating linearly.
        while self.next_t <= t:
            frac = (self.next_t - t0) / (t - t0) if t > t0 else 0.0
            self._emit(self.next_t, y0 + (y - y0) * frac)
            self.next_t += self.dt
        self.prev = (t, y)

    def _emit(self, t, value):
        self.med.append(value)
        v = float(np.median(self.med))
        out, self.zi = signal.sosfilt(self.sos, [v], zi=self.zi)
        smoothed = float(out[0])

        # Detect gross motion FIRST, on the signal itself, before it reaches
        # any of the reference windows. Freezing the references after the fact
        # is not enough: the lurch would still be sitting inside the 12 s depth
        # window when the freeze expired, and would redefine "a breath" then.
        self.motion_left = max(0.0, self.motion_left - self.dt)
        self.recover_left = max(0.0, self.recover_left - self.dt)
        if self.prev_smoothed is not None and self.norm > 0.0:
            slope = abs(smoothed - self.prev_smoothed) / self.dt / self.norm
            if slope > MOTION_SLOPE:
                self.motion_left = MOTION_HOLDOFF_S
                self.recover_left = MOTION_RECOVER_S
                self.n_motion += 1
        self.prev_smoothed = smoothed

        # Seed the baseline on the first sample so the trace does not start
        # with a large step while the average climbs from zero.
        if self.baseline is None:
            self.baseline = smoothed
        else:
            # Feed the gate the detrended value, not the raw level. A lurch
            # that displaces the link permanently shifts the absolute level;
            # a window of raw levels spanning such a shift reports an enormous
            # swing and would redefine "a breath" as that shift. The baseline
            # removes it, so the windows see only how much things are moving.
            self.baseline += self.alpha * self._gate(
                smoothed - self.baseline) * (smoothed - self.baseline)

        # Savitzky-Golay across the window centred half a window back. Nothing
        # is drawn until its whole window exists, so a drawn point is final.
        self.sg_buf.append(smoothed - self.baseline)
        if len(self.sg_buf) < self.sg_buf.maxlen:
            return
        value = float(np.dot(self.sg_coef, np.fromiter(self.sg_buf, float)))
        t_centre = t - self.sg_half * self.dt

        # Normalise into "breaths", so the axis can be fixed. See NORM_* below.
        # amp_ref is the peak-to-peak swing of normal breathing, already
        # maintained by the hold gate; half of it is the amplitude, so
        # dividing by it puts an ordinary breath at about +/-1.0.
        if self.depth_ref > 0.0:
            target = self.depth_ref / 2.0
            if self.norm <= 0.0:
                self.norm = target            # first estimate, adopt at once
            else:
                # Ease toward it. A sudden lurch spikes amp_ref, and following
                # that instantly would shrink the whole trace for a moment.
                self.norm += NORM_ALPHA * (target - self.norm)
            value /= max(self.norm, 1e-9)

        # Slew limit, in normalised units so the threshold means the same
        # thing in every room. Detection already happened up in _emit, before
        # the reference windows were fed, so a lurch never enters them.
        if self.prev_out is None:
            self.prev_out = value
        else:
            step = value - self.prev_out
            max_step = MOTION_SLOPE * self.dt
            if abs(step) > max_step:
                value = self.prev_out + np.sign(step) * max_step
            self.prev_out = value

        with self.lock:
            self.t.append(t_centre)
            self.y.append(value)
            while len(self.t) > self.max_samples:
                self.t.popleft()
                self.y.popleft()

    def _gate(self, detrended):
        """How fast should the baseline move right now? 0 = held, 1 = breathing.

        Walk through a hold: you exhale and stop. For the next 3 seconds the
        window still holds part of the last breath, so the swing is large and
        the gate stays open. Once the window contains only held-breath data the
        swing collapses to the noise floor, the ratio against the breathing
        reference drops under HOLD_LO, and the gate shuts. The baseline stops
        where it was - roughly the mid-point of your breathing - so the trace
        stays pinned at the exhaled level for as long as you hold it.

        Resume breathing and the swing recovers within a breath, the gate
        reopens, and the baseline goes back to tracking drift normally.
        """
        # During motion the windows are not fed at all, so the lurch is simply
        # never part of what "a breath" or "still breathing" is measured from.
        #
        # The timer alone is not enough. A lurch that displaces the link leaves
        # the baseline several breaths away from the signal, and it takes a few
        # time constants to catch up - long after MOTION_HOLDOFF_S expires. So
        # also refuse any value that is implausibly large next to the breathing
        # we already know about. That test ends itself: once the baseline has
        # caught up, values are normal size again and feeding resumes.
        #
        # It applies only during the recovery window after real motion. Left
        # always on it can lock itself low: a depth reference that starts too
        # small rejects exactly the larger values that would grow it, and never
        # recovers. Tied to detected motion, there is no such loop.
        plausible = (self.recover_left <= 0.0 or self.depth_ref <= 0.0
                     or abs(detrended) <= MOTION_DEPTH_MULT * self.depth_ref)
        if self.motion_left <= 0.0 and plausible:
            self.swing.append(detrended)
            self.norm_swing.append(detrended)
        if len(self.swing) < self.swing.maxlen:
            return 1.0                      # not enough history yet; track
        arr = np.fromiter(self.swing, float)
        # Percentiles, not min/max: one AGC spike should not read as breathing.
        swing = float(np.percentile(arr, 95) - np.percentile(arr, 5))

        if self.amp_ref <= 0.0:
            self.amp_ref = swing
            return 1.0
        ratio = swing / (self.amp_ref + 1e-12)
        gate = (ratio - HOLD_LO) / (HOLD_HI - HOLD_LO)
        gate = float(min(1.0, max(0.0, gate)))

        # Update the reference only while something is moving. Frozen during a
        # hold, the reference cannot fade away and silently re-open the gate,
        # so a two-minute breath-hold stays pinned just like a ten-second one.
        # ...and only while it is a chest doing the moving. Without the motion
        # check, one lurch redefines "a breath" as something huge and every
        # real breath afterwards draws tiny.
        if gate > 0.0 and self.motion_left <= 0.0:
            self.amp_ref = max(swing, self.amp_ref * self.amp_decay)
            # Same freeze rule for the depth reference, and for a second
            # reason: during a hold the swing collapses, and a divisor that
            # followed it down would magnify the held trace into nonsense.
            deep = float(np.percentile(self.norm_swing, 95)
                         - np.percentile(self.norm_swing, 5))
            self.depth_ref = max(deep, self.depth_ref * self.amp_decay)

        return HOLD_FLOOR + (1.0 - HOLD_FLOOR) * gate

    def drop_before(self, t_cut):
        """Discard drawn history older than t_cut, keeping all filter state.

        Used once after the warmup replay. The filter has to see every warmup
        sample to be settled by the time the live phase starts, but the plot
        does not have to show all of it - and showing all of it is what made
        the y-axis open tall and stay tall for half a minute.
        """
        with self.lock:
            while self.t and self.t[0] < t_cut:
                self.t.popleft()
                self.y.popleft()

    def snapshot(self):
        with self.lock:
            return np.array(self.t), np.array(self.y)


# -------------------- y-axis --------------------------------------------
# The axis is fixed at +/-Y_FIXED breaths because the trace is normalised (see
# NORM_ALPHA above). Fixed is the whole point: a given height on screen always
# means the same thing, so you can compare a shallow breath now against a deep
# one a minute ago, which an autoscaled axis makes impossible.
#
# The one thing a fixed axis must not do is silently cut off a deep breath, so
# it expands if the trace would leave the box, and comes back down on a 2 s
# time constant. In normal breathing it never moves.
Y_SHRINK_TAU = 2.0    # seconds to ease back down after an expansion
Y_HEADROOM   = 1.10   # margin when expanding past the fixed height


class YScale:
    def __init__(self, window_s=DISPLAY_S, period_s=UPDATE_PERIOD):
        self.window_s = window_s
        self.beta = min(1.0, period_s / Y_SHRINK_TAU)
        self.cur = Y_FIXED

    def update(self, t_rel, y):
        vis = y[t_rel >= -self.window_s]
        if len(vis) < 5:
            vis = y
        # Never below the fixed height; above it only to avoid clipping.
        target = max(Y_FIXED, float(np.abs(vis).max()) * Y_HEADROOM)
        if target > self.cur:
            self.cur = target                      # expand immediately
        else:
            self.cur += self.beta * (target - self.cur)   # settle back gently
            if self.cur <= Y_FIXED * 1.01:
                self.cur = Y_FIXED       # snap, so it truly comes to rest
        return self.cur


# -------------------- main -----------------------------------------------
def main():
    import matplotlib.pyplot as plt
    # Interactive backend differs by OS: 'macosx' exists only on macOS, and on
    # Windows/Linux we need Qt (if a binding is installed) or TkAgg (ships with
    # CPython). Use switch_backend, not matplotlib.use — `use` accepts a
    # backend whose GUI library is missing and only blows up later when the
    # first window is created; switch_backend imports it now and raises here.
    candidates = (['macosx'] if sys.platform == 'darwin' else []) + ['QtAgg', 'TkAgg']
    for backend in candidates:
        try:
            plt.switch_backend(backend)
            break
        except Exception:
            continue
    else:
        print('ERROR: no interactive matplotlib backend available. '
              'Install one with:  pip install pyqt6\n'
              'Or use notebooks/26_first_capture_test.py, which writes a PNG.')
        return
    print(f'[plot] backend = {plt.get_backend()}')

    # Find the board before anything else — failing here costs 2 seconds,
    # whereas failing after the 30 s warmup wastes the user's time.
    try:
        port, baud = resolve_port()
    except RuntimeError as e:
        print(f'\n{e}')
        return

    raw_buf   = RawBuffer(max_seconds=WARMUP_S + 5)
    fused_buf = FusedBuffer(max_seconds=DISPLAY_S + 5, fs=FS_TARGET)
    fused_state = {}       # filled in after warmup; reader checks each tick
    display = StreamingDisplay(FS_TARGET, BAND, max_seconds=DISPLAY_S + 5)
    stop_evt = threading.Event()
    th = threading.Thread(target=reader_thread,
                          args=(port, baud, raw_buf, fused_state,
                                fused_buf, stop_evt, display),
                          daemon=True)
    th.start()

    # ---- Stage 1: WARMUP ----
    # ASCII only in console output: the Windows console codepage mangles
    # em-dashes into mojibake. Matplotlib titles are fine, they render Unicode.
    print(f'Warming up for {WARMUP_S:.0f}s - breathe normally on the metronome.')
    t0 = time.time()
    while time.time() - t0 < WARMUP_S:
        rem = WARMUP_S - (time.time() - t0)
        print(f'  warmup: {rem:.0f}s left  ({len(raw_buf)} packets)', end='\r')
        time.sleep(1.0)
    print()
    t_arr, amp_arr = raw_buf.snapshot()
    if len(t_arr) < 200:
        print('ERROR: warmup got too few packets'); stop_evt.set(); return

    print(f'Warmup done: {len(t_arr)} packets, span {t_arr[-1]-t_arr[0]:.1f}s')
    v, mean, std, bpm0, prom0, pc_n = warmup_pick_pc(t_arr, amp_arr)

    # Replay the warmup packets through the display filter before going live.
    # Two reasons: the causal filter starts from rest and would otherwise
    # swing for its first several seconds, and this way the plot opens with
    # real history on screen instead of an empty axis that fills over 30 s.
    for ti, ai in zip(t_arr, amp_arr):
        display.push(ti, float(np.dot((ai - mean) / (std + 1e-9), v)))
    # ...but only keep the tail of it on screen. All 30 s went through the
    # filter, so it is fully settled; drawing only the last few seconds means
    # the y-axis is sized by recent breathing from the very first frame
    # instead of by whatever the largest thing in the warmup happened to be.
    display.drop_before(t_arr[-1] - PRECHARGE_KEEP_S)
    print(f'Pre-charged display with {len(t_arr)} warmup packets '
          f'(showing the last {PRECHARGE_KEEP_S:.0f}s).')

    # Install the basis last, so the reader starts projecting only once the
    # display filter is settled - otherwise live samples would interleave
    # with the replay above and arrive out of order.
    fused_state['mean'] = mean
    fused_state['std']  = std
    fused_state['v']    = v
    print(f'Locked basis (PC{pc_n}). Live phase starts now.')

    # ---- Stage 2/3: LIVE PLOT ----
    fig, ax = plt.subplots(figsize=(14, 6))
    # Make the axes track the figure size — `tight_layout` only fires
    # once; explicit margins let the plot fill the window on resize.
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.08)
    line, = ax.plot([], [], lw=1.2)
    title = ax.set_title(f'warmup BPM {bpm0:.1f} — collecting fused samples...')
    ax.set_xlabel(f'time (latest is right)   |   PC{pc_n}, warmup prom {prom0:.0f}')
    ax.set_ylabel('breaths  (1.0 = your normal depth)')
    # Reference lines at +/-1: a normal breath should just reach them.
    for lvl in (-1.0, 1.0):
        ax.axhline(lvl, ls=':', lw=0.8, color='0.6')
    ax.axhline(0.0, ls='-', lw=0.8, color='0.85')
    ax.grid(alpha=0.3)
    plt.show(block=False)

    last_bpm_update = 0.0
    last_bpm = bpm0
    y_scale = YScale()
    try:
        while plt.fignum_exists(fig.number):
            t_arr, y_arr = fused_buf.snapshot()
            t_d, y_disp = display.snapshot()
            if len(t_d) > 5:
                # Already filtered, once, when each sample arrived. Nothing
                # here recomputes history - we only slide the time axis.
                t_rel = t_d - t_d[-1]
                line.set_data(t_rel, y_disp)
                ax.set_xlim(-DISPLAY_S, 0.5)
                if len(y_disp) > 10:
                    ymax = y_scale.update(t_rel, y_disp)
                    ax.set_ylim(-ymax, ymax)

                now = time.time()
                # The display is pre-charged with warmup history, so it has
                # data before a single live sample has landed in fused_buf.
                # Guard the BPM block separately or t_arr[-1] blows up.
                if now - last_bpm_update > 1.0 and len(t_arr) > 5:
                    bpm = bpm_from_fused(t_arr, y_arr)
                    if bpm is not None and not np.isnan(bpm):
                        last_bpm = bpm
                    last_bpm_update = now
                    pps = len(t_arr) / max(1e-3, (t_arr[-1] - t_arr[0]))
                    title.set_text(f'BPM: {last_bpm:.1f}   |   '
                                   f'fused samples: {len(t_arr)}   |   '
                                   f'rate: {pps:.0f} PPS')

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(UPDATE_PERIOD)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        th.join(timeout=2.0)
        print('done.')


if __name__ == '__main__':
    main()
