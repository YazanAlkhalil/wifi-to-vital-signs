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
BASELINE_TAU_S = 8.0         # hold decays over ~8 s instead of ~1 s
                             # (soft high-pass corner ~0.02 Hz = 1.2 BPM)


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

        # Seed the baseline on the first sample so the trace does not start
        # with a large step while the average climbs from zero.
        if self.baseline is None:
            self.baseline = smoothed
        else:
            self.baseline += self.alpha * (smoothed - self.baseline)

        with self.lock:
            self.t.append(t)
            self.y.append(smoothed - self.baseline)
            while len(self.t) > self.max_samples:
                self.t.popleft()
                self.y.popleft()

    def snapshot(self):
        with self.lock:
            return np.array(self.t), np.array(self.y)


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
    print(f'Pre-charged display with {len(t_arr)} warmup packets.')

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
    ax.set_ylabel('a.u.')
    ax.grid(alpha=0.3)
    plt.show(block=False)

    last_bpm_update = 0.0
    last_bpm = bpm0
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
                    ymax = max(1e-3, np.percentile(np.abs(y_disp), 99) * 1.2)
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
