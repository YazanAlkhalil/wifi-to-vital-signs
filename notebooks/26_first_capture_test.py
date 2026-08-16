"""
26 — First live capture + BPM estimate from the two-ESP32 rig.

Captures `DURATION` seconds of CSI from the AP serial port, saves it
as a CSV that `load_csi()` can parse, then runs the paper_mode-style
preprocessing inline (DWT db4 + 40 Hz Fourier resample + PCA on all 52
subcarriers + auto-pick PC by in-band peak prominence) and reports
the Welch-peak BPM estimate. Plots fused trace + PSD.

No belt is needed — purely CSI vs metronome.

Usage:
    .venv/bin/python notebooks/26_first_capture_test.py
"""
from pathlib import Path
import os, sys, time
import numpy as np
import serial
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from csi_pipeline import load_csi, dwt_band_filter, peak_bpm  # noqa: E402
from csi_serial import resolve_port  # noqa: E402

# ---- capture config ----
# PORT/BAUD are resolved at runtime in main(): CSI_PORT / CSI_BAUD if set,
# otherwise auto-detected by listening for the board that emits CSI.
DURATION  = 60.0                      # seconds
FS_TARGET = 40.0                      # paper-mode target rate
BAND      = (0.15625, 0.625)          # 9.4–37.5 BPM (matches DWT D6+D7)
OUT_DIR   = ROOT / 'captures'
OUT_DIR.mkdir(exist_ok=True)


# ESP32-CSI-Tool prints this header once, at board boot. If we attach to a
# board that is already running we never see it, so we write it ourselves —
# otherwise pandas treats the first CSI row as column names and load_csi dies
# with "invalid literal for int()".
CSI_HEADER = ('type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,'
              'not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,'
              'ampdu_cnt,channel,secondary_channel,local_timestamp,ant,'
              'sig_len,rx_state,real_time_set,real_timestamp,len,CSI_DATA')
N_FIELDS = len(CSI_HEADER.split(','))       # 26


def capture(port, baud, duration, out_path):
    """Stream serial → CSV. Appends a host wall-clock column so the file has
    the same shape as earlier captures. Note `load_csi` reads the *firmware's*
    `real_timestamp` (column 24), since that name appears first."""
    s = serial.Serial()
    s.port = port; s.baudrate = baud; s.timeout = 0.1
    s.dtr = False; s.rts = False
    s.open()
    n_csi = n_skipped = 0
    with open(out_path, 'w') as f:
        f.write(CSI_HEADER + ',real_timestamp\n')
        f.flush()
        end = time.time() + duration
        while time.time() < end:
            line = s.readline().decode('utf-8', errors='replace').strip()
            if not line:
                continue
            if line.startswith('type,role,'):
                continue                # already written above
            if not line.startswith('CSI_DATA'):
                continue
            # The very first readline can begin mid-line, and a dropped byte
            # at 921600 baud can splice two rows together. Either way the
            # field count is wrong and the row would poison the parse.
            if line.count(',') + 1 != N_FIELDS or not line.rstrip().endswith(']'):
                n_skipped += 1
                continue
            f.write(f'{line},{time.time():.6f}\n')
            n_csi += 1
    s.close()
    if n_skipped:
        print(f'  (skipped {n_skipped} malformed line(s))')
    return n_csi


def paper_mode_bpm(t, H):
    """Inline paper_mode=True estimator. Returns (bpm, fused, fs, t_grid)."""
    data_idx = list(np.r_[6:32, 33:59])     # 52 active HT20 data subcarriers
    feats = np.abs(H[:, data_idx]).astype(np.float32)

    # Fourier-method resample to 40 Hz (paper).
    src_fs = 1.0 / np.median(np.diff(t))
    t_unif = np.arange(t[0], t[-1], 1.0 / src_fs)
    feats_unif = np.stack(
        [np.interp(t_unif, t, feats[:, k]) for k in range(feats.shape[1])],
        axis=1)
    n_target = int(round(len(t_unif) * FS_TARGET / src_fs))
    feats_u = signal.resample(feats_unif, n_target, axis=0)
    t_grid = np.arange(n_target) / FS_TARGET + t[0]

    # DWT db4, keep D6+D7.
    feats_b = dwt_band_filter(feats_u, FS_TARGET).astype(np.float32)

    # PCA fuse over all 52 columns.
    X = (feats_b - feats_b.mean(0)) / (feats_b.std(0) + 1e-9)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    # Cap Welch window at the actual signal length (short captures lose
    # one Welch segment if we ask for >60 s and only have 59).
    nperseg = min(int(FS_TARGET * 60), feats_b.shape[0])

    def in_band_prominence(pc_idx):
        pc = U[:, pc_idx] * S[pc_idx]
        f, P = signal.welch(pc, fs=FS_TARGET, nperseg=nperseg)
        m = (f >= BAND[0]) & (f <= BAND[1])
        if not m.any(): return 0.0
        return float(P[m].max() / (np.median(P[m]) + 1e-12))

    n_pcs = min(4, U.shape[1])
    proms = [in_band_prominence(i) for i in range(n_pcs)]
    pc_idx = int(np.argmax(proms))
    fused = U[:, pc_idx] * S[pc_idx]
    if abs(fused.min()) > abs(fused.max()):
        fused = -fused

    bpm, prom = peak_bpm(fused, FS_TARGET, lo=BAND[0], hi=BAND[1],
                         nperseg=nperseg, harmonic_guard=False)
    info = dict(pc_idx=pc_idx, proms=proms, prominence=prom)
    return bpm, fused, FS_TARGET, t_grid, info


def main():
    try:
        port, baud = resolve_port()
    except RuntimeError as e:
        print(f'\n{e}')
        return

    out_csv = OUT_DIR / f'capture_{int(time.time())}.csv'
    print(f'Capturing {DURATION:.0f}s from {port} @ {baud}')
    print(f'Output: {out_csv}')
    print('\n*** Start breathing on the metronome NOW (press Ctrl+C to abort) ***\n')
    t0 = time.time()
    n_csi = capture(port, baud, DURATION, out_csv)
    elapsed = time.time() - t0
    pps = n_csi / elapsed
    print(f'Done. {n_csi} CSI packets in {elapsed:.1f}s → {pps:.1f} PPS')
    if n_csi < 100:
        print('ERROR: too few packets. Is the AP streaming?'); return

    print('\nParsing CSI...')
    t, H = load_csi(out_csv)
    print(f'After HT20 filter: {len(t)} packets, span {t[-1]:.1f}s, '
          f'median rate {1.0/np.median(np.diff(t)):.1f} Hz')
    if len(t) < 60 * 20:
        print('WARNING: < 20 PPS after HT20 filter; result may be noisy.')

    bpm, fused, fs, t_grid, info = paper_mode_bpm(t, H)
    print(f'\n*** ESTIMATED BREATHING RATE: {bpm:.2f} BPM ***')
    print(f'    chose PC{info["pc_idx"]+1} (proms = '
          + ', '.join(f'{p:.1f}' for p in info["proms"]) + ')')

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        axes[0].plot(t_grid - t_grid[0], fused, lw=0.6)
        axes[0].set(title=f'Fused CSI trace — estimated {bpm:.2f} BPM (PC{info["pc_idx"]+1})',
                    xlabel='time (s)', ylabel='a.u.')
        axes[0].grid(alpha=0.3)
        f, P = signal.welch(fused, fs=fs, nperseg=int(fs*60))
        m = (f >= 0.05) & (f <= 1.0)
        axes[1].semilogy(f[m]*60, P[m])
        axes[1].axvline(bpm, color='r', ls='--', lw=1, label=f'{bpm:.2f} BPM')
        axes[1].set(xlabel='BPM', ylabel='PSD',
                    title='Power spectrum (3–60 BPM)')
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        png = out_csv.with_suffix('.png')
        fig.savefig(png, dpi=120)
        print(f'\nPlot saved: {png}')
    except Exception as e:
        print(f'(plotting skipped: {e})')


if __name__ == '__main__':
    main()
