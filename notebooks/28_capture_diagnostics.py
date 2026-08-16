"""
28 — Diagnose a CSI capture that isn't showing breathing.

Answers, in order, the questions that actually distinguish the causes:

  1. LINK HEALTH   Are we getting usable HT20 packets, fast and evenly?
                   A low HT20 fraction or a jittery rate breaks the FFT.
  2. AGC STABILITY Does the whole subcarrier bank jump at once? The ESP32's
                   automatic gain control rescales every subcarrier together;
                   those steps swamp the ~1% modulation breathing produces.
  3. SIGNAL        Is there any 0.15-0.63 Hz content at all, per subcarrier
                   and after PCA? If nothing anywhere, it's geometry.

Compares against `captures/capture_1778960307.csv`, the recording that did
yield a correct 11.66 BPM, so you get a baseline rather than raw numbers you
have to interpret cold.

Usage:
    python notebooks/28_capture_diagnostics.py <capture.csv> [--plot]
    python notebooks/28_capture_diagnostics.py --baseline
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from csi_pipeline import load_csi, dwt_band_filter  # noqa: E402

BAND = (0.15625, 0.625)          # 9.4-37.5 BPM
FS_TARGET = 40.0
DATA_IDX = list(np.r_[6:32, 33:59])
BASELINE = ROOT / 'captures' / 'capture_1778960307.csv'


def link_health(path):
    """Packet-level stats, read straight from the CSV before any filtering."""
    df = pd.read_csv(path)
    total = len(df)
    ht20 = ((df['sig_mode'] == 1) & (df['bandwidth'] == 1) & (df['len'] == 384))
    n_ht20 = int(ht20.sum())

    t = df.loc[ht20, 'real_timestamp'].to_numpy(float)
    span = t[-1] - t[0] if len(t) > 1 else 0.0
    dt = np.diff(t) if len(t) > 1 else np.array([np.nan])

    out = {
        'total_rows': total,
        'ht20_rows': n_ht20,
        'ht20_frac': n_ht20 / max(1, total),
        'span_s': span,
        'pps': n_ht20 / span if span > 0 else 0.0,
        'dt_median': float(np.median(dt)),
        'dt_p99': float(np.percentile(dt, 99)),
        'max_gap': float(np.max(dt)),
        'rssi_mean': float(df.loc[ht20, 'rssi'].mean()),
        'rssi_std': float(df.loc[ht20, 'rssi'].std()),
        'mcs_counts': df.loc[ht20, 'mcs'].value_counts().to_dict(),
        'noise_floor': float(df.loc[ht20, 'noise_floor'].mean()),
    }
    return out


def resample_amp(path):
    """(t_grid, amp[n,52]) on a uniform FS_TARGET grid."""
    t, H = load_csi(path)
    amp = np.abs(H[:, DATA_IDX]).astype(np.float32)
    grid = np.arange(0, t[-1], 1.0 / FS_TARGET)
    out = np.stack([np.interp(grid, t, amp[:, k]) for k in range(amp.shape[1])],
                   axis=1)
    return grid, out


def agc_stability(amp):
    """How much of the variation is all-subcarriers-at-once (i.e. gain steps)?

    Breathing changes subcarriers differentially — some up, some down, by
    small amounts. An AGC step multiplies the whole bank at once. If the
    common mode dominates, the AGC is the signal you're looking at.
    """
    mean_trace = amp.mean(axis=1)
    common = mean_trace / (mean_trace.mean() + 1e-9)
    d = np.abs(np.diff(common))
    jumps = int((d > 0.05).sum())          # >5% bank-wide step between samples

    # Fraction of total variance explained by the common mode.
    centred = amp - amp.mean(0)
    total_var = float((centred ** 2).sum())
    cm = centred.mean(axis=1, keepdims=True)
    common_var = float((cm ** 2).sum() * amp.shape[1])
    return {
        'common_mode_var_frac': common_var / (total_var + 1e-12),
        'gain_jumps': jumps,
        'jumps_per_min': jumps / (len(amp) / FS_TARGET / 60.0 + 1e-9),
    }


def band_content(amp):
    """In-band vs out-of-band power, per subcarrier and after PCA."""
    filt = dwt_band_filter(amp, FS_TARGET).astype(np.float32)
    nperseg = min(int(FS_TARGET * 60), filt.shape[0])

    # Per-subcarrier prominence: in-band peak over in-band median.
    proms = []
    for k in range(filt.shape[1]):
        f, P = signal.welch(filt[:, k], fs=FS_TARGET, nperseg=nperseg)
        m = (f >= BAND[0]) & (f <= BAND[1])
        if m.any():
            proms.append(float(P[m].max() / (np.median(P[m]) + 1e-12)))
    proms = np.array(proms)

    X = (filt - filt.mean(0)) / (filt.std(0) + 1e-9)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    var_frac = (S ** 2) / (S ** 2).sum()

    pcs = []
    for i in range(min(4, U.shape[1])):
        pc = U[:, i] * S[i]
        f, P = signal.welch(pc, fs=FS_TARGET, nperseg=nperseg)
        m = (f >= BAND[0]) & (f <= BAND[1])
        peak_f = float(f[m][np.argmax(P[m])]) if m.any() else float('nan')
        prom = float(P[m].max() / (np.median(P[m]) + 1e-12)) if m.any() else 0.0
        pcs.append({'pc': i + 1, 'var_frac': float(var_frac[i]),
                    'peak_bpm': peak_f * 60, 'prominence': prom})
    return proms, pcs


def report(path, label):
    print(f'\n{"=" * 66}\n{label}: {Path(path).name}\n{"=" * 66}')
    h = link_health(path)
    print('-- link health')
    print(f'   rows {h["total_rows"]}, HT20 {h["ht20_rows"]} '
          f'({h["ht20_frac"]*100:.0f}% of rows)')
    print(f'   span {h["span_s"]:.1f}s, {h["pps"]:.0f} PPS after HT20 filter')
    print(f'   packet gap: median {h["dt_median"]*1000:.1f} ms, '
          f'p99 {h["dt_p99"]*1000:.1f} ms, worst {h["max_gap"]*1000:.0f} ms')
    print(f'   RSSI {h["rssi_mean"]:.1f} +/- {h["rssi_std"]:.1f} dBm, '
          f'noise floor {h["noise_floor"]:.0f}')
    print(f'   MCS mix {h["mcs_counts"]}')

    grid, amp = resample_amp(path)
    a = agc_stability(amp)
    print('-- AGC stability')
    print(f'   common-mode variance {a["common_mode_var_frac"]*100:.1f}% of total')
    print(f'   bank-wide gain jumps >5%: {a["gain_jumps"]} '
          f'({a["jumps_per_min"]:.0f}/min)')

    proms, pcs = band_content(amp)
    print('-- breathing-band content')
    print(f'   per-subcarrier prominence: median {np.median(proms):.1f}, '
          f'best {proms.max():.1f}, '
          f'{int((proms > 5).sum())}/{len(proms)} subcarriers above 5')
    for p in pcs:
        print(f'   PC{p["pc"]}: {p["var_frac"]*100:5.1f}% var, '
              f'peak {p["peak_bpm"]:5.1f} BPM, prominence {p["prominence"]:.1f}')
    return h, a, proms, pcs


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    do_plot = '--plot' in sys.argv

    if '--baseline' in sys.argv or not args:
        report(BASELINE, 'BASELINE (known good, 11.66 BPM)')
        if not args:
            print('\nPass a capture path to compare against this baseline.')
            return

    target = Path(args[0])
    if not target.exists():
        print(f'No such file: {target}')
        return
    if '--baseline' not in sys.argv:
        report(BASELINE, 'BASELINE (known good, 11.66 BPM)')
    report(target, 'YOUR CAPTURE')

    print('\n-- how to read this')
    print('   HT20% or PPS far below baseline -> link is falling back to')
    print('     non-HT rates; move boards closer or change Wi-Fi channel.')
    print('   common-mode variance high / many gain jumps -> AGC is dominating;')
    print('     breathing is buried under gain steps.')
    print('   link and AGC fine but all prominences low -> geometry: the chest')
    print('     is not intercepting the dominant path.')

    if do_plot:
        plot(target)


def plot(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid, amp = resample_amp(path)
    filt = dwt_band_filter(amp, FS_TARGET).astype(np.float32)
    fig, ax = plt.subplots(4, 1, figsize=(11, 12))

    ax[0].plot(grid, amp.mean(1), lw=0.6)
    ax[0].set(title='Mean amplitude across subcarriers (AGC steps show as '
                    'flat shelves)', xlabel='s', ylabel='a.u.')

    im = ax[1].imshow(amp.T, aspect='auto', origin='lower',
                      extent=[0, grid[-1], 0, amp.shape[1]])
    ax[1].set(title='Raw amplitude, subcarrier vs time', xlabel='s',
              ylabel='subcarrier')
    fig.colorbar(im, ax=ax[1])

    var = filt.var(0)
    best = np.argsort(var)[-5:]
    for k in best:
        ax[2].plot(grid, filt[:, k], lw=0.6, label=f'sc {k}')
    ax[2].set(title='5 highest-variance subcarriers after band filter',
              xlabel='s', ylabel='a.u.')
    ax[2].legend(fontsize=7)

    X = (filt - filt.mean(0)) / (filt.std(0) + 1e-9)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    nperseg = min(int(FS_TARGET * 60), filt.shape[0])
    for i in range(min(3, U.shape[1])):
        f, P = signal.welch(U[:, i] * S[i], fs=FS_TARGET, nperseg=nperseg)
        m = (f > 0.05) & (f < 1.2)
        ax[3].semilogy(f[m] * 60, P[m], lw=0.8, label=f'PC{i+1}')
    ax[3].axvspan(BAND[0] * 60, BAND[1] * 60, alpha=0.15, color='green')
    ax[3].set(title='PC spectra (green = breathing band)', xlabel='BPM',
              ylabel='PSD')
    ax[3].legend()

    fig.tight_layout()
    out = Path(path).with_name(Path(path).stem + '_diag.png')
    fig.savefig(out, dpi=110)
    print(f'\nPlot saved: {out}')


if __name__ == '__main__':
    main()
