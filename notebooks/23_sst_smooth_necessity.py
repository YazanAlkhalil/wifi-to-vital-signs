"""
23 — Is SST's layer-2 smoothing needed?

Previous sweep started at smooth=5s. We never tested smooth=0 (raw SST
ridge with no smoothing). Also test exponential smoothing as a possibly-
better-feel alternative to rectangular for the live demo.

For each variant, report slide MAE on the new (n_pcs=4) fused traces.

Variants:
  - smooth=0 (raw SST)
  - rectangular smooth=1, 2, 5, 10, 20, 30 s
  - exponential half-life=2, 5, 10, 15, 30 s
"""
from pathlib import Path
import re, sys, warnings, time
import numpy as np
from scipy import signal

warnings.filterwarnings('ignore')
import os
os.environ['SSQ_GPU'] = '0'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from csi_pipeline import (
    estimate_breathing, sliding_bpm, load_belt,
)

OA = (ROOT / 'datasets' /
      'Respiration Rate Measurement Validity and Repeatability of '
      'Ubiquitous Non-contact Wi-Fi Sensing for Older Adults in Care')
SD = ROOT / 'datasets' / 'Sleep Disturbances Dataset'
BAND = (0.15625, 0.625)
WAVELET = 'morlet'
NV = 32


def find_belt_validity(target_bpm):
    for p in (OA / 'Validity' / 'GT Neulog RR Belt Sensor').glob('*.csv'):
        m = re.search(r'(\d+)\s*bpm', p.stem.lower())
        if m and int(m.group(1)) == target_bpm:
            return p


def collect_pairs():
    pairs = []
    for tgt in range(12, 29):
        csi = OA / 'Validity' / 'Wi-Fi Sensor RR' / f'Val-{tgt}BPM.csv'
        belt = find_belt_validity(tgt)
        if csi.exists() and belt:
            pairs.append(('OA-Validity', f'Val-{tgt}BPM', csi, belt))
    for n in range(1, 31):
        csi  = OA / 'Reliability' / 'Wi-Fi Sensor RR' / f'CSI-{n}.csv'
        belt = OA / 'Reliability' / 'GT Neulog RR Belt Sensor' / f'Belt-{n}.csv'
        if csi.exists() and belt.exists():
            pairs.append(('OA-Reliability', f'rep-{n}', csi, belt))
    pairs.append(('SD-Vital Signs', 'paced-12-BPM',
                  SD / 'Vital Signs' / 'Breathing - 12 BPM - CSI.csv',
                  SD / 'Vital Signs' / 'Breathing - Belt & HR.csv'))
    pairs.append(('SD-Apnoea-OSA', 'OSA',
                  SD / 'Sleep Apnoa' / 'OSA - CSI.csv',
                  SD / 'Sleep Apnoa' / 'OSA - Belt.csv'))
    pairs.append(('SD-Apnoea-CSA', 'CSA',
                  SD / 'Sleep Apnoa' / 'CSA - CSI.csv',
                  SD / 'Sleep Apnoa' / 'CSA - Belt.csv'))
    return pairs


def sst_raw(x, fs, band=BAND):
    """Compute SST and return (t, raw bpm-per-time) — no smoothing."""
    from ssqueezepy import ssq_cwt
    sos = signal.butter(2, band[1] * 4, btype='low', fs=fs, output='sos')
    x = signal.sosfiltfilt(sos, x - x.mean())
    Tx, _, ssq_freqs, _, *_ = ssq_cwt(
        x.astype(np.float32), wavelet=WAVELET, fs=fs, nv=NV,
    )
    P = np.abs(Tx)
    fmask = (ssq_freqs >= band[0]) & (ssq_freqs <= band[1])
    if not fmask.any():
        return np.array([]), np.array([])
    P_band = P[fmask, :]
    f_band = ssq_freqs[fmask]
    peak_idx = np.argmax(P_band, axis=0)
    bpm = f_band[peak_idx] * 60
    t = np.arange(len(x)) / fs
    return t, bpm


def smooth_rectangular(bpm, fs, smooth_s):
    if smooth_s <= 0: return bpm
    n = max(1, int(smooth_s * fs))
    return np.convolve(bpm, np.ones(n)/n, mode='same')


def smooth_exponential(bpm, fs, halflife_s):
    """Causal exponential moving average (mimics live behaviour).
    halflife in seconds; alpha = 1 - 0.5^(1/(halflife*fs))."""
    if halflife_s <= 0: return bpm
    alpha = 1 - 0.5 ** (1.0 / (halflife_s * fs))
    out = np.empty_like(bpm)
    out[0] = bpm[0]
    for i in range(1, len(bpm)):
        out[i] = alpha * bpm[i] + (1 - alpha) * out[i-1]
    return out


# Pre-compute fused traces (uses updated paper_mode with n_pcs=4)
print('Pre-computing fused traces (paper_mode=True with n_pcs=4 default)...')
pairs = collect_pairs()
cache = []
for ds, name, csi, belt in pairs:
    prm = estimate_breathing(csi, belt, paper_mode=True)
    belt_t, belt_x, belt_fs = load_belt(belt)
    tb, bb, _ = sliding_bpm(belt_x, belt_t, belt_fs)
    cache.append({
        'ds': ds, 'name': name,
        'fused': prm.fused, 't_grid': prm.t_grid, 'fs': prm.fs,
        'tb': tb, 'bb': bb,
    })
print(f'Cached {len(cache)} files.')


def evaluate(smoother, label):
    rows = []
    for c in cache:
        t, bpm = sst_raw(c['fused'], c['fs'])
        if len(t) < 2:
            mae = float('nan')
        else:
            bpm = smoother(bpm, c['fs'])
            t_abs = t + c['t_grid'][0]
            on = np.interp(c['tb'], t_abs, bpm)
            mae = float(np.nanmean(np.abs(on - c['bb'])))
        rows.append({'ds': c['ds'], 'mae': mae})
    return rows


def by_ds(rows):
    out = {}
    a = np.array([r['mae'] for r in rows])
    valid = ~np.isnan(a)
    out['overall'] = float(np.mean(a[valid]))
    out['max'] = float(np.max(a[valid]))
    for ds in ['OA-Validity', 'OA-Reliability', 'SD-Vital Signs',
               'SD-Apnoea-OSA', 'SD-Apnoea-CSA']:
        sub = np.array([r['mae'] for r in rows if r['ds'] == ds])
        out[ds] = float(np.mean(sub[~np.isnan(sub)])) if len(sub) else float('nan')
    return out


print(f'\n{"variant":<28} {"slide MAE":>10} {"slide max":>10} '
      f'{"OA-V":>7} {"OA-R":>7} {"VS":>6} {"OSA":>7} {"CSA":>7}')
print('-' * 95)

# Raw (no smoothing)
results = []
for label, fn in [
    ('raw (no smooth)', lambda b, fs: b),
    ('rect smooth=1s',  lambda b, fs: smooth_rectangular(b, fs, 1)),
    ('rect smooth=2s',  lambda b, fs: smooth_rectangular(b, fs, 2)),
    ('rect smooth=5s',  lambda b, fs: smooth_rectangular(b, fs, 5)),
    ('rect smooth=10s', lambda b, fs: smooth_rectangular(b, fs, 10)),
    ('rect smooth=20s', lambda b, fs: smooth_rectangular(b, fs, 20)),
    ('rect smooth=30s', lambda b, fs: smooth_rectangular(b, fs, 30)),
    ('exp halflife=1s',  lambda b, fs: smooth_exponential(b, fs, 1)),
    ('exp halflife=2s',  lambda b, fs: smooth_exponential(b, fs, 2)),
    ('exp halflife=5s',  lambda b, fs: smooth_exponential(b, fs, 5)),
    ('exp halflife=10s', lambda b, fs: smooth_exponential(b, fs, 10)),
    ('exp halflife=15s', lambda b, fs: smooth_exponential(b, fs, 15)),
    ('exp halflife=20s', lambda b, fs: smooth_exponential(b, fs, 20)),
    ('exp halflife=30s', lambda b, fs: smooth_exponential(b, fs, 30)),
]:
    t0 = time.time()
    rows = evaluate(fn, label)
    s = by_ds(rows)
    results.append((label, s))
    print(f'{label:<28} {s["overall"]:>10.3f} {s["max"]:>10.2f} '
          f'{s["OA-Validity"]:>7.2f} {s["OA-Reliability"]:>7.2f} '
          f'{s["SD-Vital Signs"]:>6.2f} {s["SD-Apnoea-OSA"]:>7.2f} '
          f'{s["SD-Apnoea-CSA"]:>7.2f}  ({time.time()-t0:.1f}s)')

best = min(results, key=lambda r: r[1]['overall'])
print(f'\nBest: {best[0]}  MAE {best[1]["overall"]:.3f}')

print('\n--- Reference ---')
print('  paper_mode=True (now n_pcs=4) Welch slide MAE:  1.748')
