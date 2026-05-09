"""
24 — Re-sweep SST wavelet × nv × halflife on the n_pcs=4 fused traces.

Original sweep (notebook 20) was on old PC1/PC2 traces with non-causal
rectangular smoothing. Now that the underlying signal is cleaner
(n_pcs=4 picks the right breathing PC) and we're tuning for live use
(causal exponential smoothing), the optimum might shift.

Sweep:
  wavelet × {gmw, morlet, bump}
  nv      × {16, 32, 64}
  halflife × {3, 5, 8, 12, 20} s

Targets the live-demo regime: exp halflife smoothing, slide MAE.
"""
from pathlib import Path
import re, sys, warnings, time, itertools
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


def sst_raw(x, fs, wavelet, nv, band=BAND):
    from ssqueezepy import ssq_cwt
    sos = signal.butter(2, band[1] * 4, btype='low', fs=fs, output='sos')
    x = signal.sosfiltfilt(sos, x - x.mean())
    Tx, _, ssq_freqs, _, *_ = ssq_cwt(
        x.astype(np.float32), wavelet=wavelet, fs=fs, nv=nv,
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


def smooth_exp(bpm, fs, halflife_s):
    if halflife_s <= 0: return bpm
    alpha = 1 - 0.5 ** (1.0 / (halflife_s * fs))
    out = np.empty_like(bpm)
    out[0] = bpm[0]
    for i in range(1, len(bpm)):
        out[i] = alpha * bpm[i] + (1 - alpha) * out[i-1]
    return out


# Pre-cache fused traces (with the now-default n_pcs=4)
print('Pre-computing fused traces (paper_mode=True, n_pcs=4)...')
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


def evaluate(wavelet, nv, halflife):
    rows = []
    for c in cache:
        t, bpm = sst_raw(c['fused'], c['fs'], wavelet, nv)
        if len(t) < 2:
            mae = float('nan')
        else:
            bpm = smooth_exp(bpm, c['fs'], halflife)
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


print(f'\n{"wavelet":<8} {"nv":<4} {"halflife":<10} {"slide MAE":>10} {"slide max":>10} '
      f'{"OA-V":>7} {"OA-R":>7} {"VS":>6} {"OSA":>7} {"CSA":>7}')
print('-' * 100)

results = []
for wavelet in ['gmw', 'morlet', 'bump']:
    for nv in [16, 32, 64]:
        for hl in [3, 5, 8, 12, 20]:
            t0 = time.time()
            rows = evaluate(wavelet, nv, hl)
            s = by_ds(rows)
            results.append((wavelet, nv, hl, s))
            print(f'{wavelet:<8} {nv:<4} {hl}s         {s["overall"]:>10.3f} {s["max"]:>10.2f} '
                  f'{s["OA-Validity"]:>7.2f} {s["OA-Reliability"]:>7.2f} '
                  f'{s["SD-Vital Signs"]:>6.2f} {s["SD-Apnoea-OSA"]:>7.2f} '
                  f'{s["SD-Apnoea-CSA"]:>7.2f}  ({time.time()-t0:.1f}s)')

best = min(results, key=lambda r: r[3]['overall'])
print(f'\nBest overall: wavelet={best[0]} nv={best[1]} halflife={best[2]}s '
      f'→ slide MAE {best[3]["overall"]:.3f}')

# Best for apnoea (OSA + CSA average)
def apn(s): return (s['SD-Apnoea-OSA'] + s['SD-Apnoea-CSA']) / 2
best_apn = min(results, key=lambda r: apn(r[3]))
print(f'Best for apnoea: wavelet={best_apn[0]} nv={best_apn[1]} halflife={best_apn[2]}s '
      f'→ avg apnoea MAE {apn(best_apn[3]):.3f}')

print('\n--- Reference (live-feasible regime) ---')
print('  Welch baseline (30s/5s):                       slide MAE  2.053')
print('  Old default SST (morlet, nv=32, exp hl=5s):    slide MAE  1.463')
