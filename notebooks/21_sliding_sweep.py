"""
21 — Sliding-window hyperparameter sweep.

Two sub-sweeps, both on the cached 50-file benchmark.

A) Welch sliding (the baseline metric):
   window in {15, 20, 25, 30, 40, 50, 60} s
   hop    in {1, 2, 3, 5, 10} s

B) SST smoothing extension (the production metric, tuned-config wavelet):
   smooth_s in {20, 25, 30, 40, 50, 60} s
   the previous sweep stopped at 20s — extending tells us whether
   the monotone-improvement trend continues.
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
    estimate_breathing, sliding_bpm, load_belt, peak_bpm,
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


def custom_sliding_welch(x, t, fs, band, win_s, hop_s):
    out_t, out_b = [], []
    n = int(win_s * fs); step = int(hop_s * fs)
    for start in range(0, len(x) - n + 1, step):
        seg = x[start:start+n]
        if seg.std() < 1e-9:
            b = float('nan')
        else:
            b, _ = peak_bpm(signal.detrend(seg), fs,
                            lo=band[0], hi=band[1],
                            nperseg=min(n, int(fs * win_s)))
        out_t.append(t[start] + win_s / 2)
        out_b.append(b)
    return np.array(out_t), np.array(out_b)


def sst_track_tuned(x, fs, band=BAND, smooth_s=20, wavelet='morlet', nv=32):
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
    n_smooth = max(1, int(smooth_s * fs))
    if n_smooth > 1:
        bpm = np.convolve(bpm, np.ones(n_smooth)/n_smooth, mode='same')
    t = np.arange(len(x)) / fs
    return t, bpm


# Pre-compute fused traces and belt data
print('Loading and pre-computing fused traces for all 50 files...')
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


def evaluate_welch(win_s, hop_s):
    rows = []
    for c in cache:
        wt, wb = custom_sliding_welch(c['fused'], c['t_grid'], c['fs'],
                                      BAND, win_s, hop_s)
        if len(wt) < 2:
            mae = float('nan')
        else:
            on = np.interp(c['tb'], wt, wb)
            mae = float(np.nanmean(np.abs(on - c['bb'])))
        rows.append({'ds': c['ds'], 'mae': mae})
    return rows


def evaluate_sst(smooth_s):
    rows = []
    for c in cache:
        st, sb = sst_track_tuned(c['fused'], c['fs'],
                                 smooth_s=smooth_s)
        if len(st) < 2:
            mae = float('nan')
        else:
            st_abs = st + c['t_grid'][0]
            on = np.interp(c['tb'], st_abs, sb)
            mae = float(np.nanmean(np.abs(on - c['bb'])))
        rows.append({'ds': c['ds'], 'mae': mae})
    return rows


def by_ds(rows):
    out = {}
    maes = np.array([r['mae'] for r in rows])
    out['overall'] = float(np.mean(maes[~np.isnan(maes)]))
    out['max']     = float(np.nanmax(maes))
    for ds in ['OA-Validity', 'OA-Reliability', 'SD-Vital Signs',
               'SD-Apnoea-OSA', 'SD-Apnoea-CSA']:
        sub = np.array([r['mae'] for r in rows if r['ds'] == ds])
        out[ds] = float(np.mean(sub[~np.isnan(sub)])) if len(sub) else float('nan')
    return out


# ============================================================
# A) Welch window × hop sweep
# ============================================================
print('\n' + '=' * 70)
print('A) Welch sliding: window × hop')
print('=' * 70)
print(f'\n{"win/hop":<10} {"slide MAE":>10} {"slide max":>10} '
      f'{"OA-V":>7} {"OA-R":>7} {"VS":>6} {"OSA":>7} {"CSA":>7}')
welch_results = []
for win_s in [15, 20, 25, 30, 40, 50, 60]:
    for hop_s in [1, 2, 3, 5, 10]:
        if hop_s > win_s / 3:  # skip silly low-overlap configs
            continue
        t0 = time.time()
        rows = evaluate_welch(win_s, hop_s)
        s = by_ds(rows)
        welch_results.append((win_s, hop_s, s))
        print(f'{win_s}s/{hop_s}s     {s["overall"]:>10.3f} {s["max"]:>10.2f} '
              f'{s["OA-Validity"]:>7.2f} {s["OA-Reliability"]:>7.2f} '
              f'{s["SD-Vital Signs"]:>6.2f} {s["SD-Apnoea-OSA"]:>7.2f} '
              f'{s["SD-Apnoea-CSA"]:>7.2f}  ({time.time()-t0:.1f}s)')

best_welch = min(welch_results, key=lambda r: r[2]['overall'])
print(f'\nWelch best: win={best_welch[0]}s hop={best_welch[1]}s  '
      f'MAE {best_welch[2]["overall"]:.3f} (vs current 30s/5s = '
      f'{[r for r in welch_results if r[0]==30 and r[1]==5][0][2]["overall"]:.3f})')


# ============================================================
# B) SST smoothing extension on tuned config
# ============================================================
print('\n' + '=' * 70)
print('B) SST smoothing extension (tuned: morlet nv=32)')
print('=' * 70)
print(f'\n{"smooth":<8} {"slide MAE":>10} {"slide max":>10} '
      f'{"OA-V":>7} {"OA-R":>7} {"VS":>6} {"OSA":>7} {"CSA":>7}')
sst_results = []
for smooth_s in [10, 15, 20, 25, 30, 40, 50, 60, 80]:
    t0 = time.time()
    rows = evaluate_sst(smooth_s)
    s = by_ds(rows)
    sst_results.append((smooth_s, s))
    print(f'{smooth_s}s        {s["overall"]:>10.3f} {s["max"]:>10.2f} '
          f'{s["OA-Validity"]:>7.2f} {s["OA-Reliability"]:>7.2f} '
          f'{s["SD-Vital Signs"]:>6.2f} {s["SD-Apnoea-OSA"]:>7.2f} '
          f'{s["SD-Apnoea-CSA"]:>7.2f}  ({time.time()-t0:.1f}s)')

best_sst = min(sst_results, key=lambda r: r[1]['overall'])
print(f'\nSST best: smooth={best_sst[0]}s  MAE {best_sst[1]["overall"]:.3f}  '
      f'(prev best at 20s = {[r for r in sst_results if r[0]==20][0][1]["overall"]:.3f})')

print('\n--- Reference ---')
print('  Welch baseline (30s/5s)            slide MAE  2.053')
print('  SST tuned (morlet nv=32 smooth=20s) slide MAE  1.581')
