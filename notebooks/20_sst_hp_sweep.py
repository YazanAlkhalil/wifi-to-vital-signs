"""
20 — SST hyperparameter sweep.

Two-stage sweep to keep runtime tractable.

Stage 1 (band only): fix wavelet=gmw, smoothing=10s, nv=32. Sweep
analysis band — primary hypothesis is that widening it past [0.156,
0.625] fixes the high-BPM-edge failures (Val-26/27/28).

Stage 2 (everything else around best band): sweep wavelet × nv ×
smoothing on the winner band from stage 1.

Reports overall slide MAE and per-dataset breakdown.
"""
from pathlib import Path
import re, sys, warnings, itertools, time
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


def sst_track(x, fs, band, smooth_s, wavelet='gmw', nv=32):
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


# Cache fused traces and belt data so we don't re-run paper_mode pipeline
# for every config.
print('Loading and pre-computing fused traces for all 50 files...')
pairs = collect_pairs()
cache = []
for ds, name, csi, belt in pairs:
    t0 = time.time()
    prm = estimate_breathing(csi, belt, paper_mode=True)
    belt_t, belt_x, belt_fs = load_belt(belt)
    tb, bb, _ = sliding_bpm(belt_x, belt_t, belt_fs)
    cache.append({
        'ds': ds, 'name': name,
        'fused': prm.fused, 't_grid': prm.t_grid, 'fs': prm.fs,
        'tb': tb, 'bb': bb,
    })
print(f'Cached {len(cache)} files.')


def evaluate_config(band, smooth_s, wavelet, nv):
    rows = []
    for c in cache:
        sst_t, sst_b = sst_track(c['fused'], c['fs'], band=band,
                                 smooth_s=smooth_s,
                                 wavelet=wavelet, nv=nv)
        if len(sst_t) < 2:
            mae = float('nan')
        else:
            sst_t_abs = sst_t + c['t_grid'][0]
            on = np.interp(c['tb'], sst_t_abs, sst_b)
            mae = float(np.nanmean(np.abs(on - c['bb'])))
        rows.append({'ds': c['ds'], 'name': c['name'], 'mae': mae})
    return rows


def summarise(rows, label, by_ds=False):
    maes = np.array([r['mae'] for r in rows])
    valid = ~np.isnan(maes)
    overall = float(np.mean(maes[valid]))
    if by_ds:
        out = {'overall': overall}
        for ds in ['OA-Validity', 'OA-Reliability', 'SD-Vital Signs',
                   'SD-Apnoea-OSA', 'SD-Apnoea-CSA']:
            sub = np.array([r['mae'] for r in rows if r['ds'] == ds])
            out[ds] = float(np.mean(sub[~np.isnan(sub)])) if len(sub) else float('nan')
        return out
    return overall


# ============================================================
# STAGE 1: band sweep
# ============================================================
print('\n' + '=' * 70)
print('STAGE 1: band sweep (gmw wavelet, 10s smoothing, nv=32)')
print('=' * 70)
bands = [
    ('paper [0.156-0.625]',     (0.15625, 0.625)),
    ('wide  [0.10-0.625]',      (0.10,    0.625)),
    ('wide  [0.10-0.70]',       (0.10,    0.70)),
    ('wide  [0.10-0.83]',       (0.10,    0.833)),
    ('wider [0.10-1.00]',       (0.10,    1.00)),
    ('low+0.5 [0.156-0.83]',    (0.15625, 0.833)),
    ('low+0.5 [0.156-1.00]',    (0.15625, 1.00)),
]
print(f'\n{"band":<25} {"slide MAE":>10} {"slide max":>10} {"OA-V":>7} {"OA-R":>7} {"OSA":>7} {"CSA":>7}')
stage1_results = []
for label, band in bands:
    t0 = time.time()
    rows = evaluate_config(band, smooth_s=10, wavelet='gmw', nv=32)
    summ = summarise(rows, label, by_ds=True)
    maes = [r['mae'] for r in rows]
    smax = float(np.nanmax(maes))
    stage1_results.append((label, band, summ['overall'], summ))
    print(f'{label:<25} {summ["overall"]:>10.3f} {smax:>10.2f} '
          f'{summ["OA-Validity"]:>7.2f} {summ["OA-Reliability"]:>7.2f} '
          f'{summ["SD-Apnoea-OSA"]:>7.2f} {summ["SD-Apnoea-CSA"]:>7.2f}  '
          f'({time.time()-t0:.1f}s)')

best_band_label, best_band, best_band_mae, _ = min(stage1_results, key=lambda r: r[2])
print(f'\nStage 1 winner: {best_band_label} (MAE {best_band_mae:.3f})')


# ============================================================
# STAGE 2: wavelet × nv × smoothing on winning band
# ============================================================
print('\n' + '=' * 70)
print(f'STAGE 2: wavelet × nv × smoothing on band {best_band}')
print('=' * 70)
print(f'\n{"config":<40} {"slide MAE":>10} {"slide max":>10} {"OA-V":>7} {"OA-R":>7} {"OSA":>7} {"CSA":>7}')
stage2_results = []
for wavelet, nv, smooth_s in itertools.product(
        ['gmw', 'morlet', 'bump'],
        [16, 32, 64],
        [5, 10, 15, 20]):
    t0 = time.time()
    rows = evaluate_config(best_band, smooth_s=smooth_s,
                           wavelet=wavelet, nv=nv)
    summ = summarise(rows, '', by_ds=True)
    maes = [r['mae'] for r in rows]
    smax = float(np.nanmax(maes))
    stage2_results.append((wavelet, nv, smooth_s, summ['overall'], summ, smax))
    label = f'{wavelet} nv={nv} smooth={smooth_s}s'
    print(f'{label:<40} {summ["overall"]:>10.3f} {smax:>10.2f} '
          f'{summ["OA-Validity"]:>7.2f} {summ["OA-Reliability"]:>7.2f} '
          f'{summ["SD-Apnoea-OSA"]:>7.2f} {summ["SD-Apnoea-CSA"]:>7.2f}  '
          f'({time.time()-t0:.1f}s)')

best_cfg = min(stage2_results, key=lambda r: r[3])
print(f'\nOverall best: wavelet={best_cfg[0]} nv={best_cfg[1]} '
      f'smooth={best_cfg[2]}s  band={best_band_label}')
print(f'  → slide MAE {best_cfg[3]:.3f}  max {best_cfg[5]:.2f}')
print(f'  Per-dataset:')
for k, v in best_cfg[4].items():
    if k != 'overall':
        print(f'    {k:<20} {v:.3f}')

print('\n--- Reference ---')
print('  Welch baseline      slide MAE  2.053')
print('  SST original config slide MAE  1.749  (band [0.156-0.625] gmw nv=32 smooth=10s)')
