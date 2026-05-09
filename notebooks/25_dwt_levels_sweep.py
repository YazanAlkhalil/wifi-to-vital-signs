"""
25 — DWT keep_levels sweep on paper_mode at fs=40 Hz.

Current default keeps {D6, D7} = [0.156, 0.625] Hz ≈ 9.4-37.5 BPM,
matching the paper.

Levels at fs=40 Hz:
  D5: [0.625,  1.25 ] Hz = 37.5-75   BPM
  D6: [0.313,  0.625] Hz = 18.75-37.5 BPM
  D7: [0.156,  0.313] Hz = 9.4-18.75 BPM
  D8: [0.078,  0.156] Hz = 4.7-9.4   BPM

Sweep alternatives:
  {6, 7}            current default
  {6}               narrow to high half
  {7}               narrow to low half
  {5, 6, 7}         widen up
  {6, 7, 8}         widen down
  {5, 6, 7, 8}      maximum width

Reports both whole MAE (uses paper_mode's Welch peak picking) and
slide MAE (with current default Welch slider, since SST tuning is
independent of DWT levels in principle).
"""
from pathlib import Path
import re, sys, time, warnings
import numpy as np
from scipy import signal

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from csi_pipeline import (
    load_csi, load_belt, dwt_band_filter, peak_bpm, sliding_bpm,
    poly_detrend,
)

OA = (ROOT / 'datasets' /
      'Respiration Rate Measurement Validity and Repeatability of '
      'Ubiquitous Non-contact Wi-Fi Sensing for Older Adults in Care')
SD = ROOT / 'datasets' / 'Sleep Disturbances Dataset'

BAND = (0.15625, 0.625)
FS_TARGET = 40.0
WELCH_S = 60


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


# Pre-load CSI features (resampled to 40 Hz). The DWT step is the
# only thing that changes per config, so we cache the resampled features.
print('Pre-loading and resampling 50 files...')
pairs = collect_pairs()
cache = []
for ds, name, csi, belt in pairs:
    t_csi, H = load_csi(csi)
    data_idx = list(np.r_[6:32, 33:59])
    feats = np.abs(H[:, data_idx]).astype(np.float32)
    src_fs = 1.0 / np.median(np.diff(t_csi))
    t_uniform = np.arange(t_csi[0], t_csi[-1], 1.0 / src_fs)
    feats_uniform = np.stack(
        [np.interp(t_uniform, t_csi, feats[:, k])
         for k in range(feats.shape[1])], axis=1)
    n_target = int(round(len(t_uniform) * FS_TARGET / src_fs))
    feats_u = signal.resample(feats_uniform, n_target, axis=0)
    t_grid = np.arange(n_target) / FS_TARGET + t_csi[0]

    belt_t, belt_x, belt_fs = load_belt(belt)
    bpm_belt, _ = peak_bpm(poly_detrend(belt_x, 3), belt_fs,
                           lo=BAND[0], hi=BAND[1],
                           nperseg=int(belt_fs * WELCH_S),
                           harmonic_guard=False)
    tb, bb, _ = sliding_bpm(belt_x, belt_t, belt_fs)

    cache.append({
        'ds': ds, 'name': name,
        'feats_u': feats_u, 't_grid': t_grid,
        'bpm_belt': bpm_belt, 'tb': tb, 'bb': bb,
    })
print(f'Cached {len(cache)} files.')


def evaluate(keep_levels):
    rows = []
    for c in cache:
        feats_b = dwt_band_filter(c['feats_u'], FS_TARGET,
                                  keep_levels=keep_levels).astype(np.float32)
        X = (feats_b - feats_b.mean(0)) / (feats_b.std(0) + 1e-9)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        nperseg = int(FS_TARGET * WELCH_S)

        def prom(idx):
            pc = U[:, idx] * S[idx]
            f, P = signal.welch(pc, fs=FS_TARGET, nperseg=nperseg)
            m = (f >= BAND[0]) & (f <= BAND[1])
            if not m.any(): return 0.0
            return float(P[m].max() / (np.median(P[m]) + 1e-12))

        n_avail = min(4, U.shape[1])
        proms = [prom(i) for i in range(n_avail)]
        pc_idx = int(np.argmax(proms))
        fused = U[:, pc_idx] * S[pc_idx]
        if abs(fused.min()) > abs(fused.max()):
            fused = -fused
        bpm_csi, _ = peak_bpm(fused, FS_TARGET, lo=BAND[0], hi=BAND[1],
                              nperseg=nperseg, harmonic_guard=False)
        wt, wb, _ = sliding_bpm(fused, c['t_grid'], FS_TARGET, band=BAND)
        on = np.interp(c['tb'], wt, wb)
        slide_mae = float(np.nanmean(np.abs(on - c['bb'])))
        rows.append({
            'ds': c['ds'], 'whole_err': abs(bpm_csi - c['bpm_belt']),
            'slide_mae': slide_mae, 'pc_idx': pc_idx,
        })
    return rows


def by_ds(rows):
    out = {}
    for key in ['whole_err', 'slide_mae']:
        a = np.array([r[key] for r in rows])
        valid = ~np.isnan(a)
        out[key] = float(np.mean(a[valid]))
        out[f'{key}_max'] = float(np.max(a[valid]))
    for ds in ['OA-Validity', 'OA-Reliability', 'SD-Vital Signs',
               'SD-Apnoea-OSA', 'SD-Apnoea-CSA']:
        sub = [r for r in rows if r['ds'] == ds]
        for key in ['whole_err', 'slide_mae']:
            a = np.array([r[key] for r in sub])
            valid = ~np.isnan(a)
            out[f'{ds}_{key}'] = float(np.mean(a[valid])) if valid.any() else float('nan')
    return out


configs = [
    ('{6,7} (current)',     {6, 7}),
    ('{6}',                 {6}),
    ('{7}',                 {7}),
    ('{5,6,7}',             {5, 6, 7}),
    ('{6,7,8}',             {6, 7, 8}),
    ('{5,6,7,8}',           {5, 6, 7, 8}),
    ('{4,5,6,7}',           {4, 5, 6, 7}),
]

print(f'\n{"levels":<22} {"whole MAE":>10} {"slide MAE":>10} '
      f'{"OA-V whl":>9} {"OA-R whl":>9} {"OSA whl":>8} {"CSA whl":>8} '
      f'{"OA-V sld":>9} {"OA-R sld":>9} {"OSA sld":>8} {"CSA sld":>8}')
print('-' * 122)

results = []
for label, levels in configs:
    t0 = time.time()
    rows = evaluate(levels)
    s = by_ds(rows)
    results.append((label, levels, s))
    print(f'{label:<22} {s["whole_err"]:>10.3f} {s["slide_mae"]:>10.3f} '
          f'{s["OA-Validity_whole_err"]:>9.2f} {s["OA-Reliability_whole_err"]:>9.2f} '
          f'{s["SD-Apnoea-OSA_whole_err"]:>8.2f} {s["SD-Apnoea-CSA_whole_err"]:>8.2f} '
          f'{s["OA-Validity_slide_mae"]:>9.2f} {s["OA-Reliability_slide_mae"]:>9.2f} '
          f'{s["SD-Apnoea-OSA_slide_mae"]:>8.2f} {s["SD-Apnoea-CSA_slide_mae"]:>8.2f}  '
          f'({time.time()-t0:.1f}s)')

best_whole = min(results, key=lambda r: r[2]['whole_err'])
best_slide = min(results, key=lambda r: r[2]['slide_mae'])
print(f'\nBest whole MAE: {best_whole[0]} → {best_whole[2]["whole_err"]:.3f}')
print(f'Best slide MAE: {best_slide[0]} → {best_slide[2]["slide_mae"]:.3f}')

print('\n--- Reference (current defaults, n_pcs=4) ---')
print('  Whole MAE  0.348   Slide MAE  1.748')
