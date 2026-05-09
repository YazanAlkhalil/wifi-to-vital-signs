"""
22 — paper_mode PC selection sweep.

Currently auto_prom picks PC1 vs PC2 by peak/median in-band prominence.
On apnoea data, body motion can dominate PC1/PC2 and breathing might
land in PC3 or PC4. Sweep:
  A) number of PCs considered: 2, 3, 4, 6, 8, 16
  B) scoring metric: prominence (current), bnr (peak/total in-band),
     concentration (in-band power / total power), sinusoidality

Reports both whole-record MAE and slide MAE.
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


# Pre-compute the SVD basis once per file (the expensive part).
print('Pre-computing per-file SVD bases (paper preprocessing)...')
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
    feats_b = dwt_band_filter(feats_u, FS_TARGET).astype(np.float32)
    X = (feats_b - feats_b.mean(0)) / (feats_b.std(0) + 1e-9)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    t_grid = np.arange(n_target) / FS_TARGET + t_csi[0]

    belt_t, belt_x, belt_fs = load_belt(belt)
    bpm_belt, _ = peak_bpm(poly_detrend(belt_x, 3), belt_fs,
                           lo=BAND[0], hi=BAND[1],
                           nperseg=int(belt_fs * WELCH_S),
                           harmonic_guard=False)
    tb, bb, _ = sliding_bpm(belt_x, belt_t, belt_fs)

    cache.append({
        'ds': ds, 'name': name,
        'U': U, 'S': S, 't_grid': t_grid,
        'bpm_belt': bpm_belt, 'tb': tb, 'bb': bb,
    })
print(f'Cached {len(cache)} files.')


def score_pc(pc, fs, band, welch_s, method):
    """Return scalar score; higher = more breathing-like."""
    nperseg = min(len(pc), int(fs * welch_s))
    f, P = signal.welch(pc, fs=fs, nperseg=nperseg)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        return 0.0
    P_in = P[m]
    P_out = P[~m]
    if method == 'prominence':
        return float(P_in.max() / (np.median(P_in) + 1e-12))
    elif method == 'bnr':
        return float(P_in.max() / (P_in.sum() + 1e-12))
    elif method == 'concentration':
        return float(P_in.sum() / (P.sum() + 1e-12))
    elif method == 'sinusoidality':
        # Autocorrelation lag-1-peak height in lag range corresponding
        # to band. Higher = cleaner periodicity.
        x = pc - pc.mean()
        n = len(x)
        ff = np.fft.rfft(x, n=2*n)
        ac = np.fft.irfft(ff * np.conj(ff), n=2*n)[:n]
        ac = ac / (ac[0] + 1e-12)
        lag_lo = int(round(60.0 / (band[1]*60) * fs))
        lag_hi = int(round(60.0 / (band[0]*60) * fs))
        if lag_hi >= n or lag_lo >= lag_hi:
            return 0.0
        seg = ac[lag_lo:lag_hi]
        return float(seg.max())
    raise ValueError(method)


def evaluate(n_pcs, method):
    rows = []
    for c in cache:
        U, S = c['U'], c['S']
        n_avail = min(n_pcs, U.shape[1])
        scores = [score_pc(U[:, i] * S[i], FS_TARGET, BAND, WELCH_S, method)
                  for i in range(n_avail)]
        pc_idx = int(np.argmax(scores))
        fused = U[:, pc_idx] * S[pc_idx]
        if abs(fused.min()) > abs(fused.max()):
            fused = -fused
        bpm_csi, _ = peak_bpm(fused, FS_TARGET, lo=BAND[0], hi=BAND[1],
                              nperseg=int(FS_TARGET * WELCH_S),
                              harmonic_guard=False)
        # Slide
        wt, wb, _ = sliding_bpm(fused, c['t_grid'], FS_TARGET, band=BAND)
        on = np.interp(c['tb'], wt, wb)
        slide_mae = float(np.nanmean(np.abs(on - c['bb'])))
        rows.append({
            'ds': c['ds'], 'name': c['name'],
            'whole_err': abs(bpm_csi - c['bpm_belt']),
            'slide_mae': slide_mae,
            'pc_idx': pc_idx,
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
    # Fraction of files where chosen PC was not 1 or 2
    pc_dist = {}
    for r in rows:
        pc_dist[r['pc_idx']] = pc_dist.get(r['pc_idx'], 0) + 1
    out['pc_dist'] = pc_dist
    return out


# ============================================================
# Sweep
# ============================================================
print('\n' + '=' * 70)
print('PC selection sweep: n_pcs × scoring method')
print('=' * 70)
print(f'\n{"n_pcs":<5} {"method":<14} {"whole MAE":>10} {"slide MAE":>10} '
      f'{"OA-V whl":>9} {"OA-R whl":>9} {"OSA whl":>8} {"CSA whl":>8} '
      f'{"OA-V sld":>9} {"OA-R sld":>9} {"OSA sld":>8} {"CSA sld":>8} '
      f'{"PC dist":>20}')

results = []
for n_pcs in [2, 3, 4, 6, 8, 16]:
    for method in ['prominence', 'bnr', 'concentration', 'sinusoidality']:
        t0 = time.time()
        rows = evaluate(n_pcs, method)
        s = by_ds(rows)
        results.append((n_pcs, method, s))
        pcd = ', '.join(f'{k}:{v}' for k, v in sorted(s['pc_dist'].items()))
        print(f'{n_pcs:<5} {method:<14} {s["whole_err"]:>10.3f} {s["slide_mae"]:>10.3f} '
              f'{s["OA-Validity_whole_err"]:>9.2f} {s["OA-Reliability_whole_err"]:>9.2f} '
              f'{s["SD-Apnoea-OSA_whole_err"]:>8.2f} {s["SD-Apnoea-CSA_whole_err"]:>8.2f} '
              f'{s["OA-Validity_slide_mae"]:>9.2f} {s["OA-Reliability_slide_mae"]:>9.2f} '
              f'{s["SD-Apnoea-OSA_slide_mae"]:>8.2f} {s["SD-Apnoea-CSA_slide_mae"]:>8.2f} '
              f'  {pcd:<20}  ({time.time()-t0:.1f}s)')

# Best by whole MAE
best_whole = min(results, key=lambda r: r[2]['whole_err'])
best_slide = min(results, key=lambda r: r[2]['slide_mae'])
print(f'\nBest whole MAE: n_pcs={best_whole[0]} method={best_whole[1]} → {best_whole[2]["whole_err"]:.3f}')
print(f'Best slide MAE: n_pcs={best_slide[0]} method={best_slide[1]} → {best_slide[2]["slide_mae"]:.3f}')

print('\n--- Reference (current default n_pcs=2 prominence) ---')
print('  Whole MAE  0.393   Slide MAE  2.053')
