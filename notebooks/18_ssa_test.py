"""
18 — Singular Spectrum Analysis (SSA) for breathing reconstruction.

Hypothesis (from literature survey, recommendation #6): SSA decomposes
the signal into orthogonal components ranked by eigenvalue. Breathing
typically appears as a sine pair in two adjacent components. Higher
components carry harmonics (which we drop) and noise. The literature
calls SSA "insensitive to short bursts" — which is precisely the
apnoea regime where SST regressed (OSA 2.08 → 2.39).

Algorithm:
  1. Build Hankel matrix X of shape (L, K=N-L+1) from x[0..N-1].
  2. SVD: X = U S Vt.
  3. Each component i contributes X_i = sigma_i * u_i * v_i^T.
  4. Anti-diagonal-average each X_i back to a length-N time series x_i.
  5. Score each x_i by in-band BNR; pick top-2 → sum → breathing signal.
  6. Welch peak on the reconstructed signal.

Tested on whole-record and sliding-window, against paper Welch baseline.
"""
from pathlib import Path
import re, sys
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from csi_pipeline import (
    estimate_breathing, sliding_bpm, load_belt, peak_bpm,
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


def hankelize(u, v, sigma):
    """Anti-diagonal averaging of sigma * u * v.T → length-N time series.
    L = len(u), K = len(v), N = L + K - 1."""
    L, K = len(u), len(v)
    N = L + K - 1
    X = sigma * np.outer(u, v)
    out = np.zeros(N)
    counts = np.zeros(N)
    for i in range(L):
        for j in range(K):
            out[i + j] += X[i, j]
            counts[i + j] += 1
    return out / counts


def hankelize_fast(u, v, sigma):
    """Vectorised anti-diagonal averaging via 2D summed-area trick.
    Equivalent to the loop above but ~100× faster."""
    L, K = len(u), len(v)
    N = L + K - 1
    X = sigma * np.outer(u, v)
    # For each anti-diagonal d in [0, N), sum elements where i+j=d.
    out = np.zeros(N)
    counts = np.zeros(N)
    for d in range(N):
        i_lo = max(0, d - K + 1); i_hi = min(L - 1, d)
        i = np.arange(i_lo, i_hi + 1)
        j = d - i
        out[d] = X[i, j].sum()
        counts[d] = len(i)
    return out / counts


def ssa_breathing(x, fs, band=(0.15625, 0.625), L=None,
                  n_components=20, n_keep=2):
    """SSA reconstruction of the breathing signal.

    Returns reconstructed signal (length N).

    L = window length; defaults to min(N//2, 10*period_max).
    n_components = how many SVD components to compute (full SVD is
        too slow for long signals; we keep the top n_components only).
    n_keep = how many in-band-BNR-best components to sum into the
        reconstructed breathing signal."""
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    N = len(x)
    if L is None:
        # Period at lo edge in samples — make L several periods long.
        period_max = int(fs / band[0])              # samples
        L = min(N // 2, max(period_max * 3, 100))
    K = N - L + 1
    if K < 2 or L < 2:
        return x.copy()

    # Build Hankel matrix
    H = np.lib.stride_tricks.sliding_window_view(x, L).T  # (L, K)
    # H is (L, K); using sliding_window_view gives (K, L) → transpose

    # Truncated SVD via numpy (full SVD is fine for L up to ~600)
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    n_components = min(n_components, len(S))

    # Reconstruct each component, score by in-band BNR
    bnrs = np.zeros(n_components)
    recons = []
    nperseg = min(N, int(fs * 60))
    for i in range(n_components):
        x_i = hankelize_fast(U[:, i], Vt[i, :], S[i])
        recons.append(x_i)
        f, P = signal.welch(x_i, fs=fs, nperseg=nperseg)
        m = (f >= band[0]) & (f <= band[1])
        if not m.any():
            continue
        bnrs[i] = float(P[m].max() / (np.median(P[m]) + 1e-12))

    # Pick top n_keep by BNR
    top = np.argsort(bnrs)[-n_keep:]
    breathing = np.zeros(N)
    for i in top:
        breathing += recons[i]
    return breathing


def evaluate(csi, belt):
    prm = estimate_breathing(csi, belt, paper_mode=True)
    fused = prm.fused
    fs = prm.fs

    welch_bpm = prm.bpm_csi

    # Whole-record SSA
    band = (0.15625, 0.625)
    breath = ssa_breathing(fused, fs, band=band, n_components=15, n_keep=2)
    ssa_bpm, _ = peak_bpm(breath, fs, lo=band[0], hi=band[1],
                          nperseg=min(len(breath), int(fs * 60)),
                          harmonic_guard=False)

    belt_t, belt_x, belt_fs = load_belt(belt)
    tb, bb, _ = sliding_bpm(belt_x, belt_t, belt_fs)

    welch_t, welch_b, _ = sliding_bpm(fused, prm.t_grid, fs, band=band)

    # Sliding SSA: re-run per window. Use smaller n_components for speed.
    win_s, hop_s = 30, 5
    n = int(win_s * fs); step = int(hop_s * fs)
    ssa_t, ssa_b = [], []
    for start in range(0, len(fused) - n + 1, step):
        seg = fused[start:start+n]
        if seg.std() < 1e-9:
            continue
        try:
            br = ssa_breathing(seg, fs, band=band, n_components=10, n_keep=2)
            b, _ = peak_bpm(br, fs, lo=band[0], hi=band[1],
                            nperseg=min(len(br), int(fs * win_s)),
                            harmonic_guard=False)
        except Exception:
            b = float('nan')
        ssa_t.append(prm.t_grid[start] + win_s / 2)
        ssa_b.append(b)
    ssa_t, ssa_b = np.array(ssa_t), np.array(ssa_b)

    def slide_mae(t, b):
        if len(t) < 2:
            return float('nan')
        on = np.interp(tb, t, b)
        return float(np.nanmean(np.abs(on - bb)))

    return {
        'belt_bpm':         prm.bpm_belt,
        'welch_bpm':        welch_bpm,
        'ssa_bpm':          ssa_bpm,
        'welch_whole_err':  abs(welch_bpm - prm.bpm_belt),
        'ssa_whole_err':    abs(ssa_bpm - prm.bpm_belt),
        'welch_slide_mae':  slide_mae(welch_t, welch_b),
        'ssa_slide_mae':    slide_mae(ssa_t, ssa_b),
    }


pairs = collect_pairs()
print(f'Evaluating {len(pairs)} files...')
rows = []
for ds, name, csi, belt in pairs:
    try:
        r = evaluate(csi, belt)
    except Exception as e:
        print(f'  {ds:>17} / {name:<15}  FAILED: {e}')
        continue
    r['ds'] = ds; r['name'] = name
    rows.append(r)
    print(f'  {ds:>17} / {name:<15}  belt {r["belt_bpm"]:5.2f} | '
          f'W {r["welch_bpm"]:5.2f} S {r["ssa_bpm"]:5.2f}  '
          f'whole err W {r["welch_whole_err"]:5.2f} → S {r["ssa_whole_err"]:5.2f}  '
          f'slide W {r["welch_slide_mae"]:5.2f} → S {r["ssa_slide_mae"]:5.2f}')


def summarise(rows, name):
    cols = [
        ('paper Welch (default)', 'welch_whole_err', 'welch_slide_mae'),
        ('paper + SSA',           'ssa_whole_err',   'ssa_slide_mae'),
    ]
    print(f'\n--- {name}  (n={len(rows)}) ---')
    print(f'  {"":<22} {"whole MAE":>10} {"whole max":>10} {"≤1":>8} {"slide MAE":>10} {"slide max":>10}')
    for cname, wkey, skey in cols:
        w = np.array([r[wkey] for r in rows])
        s = np.array([r[skey] for r in rows])
        valid_w = ~np.isnan(w); valid_s = ~np.isnan(s)
        print(f'  {cname:<22} {w[valid_w].mean():>10.3f} {w[valid_w].max():>10.2f} '
              f'{int(np.sum(w[valid_w]<=1)):>4}/{int(valid_w.sum()):<3} '
              f'{s[valid_s].mean():>10.3f} {s[valid_s].max():>10.2f}')


print('\n' + '=' * 70)
print('OVERALL (all files)')
print('=' * 70)
summarise(rows, 'All datasets combined')

for ds in ['OA-Validity', 'OA-Reliability', 'SD-Vital Signs', 'SD-Apnoea-OSA', 'SD-Apnoea-CSA']:
    sub = [r for r in rows if r['ds'] == ds]
    if sub:
        summarise(sub, ds)
