"""
19 — Welch + SST + SSA ensemble for sliding-window MAE.

Question: does combining the three sliding methods clearly beat
SST-alone? If not, ship SST.

Two ensembles:
  - oracle: per-window pick whichever of {welch, sst, ssa} is closest
    to ground truth. Upper bound — if this is ≈ SST, no ensemble
    can help.
  - practical: per-window inter-method agreement. If all three agree
    within `tol` BPM, take median. Else if any pair agrees, take
    their mean. Else fall back to Welch (most reliable per the
    Validity finding).

Comparing against:
  - Welch (current default)
  - SST (best single method)
  - SSA (best on CSA)
"""
from pathlib import Path
import re, sys, warnings
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


# ------------------ SST (from notebook 17) ------------------
def sst_track(x, fs, band=BAND, smooth_s=10):
    from ssqueezepy import ssq_cwt
    sos = signal.butter(2, band[1] * 4, btype='low', fs=fs, output='sos')
    x = signal.sosfiltfilt(sos, x - x.mean())
    Tx, _, ssq_freqs, _, *_ = ssq_cwt(x.astype(np.float32), fs=fs)
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


# ------------------ SSA (from notebook 18) ------------------
def hankelize_fast(u, v, sigma):
    L, K = len(u), len(v)
    N = L + K - 1
    X = sigma * np.outer(u, v)
    out = np.zeros(N); counts = np.zeros(N)
    for d in range(N):
        i_lo = max(0, d - K + 1); i_hi = min(L - 1, d)
        i = np.arange(i_lo, i_hi + 1); j = d - i
        out[d] = X[i, j].sum(); counts[d] = len(i)
    return out / counts


def ssa_breathing(x, fs, band=BAND, L=None, n_components=10, n_keep=2):
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    N = len(x)
    if L is None:
        period_max = int(fs / band[0])
        L = min(N // 2, max(period_max * 3, 100))
    K = N - L + 1
    if K < 2 or L < 2: return x.copy()
    H = np.lib.stride_tricks.sliding_window_view(x, L).T
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    n_components = min(n_components, len(S))
    bnrs = np.zeros(n_components); recons = []
    nperseg = min(N, int(fs * 60))
    for i in range(n_components):
        x_i = hankelize_fast(U[:, i], Vt[i, :], S[i])
        recons.append(x_i)
        f, P = signal.welch(x_i, fs=fs, nperseg=nperseg)
        m = (f >= band[0]) & (f <= band[1])
        if m.any():
            bnrs[i] = float(P[m].max() / (np.median(P[m]) + 1e-12))
    top = np.argsort(bnrs)[-n_keep:]
    breathing = np.zeros(N)
    for i in top: breathing += recons[i]
    return breathing


# ------------------ Per-window estimators on a common time grid ------------------
def per_window_estimates(fused, t_grid, fs, win_s=30, hop_s=5):
    """Returns common time grid + per-window estimates from each method.
    All three methods report on the same window centres."""
    n = int(win_s * fs); step = int(hop_s * fs)
    centres = []
    welch_b, ssa_b = [], []

    # SST is computed once on the full trace, then sampled at centres.
    sst_t_local, sst_b_full = sst_track(fused, fs, band=BAND, smooth_s=hop_s)
    sst_t_abs = sst_t_local + t_grid[0] if len(sst_t_local) > 0 else np.array([])

    for start in range(0, len(fused) - n + 1, step):
        seg = fused[start:start+n]
        t_mid = t_grid[start] + win_s / 2
        centres.append(t_mid)

        # Welch
        if seg.std() < 1e-9:
            welch_b.append(np.nan)
        else:
            b, _ = peak_bpm(signal.detrend(seg), fs, lo=BAND[0], hi=BAND[1],
                            nperseg=min(n, int(fs * win_s)))
            welch_b.append(b)

        # SSA per window
        try:
            br = ssa_breathing(seg, fs, n_components=10, n_keep=2)
            b, _ = peak_bpm(br, fs, lo=BAND[0], hi=BAND[1],
                            nperseg=min(len(br), int(fs * win_s)),
                            harmonic_guard=False)
            ssa_b.append(b)
        except Exception:
            ssa_b.append(np.nan)

    centres = np.array(centres)
    welch_b = np.array(welch_b)
    ssa_b = np.array(ssa_b)

    # Sample SST at the same centres
    if len(sst_t_abs) >= 2:
        sst_b = np.interp(centres, sst_t_abs, sst_b_full)
    else:
        sst_b = np.full_like(centres, np.nan)

    return centres, welch_b, sst_b, ssa_b


def practical_ensemble(welch, sst, ssa, tol=1.5):
    """Per-window selector. If all three agree within tol, use median.
    Else if any pair agrees within tol, use their mean. Else Welch.
    Returns (bpm, source_label) per window."""
    out = np.zeros_like(welch)
    src = np.empty(len(welch), dtype=object)
    for i in range(len(welch)):
        w, s, a = welch[i], sst[i], ssa[i]
        valid = [(name, v) for name, v in (('w', w), ('s', s), ('a', a))
                 if not np.isnan(v)]
        if len(valid) == 0:
            out[i] = np.nan; src[i] = 'none'
            continue
        if len(valid) == 1:
            out[i] = valid[0][1]; src[i] = valid[0][0]
            continue
        # All-three first
        if len(valid) == 3:
            vals = np.array([v for _, v in valid])
            if vals.max() - vals.min() <= tol:
                out[i] = float(np.median(vals)); src[i] = 'all3'
                continue
        # Pair check
        best_pair, best_diff = None, float('inf')
        for a_idx in range(len(valid)):
            for b_idx in range(a_idx + 1, len(valid)):
                d = abs(valid[a_idx][1] - valid[b_idx][1])
                if d < best_diff:
                    best_diff = d
                    best_pair = (valid[a_idx], valid[b_idx])
        if best_pair and best_diff <= tol:
            out[i] = (best_pair[0][1] + best_pair[1][1]) / 2
            src[i] = best_pair[0][0] + best_pair[1][0]
            continue
        # Fall back to Welch (or first valid if Welch is NaN)
        if not np.isnan(w):
            out[i] = w; src[i] = 'w_fallback'
        else:
            out[i] = valid[0][1]; src[i] = valid[0][0] + '_fallback'
    return out, src


def evaluate(csi, belt):
    prm = estimate_breathing(csi, belt, paper_mode=True)
    fs = prm.fs
    fused = prm.fused

    belt_t, belt_x, belt_fs = load_belt(belt)
    tb, bb, _ = sliding_bpm(belt_x, belt_t, belt_fs)

    centres, welch_b, sst_b, ssa_b = per_window_estimates(fused, prm.t_grid, fs)

    # Belt sampled at our centres for oracle-truth
    belt_at_centres = np.interp(centres, tb, bb)

    # Oracle: pick best of three per window
    err_w = np.abs(welch_b - belt_at_centres)
    err_s = np.abs(sst_b   - belt_at_centres)
    err_a = np.abs(ssa_b   - belt_at_centres)
    stack = np.stack([err_w, err_s, err_a], axis=1)
    stack_safe = np.where(np.isnan(stack), np.inf, stack)
    oracle_pick = np.argmin(stack_safe, axis=1)
    oracle_b = np.where(oracle_pick == 0, welch_b,
              np.where(oracle_pick == 1, sst_b, ssa_b))

    # Practical ensemble
    ens_b, ens_src = practical_ensemble(welch_b, sst_b, ssa_b, tol=1.5)

    def slide_mae(estim):
        on = np.interp(tb, centres, estim)
        return float(np.nanmean(np.abs(on - bb)))

    # Source frequencies for the practical ensemble
    src_counts = {}
    for s in ens_src:
        src_counts[s] = src_counts.get(s, 0) + 1

    return {
        'belt_bpm':       prm.bpm_belt,
        'welch_slide':    slide_mae(welch_b),
        'sst_slide':      slide_mae(sst_b),
        'ssa_slide':      slide_mae(ssa_b),
        'oracle_slide':   slide_mae(oracle_b),
        'ensemble_slide': slide_mae(ens_b),
        'src_counts':     src_counts,
        # How often each method "wins" the oracle
        'oracle_win_w':   float(np.mean(oracle_pick == 0)),
        'oracle_win_s':   float(np.mean(oracle_pick == 1)),
        'oracle_win_a':   float(np.mean(oracle_pick == 2)),
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
    print(f'  {ds:>17} / {name:<15}  '
          f'W {r["welch_slide"]:5.2f}  S {r["sst_slide"]:5.2f}  '
          f'A {r["ssa_slide"]:5.2f}  ORC {r["oracle_slide"]:5.2f}  '
          f'ENS {r["ensemble_slide"]:5.2f}   '
          f'orc-pick W{r["oracle_win_w"]:.0%}/S{r["oracle_win_s"]:.0%}/A{r["oracle_win_a"]:.0%}')


def summarise(rows, name):
    cols = [
        ('Welch (default)',  'welch_slide'),
        ('SST (best single)','sst_slide'),
        ('SSA',              'ssa_slide'),
        ('Oracle (best/3)',  'oracle_slide'),
        ('Practical ens.',   'ensemble_slide'),
    ]
    print(f'\n--- {name}  (n={len(rows)}) ---')
    print(f'  {"":<22} {"slide MAE":>10} {"slide max":>10}')
    for cname, key in cols:
        s = np.array([r[key] for r in rows])
        valid = ~np.isnan(s)
        print(f'  {cname:<22} {s[valid].mean():>10.3f} {s[valid].max():>10.2f}')


print('\n' + '=' * 70)
print('OVERALL (all files)')
print('=' * 70)
summarise(rows, 'All datasets combined')

# Aggregate practical-ensemble source frequencies
agg_src = {}
for r in rows:
    for k, v in r['src_counts'].items():
        agg_src[k] = agg_src.get(k, 0) + v
total_src = sum(agg_src.values())
print(f'\nPractical-ensemble per-window source breakdown (total {total_src} windows):')
for k in sorted(agg_src, key=lambda k: -agg_src[k]):
    print(f'  {k:<15} {agg_src[k]:>5}  ({agg_src[k]/total_src:.1%})')

# Aggregate oracle-pick rates
ws = np.mean([r['oracle_win_w'] for r in rows])
ss = np.mean([r['oracle_win_s'] for r in rows])
a_s = np.mean([r['oracle_win_a'] for r in rows])
print(f'\nOracle-pick rates (avg per file): Welch {ws:.1%}, SST {ss:.1%}, SSA {a_s:.1%}')

for ds in ['OA-Validity', 'OA-Reliability', 'SD-Vital Signs', 'SD-Apnoea-OSA', 'SD-Apnoea-CSA']:
    sub = [r for r in rows if r['ds'] == ds]
    if sub:
        summarise(sub, ds)
