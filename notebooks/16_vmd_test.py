"""
16 — VMD test on the fused trace.

Hypothesis (from literature): Variational Mode Decomposition produces
K narrowband IMFs by minimising bandwidth around adaptive centre
frequencies. Picking the IMF whose centre frequency falls in the
breathing band [0.1, 0.7] Hz and has highest BNR should be more robust
to harmonic confusion than direct PSD peak picking on the fused trace.

Strategy: run paper_mode=True to get the fused trace, then apply VMD
with K=4 to that trace. Pick best IMF by in-band BNR. Compare BPM to
the standard Welch peak.

This is a CHEAP test — we're not replacing DWT, just adding a VMD
post-fusion step. If it helps, we can move to per-subcarrier VMD.
"""
from pathlib import Path
import re, sys
import numpy as np
from scipy import signal
from vmdpy import VMD

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from csi_pipeline import (
    estimate_breathing, sliding_bpm, load_belt, peak_bpm,
)

OA = (ROOT / 'datasets' /
      'Respiration Rate Measurement Validity and Repeatability of '
      'Ubiquitous Non-contact Wi-Fi Sensing for Older Adults in Care')
SD = ROOT / 'datasets' / 'Sleep Disturbances Dataset'

# VMD hyperparameters from VMD-HHT JEIT 2025
VMD_K = 4         # 4 modes: trend + breathing + harmonic/HR + noise
VMD_ALPHA = 2000  # bandwidth constraint; higher = narrower bands
VMD_TAU = 0.0     # noise tolerance
VMD_DC = 0        # no DC mode forced
VMD_INIT = 1      # uniform init of centre freqs
VMD_TOL = 1e-7


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


def vmd_pick(x, fs, band=(0.1, 0.7)):
    """Run VMD; pick the IMF whose centre frequency is in `band` and
    has highest in-band BNR. Returns (best_imf, bpm, centre_freq).

    VMD requires even-length input. The signal is pre-bandpassed to
    `band` so VMD's centre-frequency search can't drift outside the
    band — fixes the high-BPM regression where uniform-init modes
    were getting trapped in the lower band."""
    n = (len(x) // 2) * 2
    x = x[:n].astype(np.float64)
    # Pre-bandpass: VMD's uniform init places K modes evenly in [0, fs/2].
    # That's wasteful when we know breathing is in `band`. Bandpass first
    # so out-of-band energy doesn't dominate the bandwidth-minimisation
    # objective.
    sos = signal.butter(4, band, btype='band', fs=fs, output='sos')
    x = signal.sosfiltfilt(sos, x)
    u, _, omega = VMD(x, VMD_ALPHA, VMD_TAU, VMD_K, VMD_DC, VMD_INIT, VMD_TOL)
    # `u` is (K, N); each row is an IMF.
    # `omega` is (n_iter, K), final centre freqs in normalised units (0..0.5).
    centre_hz = omega[-1] * fs

    nperseg = min(len(x), int(fs * 60))
    best_bnr = -1
    best_idx = -1
    best_bpm = float('nan')
    for k in range(VMD_K):
        f, P = signal.welch(u[k], fs=fs, nperseg=nperseg)
        m = (f >= band[0]) & (f <= band[1])
        if not m.any():
            continue
        peak = float(P[m].max())
        med = float(np.median(P[m]))
        bnr = peak / (med + 1e-12)
        # Only consider IMFs whose own centre frequency is in band
        if not (band[0] <= centre_hz[k] <= band[1]):
            continue
        if bnr > best_bnr:
            best_bnr = bnr
            best_idx = k
            # Sub-bin peak interpolation already done by peak_bpm
            best_bpm, _ = peak_bpm(u[k], fs, lo=band[0], hi=band[1],
                                   nperseg=nperseg, harmonic_guard=False)

    if best_idx < 0:
        # Nothing in band; fall back
        return None, float('nan'), float('nan')
    return u[best_idx], best_bpm, centre_hz[best_idx]


def evaluate(csi, belt):
    prm = estimate_breathing(csi, belt, paper_mode=True)
    fused = prm.fused
    fs = prm.fs

    welch_bpm = prm.bpm_csi
    vmd_imf, vmd_bpm, vmd_centre = vmd_pick(fused, fs, band=(0.15625, 0.625))

    belt_t, belt_x, belt_fs = load_belt(belt)
    tb, bb, _ = sliding_bpm(belt_x, belt_t, belt_fs)

    # Sliding for both
    welch_t, welch_b, _ = sliding_bpm(fused, prm.t_grid, fs, band=(0.15625, 0.625))

    # VMD sliding: re-run VMD per window (slow but tractable for ~50 files)
    win_s, hop_s = 30, 5
    n = int(win_s * fs); step = int(hop_s * fs)
    vmd_t, vmd_b = [], []
    for start in range(0, len(fused) - n + 1, step):
        seg = fused[start:start+n]
        if seg.std() < 1e-9:
            continue
        try:
            _, b, _ = vmd_pick(seg, fs, band=(0.15625, 0.625))
        except Exception:
            b = float('nan')
        vmd_t.append(prm.t_grid[start] + win_s / 2)
        vmd_b.append(b if not np.isnan(b) else welch_b[len(vmd_t)-1] if len(welch_b) > len(vmd_t)-1 else float('nan'))
    vmd_t, vmd_b = np.array(vmd_t), np.array(vmd_b)

    def slide_mae(t, b):
        if len(t) < 2:
            return float('nan')
        on = np.interp(tb, t, b)
        return float(np.nanmean(np.abs(on - bb)))

    return {
        'belt_bpm':         prm.bpm_belt,
        'welch_bpm':        welch_bpm,
        'vmd_bpm':          vmd_bpm,
        'vmd_centre':       vmd_centre,
        'welch_whole_err':  abs(welch_bpm - prm.bpm_belt),
        'vmd_whole_err':    abs(vmd_bpm - prm.bpm_belt) if not np.isnan(vmd_bpm) else float('nan'),
        'welch_slide_mae':  slide_mae(welch_t, welch_b),
        'vmd_slide_mae':    slide_mae(vmd_t, vmd_b),
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
    r['ds'] = ds
    r['name'] = name
    rows.append(r)
    cf = f"{r['vmd_centre']*60:.1f}" if not np.isnan(r['vmd_centre']) else "----"
    print(f'  {ds:>17} / {name:<15}  belt {r["belt_bpm"]:5.2f} | '
          f'W {r["welch_bpm"]:5.2f} V {r["vmd_bpm"]:5.2f} (centre {cf}bpm)  '
          f'whole err W {r["welch_whole_err"]:5.2f} → V {r["vmd_whole_err"] if not np.isnan(r["vmd_whole_err"]) else float("nan"):5.2f}  '
          f'slide W {r["welch_slide_mae"]:5.2f} → V {r["vmd_slide_mae"]:5.2f}')


def summarise(rows, name):
    cols = [
        ('paper Welch (default)', 'welch_whole_err', 'welch_slide_mae'),
        ('paper + VMD-pick',      'vmd_whole_err',   'vmd_slide_mae'),
    ]
    print(f'\n--- {name}  (n={len(rows)}) ---')
    print(f'  {"":<22} {"whole MAE":>10} {"whole max":>10} {"≤1":>8} {"slide MAE":>10} {"slide max":>10}')
    for cname, wkey, skey in cols:
        w = np.array([r[wkey] for r in rows])
        s = np.array([r[skey] for r in rows])
        valid_w = ~np.isnan(w)
        valid_s = ~np.isnan(s)
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
