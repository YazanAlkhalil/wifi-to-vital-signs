"""
11 — Overall RR comparison: enhanced vs paper_mode across every
breathing-rate file we have.

Datasets:
  - Older-adults Validity (17 paced rates, 12-28 BPM)
  - Older-adults Reliability (30 reps at ~14 BPM)
  - Sleep Disturbances Vital Signs (paced 12 BPM, 1 file)
  - Sleep Disturbances Sleep Apnoa OSA (1 file, paced ~12 with 5 holds)
  - Sleep Disturbances Sleep Apnoa CSA (1 file, paced ~12 with 5 holds)

Total: 50 files.

Reports two error views:
  - whole-record |csi - belt| (single number per file)
  - sliding 30s MAE per file (averaged over windows)
"""
from pathlib import Path
import re, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from csi_pipeline import estimate_breathing, sliding_bpm, load_belt

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
    # Older-adults Validity
    for tgt in range(12, 29):
        csi = OA / 'Validity' / 'Wi-Fi Sensor RR' / f'Val-{tgt}BPM.csv'
        belt = find_belt_validity(tgt)
        if csi.exists() and belt:
            pairs.append(('OA-Validity', f'Val-{tgt}BPM', csi, belt))
    # Older-adults Reliability
    for n in range(1, 31):
        csi  = OA / 'Reliability' / 'Wi-Fi Sensor RR' / f'CSI-{n}.csv'
        belt = OA / 'Reliability' / 'GT Neulog RR Belt Sensor' / f'Belt-{n}.csv'
        if csi.exists() and belt.exists():
            pairs.append(('OA-Reliability', f'rep-{n}', csi, belt))
    # Sleep Disturbances paced 12
    pairs.append(('SD-Vital Signs', 'paced-12-BPM',
                  SD / 'Vital Signs' / 'Breathing - 12 BPM - CSI.csv',
                  SD / 'Vital Signs' / 'Breathing - Belt & HR.csv'))
    # Sleep Disturbances apnoea
    pairs.append(('SD-Apnoea-OSA', 'OSA',
                  SD / 'Sleep Apnoa' / 'OSA - CSI.csv',
                  SD / 'Sleep Apnoa' / 'OSA - Belt.csv'))
    pairs.append(('SD-Apnoea-CSA', 'CSA',
                  SD / 'Sleep Apnoa' / 'CSA - CSI.csv',
                  SD / 'Sleep Apnoa' / 'CSA - Belt.csv'))
    return pairs


def evaluate(label, csi, belt):
    enh = estimate_breathing(csi, belt)
    pap = estimate_breathing(csi, belt, paper_mode='strict')   # literal paper replication
    prm = estimate_breathing(csi, belt, paper_mode=True)       # default = paper + auto-prominence PC pick
    belt_t, belt_x, belt_fs = load_belt(belt)
    tb, bb, _ = sliding_bpm(belt_x, belt_t, belt_fs)
    def slide_mae(r):
        tc, bc, _ = sliding_bpm(r.fused, r.t_grid, r.fs)
        on = np.interp(tb, tc, bc)
        return float(np.nanmean(np.abs(on - bb)))
    return {
        'belt_bpm':       enh.bpm_belt,
        'enh_bpm':        enh.bpm_csi,
        'pap_bpm':        pap.bpm_csi,
        'prm_bpm':        prm.bpm_csi,
        'enh_whole_err':  abs(enh.bpm_csi - enh.bpm_belt),
        'pap_whole_err':  abs(pap.bpm_csi - pap.bpm_belt),
        'prm_whole_err':  abs(prm.bpm_csi - prm.bpm_belt),
        'enh_slide_mae':  slide_mae(enh),
        'pap_slide_mae':  slide_mae(pap),
        'prm_slide_mae':  slide_mae(prm),
    }


pairs = collect_pairs()
print(f'Evaluating {len(pairs)} files across {len(set(p[0] for p in pairs))} datasets...')
rows = []
for ds, name, csi, belt in pairs:
    r = evaluate(name, csi, belt)
    r['ds'] = ds
    r['name'] = name
    rows.append(r)
    print(f'  {ds:>17} / {name:<15}  '
          f'whole: e {r["enh_whole_err"]:5.2f} p {r["pap_whole_err"]:5.2f} a {r["prm_whole_err"]:5.2f}  '
          f'| slide: e {r["enh_slide_mae"]:5.2f} p {r["pap_slide_mae"]:5.2f} a {r["prm_slide_mae"]:5.2f}')


def summarise(rows, name):
    cols = [('enhanced','enh_whole_err','enh_slide_mae'),
            ('paper',   'pap_whole_err','pap_slide_mae'),
            ('auto_prom','prm_whole_err','prm_slide_mae')]
    print(f'\n--- {name}  (n={len(rows)}) ---')
    print(f'  {"":<10} {"whole MAE":>10} {"whole max":>10} {"≤1 BPM":>10} {"slide MAE":>10} {"slide max":>10}')
    for cname, wkey, skey in cols:
        w = np.array([r[wkey] for r in rows])
        s = np.array([r[skey] for r in rows])
        print(f'  {cname:<10} {w.mean():>10.3f} {w.max():>10.2f} '
              f'{int(np.sum(w<=1)):>5}/{len(w):<3} {s.mean():>10.3f} {s.max():>10.2f}')


print('\n' + '=' * 60)
print('OVERALL (all 50 files)')
print('=' * 60)
summarise(rows, 'All datasets combined')

# Per-dataset breakdown
for ds in ['OA-Validity', 'OA-Reliability', 'SD-Vital Signs', 'SD-Apnoea-OSA', 'SD-Apnoea-CSA']:
    sub = [r for r in rows if r['ds'] == ds]
    if sub:
        summarise(sub, ds)

# Oracle: per-file pick the best of all three.
print('\n--- ORACLE (per-file best of {enh, paper, auto_prom}) ---')
oracle_whole = np.array([min(r['enh_whole_err'], r['pap_whole_err'], r['prm_whole_err']) for r in rows])
oracle_slide = np.array([min(r['enh_slide_mae'], r['pap_slide_mae'], r['prm_slide_mae']) for r in rows])
print(f'  Whole MAE: {oracle_whole.mean():.3f}   max: {oracle_whole.max():.2f}   within 1: {int(np.sum(oracle_whole<=1))}/{len(oracle_whole)}')
print(f'  Slide MAE: {oracle_slide.mean():.3f}   max: {oracle_slide.max():.2f}')
