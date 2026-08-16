"""
29 — The three-capture A/B test for "why is breathing not showing up?".

Records three labelled 60 s captures and prints the one number that matters
for each: the strongest in-band peak prominence, i.e. how far a single
breathing-rate frequency stands out above its neighbours.

  1. control     nobody near the boards      -> the noise floor
  2. breathhold  you in position, still      -> your body, but no breathing
  3. paced       you breathing on a metronome-> what we are trying to detect

Reading the result:
  paced ~ control                -> geometry is wrong; chest is not in the path
  paced >> breathhold ~ control  -> the rig works, the live demo picks badly
  all three high                 -> something periodic in the room, not you

Files are written to captures/ as abtest_<run>_<label>.csv so they can be
compared later, and so someone else can tell which is which.

Usage:
    .venv\\Scripts\\python.exe notebooks\\29_ab_test.py            (all three)
    .venv\\Scripts\\python.exe notebooks\\29_ab_test.py paced      (just one)
"""
from pathlib import Path
import importlib.util
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))

# The capture + preprocessing code lives in a numerically-named module, so it
# has to be loaded by path rather than imported by name.
_spec = importlib.util.spec_from_file_location(
    'capture_test', ROOT / 'notebooks' / '26_first_capture_test.py')
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

from csi_pipeline import load_csi  # noqa: E402

DURATION = 60.0
OUT_DIR = ROOT / 'captures'
DATA_IDX = list(np.r_[6:32, 33:59])      # 52 active HT20 data subcarriers

# Metronome at 30 beats/min, inhaling on one beat and exhaling on the next,
# gives 15 breaths/min. A correct detection should peak here.
METRONOME_BPM = 30
TARGET_BPM = METRONOME_BPM / 2           # 15 breaths per minute
BPM_TOLERANCE = 1.5

STEPS = [
    ('control', [
        'Leave the room, or stand well away from both boards.',
        'Nothing should move for the next minute. This measures the',
        'noise floor - what the link does with no one in it.',
    ]),
    ('breathhold', [
        'Sit in your measurement position: chest between the two boards,',
        'boards at sternum height, ~1-1.5 m apart, both resting on',
        'something solid that will not move.',
        '',
        'Stay still. Hold your breath as long as is comfortable, then',
        'breathe as shallowly as you can manage for the rest.',
    ]),
    ('paced', [
        'Same position, do not move the boards between runs.',
        '',
        f'Set a metronome to {METRONOME_BPM} BPM. Inhale on one beat, exhale',
        f'on the next - that is {TARGET_BPM:.0f} breaths per minute. Breathe',
        'deeply and regularly. Keep everything except your chest still.',
    ]),
]


def prominence(csv_path):
    """Best in-band peak prominence across the first 4 PCs.

    Same measure 28_capture_diagnostics.py reports, so the numbers are
    directly comparable between the two scripts.
    """
    from scipy import signal
    from csi_pipeline import dwt_band_filter

    t, H = load_csi(csv_path)
    amp = np.abs(H[:, DATA_IDX]).astype(np.float32)
    grid = np.arange(0, t[-1], 1.0 / _m.FS_TARGET)
    feats = np.stack([np.interp(grid, t, amp[:, k])
                      for k in range(amp.shape[1])], axis=1)
    filt = dwt_band_filter(feats, _m.FS_TARGET).astype(np.float32)

    X = (filt - filt.mean(0)) / (filt.std(0) + 1e-9)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    nperseg = min(int(_m.FS_TARGET * 60), filt.shape[0])

    best = (0.0, float('nan'), 0)
    for i in range(min(4, U.shape[1])):
        f, P = signal.welch(U[:, i] * S[i], fs=_m.FS_TARGET, nperseg=nperseg)
        m = (f >= _m.BAND[0]) & (f <= _m.BAND[1])
        if not m.any():
            continue
        prom = float(P[m].max() / (np.median(P[m]) + 1e-12))
        if prom > best[0]:
            best = (prom, float(f[m][np.argmax(P[m])]) * 60, i + 1)
    return best          # (prominence, bpm, pc_number)


def run_step(label, instructions, run_id, port, baud):
    print('\n' + '=' * 68)
    print(f'CAPTURE: {label}')
    print('=' * 68)
    for line in instructions:
        print('  ' + line)
    print()
    input('  Press ENTER when you are ready to start the 60 s capture...')

    for n in (3, 2, 1):
        print(f'  starting in {n}...', end='\r', flush=True)
        time.sleep(1)
    print('  RECORDING - hold the condition for 60 seconds.        ')

    out = OUT_DIR / f'abtest_{run_id}_{label}.csv'
    n_csi = _m.capture(port, baud, DURATION, out)
    print(f'  done: {n_csi} CSI rows -> {out.name}')

    if n_csi < 500:
        print('  WARNING: very few packets. Check the boards are still linked.')
        return out, None

    prom, bpm, pc = prominence(out)
    print(f'  best in-band prominence: {prom:.1f} at {bpm:.1f} BPM (PC{pc})')
    return out, (prom, bpm, pc)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    wanted = [a for a in sys.argv[1:] if not a.startswith('-')]
    steps = [s for s in STEPS if not wanted or s[0] in wanted]
    if not steps:
        print(f'Unknown label. Choose from: {", ".join(s[0] for s in STEPS)}')
        return 1

    try:
        port, baud = _m.resolve_port()
    except RuntimeError as e:
        print(f'\n{e}')
        return 1

    run_id = int(time.time())
    print(f'\nRun {run_id} - {len(steps)} capture(s) of {DURATION:.0f}s each.')
    print('Do not move the boards once the breathhold capture has started.')

    results = {}
    for label, instructions in steps:
        _path, res = run_step(label, instructions, run_id, port, baud)
        results[label] = res

    print('\n' + '=' * 68)
    print('SUMMARY')
    print('=' * 68)
    for label, res in results.items():
        if res is None:
            print(f'  {label:11} capture failed')
        else:
            prom, bpm, pc = res
            print(f'  {label:11} prominence {prom:7.1f}  at {bpm:5.1f} BPM  (PC{pc})')

    print(f'\n  Baseline for reference: the recording that worked scored 138.5')
    print(f'  at 12.1 BPM. Files are in captures/abtest_{run_id}_*.csv')

    ok = all(v is not None for v in results.values())
    if ok and {'control', 'paced'} <= results.keys():
        c = results['control'][0]
        p = results['paced'][0]
        print()
        paced_bpm = results['paced'][1]
        on_target = abs(paced_bpm - TARGET_BPM) <= BPM_TOLERANCE
        if p > 3 * c and p > 20 and on_target:
            print(f'  -> paced beats control AND peaks at {paced_bpm:.1f} BPM, '
                  f'which matches')
            print(f'     the {TARGET_BPM:.0f} BPM you were breathing. The rig '
                  f'detects breathing;')
            print('     the problem is in how the live demo picks its PC.')
        elif p > 3 * c and p > 20:
            print(f'  -> paced beats control, but peaks at {paced_bpm:.1f} BPM '
                  f'rather than the')
            print(f'     {TARGET_BPM:.0f} BPM you were breathing. Something '
                  f'periodic is being')
            print('     picked up, but it is probably not your chest.')
        elif p < 1.5 * c:
            print('  -> paced is no better than an empty room. Geometry: your')
            print('     chest is not intercepting the dominant path. Reposition')
            print('     the boards either side of your torso and repeat.')
        else:
            print('  -> weak but present. Try moving the boards further apart,')
            print('     and breathe more deeply.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
