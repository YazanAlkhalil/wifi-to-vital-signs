"""
30 — Sweep Hampel despiking parameters over the A/B captures.

The point of the sweep is NOT to find a setting that reports 15 BPM on the
paced capture. With 25 parameter combinations and a noisy spectrum, some
setting will land near any target you pick, purely by chance. That is how
you talk yourself into a detection that is not there.

So every setting is scored on all three captures and judged by contrast:

    a setting "wins" only if it finds the paced rate in `paced`
    AND does not find it in `control`.

`control` was recorded with nobody in the room. Anything that fires there is
a false positive by construction, which makes it the honest negative control.
The known-good capture is swept too, as a positive control: settings that
break a signal we know is real are not worth having.

Usage:
    .venv\\Scripts\\python.exe notebooks\\30_hampel_sweep.py
"""
from pathlib import Path
import itertools
import sys

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from csi_pipeline import load_csi, dwt_band_filter, hampel_columns  # noqa: E402

DATA_IDX = list(np.r_[6:32, 33:59])
FS = 40.0
BAND = (0.15625, 0.625)

WINDOWS = [0, 5, 11, 21, 31]          # 0 = no Hampel, the current behaviour
SIGMAS = [1.5, 2.0, 2.5, 3.0, 4.0]

RUN = 'abtest_1786834131'
CAPS = {
    'paced':      (ROOT / 'captures' / f'{RUN}_paced.csv', 15.0),
    'breathhold': (ROOT / 'captures' / f'{RUN}_breathhold.csv', None),
    'control':    (ROOT / 'captures' / f'{RUN}_control.csv', None),
    'baseline':   (ROOT / 'captures' / 'capture_1778960307.csv', 12.0),
}
TOL = 1.5           # BPM either side of target that counts as a hit


def features(path):
    """Resampled amplitude, computed once per capture and reused."""
    t, H = load_csi(path)
    amp = np.abs(H[:, DATA_IDX]).astype(np.float32)
    grid = np.arange(0, t[-1], 1.0 / FS)
    return np.stack([np.interp(grid, t, amp[:, k])
                     for k in range(amp.shape[1])], axis=1)


def score(feats, win, sigma):
    """(prominence, peak_bpm) for the best of the first four PCs."""
    x = hampel_columns(feats, win=win, n_sigmas=sigma) if win else feats
    filt = dwt_band_filter(x, FS).astype(np.float32)
    X = (filt - filt.mean(0)) / (filt.std(0) + 1e-9)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    nperseg = min(int(FS * 60), filt.shape[0])
    best = (0.0, float('nan'))
    for i in range(min(4, U.shape[1])):
        f, P = signal.welch(U[:, i] * S[i], fs=FS, nperseg=nperseg)
        m = (f >= BAND[0]) & (f <= BAND[1])
        if not m.any():
            continue
        prom = float(P[m].max() / (np.median(P[m]) + 1e-12))
        if prom > best[0]:
            best = (prom, float(f[m][np.argmax(P[m])]) * 60)
    return best


def main():
    print('Loading captures...')
    feats = {}
    for name, (path, _) in CAPS.items():
        if not path.exists():
            print(f'  missing: {path.name}')
            continue
        feats[name] = features(path)
        print(f'  {name:11} {feats[name].shape[0]} samples')

    print(f'\nSweeping {len(WINDOWS)}x{len(SIGMAS)} settings over '
          f'{len(feats)} captures.')
    print(f'A "hit" = strongest in-band peak within {TOL} BPM of the target.\n')

    hdr = (f'{"win":>4} {"sigma":>6} | {"paced":>16} | {"control":>16} | '
           f'{"breathhold":>16} | {"baseline":>16} | verdict')
    print(hdr)
    print('-' * len(hdr))

    winners = []
    for win, sigma in itertools.product(WINDOWS, SIGMAS):
        if win == 0 and sigma != SIGMAS[0]:
            continue            # no-Hampel does not depend on sigma
        cells = {}
        for name in feats:
            prom, bpm = score(feats[name], win, sigma)
            cells[name] = (prom, bpm)

        def fmt(name):
            if name not in cells:
                return f'{"-":>16}'
            p, b = cells[name]
            return f'{p:7.1f} @ {b:5.1f}'

        paced_hit = ('paced' in cells
                     and abs(cells['paced'][1] - CAPS['paced'][1]) <= TOL)
        # A control that peaks near the paced rate is a false positive.
        ctrl_hit = ('control' in cells
                    and abs(cells['control'][1] - CAPS['paced'][1]) <= TOL)
        base_ok = ('baseline' in cells
                   and abs(cells['baseline'][1] - CAPS['baseline'][1]) <= TOL)

        if paced_hit and not ctrl_hit:
            verdict = 'WIN' if base_ok else 'win (breaks baseline)'
            winners.append((win, sigma, cells))
        elif paced_hit and ctrl_hit:
            verdict = 'false positive'
        elif not base_ok:
            verdict = 'breaks baseline'
        else:
            verdict = '.'

        label = 'none' if win == 0 else str(win)
        sig = '-' if win == 0 else f'{sigma:.1f}'
        print(f'{label:>4} {sig:>6} | {fmt("paced")} | {fmt("control")} | '
              f'{fmt("breathhold")} | {fmt("baseline")} | {verdict}')

    print()
    if winners:
        print(f'{len(winners)} setting(s) found the paced rate without firing '
              f'on the empty room:')
        for win, sigma, cells in winners:
            print(f'  win={win} sigma={sigma}: paced '
                  f'{cells["paced"][0]:.1f} @ {cells["paced"][1]:.1f} BPM')
        print('\nTreat these as a lead, not a result. Confirm by recording a '
              'fresh paced capture\nat a different rate and checking the same '
              'setting finds that rate too.')
    else:
        print('No setting found the paced rate in `paced` without also firing '
              'on `control`.')
        print('Despiking is not the missing ingredient: the periodic component '
              'is absent from the\ndata, not hidden under spikes.')


if __name__ == '__main__':
    main()
