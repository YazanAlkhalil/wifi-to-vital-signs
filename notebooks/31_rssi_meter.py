"""
31 — Live placement meter. Move the boards, watch the numbers.

Prints one line per second describing what the link is actually doing, so
you can position the boards against measured targets instead of guessing
and then discovering the problem 60 seconds later in a capture.

Columns:
  PPS      usable HT20 packets per second. Below ~20 nothing else matters.
  RSSI     mean +/- standard deviation over the window.
           The recording that worked sat at -72.2 +/- 0.8 dBm.
           A link much stronger than that is dominated by the direct path,
           which chest motion barely perturbs.
  |H|      mean subcarrier amplitude, raw int8 units.
  MOD%     the number to maximise. Typical per-subcarrier swing inside the
           breathing band, as a percentage of mean amplitude. On this
           meter's scale the working capture reads 3.75% and the failing
           ones read 1.4-1.7%. Getting above ~3 is the goal.
  VERDICT  a rough reading of the two numbers together.

MOD% needs a filled window, so it stays blank for the first WINDOW_S seconds
and is meaningless if you are moving around. To read it properly: set the
boards down, sit still, breathe normally, and watch for ten seconds.

This measures how much the channel is being modulated. It cannot tell
breathing from any other movement, so keep the room still while reading it.

Usage:
    .venv\\Scripts\\python.exe notebooks\\31_rssi_meter.py
    .venv\\Scripts\\python.exe notebooks\\31_rssi_meter.py --window 20
"""
from pathlib import Path
import argparse
import collections
import sys
import time

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from csi_pipeline import parse_csi_bytes  # noqa: E402
from csi_serial import resolve_port  # noqa: E402

DATA_IDX = list(np.r_[6:32, 33:59])
FS = 40.0
BAND = (0.15625, 0.625)          # 9.4-37.5 BPM

# Measured from captures/capture_1778960307.csv, the run that recovered
# 11.66 BPM against a 12 BPM metronome.
RSSI_TARGET = -72.2

# MOD% is computed with the Butterworth band below, which reads lower than
# the pipeline's wider DWT band: the working capture scores 3.75 here versus
# 4.69 there. These constants are on THIS scale -- do not paste the DWT
# numbers in. Verified against all four stored captures; the separation
# between working (3.75) and failing (1.39-1.78) survives either filter.
MOD_TARGET = 3.75
MOD_FLOOR = 1.8                  # the failing captures sat at or below this

RSSI_FIELD = 3                   # type,role,mac,rssi,...
N_FIELDS = 26


def read_lines(ser, buf):
    """Yield complete lines from a bounded read.

    Deliberately avoids readline(): it returns only on a newline or an empty
    read, so a port streaming bytes with no newlines blocks forever no matter
    what the port timeout is.
    """
    chunk = ser.read(4096)
    if not chunk:
        return
    buf.extend(chunk)
    if len(buf) > (1 << 20):
        del buf[:-4096]
    while True:
        i = buf.find(b'\n')
        if i < 0:
            return
        raw = bytes(buf[:i])
        del buf[:i + 1]
        yield raw.decode('utf-8', errors='replace').strip()


def modulation_pct(times, amps):
    """Per-subcarrier swing in the breathing band, as % of mean amplitude.

    Returns None when there is not enough data to filter meaningfully.
    """
    if len(times) < 200:
        return None
    t = np.asarray(times)
    span = t[-1] - t[0]
    if span < 12.0:
        return None

    A = np.asarray(amps, dtype=np.float32)
    grid = np.arange(0, span, 1.0 / FS)
    if len(grid) < 100:
        return None
    feats = np.stack([np.interp(grid, t - t[0], A[:, k])
                      for k in range(A.shape[1])], axis=1)

    # Butterworth rather than the pipeline's DWT: the live window is short and
    # a fixed-level wavelet decomposition warns and misbehaves below ~2^7
    # samples. sosfiltfilt is zero-phase, so the swing it reports is honest.
    sos = signal.butter(4, [BAND[0], BAND[1]], btype='band', fs=FS, output='sos')
    padlen = 3 * (sos.shape[0] * 2)
    if feats.shape[0] <= padlen + 1:
        return None
    filt = signal.sosfiltfilt(sos, feats, axis=0)

    band_sigma = float(np.median(filt.std(axis=0)))
    mean_amp = float(feats.mean())
    if mean_amp <= 0:
        return None
    return 100.0 * band_sigma / mean_amp


def verdict(pps, rssi_mean, mod, elapsed):
    # The first second holds a partial window, so its rate is meaningless.
    if elapsed < 3.0:
        return 'starting...'
    if pps < 20:
        return 'LINK WEAK - too few HT20 packets'
    if mod is None:
        return 'filling window...'
    if mod >= MOD_TARGET:
        return 'GOOD - at or above the level that worked'
    if mod >= 3.0:
        return 'PROMISING - keep going this direction'
    if mod <= MOD_FLOOR:
        if rssi_mean > RSSI_TARGET + 5:
            return 'WEAK - link too strong; move boards apart'
        return 'WEAK - little modulation; try moving/reorienting'
    return 'MARGINAL'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=float, default=25.0,
                    help='rolling window in seconds (default 25)')
    args = ap.parse_args()
    window_s = args.window

    import serial
    try:
        port, baud = resolve_port()
    except RuntimeError as e:
        print(f'\n{e}')
        return 1

    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.1
    ser.dtr = False
    ser.rts = False
    ser.open()

    print(f'\nReading {port} @ {baud}, {window_s:.0f}s rolling window.')
    print(f'Targets from the capture that worked: '
          f'RSSI {RSSI_TARGET:.1f} dBm, MOD {MOD_TARGET:.2f}%')
    print('Set the boards down and stay still to read MOD%. Ctrl+C to stop.\n')
    print(f'{"time":>6} {"PPS":>5} {"RSSI":>16} {"|H|":>7} {"MOD%":>7}  verdict')
    print('-' * 78)

    times = collections.deque()
    amps = collections.deque()
    rssis = collections.deque()
    buf = bytearray()
    t0 = time.time()
    last_print = 0.0
    best_mod = 0.0

    try:
        while True:
            for line in read_lines(ser, buf):
                if not line.startswith('CSI_DATA'):
                    continue
                # 26 fields = 25 commas, so split at most 25 times and the
                # CSI byte blob (which contains no commas) lands in parts[25].
                parts = line.split(',', N_FIELDS - 1)
                if len(parts) < N_FIELDS:
                    continue
                try:
                    if (int(parts[5]) != 1 or int(parts[7]) != 1
                            or int(parts[24]) != 384):
                        continue
                    rssi = float(parts[RSSI_FIELD])
                    b = parse_csi_bytes(parts[25])
                    if len(b) != 128:
                        continue
                    imag = b[0::2].astype(np.float32)
                    real = b[1::2].astype(np.float32)
                    amp = np.sqrt(real * real + imag * imag)[DATA_IDX]
                except Exception:
                    continue

                now = time.time()
                times.append(now)
                amps.append(amp)
                rssis.append(rssi)
                cutoff = now - window_s
                while times and times[0] < cutoff:
                    times.popleft()
                    amps.popleft()
                    rssis.popleft()

            now = time.time()
            if now - last_print < 1.0:
                time.sleep(0.02)
                continue
            last_print = now

            elapsed = now - t0
            span = (times[-1] - times[0]) if len(times) > 1 else 0.0
            pps = len(times) / span if span > 0 else 0.0
            if not times:
                print(f'{elapsed:6.0f} {0:5.0f} {"- no packets -":>16} '
                      f'{"-":>7} {"-":>7}  waiting for CSI', flush=True)
                continue

            r = np.asarray(rssis)
            A = np.asarray(amps, dtype=np.float32)
            mod = modulation_pct(list(times), A)
            if mod is not None:
                best_mod = max(best_mod, mod)

            rssi_txt = f'{r.mean():6.1f} +/-{r.std():4.1f}'
            mod_txt = f'{mod:6.2f}' if mod is not None else f'{"-":>6}'
            # flush: stdout is block-buffered when redirected to a file or
            # pipe, which would hold a live meter's lines back for minutes.
            print(f'{elapsed:6.0f} {pps:5.0f} {rssi_txt:>16} '
                  f'{A.mean():7.2f} {mod_txt:>7}  '
                  f'{verdict(pps, r.mean(), mod, elapsed)}', flush=True)

    except KeyboardInterrupt:
        print('\n\nstopped.')
        if best_mod:
            print(f'best MOD% seen this session: {best_mod:.2f}% '
                  f'(target {MOD_TARGET:.2f}%, failing captures were <= '
                  f'{MOD_FLOOR:.1f}%)')
            if best_mod >= 3.0:
                print('That is worth recording. Run notebooks/29_ab_test.py '
                      'without moving anything.')
            else:
                print('Still low. Try: boards further apart, higher off the '
                      'desk, antennas clear of\nmetal and screens, and sit to '
                      'the side of the line between them rather than in it.')
    finally:
        ser.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
