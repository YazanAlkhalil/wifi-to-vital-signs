"""
Environment check — run this first on a fresh machine.

Confirms every dependency imports, lists the serial ports the OS is willing
to hand us, and re-runs the DSP pipeline on the recorded capture in
`captures/` so you know the whole analysis chain works before you touch any
hardware.

Usage:
    .venv\\Scripts\\python.exe verify_env.py      (Windows)
    .venv/bin/python verify_env.py               (macOS / Linux)
"""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
EXPECTED_BPM = 11.66          # what the saved 12 BPM metronome capture yields
REFERENCE_CAPTURE = ROOT / 'captures' / 'capture_1778960307.csv'


def check_imports():
    print('== packages ==')
    ok = True
    for name in ('numpy', 'scipy', 'pandas', 'matplotlib', 'pywt',
                 'serial', 'vmdpy', 'ssqueezepy'):
        try:
            mod = __import__(name)
            print(f'  {name:12} {getattr(mod, "__version__", "ok")}')
        except Exception as e:
            print(f'  {name:12} MISSING ({e})')
            ok = False
    print(f'  {"python":12} {sys.version.split()[0]}')
    return ok


def check_ports():
    print('\n== serial ports ==')
    try:
        import serial.tools.list_ports as lp
    except Exception as e:
        print(f'  pyserial unavailable: {e}')
        return
    ports = list(lp.comports())
    if not ports:
        print('  none found — fine for offline work. For the hardware rig this '
              'usually means the CP210x/CH340 USB-serial driver is missing; '
              'see SETUP.md section B4.')
        return
    for p in ports:
        print(f'  {p.device:12} {p.description}')


def check_pipeline():
    print('\n== DSP pipeline on recorded capture ==')
    if not REFERENCE_CAPTURE.exists():
        print(f'  skipped — {REFERENCE_CAPTURE.name} not present')
        return True

    # The capture scripts are numbered, so they aren't importable by name.
    sys.path.insert(0, str(ROOT / 'notebooks'))
    path = ROOT / 'notebooks' / '26_first_capture_test.py'
    spec = importlib.util.spec_from_file_location('capture_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    t, H = mod.load_csi(REFERENCE_CAPTURE)
    bpm, _fused, _fs, _grid, info = mod.paper_mode_bpm(t, H)
    print(f'  packets  {len(t)}')
    print(f'  span     {t[-1] - t[0]:.1f} s')
    print(f'  BPM      {bpm:.2f}  (PC{info["pc_idx"] + 1})')

    if abs(bpm - EXPECTED_BPM) < 0.1:
        print(f'  matches the expected {EXPECTED_BPM} BPM — pipeline is good.')
        return True
    print(f'  WARNING: expected ~{EXPECTED_BPM} BPM. A dependency version may '
          f'have shifted behaviour.')
    return False


if __name__ == '__main__':
    imports_ok = check_imports()
    check_ports()
    pipeline_ok = check_pipeline() if imports_ok else False
    print('\n' + ('ALL GOOD' if imports_ok and pipeline_ok
                  else 'SOMETHING NEEDS ATTENTION — see above'))
    sys.exit(0 if imports_ok and pipeline_ok else 1)
