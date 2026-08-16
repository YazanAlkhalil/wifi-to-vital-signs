"""
Report which serial port is streaming CSI, and at what baud rate.

The capture scripts now auto-detect this themselves, so you only need this
when something is wrong and you want to see every port's behaviour rather
than just the winner.

Read-only and safe: it holds DTR/RTS low, so it will not reset the boards.

Usage:
    .venv\\Scripts\\python.exe sniff_ports.py      (Windows)
    .venv/bin/python sniff_ports.py                (macOS / Linux)
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'notebooks'))
from csi_serial import DEFAULT_BAUDS, candidate_ports, listen  # noqa: E402

LISTEN_SECONDS = 3.0


def main():
    ports = candidate_ports()
    if not ports:
        print('No serial ports found.\n'
              '  Windows: the CP210x driver is probably missing — SETUP.md B4.\n'
              '  macOS:   install the Silicon Labs driver, then approve it in\n'
              '           System Settings > Privacy & Security.\n'
              '  Linux:   add yourself to the dialout group.')
        return 1

    print(f'{len(ports)} port(s). Listening {LISTEN_SECONDS:.0f}s at each baud.\n')
    found = []
    for p in ports:
        print(f'{p.device}  ({p.description})')
        for baud in DEFAULT_BAUDS:
            try:
                n_lines, n_csi, sample = listen(p.device, baud, LISTEN_SECONDS)
            except Exception as e:
                print(f'   {baud:>7} baud: cannot open — {e}')
                continue

            pps = n_csi / LISTEN_SECONDS
            print(f'   {baud:>7} baud: {n_lines:5d} lines, {n_csi:5d} CSI '
                  f'({pps:.1f} PPS)')
            if n_csi > 0:
                role = sample.split(',')[1] if ',' in sample else '?'
                print(f'      role={role}  {sample[:80]}')
                found.append((p.device, baud, pps))
                break       # right baud for this port; don't try the others

    print()
    if not found:
        print('No CSI on any port. Likely causes, in order:\n'
              '  - the RX board (active_ap) is not plugged in — the TX board\n'
              '    is silent by design\n'
              '  - the boards have not associated (SSID/password must match)\n'
              '  - firmware flashed at a baud outside '
              f'{list(DEFAULT_BAUDS)} — set CSI_BAUD')
        return 1

    for device, baud, pps in found:
        print(f'CSI source: {device} at {baud} baud ({pps:.1f} PPS)')
        print('  The capture scripts find this on their own - you only need\n'
              '  to set CSI_PORT if you want to override the choice.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
