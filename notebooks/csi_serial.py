"""
Find the ESP32 that is streaming CSI, on any OS.

The rig has two boards. Only the RX one (ESP32-CSI-Tool's `active_ap`) prints
`CSI_DATA` lines; the TX board (`active_sta`) is silent. Which serial device
each lands on is not stable — it follows the USB socket, so replugging or
swapping cables renames them (COM3/COM4 on Windows, /dev/cu.usbserial-* on
macOS, /dev/ttyUSB* on Linux).

Rather than hardcode a name, `find_csi_port()` listens to each candidate port
and returns the one actually emitting CSI.

Detection is skipped entirely if CSI_PORT is set, so an explicit choice always
wins and costs nothing.
"""
import os
import sys
import time

import serial
import serial.tools.list_ports as list_ports

# 921600 is what this rig's firmware is flashed at; 1843200 is the rate the
# reference paper used; 115200 is the ESP-IDF default someone might leave in.
DEFAULT_BAUDS = (921600, 1843200, 115200)

# USB-serial bridges found on ESP32 devkits: CP210x (Silicon Labs), CH340
# (WCH), FTDI, and the ESP32-S2/S3 native USB.
KNOWN_VIDS = {0x10C4, 0x1A86, 0x0403, 0x303A}


def candidate_ports():
    """Serial ports worth probing, most likely first."""
    ports = list(list_ports.comports())
    # A USB-serial bridge we recognise beats a random built-in port (on macOS
    # especially, /dev/cu.Bluetooth-* shows up and never answers).
    known = [p for p in ports if p.vid in KNOWN_VIDS]
    other = [p for p in ports if p.vid not in KNOWN_VIDS
             and 'Bluetooth' not in (p.device + (p.description or ''))]
    return known + other


_MAX_BUFFER = 1 << 20        # 1 MB; discard older bytes past this


def listen(device, baud, seconds):
    """Return (total_lines, csi_lines, first_csi_line) heard on a port.

    Holds DTR/RTS low so opening the port does not reset the board.

    Deliberately does NOT use `readline()`. pyserial's readline returns only
    on a newline or an empty read, so a port streaming continuous bytes with
    no newline in them — which is exactly what you get reading a board at the
    wrong baud rate — blocks forever regardless of the port timeout. Probing
    must be bounded by wall-clock time, so we do fixed-size reads and split
    the lines ourselves.
    """
    s = serial.Serial()
    s.port = device
    s.baudrate = baud
    s.timeout = 0.1          # caps how long a single read() can block
    s.dtr = False
    s.rts = False
    s.open()
    try:
        n_lines = n_csi = 0
        sample = None
        buf = bytearray()
        end = time.time() + seconds
        while time.time() < end:
            chunk = s.read(4096)
            if not chunk:
                continue
            buf.extend(chunk)
            # A stream with no newlines at all would otherwise grow forever.
            if len(buf) > _MAX_BUFFER:
                del buf[:-4096]
            while True:
                i = buf.find(b'\n')
                if i < 0:
                    break
                raw = bytes(buf[:i])
                del buf[:i + 1]
                line = raw.decode('utf-8', errors='replace').strip()
                if not line:
                    continue
                n_lines += 1
                if line.startswith('CSI_DATA'):
                    n_csi += 1
                    if sample is None:
                        sample = line
        return n_lines, n_csi, sample
    finally:
        s.close()


def find_csi_port(bauds=DEFAULT_BAUDS, listen_s=1.5, verbose=True):
    """Locate the CSI-emitting board. Returns (device, baud).

    Raises RuntimeError if nothing is streaming CSI, with a message naming the
    likely cause rather than just failing to open a port later on.
    """
    ports = candidate_ports()
    if not ports:
        raise RuntimeError(
            'No serial ports found at all.\n'
            '  Windows: the CP210x driver is probably missing — see SETUP.md B4.\n'
            '  macOS:   install the Silicon Labs driver and approve it in\n'
            '           System Settings > Privacy & Security.\n'
            '  Linux:   add yourself to the dialout group.')

    if verbose:
        names = ', '.join(p.device for p in ports)
        print(f'[csi] probing {len(ports)} port(s): {names}')

    # Sweep by baud rate rather than by port: the first baud is overwhelmingly
    # the likely one, so we check every port at 921600 before trying anything
    # exotic. Finds the usual case in ~1.5 s per port instead of ~4.5 s.
    for baud in bauds:
        for p in ports:
            try:
                _n_lines, n_csi, sample = listen(p.device, baud, listen_s)
            except Exception as e:
                if verbose:
                    print(f'[csi]   {p.device} @ {baud}: cannot open ({e})')
                continue

            if n_csi > 0:
                role = sample.split(',')[1] if sample and ',' in sample else '?'
                if verbose:
                    pps = n_csi / listen_s
                    print(f'[csi]   {p.device} @ {baud}: {n_csi} CSI '
                          f'(~{pps:.0f} PPS, role={role}) -> using this')
                return p.device, baud
            if verbose:
                print(f'[csi]   {p.device} @ {baud}: silent')

    raise RuntimeError(
        'Serial ports exist, but none is streaming CSI.\n'
        '  - Is the RX board (active_ap) plugged in? The TX board is silent\n'
        '    by design and will never print CSI.\n'
        '  - Have the two boards associated? Their SSID and password must\n'
        '    match. Check with: idf.py monitor\n'
        f'  - Was the firmware flashed at a baud rate outside {list(bauds)}?\n'
        '    If so, set CSI_BAUD to it.')


def resolve_port(verbose=True):
    """(device, baud) for the capture scripts.

    Honours CSI_PORT / CSI_BAUD when set; otherwise auto-detects. This is what
    the scripts call at startup.
    """
    env_port = os.environ.get('CSI_PORT')
    env_baud = os.environ.get('CSI_BAUD')
    bauds = (int(env_baud),) if env_baud else DEFAULT_BAUDS

    if env_port:
        baud = int(env_baud) if env_baud else DEFAULT_BAUDS[0]
        # Don't trust the override blindly. A CSI_PORT left over from an
        # earlier session is a stale shell variable, and the board may have
        # since been replugged onto a different port. Confirm it is actually
        # streaming before committing to a capture that runs for minutes.
        try:
            _n_lines, n_csi, _sample = listen(env_port, baud, 1.5)
        except Exception as e:
            # Plain ASCII: the Windows console codepage mangles em-dashes.
            print(f'[csi] CSI_PORT={env_port} cannot be opened ({e}) '
                  f'- falling back to auto-detect')
        else:
            if n_csi > 0:
                if verbose:
                    print(f'[csi] using CSI_PORT={env_port} @ {baud}')
                return env_port, baud
            print(f'[csi] CSI_PORT={env_port} @ {baud} is silent '
                  f'- ignoring it and auto-detecting instead')
            print(f'[csi] (clear it with:  '
                  + ('Remove-Item Env:CSI_PORT' if sys.platform == 'win32'
                     else 'unset CSI_PORT') + ')')

    return find_csi_port(bauds=bauds, verbose=verbose)


if __name__ == '__main__':
    # Same path the capture scripts take, so running this module directly
    # reproduces exactly what they will do.
    try:
        device, baud = resolve_port()
    except RuntimeError as e:
        print(f'\n{e}')
        sys.exit(1)
    print(f'\nCSI source: {device} at {baud} baud')
    if sys.platform == 'win32':
        print(f'  $env:CSI_PORT="{device}"; $env:CSI_BAUD="{baud}"')
    else:
        print(f'  export CSI_PORT={device} CSI_BAUD={baud}')
