# Setup guide — reproducing the Wi-Fi CSI breathing rig

How to get this project running from scratch on a fresh machine, and how to
build the two-ESP32 hardware rig that produces the live captures.

Two independent halves. **Part A** (software) works on its own — you can run
the whole DSP pipeline against the recorded datasets and the saved captures in
`captures/` with no hardware at all. **Part B** (hardware) is only needed to
record new data.

---

## Part A — Software environment

### A1. What you need

- **Python 3.12.** Not 3.14 — `numba` (pulled in by `ssqueezepy`) has no wheel
  for 3.14 yet and will try to build from source and fail. 3.12 has wheels for
  everything in `requirements.txt`.
- **Git**, to clone the repo.
- ~2 GB of disk for the venv (the Jupyter stack is most of it).

### A2. Create the virtual environment

A virtual environment ("venv") is a private folder of Python packages for this
project only, so installs here can't break other Python work on the machine.

**Windows (PowerShell):**

```powershell
cd "path\to\grad project"
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux:**

```bash
cd "path/to/grad project"
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Everywhere below, `.venv\Scripts\python.exe` (Windows) and `.venv/bin/python`
(macOS/Linux) are interchangeable — pick the one for your OS.

> **If the install dies partway with `[WinError 32] The process cannot access
> the file`:** another process still holds a file inside `.venv`. Usually it's
> an earlier `pip` run that hasn't exited, or a Jupyter kernel, or antivirus
> scanning the folder. Close them (`Get-Process python`), then re-run the same
> `pip install -r requirements.txt` — pip is resumable and skips what's already
> installed. You do **not** need to delete the venv and start over.

### A3. Verify it worked

`verify_env.py` at the repo root checks all three things at once — that every
package imports, what serial ports the OS is offering, and that the DSP still
reproduces a known result:

```powershell
.venv\Scripts\python.exe verify_env.py
```

Expected on a good install (the "none found" for ports is normal and fine
until you've dealt with the drivers in **B4**):

```
== packages ==
  numpy        2.5.2
  ...
== serial ports ==
  none found — ...
== DSP pipeline on recorded capture ==
  packets  5721
  span     59.4 s
  BPM      11.66  (PC2)
  matches the expected 11.66 BPM — pipeline is good.

ALL GOOD
```

That last block is the real test. `captures/capture_1778960307.csv` is a genuine
60-second recording from the rig, taken while breathing to a ~12 BPM metronome.
Getting 11.66 BPM back out means the entire analysis chain — CSI byte decoding,
subcarrier masking, resampling, DWT filtering, PCA fusion, Welch spectrum — is
working, with no hardware involved.

Versions this was last verified against (Windows 11, Python 3.12.10):

| package | version |
|---|---|
| numpy | 2.5.2 |
| scipy | 1.18.0 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |
| pywavelets (`pywt`) | 1.8.0 |
| pyserial (`serial`) | 3.5 |
| vmdpy, ssqueezepy | latest |

Note `requirements.txt` pins no versions, so a fresh install picks up whatever
is current. If something breaks later, this table is what a known-good install
looked like.

### A4. Jupyter (for `notebooks/01_breathing_dsp_baseline.ipynb`)

Register the venv as a kernel so the notebook UI can find it:

```powershell
.venv\Scripts\python.exe -m ipykernel install --user --name grad-project --display-name "Python (grad project)"
.venv\Scripts\jupyter.exe lab
```

Then pick **Python (grad project)** as the kernel.

### A5. Datasets

`datasets/` is in `.gitignore` — it is large and does not travel with the repo.
A collaborator cloning fresh gets the code but not the data. The two primary
datasets and their layouts are described in `README.md`; the format traps
(CSV trailer rows, the apostrophe in the belt time column, which of `Arb1`/
`Arb2` is the belt) are documented in `CLAUDE.md`. Read that before parsing
anything by hand — those cost real debugging time to find.

---

## Part B — The two-ESP32 hardware rig

### B1. Hardware

- **Two ESP32-DevKitC-VE boards.** This is what the reference paper used
  (`papers/Non-contact_Wi-Fi_Sensing_of_Respiration_Rate_for_.pdf`, §III.A:
  *"two ESP32-DevKitC-VE embedded devices … one programmed to act as the Access
  Point or Transmitter (TX), and the other set as the Receiver (RX)"*). Any
  ESP32 devkit with the same WROOM-32 module works; the 802.11n HT20 CSI format
  the code decodes is the same.
- **Two USB cables that carry data.** Charge-only cables are a common and
  maddening failure — the board powers up, its LED lights, and no serial port
  ever appears.
- One host computer. Only the RX board needs to stay plugged into it during a
  capture; the TX just needs power.

### B2. Firmware — which one

**[ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool)** by
Steven M. Hernandez. Two independent confirmations:

1. The paper cites it directly (§III.B.1: *"We use the esp32-CSI-tool to obtain
   CSI data using the IEEE 802.11n 2.4 GHz Wi-Fi standard [55]"*, where
   reference [55] is Hernandez & Bulut, WoWMoM 2020).
2. The CSV header in `captures/` is this tool's exact output format —
   `type,role,mac,rssi,rate,sig_mode,…,real_time_set,real_timestamp,len,CSI_DATA`.
   The `role` and `real_time_set` columns are specific to it. The bundled
   dataset CSVs share the header, which is why one loader parses both.

Two of its example projects matter:

| project | role | flash it to |
|---|---|---|
| `active_sta` | station, sends packets | the **TX** board |
| `active_ap`  | access point, receives and **prints CSI over serial** | the **RX** board |

The capture files have `role` = `AP`, so the RX board — the one you read from —
is the `active_ap` one. That is the board whose port goes into `CSI_PORT`.

### B3. Flashing

ESP32-CSI-Tool is an ESP-IDF project, not Arduino. It targets the **ESP-IDF
v4.x** line; recent v5.x releases changed Wi-Fi and build APIs and the examples
may not compile unmodified. If the build errors out on missing headers or
changed function signatures, checking out an IDF v4.3 tag is the first thing to
try.

```bash
git clone https://github.com/StevenMHernandez/ESP32-CSI-Tool.git
cd ESP32-CSI-Tool
```

Install ESP-IDF per Espressif's own guide (on Windows, the **ESP-IDF Tools
Installer** is far less painful than doing it by hand), then open the "ESP-IDF
Command Prompt" it creates and, for each board in turn:

```bash
cd active_ap          # or active_sta for the other board
idf.py set-target esp32
idf.py menuconfig     # see settings below
idf.py -p COM5 flash monitor    # macOS/Linux: -p /dev/cu.usbserial-4
```

In `menuconfig`, the settings that matter:

- **Serial baud rate.** The repo's capture scripts default to **921600**. The
  paper used 1843200. They must match between firmware and script — if they
  don't, you get garbage bytes or nothing at all. Either flash at 921600, or
  flash at 1843200 and run the scripts with `CSI_BAUD=1843200`.
- **Wi-Fi SSID / password**, identical on both boards so the STA associates
  with the AP.
- **Packet rate.** The paper ran 120 packets/s. The saved capture came in at
  ~96 PPS after filtering. Anything comfortably above ~20 PPS is enough for
  breathing (0.15–0.63 Hz); more is better for heart rate.

Press `Ctrl+]` to leave `idf.py monitor` — and **close the monitor before
running any capture script**, since only one process can hold a serial port.

### B4. Serial drivers — the thing that will actually block you

ESP32-DevKitC boards talk to the host through a **Silicon Labs CP2102**
USB-to-UART bridge. Windows ships no driver for it. Without one the boards
enumerate as USB devices but **no COM port is created**, so `pyserial` sees
nothing and the capture scripts have no port to open. This is the single most
likely thing to stop a fresh Windows machine.

**How to tell this is your problem.** The boards show up, but broken:

```powershell
Get-PnpDevice | Where-Object { $_.InstanceId -match 'VID_10C4|VID_1A86' } |
    Select-Object Status, FriendlyName, InstanceId, Problem | Format-List
```

```
Status       : Error
FriendlyName : CP2102 USB to UART Bridge Controller
InstanceId   : USB\VID_10C4&PID_EA60\0001
Problem      : CM_PROB_FAILED_INSTALL          <-- driver missing
```

**Fix (Windows).** These are the exact steps that worked here. You need local
administrator rights — the last command triggers a UAC prompt.

```powershell
# 1. Download the CP210x Universal Windows Driver from Silicon Labs
$zip = "$env:USERPROFILE\Downloads\CP210x_Universal_Windows_Driver.zip"
$dir = "$env:USERPROFILE\Downloads\CP210x_Driver"
Invoke-WebRequest -UseBasicParsing -OutFile $zip `
  -Uri "https://www.silabs.com/documents/public/software/CP210x_Universal_Windows_Driver.zip"
Expand-Archive -Path $zip -DestinationPath $dir -Force

# 2. Sanity-check before installing anything system-wide.
#    Expect: Status = Valid, signed by "Microsoft Windows Hardware
#    Compatibility Publisher", and the INF listing our exact hardware ID.
Get-AuthenticodeSignature "$dir\silabser.cat" | Select-Object Status
Select-String -Path "$dir\silabser.inf" -Pattern 'EA60' | Select-Object -First 1

# 3. Install into the driver store and bind it to the attached boards.
#    A UAC prompt appears — accept it.
Start-Process cmd.exe -Verb RunAs -Wait `
  -ArgumentList '/c pnputil /add-driver "%USERPROFILE%\Downloads\CP210x_Driver\silabser.inf" /install & pause'
```

A successful run prints:

```
Driver package added successfully.
Published Name:         oem64.inf
Driver package installed on device: USB\VID_10C4&PID_EA60\0001
Driver package installed on device: USB\VID_10C4&PID_EA60\5&1e409185&0&4
```

Re-running the `Get-PnpDevice` check above should now show `Status : OK`,
`Problem : CM_PROB_NONE`, and a COM number in the name. No reboot or replug was
needed. If yours still shows an error, unplug and replug both boards.

**macOS / Linux.** On macOS the driver also comes from Silicon Labs and needs
approval under System Settings → Privacy & Security after installing. On Linux
the `cp210x` module is already in the kernel; you only need your user in the
`dialout` group (`sudo usermod -aG dialout $USER`, then log out and back in).

**Clone boards.** Some cheaper ESP32 boards use a **CH340** chip (`VID_1A86`)
rather than a CP2102. Same failure mode, different driver — get it from WCH.

### B4a. Work out which port is which

The capture scripts work this out themselves (see **Part C**), so you can skip
this section unless something is wrong. `sniff_ports.py` shows what *every*
port is doing rather than just naming the winner. It's read-only and holds
DTR/RTS low, so it won't reset the boards:

```powershell
.venv\Scripts\python.exe sniff_ports.py
```

Output on this machine, with the rig powered and associated:

```
COM3  (Silicon Labs CP210x USB to UART Bridge (COM3))
    921600 baud:     0 lines,     0 CSI (0.0 PPS)
   1843200 baud:     0 lines,     0 CSI (0.0 PPS)
    115200 baud:     0 lines,     0 CSI (0.0 PPS)
COM4  (Silicon Labs CP210x USB to UART Bridge (COM4))
    921600 baud:   400 lines,   399 CSI (99.8 PPS)
      CSI_DATA,AP,1C:C3:AB:B4:06:E0,-38,11,1,4,1,1,1,1,0,0,0,-96,1,6,1,...

CSI source: COM4 at 921600 baud (99.8 PPS)
```

The silent port is the station; the chatty one is the AP. The `AP` in the second
field of a `CSI_DATA` line is the confirmation.

**Port names are not stable.** They follow the USB socket, so moving a cable to
a different port renames them — the two boards above swapped from COM4/COM3 to
COM3/COM4 during this session simply by being replugged. That is exactly why
the scripts detect rather than remember, and why you shouldn't write a port
name into anything permanent.

If **no** port produces CSI, the script lists the likely causes — most often
the two boards haven't associated because their SSID/password don't match.

### B5. Physical placement

From the paper's setup: TX and RX on either side of the subject, roughly chest
height, a couple of metres apart, with the person's torso between them so chest
movement modulates the direct path. Keep both boards still — the pipeline
cannot tell chest motion from board motion. Nobody else should be moving in the
room during a capture.

---

## Part C — Running a capture

**You don't need to configure a port.** Both capture scripts auto-detect it at
startup: they list the serial devices, listen to each for ~1.5 s, and use
whichever one is emitting `CSI_DATA`. This works the same on Windows, macOS and
Linux, and survives replugging the boards or swapping the cables.

```
[csi] probing 2 port(s): COM3, COM4
[csi]   COM3 @ 921600: 150 CSI (~100 PPS, role=AP) -> using this
```

Set `CSI_PORT` only to override that choice; `CSI_BAUD` likewise if your
firmware uses a rate outside 921600 / 1843200 / 115200.

### C1. Record 60 seconds and get a number

```powershell
.venv\Scripts\python.exe notebooks\26_first_capture_test.py
```

```bash
.venv/bin/python notebooks/26_first_capture_test.py
```

Breathe on a metronome for the full minute. It writes
`captures/capture_<unix-time>.csv` plus a PNG of the fused trace and its power
spectrum, and prints the estimated BPM. `DURATION` at the top of the file
controls the length.

### C2. Live plot

```powershell
.venv\Scripts\python.exe notebooks\27_live_demo.py
```

First 30 s is a warm-up: it collects CSI, runs the full preprocessing once, and
locks onto the principal component that best shows breathing. After that every
incoming packet is projected onto that fixed direction, so the trace is stable
second to second and holding your breath visibly flattens it. BPM updates about
once a second.

### C3. Troubleshooting

| symptom | cause |
|---|---|
| `No serial ports found at all` | driver problem — see **B4** |
| `Serial ports exist, but none is streaming CSI` | only the TX board is plugged in, or the boards haven't associated (SSID/password must match), or the firmware baud isn't one of the three probed — set `CSI_BAUD` |
| `ERROR: warmup got too few packets` after 30 s | a stale `CSI_PORT` in that shell pointing at the wrong board. The scripts now detect this in ~1.5 s and auto-correct, but you can clear it with `Remove-Item Env:CSI_PORT` (PowerShell) or `unset CSI_PORT` |
| `SerialException: could not open port` | `idf.py monitor` or another capture script still holds it — only one process at a time |
| `ERROR: too few packets` | boards too far apart, or association dropped mid-capture |
| `skipped N malformed line(s)` | normal in small numbers at 921600 baud; hundreds means a baud mismatch |
| BPM is wildly wrong | someone moved, or a board moved; re-record |
| live plot exits with "no interactive backend" | `pip install pyqt6`, or use script 26, which writes a PNG and needs no window |

---

## What was changed to make this portable

The two capture scripts were macOS-only. Both now:

- read the port from `CSI_PORT` (and baud from `CSI_BAUD`), defaulting per OS
  instead of hardcoding `/dev/cu.usbserial-4`;
- in `27_live_demo.py`, select a matplotlib backend by platform — `macosx`
  exists only on macOS, so on Windows/Linux it now tries `QtAgg` then `TkAgg`
  and prints which one it got.

Two new helper scripts at the repo root, so a collaborator can diagnose without
reading any code:

- **`verify_env.py`** — checks packages, lists serial ports, and reproduces
  11.66 BPM from the recorded capture. Separates "broken install" from "driver
  problem".
- **`sniff_ports.py`** — listens to every serial port at each plausible baud
  rate and reports what each one is doing. Diagnostic only now.
- **`notebooks/csi_serial.py`** — the shared auto-detection used by both
  capture scripts. `resolve_port()` honours `CSI_PORT`/`CSI_BAUD` when set and
  otherwise finds the CSI-emitting board, so no port name is hardcoded
  anywhere.

Also fixed in `26_first_capture_test.py`: it used to write the CSV header only
if it happened to catch the firmware's boot-time header line. Attach to a board
that is already running — the normal case — and the file had no header, so
`load_csi` read the first data row as column names and died with
`invalid literal for int() with base 10: 'SI_DAT'`. It now always writes the
header, and drops truncated or spliced lines instead of letting them poison the
parse.

Nothing in the DSP itself changed; `26_first_capture_test.py` reproduces the
same 11.66 BPM on the saved capture as before.
