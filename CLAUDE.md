# CLAUDE.md

Project-specific notes for Claude Code. Read this before doing any work in this repo.

## What this project is

Graduation project: estimate vital signs from Wi-Fi CSI (Channel State Information), no contact sensors. Targets in priority order:

1. **Breathing rate** — must work.
2. **Heart rate** — nice to have.
3. **Sleep stages / disturbances** — stretch.

User is a college student. Read `README.md` for the longer write-up.

## Working directory

The project is worked on from two machines. Check which one you're on before
assuming paths:

- **macOS (original)**: `/Users/kpj/college/groudation project` (yes, "groudation" — typo in the path, leave it).
- **Windows**: `C:\Users\Lenovo\Downloads\grad project`.

## Environment

Full from-scratch instructions — both OSes, plus the ESP32 firmware and driver
setup — are in **`SETUP.md`**. Summary:

- Venv at `.venv/`. **Python 3.12 on Windows** (3.14 has no `numba` wheel, which
  `ssqueezepy` needs, so the install fails); 3.14 on the original Mac.
- Run binaries directly: `.venv\Scripts\python.exe` (Windows), `.venv/bin/python` (macOS).
- Deps in `requirements.txt` — **unpinned**, so versions drift between machines.
  Known-good versions are tabled in `SETUP.md` §A3.
- `python verify_env.py` checks imports + serial ports + reproduces 11.66 BPM
  from `captures/capture_1778960307.csv`. Run it first when anything looks off.
- Jupyter kernel registered as **`Python (groudation)`** on the Mac — that's the one to pick in the notebook UI.
- Serial-dependent scripts (`26_*`, `27_*`) **auto-detect** the board via
  `notebooks/csi_serial.py:resolve_port()` — it probes each port for `CSI_DATA`
  lines. `CSI_PORT`/`CSI_BAUD` override it. Never hardcode a port name: COM
  numbers follow the USB socket and change on replug.
- **NumPy is 2.x**: `np.fromstring` is gone — use `np.array(s.split(), dtype=...)`. `Series.to_numpy()` returns a read-only view, so call `.copy()` before any in-place op (`-=`, `+=`).

## Datasets

Four datasets in `datasets/`. We are mostly using the first two:

1. `Respiration Rate Measurement Validity and Repeatability ... Older Adults in Care/` — Validity/ + Reliability/ splits, paired Neulog belt + Wi-Fi CSI. Has bundled `*.ipynb` extraction notebooks (load + plot only).
2. `Sleep Disturbances Dataset/` — `Vital Signs/` (paced 12 BPM CSI + belt + HR), `Sleep Apnoa/` (CSA + OSA, paired CSI/belt), and event folders (`Confusional Arousals/`, `Leg Restlessness/`, `Posture Changes/`). Has `scripts/wi_fi_csi_extraction.py` and `scripts/ground_truth_extraction.py` (load + plot only).
3. `MultiPatient Elderly Respiration ... Digital Twin Technology/` — still a `.rar`, not extracted. Reserve.
4. `chinese data/` — not inspected. Reserve.

Default to datasets 1 and 2. Don't touch 3 or 4 unless asked or unless 1+2 are clearly insufficient.

## Data format gotchas

These are not derivable from the code — they are the result of opening the raw files and getting bitten. Don't re-derive.

### CSI CSV (ESP32, both primary datasets)

- `CSI_DATA` column: string `"[110 96 6 0 ... ]"` of **128 signed bytes** = 64 complex subcarriers.
- ESP-IDF byte order: **interleaved imag/real** — byte 0 = imag(sc0), byte 1 = real(sc0), byte 2 = imag(sc1), …
- Per subcarrier: `amp = sqrt(re² + im²)`, `phase = arctan2(im, re)`.
- **Active data subcarriers (HT20): indices `[6:32] + [33:59]` = 52.** The other 12 are guards / DC null / pilots. The dataset authors use this exact mask in their bundled scripts. Use this rather than a variance threshold.
- Time column: `real_timestamp` (Unix seconds). **Packet rate is jittery.** Resample onto a uniform grid (e.g. 20 Hz) before any FFT.
- Median packet rate: ~111 Hz on sleep-disturbances `Breathing - 12 BPM - CSI.csv`; ~40 Hz on the older-adults dataset.

### Belt CSV (Neulog logger)

- Sampling rate: **50 Hz on the sleep-disturbances dataset, 100 Hz on the older-adults dataset.** Don't assume.
- Time column starts with a literal apostrophe (`'0:0:0.04`) — Excel artifact. Strip it before parsing H:M:S.
- Two channels: `Arb1` and `Arb2`. **In `Sleep Disturbances Dataset/Vital Signs/Breathing - Belt & HR.csv`, `Arb2` is the chest belt (smooth ~5 s waves at 12 BPM); `Arb1` is the HR sensor waveform (sharper / squarer pulses).** Confirmed empirically — column ordering is not labeled in the CSV. Same convention is *probably* but not *confirmed* across the other sleep-disturbance files.
- **The CSV has trailer rows** after the data: `"Attention! Any change in the file may prevent reading its data."`, `"dot_sign."`, `"coma_sign;"`, plus a `"sys info:[...]"` metadata row, plus a couple of NaN rows. Filter rows where `Time` is a string with **exactly two colons** before parsing — anything else explodes the H:M:S split.

## Pipeline that's already working

`notebooks/01_breathing_dsp_baseline.ipynb` runs end-to-end on the paced 12 BPM file. Result: belt BPM = 12.00, CSI BPM = 12.00, PC1 explains 94% of variance across the strongest subcarriers.

Steps (don't re-derive these — extend them):

1. Parse belt + CSI, drop CSV trailers, decode CSI bytes to (n_packets, 64) amplitude.
2. Mask to the 52 documented data subcarriers.
3. Interpolate each subcarrier onto a uniform 20 Hz grid.
4. Per-subcarrier: scipy `detrend` → 4th-order Butterworth bandpass 0.1–0.5 Hz with `sosfiltfilt` (zero-phase).
5. Rank subcarriers by post-filter variance, keep the top quartile.
6. PCA fuse via SVD; flip PC1 sign if needed.
7. Welch PSD on the fused trace (1-minute segment), find peak in 0.1–0.5 Hz band, multiply by 60 → BPM.

Heart rate would use the same skeleton with band 0.8–2.5 Hz (and probably needs phase, not just amplitude).

## How the user wants to work

- They are guiding scope. Don't unilaterally swap datasets or jump ahead to ML when DSP isn't done. Targets are breathing first, HR nice-to-have, sleep stages stretch — respect that ordering.
- Brief explanations of ML/DSP choices are welcome when introducing a concept (they asked "what is DSP" once); don't over-explain basics once introduced.
- They want the next step suggested at the end of each chunk of work, not implemented automatically.

## Memory location

`/Users/kpj/.claude/projects/-Users-kpj-college-groudation-project/memory/` — `MEMORY.md` indexes the rest. Update when project state changes meaningfully.
