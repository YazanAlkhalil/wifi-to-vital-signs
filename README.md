# Vital Signs Estimation from Wi-Fi CSI

Graduation project: train a model to estimate vital signs from Wi-Fi Channel State Information (CSI), without any contact sensor on the body.

## Goal

From raw CSI streams, predict:

1. **Breathing rate** — primary target. Must work.
2. **Heart rate** — nice to have. Smaller signal in CSI, harder than breathing.
3. **Sleep stages / sleep disturbances** — stretch goal. Classify events such as apnoea, posture changes, leg restlessness, confusional arousals, or coarse sleep stages.

The model should generalize across people and environments (different rooms, different distances between Tx/Rx and the subject).

## Why CSI

CSI captures fine-grained per-subcarrier channel response of a Wi-Fi link. Chest motion from breathing — and, more weakly, the heartbeat — modulates multipath, which shows up as a quasi-periodic component in the CSI amplitude/phase across time. Unlike a chest belt or pulse oximeter, it is fully non-contact and works through clothing and bedding, which makes it attractive for elderly care and long-term sleep monitoring.

## Datasets

We have four datasets in `datasets/`. Most of the work will use the first two below.

### 1. Respiration Rate Measurement — Validity and Repeatability (older adults in care)
`datasets/Respiration Rate Measurement Validity and Repeatability of Ubiquitous Non-contact Wi-Fi Sensing for Older Adults in Care/`

- **Validity/** and **Reliability/** splits.
- Each split contains:
  - `GT Neulog RR Belt Sensor/` — ground-truth respiration rate from a Neulog chest belt.
  - `Wi-Fi Sensor RR/` — Wi-Fi CSI recordings.
- Bundled extraction notebooks: `Belt Data Extraction.ipynb`, `Wi-Fi CSI Extraction.ipynb`.
- Population: older adults in a care setting. Good for the breathing-rate task and for reliability/validity claims.

### 2. Sleep Disturbances Dataset
`datasets/Sleep Disturbances Dataset/`

- **Vital Signs/** — `Breathing - 12 BPM - CSI.csv`, `Breathing - Belt & HR.csv` (CSI + belt + HR ground truth).
- **Sleep Apnoa/** — paired `*-CSI.csv` and `*-Belt.csv` for both CSA (central) and OSA (obstructive) apnoea.
- **Confusional Arousals/**, **Leg Restlessness/**, **Posture Changes/** — labeled disturbance events.
- **scripts/** — author-provided demo scripts (`wi_fi_csi_extraction.py`, `ground_truth_extraction.py`). They only load + plot; we go further.

This is the dataset for the sleep-stages / sleep-disturbances stretch goal, and also gives us a second source of breathing + HR ground truth.

### 3. MultiPatient Elderly Respiration (Digital Twin Technology)
`datasets/MultiPatient Elderly Respiration dataset in Digital Twin Technology/`
- Currently a single `.rar` archive — needs extraction before use.
- Likely a secondary source for cross-subject breathing-rate evaluation.

### 4. Chinese data
`datasets/chinese data/`
- Contents not yet inspected. Held in reserve.

## Approach (planned)

Standard pipeline, refined as we go:

1. **Load & sanity-check** CSI + ground-truth belt/HR for one recording end-to-end before scaling up.
2. **Preprocess CSI**
   - Amplitude (and possibly phase) per subcarrier per antenna pair.
   - Outlier subcarrier removal, denoising (e.g. Hampel, low-pass), detrending.
   - Resample to a fixed rate; align with belt/HR ground truth on a common timeline.
3. **Feature / representation**
   - Classical baseline: bandpass into the breathing band (~0.1–0.5 Hz) and HR band (~0.8–2.5 Hz), pick dominant frequency per subcarrier, fuse across subcarriers (PCA / variance-weighted).
   - Learned: spectrogram / 2D (subcarrier × time) tensors fed to a CNN or 1D-CNN+GRU.
4. **Models**
   - Regression head for breathing rate (BPM) and heart rate (BPM).
   - Classification head for sleep-disturbance events.
5. **Evaluation**
   - Subject-disjoint splits (no leakage of a subject across train/test).
   - Breathing/HR: MAE in BPM, Bland–Altman against the belt.
   - Disturbances: per-class precision/recall, event-level F1.

## Repo layout

```
datasets/    # raw data — large, do not modify
notebooks/   # exploration, preprocessing, training experiments
report/      # written report and figures
README.md
```

## Data format reference

Both primary datasets use the same ESP32 CSI capture format. Worth recording up front because the CSV trailers and byte layout will trip you up.

### CSI CSV (ESP32 / 802.11n HT20)
- One row per packet. The `CSI_DATA` column is a string like `"[110 96 6 0 ... ]"` of **128 signed bytes** = 64 complex subcarriers.
- ESP-IDF byte layout: **interleaved imag/real pairs** — byte 0 = imag(sc0), byte 1 = real(sc0), byte 2 = imag(sc1), …
- Per subcarrier: `amp = sqrt(re² + im²)`, `phase = arctan2(im, re)`.
- **Active data subcarriers**: indices `[6:32] + [33:59]` = **52 subcarriers**. The rest are guards / DC null / pilots and should be dropped before any frequency analysis. (This mask is what the dataset authors document in their bundled scripts; matches the 802.11n HT20 spec.)
- Time axis: use the `real_timestamp` column (Unix seconds). Packet rate is jittery — typically ~100 Hz on the sleep-disturbances dataset and ~40 Hz on the older-adults dataset — so resample onto a uniform grid before any FFT.

### Belt CSV (Neulog logger)
- Sleep-disturbances dataset: 50 Hz. Older-adults dataset: 100 Hz. Don't assume.
- Time column starts with a literal apostrophe (`'0:0:0.04`) — Excel artifact. Two channels: `Arb1` and `Arb2`.
- **In the sleep-disturbances Vital Signs file: `Arb2` is the chest belt (smooth ~5 s breathing waves), `Arb1` is the HR sensor waveform (sharper / squarer pulses).** Confirmed empirically — column ordering is not labeled in the CSV.
- The CSV has trailer rows after the data: `"Attention! Any change in the file may prevent reading its data."`, `"dot_sign."`, `"coma_sign;"`, plus a `"sys info:[...]"` metadata row and a couple of NaN rows. Filter rows where `Time` is a string with exactly two colons before parsing.

## Status

DSP baseline working end-to-end on `Sleep Disturbances Dataset/Vital Signs/`. From the 5-minute paced recording, both the belt and the fused CSI signal recover **12.00 BPM exactly** — pipeline validated.

Notebook: `notebooks/01_breathing_dsp_baseline.ipynb`. Steps: parse → resample CSI to uniform 20 Hz → 0.1–0.5 Hz Butterworth bandpass per subcarrier → keep top-quartile by in-band variance → PCA fuse → Welch PSD → read peak. PC1 captures 94% of variance across the kept subcarriers, which is a strong signal that breathing is the dominant shared component.

**Next step**: slide a 30 s window (1 s hop) across the recording, compute CSI vs belt BPM in each window, and plot the per-window error. That gives a real metric instead of one number, and is the first thing a learned model will need to beat.
