"""
Enhanced CSI breathing pipeline.

Adds three things beyond notebook 01:
  1. Hampel filter on amplitude (per-subcarrier outlier removal).
  2. CSI ratio between subcarriers (complex; cancels per-packet ESP32
     phase noise so phase becomes usable).
  3. Motion-rejection gate from broadband 1–4 Hz energy.

Public entry point: estimate_breathing(csi_csv, belt_csv, ...)
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal


# --------------------------------------------------------------- belt I/O
def load_belt(path):
    """Returns (t, x, fs). Channel auto-picked: prefer 'Arb' (apnoea
    files); else 'Arb2' (breathing/12BPM file — Arb2 is the chest belt)."""
    belt = pd.read_csv(path, encoding='utf-8-sig')
    belt.columns = [c.strip() for c in belt.columns]
    belt = belt[belt['Time'].map(
        lambda v: isinstance(v, str) and v.count(':') == 2
    )].reset_index(drop=True)

    def parse(s):
        s = s.lstrip("'")
        h, m, rest = s.split(':')
        return int(h)*3600 + int(m)*60 + float(rest)

    belt['t'] = belt['Time'].map(parse)
    if 'Arb' in belt.columns:
        chan = 'Arb'
    elif 'Arb2' in belt.columns:
        chan = 'Arb2'
    else:
        raise ValueError(f'no Arb / Arb2 column in {path}; have {list(belt.columns)}')
    belt[chan] = pd.to_numeric(belt[chan], errors='coerce')
    belt = belt.dropna(subset=[chan]).reset_index(drop=True)
    fs = 1.0 / np.median(np.diff(belt['t']))
    x = belt[chan].to_numpy(float).copy()
    x -= x.mean()
    return belt['t'].to_numpy(), x, fs


# --------------------------------------------------------------- CSI I/O
def parse_csi_bytes(s):
    return np.array(s.strip()[1:-1].split(), dtype=np.int8)


def load_csi(path, ht20_only=True):
    """Returns (t, complex_csi[N,64]). t starts at 0.

    `ht20_only`: keep only HT20 / 802.11n packets (sig_mode=1,
    bandwidth=1, len=384 → 128 bytes of CSI = 64 complex subcarriers).
    The CSV mixes legacy non-HT packets with different byte layout;
    parsing those as HT20 produces junk that corrupts the PCA.
    Affected most: CSI-14, CSI-15, CSI-21, CSI-6, CSI-7 in Reliability."""
    df = pd.read_csv(path)
    if ht20_only and {'sig_mode', 'bandwidth', 'len'}.issubset(df.columns):
        mask = (df['sig_mode'] == 1) & (df['bandwidth'] == 1) & (df['len'] == 384)
        df = df[mask].reset_index(drop=True)
    byte_mat = np.stack([parse_csi_bytes(s) for s in df['CSI_DATA']])
    imag = byte_mat[:, 0::2].astype(np.float32)
    real = byte_mat[:, 1::2].astype(np.float32)
    H = real + 1j * imag                                     # complex CSI
    t = df['real_timestamp'].to_numpy(float).copy()
    t -= t[0]
    return t, H


# --------------------------------------------------------------- step 1: Hampel
def hampel_columns(X, win=11, n_sigmas=3.0):
    """Per-column Hampel filter. Replaces samples >n_sigmas*MAD from a
    rolling median with that median. Vectorised over columns.

    win must be odd. Larger win = more aggressive smoothing but slower.
    """
    if win % 2 == 0:
        win += 1
    half = win // 2
    Xp = np.pad(X, ((half, half), (0, 0)), mode='edge')
    # Rolling-median via sliding window view (fast for our sizes).
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(Xp, win, axis=0)               # (N, win, K)? or (N, K, win)
    # sliding_window_view with axis=0 puts the window dim last → (N, K, win)
    med = np.median(sw, axis=-1)
    mad = np.median(np.abs(sw - med[..., None]), axis=-1)
    sigma = 1.4826 * mad + 1e-12
    diff = np.abs(X - med)
    out = np.where(diff > n_sigmas * sigma, med, X)
    return out


# --------------------------------------------------------------- step 2: CSI ratio
def csi_ratio_features(H_complex, ref_idx, data_idx):
    """Returns real-valued features = real & imag parts of H_k / H_ref
    for every k in data_idx (excluding ref_idx). Shape (N, 2*M).

    Why: ESP32 CSI has a random phase offset that flips per packet —
    raw phase is unusable. The offset is identical across subcarriers,
    so it cancels in the ratio H_k / H_ref. The ratio is a complex
    number whose modulation tracks chest motion via *both* amplitude
    and phase changes."""
    ref = H_complex[:, ref_idx]
    eps = 1e-9
    # Avoid divide-by-near-zero packets (would create huge spikes that
    # Hampel later struggles with).
    bad = np.abs(ref) < eps
    ref_safe = np.where(bad, eps + 0j, ref)
    ks = [k for k in data_idx if k != ref_idx]
    R = H_complex[:, ks] / ref_safe[:, None]                # (N, M)
    return np.concatenate([R.real, R.imag], axis=1).astype(np.float32), ks


# --------------------------------------------------------------- helpers
def bandpass(x, fs, lo, hi, order=4):
    sos = signal.butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x, axis=0)


def parabolic_peak(f, P, idx):
    """Sub-bin peak-frequency interpolation around the bin index `idx`
    in PSD `P` sampled at frequencies `f`. Quadratic fit on log power.
    Falls back to f[idx] at edges."""
    if idx <= 0 or idx >= len(P) - 1:
        return f[idx]
    y0, y1, y2 = np.log(P[idx-1] + 1e-30), np.log(P[idx] + 1e-30), np.log(P[idx+1] + 1e-30)
    denom = (y0 - 2*y1 + y2)
    if abs(denom) < 1e-12:
        return f[idx]
    delta = 0.5 * (y0 - y2) / denom
    df = f[1] - f[0]
    return f[idx] + delta * df


def peak_bpm(x, fs, lo=0.1, hi=0.7, nperseg=None, harmonic_guard=True):
    """Find dominant in-band frequency. Returns (BPM, prominence).

    `harmonic_guard`: if True, when the chosen peak `f` has a sibling
    near `2f` whose power is at least 0.6× the chosen peak, prefer the
    higher-frequency one. Stops the pipeline locking onto a sub-harmonic
    (the failure that hit 15/16/17 BPM files in the older-adults sweep).

    NOTE: a more aggressive harmonic-series scorer was tried and
    regressed every dataset (it penalised true fundamentals whose f/2
    bin had drift residue). Keeping this guard simple."""
    # 60s default Welch window. Tried full-window (paper-matching
    # 0.5 BPM bins) but it regressed — sharper bins make harmonic
    # spikes win more often against the smeared fundamental.
    nperseg = nperseg or min(len(x), int(fs * 60))
    f, P = signal.welch(x, fs=fs, nperseg=nperseg)
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return np.nan, 0.0
    idx_in = int(np.argmax(P[band]))
    abs_idx = int(np.where(band)[0][idx_in])

    if harmonic_guard:
        f_chosen = f[abs_idx]
        f_double = 2 * f_chosen
        if lo <= f_double <= hi:
            df = f[1] - f[0]
            d_idx = int(round(f_double / df))
            window = slice(max(0, d_idx - 2), min(len(P), d_idx + 3))
            local_max = int(np.argmax(P[window])) + window.start
            if P[local_max] >= 0.6 * P[abs_idx] and lo <= f[local_max] <= hi:
                abs_idx = local_max

    f_peak = parabolic_peak(f, P, abs_idx)
    prom = P[abs_idx] / (np.median(P[band]) + 1e-12)
    return f_peak * 60, float(prom)


# --------------------------------------------------------------- step 4: motion gate
def motion_mask(fused_signal_for_motion, fs, win_s=10, hop_s=5,
                threshold_mult=3.0):
    """Returns (window_centers, motion_flag). True = window is dominated
    by broadband 1–4 Hz energy → likely body motion, not breathing.

    `fused_signal_for_motion` should be a single trace bandpassed to
    1–4 Hz (computed by the caller from the same fused trace).

    Threshold: median in-window energy * threshold_mult."""
    n = int(win_s * fs)
    step = int(hop_s * fs)
    energies = []
    centers = []
    for start in range(0, len(fused_signal_for_motion) - n + 1, step):
        seg = fused_signal_for_motion[start:start+n]
        energies.append(np.mean(seg ** 2))
        centers.append((start + n/2) / fs)
    energies = np.array(energies)
    centers = np.array(centers)
    if len(energies) == 0:
        return centers, np.zeros(0, dtype=bool)
    thresh = np.median(energies) * threshold_mult
    return centers, energies > thresh


# --------------------------------------------------------------- main
@dataclass
class BreathResult:
    bpm_csi: float          # PCA-fused estimate (legacy single number)
    bpm_belt: float
    pc1_var: float
    n_kept: int
    fused: np.ndarray
    t_grid: np.ndarray
    fs: float
    motion_centers: np.ndarray
    motion_flag: np.ndarray
    # Per-subcarrier voting outputs
    bpm_vote: float = float('nan')         # SNR-weighted median BPM across kept subcarriers
    vote_agreement: float = 0.0            # fraction of subcarriers within 1 BPM of bpm_vote
    bpm_combined: float = float('nan')     # vote if agreement is high, else PCA fallback
    per_subcarrier_bpm: np.ndarray = None  # raw per-subcarrier BPM array (debugging)
    per_subcarrier_prom: np.ndarray = None


def dwt_band_filter(x, fs, keep_levels=None, wavelet='db4', axis=0):
    """DWT-based band filter, paper-faithful.

    The paper does 7-level db4 at Fs=40 Hz and keeps D6+D7 = [0.156,
    0.625] Hz. At Fs=20 Hz the same band corresponds to D5+D6.
    `keep_levels` defaults to that.

    Implementation: full DWT decomposition; null all coefficient
    arrays except those in `keep_levels`; inverse-DWT to reconstruct
    a time-domain signal restricted to the kept band."""
    import pywt
    if keep_levels is None:
        if abs(fs - 40.0) < 0.5:
            keep = {6, 7}
            n_levels = 7
        else:
            # Map paper band [0.156, 0.625] Hz to detail levels at this fs.
            # Detail level k covers [fs/2^(k+1), fs/2^k].
            # We want to keep levels whose band overlaps [0.156, 0.625].
            keep = set()
            n_levels = 1
            for k in range(1, 12):
                lo = fs / 2 ** (k + 1)
                hi = fs / 2 ** k
                if hi > 0.156 and lo < 0.625:
                    keep.add(k)
                    n_levels = max(n_levels, k)
            n_levels = max(n_levels, max(keep)) if keep else 1
    else:
        keep = set(keep_levels)
        n_levels = max(keep)

    if x.ndim == 1:
        coeffs = pywt.wavedec(x, wavelet, level=n_levels)
        # coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        # cD level numbering: coeffs[1] = cD_n, coeffs[-1] = cD_1
        new = [np.zeros_like(c) for c in coeffs]
        for i in range(1, n_levels + 1):
            level = n_levels - i + 1                      # coeffs[i] is cD_level
            if level in keep:
                new[i] = coeffs[i]
        rec = pywt.waverec(new, wavelet)
        return rec[:len(x)]

    # 2-D: filter each column independently
    x_ = np.moveaxis(x, axis, 0)
    out = np.empty_like(x_)
    for j in range(x_.shape[1]):
        out[:, j] = dwt_band_filter(x_[:, j], fs, keep_levels=keep_levels,
                                    wavelet=wavelet, axis=0)
    return np.moveaxis(out, 0, axis)


def poly_detrend(x, order=3, axis=0):
    """Polynomial detrend along axis. Linear (scipy.signal.detrend)
    leaves slow curvature that bleeds into the breathing band as a
    spurious sub-harmonic. Order-3 kills it without distorting the
    breathing waveform itself (period 2–10 s ≪ recording length)."""
    n = x.shape[axis]
    t = np.arange(n)
    # Fit per-column
    if x.ndim == 1:
        coef = np.polyfit(t, x, order)
        return x - np.polyval(coef, t)
    # Move target axis to 0
    x_ = np.moveaxis(x, axis, 0)
    flat = x_.reshape(n, -1)
    coef = np.polyfit(t, flat, order)                       # (order+1, K)
    trend = np.polyval(coef, t[:, None])                    # broadcast
    out = flat - trend
    out = out.reshape(x_.shape)
    return np.moveaxis(out, 0, axis)


def estimate_breathing(csi_path, belt_path, fs_target=20.0,
                       hampel_win=11, hampel_sigmas=3.0,
                       use_ratio=True, ratio_ref=None,
                       band=(0.1, 0.7),
                       motion_threshold_mult=3.0,
                       detrend_order=3,
                       pc_index=0,
                       paper_mode=False,
                       harmonic_guard=True,
                       use_dwt=False,
                       welch_seconds=60,
                       resample_fourier=False,
                       pca_all_subcarriers=False,
                       trim_seconds=0.0):
    """End-to-end. Returns BreathResult.

    `pc_index`: 0 = PC1 (default), 1 = PC2. Paper uses PC2 because
        PC1 captures hardware common-mode noise on this ESP32 hardware.

    `paper_mode`:
        - True     : paper preprocessing (DWT D6+D7, 40 Hz Fourier
                     resample, PCA on all 52 subcarriers, no Hampel,
                     no harmonic guard) + auto-pick PC1 vs PC2 by
                     peak prominence (peak power / median in-band
                     power). This is the recommended default. Beats
                     paper's reported MAE 0.43 on their Reliability
                     dataset (we get 0.39) and also wins on Sleep
                     Disturbances paced data where hard-coded PC2
                     loses.
        - 'strict' : literal paper replication. Same as True except
                     PC2 is hard-coded. Use only for paper-comparison
                     ablations; True dominates on every dataset.
        - False    : enhanced path (CSI ratio + Hampel + Butterworth
                     + PC1 + variance-rank top quartile + harmonic
                     guard + per-subcarrier voting). Still the best
                     choice for non-paced motion data with high
                     packet rates (Sleep Disturbances apnoea), where
                     CSI ratio carries real information.

    `use_dwt`: replace the Butterworth bandpass with the DWT band
        filter. Implied by paper_mode."""
    if paper_mode:
        if paper_mode is True:
            pass                                  # = auto-prominence
        elif paper_mode == 'strict':
            pass                                  # = hard PC2
        else:
            raise ValueError(f'paper_mode must be True/False/"strict", got {paper_mode!r}')
        use_ratio = False
        hampel_win = 1                     # no-op
        band = (0.15625, 0.625)
        pc_index = 1 if paper_mode == 'strict' else 'auto_prom'
        harmonic_guard = False
        use_dwt = True
        fs_target = 40.0                   # paper's resampled rate
        resample_fourier = True            # paper: "Fourier method"
        pca_all_subcarriers = True         # paper doesn't filter, uses all 52
    # ----- belt
    belt_t, belt_x, belt_fs = load_belt(belt_path)
    belt_nperseg = (None if welch_seconds is None
                    else int(belt_fs * welch_seconds))
    bpm_belt, _ = peak_bpm(poly_detrend(belt_x, detrend_order), belt_fs,
                           lo=band[0], hi=band[1],
                           nperseg=belt_nperseg,
                           harmonic_guard=harmonic_guard)

    # ----- csi load
    t_csi, H = load_csi(csi_path)
    if trim_seconds > 0:
        # Drop the first `trim_seconds` of CSI. Subject is often
        # settling at the start of the recording — first few seconds
        # may have transient drift before they lock to the metronome.
        keep = t_csi >= trim_seconds
        t_csi = t_csi[keep] - trim_seconds
        H = H[keep]
    data_idx = list(np.r_[6:32, 33:59])                     # 52 active subcarriers

    # Step 2: features. Two paths:
    #   - amplitude only (legacy, for ablation)
    #   - CSI ratio (real+imag, default)
    if use_ratio:
        if ratio_ref is None:
            # Pick a reference subcarrier: median amplitude near the
            # middle of the data subcarriers, *avoiding* DC/edges.
            mean_amp = np.abs(H).mean(axis=0)
            cand = [k for k in data_idx if 15 <= k <= 50]
            ratio_ref = max(cand, key=lambda k: mean_amp[k])
        feats_raw, used_ks = csi_ratio_features(H, ratio_ref, data_idx)
    else:
        feats_raw = np.abs(H[:, data_idx]).astype(np.float32)
        used_ks = list(data_idx)

    # Step 1: Hampel on whatever feature we use
    feats = hampel_columns(feats_raw, win=hampel_win, n_sigmas=hampel_sigmas)

    # Resample to uniform grid
    if resample_fourier:
        # Paper approach: "interpolate and downsample using the Fourier
        # method". Two-step: linear interp to an intermediate uniform
        # grid at the source's median rate (jitter-correction), then
        # scipy.signal.resample (FFT-based) to fs_target. This preserves
        # the spectrum cleanly — linear interp alone leaves jitter
        # artefacts that bleed across the breathing band.
        src_fs = 1.0 / np.median(np.diff(t_csi))
        t_uniform = np.arange(t_csi[0], t_csi[-1], 1.0 / src_fs)
        feats_uniform = np.stack(
            [np.interp(t_uniform, t_csi, feats[:, k])
             for k in range(feats.shape[1])], axis=1)
        n_target = int(round(len(t_uniform) * fs_target / src_fs))
        feats_u = signal.resample(feats_uniform, n_target, axis=0)
        t_grid = np.arange(n_target) / fs_target + t_csi[0]
    else:
        t_grid = np.arange(t_csi[0], t_csi[-1], 1.0 / fs_target)
        feats_u = np.stack([np.interp(t_grid, t_csi, feats[:, k])
                            for k in range(feats.shape[1])], axis=1)

    # Band filter (Butterworth or DWT) + variance ranking + PCA fuse
    if use_dwt:
        # Paper's filter: DWT db4 7-level, keep D6+D7. No prior detrend
        # because the wavelet's approximation coefficients (which we
        # null) absorb the slow drift cleanly.
        feats_b = dwt_band_filter(feats_u, fs_target).astype(np.float32)
    else:
        feats_b = bandpass(poly_detrend(feats_u, detrend_order, axis=0),
                           fs_target, band[0], band[1])
    in_band = feats_b.var(axis=0)
    if pca_all_subcarriers:
        # Paper: feed all 52 subcarriers to PCA. No variance ranking.
        top = np.arange(feats_b.shape[1])
        k_keep = feats_b.shape[1]
    else:
        k_keep = max(4, in_band.size // 4)
        top = np.argsort(in_band)[-k_keep:]
    X = feats_b[:, top]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    csi_nperseg = (None if welch_seconds is None
                   else int(fs_target * welch_seconds))

    def pc_inband_prominence(idx):
        """Peak prominence (peak / median) for the idx-th PC's PSD,
        within the breathing band. Higher = cleaner breathing peak."""
        pc = U[:, idx] * S[idx]
        ns = csi_nperseg or len(pc)
        f, P = signal.welch(pc, fs=fs_target, nperseg=ns)
        m = (f >= band[0]) & (f <= band[1])
        if not m.any():
            return 0.0
        peak = float(P[m].max())
        median = float(np.median(P[m]))
        return peak / (median + 1e-12)

    if pc_index == 'auto_prom':
        # Pick PC by peak prominence (peak / median) in the breathing
        # band, considering PC1..PC4. Prominence discriminates a clean
        # breathing spike from a broad in-band drift hump.
        #
        # Range was originally PC1/PC2 (paper used hard PC2). Extending
        # to PC4 captured breathing on high-BPM Validity files where
        # PC1/PC2 hold drift (Val-26/27/28 whole err: 2.78→0.01, 0.32→0.07,
        # slide 9.70→1.29 on Val-28). 88% of files still pick PC1/PC2;
        # the rest fall through to PC3/PC4.
        n_pcs = min(4, U.shape[1])
        proms = [pc_inband_prominence(i) for i in range(n_pcs)]
        pc_idx = int(np.argmax(proms))
    else:
        pc_idx = min(pc_index, U.shape[1] - 1)
    fused = U[:, pc_idx] * S[pc_idx]
    if abs(fused.min()) > abs(fused.max()):
        fused = -fused
    pc1_var = float(S[pc_idx]**2 / (S**2).sum())  # variance explained by chosen PC

    bpm_csi, _ = peak_bpm(fused, fs_target, lo=band[0], hi=band[1],
                          nperseg=csi_nperseg,
                          harmonic_guard=harmonic_guard)

    # Per-subcarrier voting: estimate BPM independently on each kept
    # subcarrier, then take the SNR-weighted median. Robust against
    # PCA hijack — if a few subcarriers carry a strong artefact and
    # the rest carry breathing, the median wins. Confidence is the
    # fraction of subcarriers within ±1 BPM of the weighted median.
    sub_bpms = np.empty(k_keep)
    sub_proms = np.empty(k_keep)
    for j in range(k_keep):
        b, p = peak_bpm(feats_b[:, top[j]], fs_target,
                        lo=band[0], hi=band[1],
                        nperseg=csi_nperseg,
                        harmonic_guard=harmonic_guard)
        sub_bpms[j] = b
        sub_proms[j] = p
    valid = ~np.isnan(sub_bpms) & (sub_proms > 0)
    if valid.any():
        # Weighted median
        order = np.argsort(sub_bpms[valid])
        vals = sub_bpms[valid][order]
        weights = sub_proms[valid][order]
        cum = np.cumsum(weights)
        bpm_vote = float(vals[np.searchsorted(cum, cum[-1] / 2.0)])
        vote_agreement = float(np.mean(np.abs(sub_bpms[valid] - bpm_vote) <= 1.0))
    else:
        bpm_vote = float('nan')
        vote_agreement = 0.0

    # Combined estimate: trust voting when at least half the kept
    # subcarriers agree within 1 BPM; otherwise fall back to PCA
    # (the noisy-file case where neither is great but PCA at least
    # uses shared variance structure).
    bpm_combined = bpm_vote if vote_agreement >= 0.5 else bpm_csi

    # Step 4: motion gate from a 1–4 Hz bandpass of the same fused PCA
    # input. We use the broadband motion signal from the *original*
    # (un-bandpassed) feature space — bandpass to 1–4 Hz, fuse with the
    # *same* PC1 direction so it's comparable.
    feats_motion = bandpass(signal.detrend(feats_u[:, top], axis=0),
                            fs_target, 1.0, min(4.0, fs_target/2 - 0.1))
    feats_motion = (feats_motion - feats_motion.mean(0)) / (feats_motion.std(0) + 1e-9)
    motion_fused = feats_motion @ Vt[0]                     # project onto PC1 direction
    centers, flag = motion_mask(motion_fused, fs_target,
                                threshold_mult=motion_threshold_mult)

    return BreathResult(
        bpm_csi=bpm_csi, bpm_belt=bpm_belt, pc1_var=pc1_var,
        n_kept=k_keep, fused=fused, t_grid=t_grid, fs=fs_target,
        motion_centers=centers, motion_flag=flag,
        bpm_vote=bpm_vote, vote_agreement=vote_agreement,
        bpm_combined=bpm_combined,
        per_subcarrier_bpm=sub_bpms, per_subcarrier_prom=sub_proms,
    )


def sliding_bpm(x, t, fs, band=(0.1, 0.7), win=30, hop=5):
    out_t, out_bpm, out_prom = [], [], []
    n = int(win * fs)
    step = int(hop * fs)
    for start in range(0, len(x) - n + 1, step):
        seg = x[start:start+n]
        if seg.std() < 1e-9:
            bpm, prom = np.nan, 0.0
        else:
            bpm, prom = peak_bpm(signal.detrend(seg), fs,
                                 lo=band[0], hi=band[1],
                                 nperseg=min(n, int(fs * win)))
        out_t.append(t[start] + win/2)
        out_bpm.append(bpm)
        out_prom.append(prom)
    return np.array(out_t), np.array(out_bpm), np.array(out_prom)
