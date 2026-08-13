"""
Manual loader for the raw .mat files, for when you
already have s1.mat, s2.mat, ... downloaded locally.
"""

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt

N_EEG_CHANNELS = 64          # rows 0-63; rows 64-67 are EMG, dropped here
SRATE_EXPECTED = 512
BANDPASS = (8.0, 30.0)       # mu + beta band
CUE_TO_EPOCH_START = 0.5     # seconds after cue
CUE_TO_EPOCH_END = 2.5       # seconds after cue


def load_raw_struct(mat_path):
    """
    Load the .mat file and unwrap MATLAB's nested struct format.
    """
    d = sio.loadmat(mat_path)
    eeg = d['eeg'][0, 0]

    srate = int(eeg['srate'][0, 0])
    if srate != SRATE_EXPECTED:
        print(f"[warn] {mat_path}: srate={srate}, expected {SRATE_EXPECTED}")

    return {
        "srate": srate,
        "frame_ms": eeg['frame'][0],                 # e.g. [-2000, 5000]
        "imagery_left": eeg['imagery_left'],           # (68, n_samples)
        "imagery_right": eeg['imagery_right'],          # (68, n_samples)
        "imagery_event": eeg['imagery_event'][0],        # (n_samples,)
        "subject": str(eeg['subject'][0]),
    }


def bandpass_filter(data, srate, band=BANDPASS, order=4):
    """
    Zero-phase Butterworth band-pass, 8-30 Hz (mu + beta).
    """
    nyq = srate / 2.0
    b, a = butter(order, [band[0] / nyq, band[1] / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def epoch_trials(filtered_data, event_vec, srate,
                  tmin=CUE_TO_EPOCH_START, tmax=CUE_TO_EPOCH_END):
    """
    Cut the continuous filtered signal into per-trial epochs
    around each cue onset.
    """
    onsets = np.where(event_vec == 1)[0]
    start_offset = int(tmin * srate)
    end_offset = int(tmax * srate)
    epoch_len = end_offset - start_offset

    epochs = np.zeros((len(onsets), filtered_data.shape[0], epoch_len),
                       dtype=np.float32)
    for i, onset in enumerate(onsets):
        s, e = onset + start_offset, onset + end_offset
        if s < 0 or e > filtered_data.shape[1]:
            epochs[i] = np.nan  # trial falls outside recorded buffer -- flagged below
            continue
        epochs[i] = filtered_data[:, s:e]

    return epochs


def reject_bad_trials(epochs, z_thresh=5.0):
    """
    Robust, unit-agnostic bad-trial rejection.
    """
    peak_amp = np.nanmax(np.abs(epochs), axis=(1, 2))  # per trial
    median = np.nanmedian(peak_amp)
    mad = np.nanmedian(np.abs(peak_amp - median)) + 1e-9
    robust_z = 0.6745 * (peak_amp - median) / mad

    nan_mask = np.isnan(peak_amp)
    bad_mask = (robust_z > z_thresh) | nan_mask
    good_idx = np.where(~bad_mask)[0]
    return good_idx, bad_mask


def load_subject_local(mat_path, verbose=True):
    """
    Full pipeline for ONE subject's .mat file: this is the drop-in
    replacement for the MOABB-based `load_subject_data()` from before --
    same output contract (X, y), so nothing downstream (client.py,
    run_experiment.py, eegnet.py) needs to change.

    Returns
    -------
    X : np.ndarray, shape (n_good_trials, 64, n_times)
        EEG-only (EMG channels dropped), filtered, epoched, bad trials removed
    y : np.ndarray, shape (n_good_trials,)
        0 = left hand MI, 1 = right hand MI
    """
    raw = load_raw_struct(mat_path)
    srate = raw["srate"]

    all_X, all_y = [], []
    for class_label, key in [(0, "imagery_left"), (1, "imagery_right")]:
        eeg_data = raw[key][:N_EEG_CHANNELS]

        filtered = bandpass_filter(eeg_data, srate)
        epochs = epoch_trials(filtered, raw["imagery_event"], srate)
        good_idx, bad_mask = reject_bad_trials(epochs)

        if verbose:
            print(f"  [{raw['subject']}/{key}] {len(epochs)} trials -> "
                  f"{len(good_idx)} kept, {bad_mask.sum()} rejected")

        all_X.append(epochs[good_idx])
        all_y.append(np.full(len(good_idx), class_label, dtype=np.int64))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y


if __name__ == "__main__":
    import sys
    # Need to test throughout the 52 subjects
    # The subject 1 is loaded and processed to check the validation of the code.
    path = sys.argv[1] if len(sys.argv) > 1 else "~/eeg_data/s1.mat"
    print(f"Loading {path} ...")
    X, y = load_subject_local(path)
    print(f"\nFinal shapes: X={X.shape}, y={y.shape}")
    print(f"Class balance: left={np.sum(y==0)}, right={np.sum(y==1)}")
    print(f"Value range after filtering: min={X.min():.2f} max={X.max():.2f} "
          f"mean={X.mean():.4f} std={X.std():.2f}")
