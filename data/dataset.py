"""
Data loading & preprocessing for the EEG Motor Imagery (MI) dataset.

Loading backend is available:
  - load_subject_data_local() : manual .mat loader, for locally downloaded
                                 files (see data/manual_loader.py). Use this
                                 one if you already have s1.mat, s2.mat, ...
"""

import os
import numpy as np
from data.manual_loader import load_subject_local

BAD_SUBJECTS = [29, 34]

BANDPASS = (8.0, 30.0)   # mu + beta band
TMIN, TMAX = 0.5, 2.5    # seconds post-cue


def get_all_subjects():
    """Return the list of usable subject IDs (1-52, minus bad subjects)."""
    all_subjects = [s for s in range(1, 53) if s not in BAD_SUBJECTS]
    return all_subjects


def load_subject_data_local(subject_id, data_dir):
    """
    Local-file counterpart to load_subject_data() below. Expects files
    named like s1.mat, s2.mat, ... in `data_dir` (GigaDB's standard naming
    for this dataset). This is what a real federated client would run --
    entirely on its own machine, touching only its own subject's file.
    """
    mat_path = os.path.join(data_dir, f"s{subject_id}.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(
            f"Expected {mat_path} -- check that your local data directory "
            f"uses GigaDB's 's<N>.mat' naming, or adjust the path pattern."
        )
    X, y = load_subject_local(mat_path, verbose=False)
    return X, y


def train_test_split_per_client(X, y, test_frac=0.2, seed=42):
    """
    Local 80/20 split WITHIN one client's own data.
    This is local train/local test.
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def normalize_per_client(X_train, X_test):
    """
    Z-score normalization computed from THIS client's own training data only.
    Never normalize using statistics pooled across clients -- that would leak
    information across the federation boundary and defeats the point of FL.
    """
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm


def make_client_split(unseen_fraction=0.1, seed=42):
    """
    Decide which subjects are "training clients" (participate in
    federated rounds) vs. "unseen clients" (held out entirely, used only to
    test generalization to a brand-new client at the end).
    """
    subjects = get_all_subjects()
    rng = np.random.RandomState(seed)
    subjects = [int(s) for s in rng.permutation(subjects)]

    n_unseen = max(1, int(len(subjects) * unseen_fraction))
    unseen_clients = sorted(subjects[:n_unseen])
    training_clients = sorted(subjects[n_unseen:])

    return training_clients, unseen_clients


if __name__ == "__main__":
    # quick smoke test on one subject
    train_clients, unseen_clients = make_client_split()
    print(f"Training clients ({len(train_clients)}): {train_clients}")
    print(f"Unseen/generalization clients ({len(unseen_clients)}): {unseen_clients}")
