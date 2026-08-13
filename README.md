# Federated EEG Motor Imagery: FedProx vs. FedSplit

Code for the pipeline discussed in our conversation: federated learning on
the EEG motor imagery dataset, one subject = one client, comparing
FedAvg (baseline), FedProx, and a from-scratch FedSplit implementation.


**Data loading is manual** (`data/manual_loader.py`), for locally downloaded
GigaDB `.mat` files -- Verified against a real
uploaded `s1.mat`: 198/200 trials kept after filtering + bad-trial rejection.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate           # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install "flwr[simulation]"     # required for local multi-client simulation
```

Put your downloaded GigaDB files in one folder, named `s1.mat`, `s2.mat`, ...
(GigaDB's own naming convention) -- e.g. `~/eeg_data/s1.mat`.

## 2. Project layout

```
data/
  manual_loader.py   Step-by-step raw .mat parsing, filtering, epoching,
                      bad-trial rejection -- the core of this update
  dataset.py         Per-client splitting, normalization, client
                      list management. load_subject_data_local() is the
                      function that wraps manual_loader.py for the rest
                      of the pipeline.
models/
  eegnet.py      The shared CNN architecture
scripts/
  client.py      Flower client, handles both FedProx and FedSplit local updates
  run_experiment.py   Main entry point, runs one strategy end-to-end
  compare_results.py  Builds comparison table + plots across strategies
strategies/
  fedsplit.py         Custom FedSplit strategy (not built into Flower)
  tracking_fedavg.py  Small wrapper so FedAvg/FedProx expose their final model
results/          JSON results + plots land here
```

## 3. Understanding the raw .mat structure (why manual_loader.py works the way it does)

Confirmed directly against a real subject file:
- `eeg['srate']` = 512 Hz
- `eeg['imagery_left']` / `eeg['imagery_right']`: shape `(68, n_samples)` --
  rows 0-63 are the 64 EEG channels, rows 64-67 are 4 EMG channels (dropped
  for classification here)
- `eeg['imagery_event']` fires `1` at each trial's **cue onset** (t=0);
  100 onsets per class, spaced exactly 3584 samples (7.0s) apart
- `eeg['frame']` = `[-2000, 5000]` (ms): the valid window is 2s before the
  cue to 5s after
- `eeg['bad_trial_indices']` exists in the struct but was **empty** for the
  subject we tested -- don't rely on it always being populated; the loader
  computes its own rejection instead
- **Raw amplitudes are not calibrated microvolts.** Post-bandpass values run
  to +-12,000+, not the +-50uV typical of real EEG. This means the original
  paper's fixed +-100uV bad-trial threshold can't be reused literally --
  `manual_loader.py` uses a robust (MAD-based) per-subject outlier threshold
  instead, which achieves the same goal without assuming a specific
  calibration.

## 4. Running the three experiments

```bash
# Baseline (no non-IID robustness)
python scripts/run_experiment.py --strategy fedavg --rounds 30 --clients 45 --data_dir ~/eeg_data

# Your main method
python scripts/run_experiment.py --strategy fedprox --rounds 30 --clients 45 --mu 1.0 --data_dir ~/eeg_data

# Secondary comparison, reduced scale (Step 5.2 -- FedSplit's inner loop is
# more expensive per round, so we compare it fairly at a smaller client count)
python scripts/run_experiment.py --strategy fedsplit --rounds 30 --clients 15 --data_dir ~/eeg_data
```

Each run saves `results/<strategy>_results.json` with:
- per-round convergence (loss + accuracy + precision + recall + f1_score)
- wall-clock time
- per-unseen-client accuracy (the generalization-to-a-new-subject test)

### Resuming an interrupted FedSplit run

Use the same data and training settings as the original run. `--rounds` is
the total target, so a checkpoint saved after round 28 of a 50-round run
automatically runs rounds 29-50:

```bash
python scripts/run_experiment.py --strategy fedsplit --rounds 50 --clients 45 \
    --fedsplit_mu 1.0 --fedsplit_steps 3 --fraction_fit 1.0 \
    --local_epochs 5 --batch_size 32 --data_dir ~/eeg_data \
    --resume_checkpoint results/fedsplit_checkpoint.npz
```

## 5. Tuning FedProx's mu

Run fedprox with a few different `--mu` values and compare `unseen_mean_accuracy`
in the resulting JSON files:

```bash
for mu in 0.001 0.01 0.1 1.0; do
    python scripts/run_experiment.py --strategy fedprox --rounds 20 --clients 45 --mu $mu --data_dir ~/eeg_data
    mv results/fedprox_results.json results/fedprox_mu${mu}.json
done
```

## 6. Comparing everything

```bash
python scripts/compare_results.py
```

Produces `results/convergence_comparison.png` and `results/unseen_client_spread.png`,
plus a printed summary table (accuracy, wall-clock time, per-client spread) --
ready to drop into your results section.

## 7. Notes on methods

- **Bad-trial rejection** uses a robust per-subject amplitude-outlier
  threshold (median + MAD) instead of the original paper's fixed +-100uV
  rule, because this .mat export's raw units aren't calibrated microvolts.
  State this adaptation explicitly rather than citing the paper's exact
  threshold as your method.
- **FedSplit** is implemented from scratch (Peaceman-Rachford splitting,
  Pathak & Wainwright 2020) since no standard FL library ships it. The
  paper's convergence guarantees are proven for convex objectives; EEGNet is
  non-convex, so report this as an empirical adaptation, not a
  theorem-backed result.
- **Feature extraction**: `data/features.py` is provided if you decide to
  go the hand-crafted-feature route instead of raw-epoch EEGNet -- but the
  default pipeline (`run_experiment.py`) uses raw filtered epochs directly,
  which avoids the federated-feature-selection problem entirely.
- **Client resource limits**: `client_resources={"num_cpus": 1, "num_gpus": 0}`
  in `run_experiment.py` runs everything on CPU by default for portability.
  If you have a GPU, increase `num_gpus` and edit `client.py`'s `device`
  handling for a large speed-up with ~50 simulated clients.
- You may see a Flower deprecation warning about `client_fn(cid)` vs.
  `client_fn(context)` signatures -- this is harmless with the pinned
  Flower version in requirements.txt; a future Flower major version may
  require updating to the `Context`-based signature.

## 8. Validated

- `data/manual_loader.py` was run against a **real uploaded s1.mat** and
  correctly parsed 198/200 trials (64 EEG channels, 1024 samples/trial at
  0.5-2.5s post-cue, 512 Hz).
- The full chain (manual loader -> EEGNet -> Flower client -> FedProx
  aggregation) was run end-to-end on that same real data and completed
  3 simulation rounds without errors.
- FedSplit's custom strategy was separately validated on synthetic data
  shaped like Cho2017 epochs (real-data FedSplit run needs multiple real
  subjects, which requires your full local download).
