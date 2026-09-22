"""
Convert Track2p longitudinal calcium imaging data for neural decoder training.

Decisions and justifications:
- Neural data: Compute dF/F from raw fluorescence (F.npy) and neuropil (Fneu.npy) using
  Suite2p default parameters as described in the paper: neucoeff=0.7, maximin baseline,
  sig_baseline=10, win_baseline=60s at 30 Hz.
- Temporal binning: Average both neural and behavioral data in bins of 10 consecutive frames,
  as described in the paper's decoding methods section. Effective rate: 3 Hz.
- Motion energy: When camera frames are missing (ME length < neural frames), interpolate ME
  to match neural frame count before binning, as suggested by data README.
- Trials: Split each session into 60-second non-overlapping trials. At 3 Hz after binning,
  each trial has 180 time bins. Incomplete final trials are discarded.
- Output: Motion energy discretized into 5 equal-percentile (quintile) bins per session.
- Input: Time elapsed from beginning of session in seconds (time-varying).
- No additional neuron filtering: Data is already from Track2p (only neurons tracked across
  all days for each mouse). iscell is all 1s.
- Brain region: barrel cortex (S1BF) for all neurons.
"""

import os
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d

DATA_DIR = '/app/data'
OUTPUT_PATH = '/app/converted_data.pkl'

FS = 30.0           # imaging rate Hz
BIN_SIZE = 10       # frames to average (paper: "bins of 10 consecutive timestamps")
TRIAL_DUR = 60      # trial duration in seconds
NEUCOEFF = 0.7      # Suite2p default neuropil coefficient
SIG_BASELINE = 10.0 # Suite2p default baseline sigma (frames)
WIN_BASELINE = 60.0 # Suite2p default baseline window (seconds)
N_BINS_OUTPUT = 5   # number of percentile bins for motion energy


def compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF,
                sig_baseline=SIG_BASELINE, win_baseline=WIN_BASELINE):
    """Compute dF/F using Suite2p default parameters (maximin baseline)."""
    # Neuropil subtraction
    Fc = F - neucoeff * Fneu

    # Maximin baseline estimation
    win = int(win_baseline * fs)
    Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)

    # Baseline-subtracted fluorescence (Suite2p's preprocess returns Fc - Flow,
    # consistent with track2p code and paper's "baseline corrected fluorescence")
    dff = Fc - Flow
    return dff


def bin_data(data, bin_size, axis=-1):
    """Average data in non-overlapping bins along specified axis."""
    n = data.shape[axis]
    n_bins = n // bin_size
    trimmed = n_bins * bin_size
    if axis == -1 or axis == 1:
        return data[:, :trimmed].reshape(data.shape[0], n_bins, bin_size).mean(axis=2)
    else:  # axis == 0
        return data[:trimmed].reshape(n_bins, bin_size).mean(axis=1)


def interpolate_me(me, target_len):
    """Interpolate motion energy to match neural frame count."""
    if len(me) == target_len:
        return me
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, target_len)
    return np.interp(x_target, x_orig, me)


def main():
    subjects = sorted([d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])

    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx = []

    bins_per_trial = int(TRIAL_DUR * FS / BIN_SIZE)  # 180

    for si, subject in enumerate(subjects):
        subject_dir = os.path.join(DATA_DIR, subject)
        sessions = sorted([d for d in os.listdir(subject_dir)
                          if os.path.isdir(os.path.join(subject_dir, d))])

        for sess in sessions:
            sess_dir = os.path.join(subject_dir, sess)
            s2p_dir = os.path.join(sess_dir, 'suite2p', 'plane0')
            move_dir = os.path.join(sess_dir, 'move_deve')

            # Load neural data
            F = np.load(os.path.join(s2p_dir, 'F.npy'))
            Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
            n_neurons, n_frames = F.shape

            # Compute dF/F
            dff = compute_dff(F, Fneu)

            # Load and align motion energy
            me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
            me = interpolate_me(me, n_frames)

            # Bin both streams
            dff_binned = bin_data(dff, BIN_SIZE, axis=1)  # (n_neurons, n_bins)
            me_binned = bin_data(me.reshape(1, -1), BIN_SIZE, axis=1).squeeze()  # (n_bins,)

            n_bins_total = dff_binned.shape[1]
            n_trials = n_bins_total // bins_per_trial

            if n_trials < 2:
                print(f"Skipping {subject}/{sess}: only {n_trials} trials")
                continue

            # Discretize motion energy into 5 equal-percentile bins for this session
            percentiles = np.linspace(0, 100, N_BINS_OUTPUT + 1)
            bin_edges = np.percentile(me_binned, percentiles)
            # Use digitize; clip to valid range [0, N_BINS_OUTPUT-1]
            me_discrete = np.digitize(me_binned, bin_edges[1:-1])  # values 0..N_BINS_OUTPUT-1

            # Time in seconds from session start (after binning)
            time_bin_dur = BIN_SIZE / FS  # seconds per bin
            time_vec = np.arange(n_bins_total) * time_bin_dur

            # Split into trials
            session_neural = []
            session_input = []
            session_output = []

            for t in range(n_trials):
                start = t * bins_per_trial
                end = start + bins_per_trial

                session_neural.append(dff_binned[:, start:end])
                # Input: time elapsed from beginning of session
                session_input.append(time_vec[start:end].reshape(1, -1))
                # Output: discretized motion energy
                session_output.append(me_discrete[start:end].reshape(1, -1))

            all_neural.append(session_neural)
            all_input.append(session_input)
            all_output.append(session_output)
            all_subject_idx.append(si)
            all_brain_region_idx.append(np.zeros(n_neurons, dtype=int))

    # Build output dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': subjects,
        'subject_idx': np.array(all_subject_idx),

        'brain_regions': ['S1BF'],  # barrel cortex (somatosensory area, barrel field)
        'brain_region_idx': all_brain_region_idx,

        'input_names': ['time_s'],
        'output_names': ['motion_energy'],
        'output_values': [
            [f'bin_{i}' for i in range(N_BINS_OUTPUT)]
        ],

        'metadata': {
            'task_description': 'Decode motion energy (5 quintile bins) from barrel cortex calcium imaging during spontaneous behavior',
            'time_bin_size': BIN_SIZE / FS * 1000,  # in ms = 333.33
            'temporal_alignment_event': 'Start of imaging session',
            'off_start': 0.0,
            'off_end': float(TRIAL_DUR),
            'sampling_rate_hz': FS,
            'bin_size_frames': BIN_SIZE,
            'trial_duration_s': TRIAL_DUR,
            'dff_method': 'Suite2p default (neucoeff=0.7, maximin baseline, sig_baseline=10, win_baseline=60s)',
        }
    }

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(data, f)

    # Print summary
    n_sessions = len(all_neural)
    n_trials_total = sum(len(s) for s in all_neural)
    print(f"Saved {n_sessions} sessions, {n_trials_total} total trials")
    print(f"Subjects: {subjects}")
    for i, (neural, subj) in enumerate(zip(all_neural, all_subject_idx)):
        n_neurons = neural[0].shape[0] if neural else 0
        n_trials = len(neural)
        print(f"  Session {i}: subject={subjects[subj]}, neurons={n_neurons}, trials={n_trials}")


if __name__ == '__main__':
    main()
