"""
Convert Track2p longitudinal calcium imaging data to decoder format.

Reference: Majnik et al. 2025, "Longitudinal tracking of neuronal activity
from the same cells in the developing brain using Track2p"

Data processing follows the methods described in the paper:
- dF/F computed using Suite2p default parameters (neuropil correction + maximin baseline)
- Neural and behavioral data binned in 10-frame bins (as in paper's decoding analysis)
- Sessions split into 2-minute blocks (trials, matching paper's cross-validation scheme)
- Motion energy normalized and discretized into 5 equal-percentile bins
"""

import os
import sys
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d

DATA_DIR = '/app/data'
BIN_SIZE = 10  # frames per bin (paper: "averaging in bins of 10 consecutive timestamps")
FS = 30  # imaging rate in Hz
TRIAL_DURATION_S = 120  # 2 minutes per trial (paper: "consecutive 2 minute blocks")
N_BINS_OUTPUT = 5  # number of percentile bins for motion energy

# Suite2p default parameters (from ops.npy)
NEUCOEFF = 0.7
WIN_BASELINE = 60.0  # seconds
SIG_BASELINE = 10.0  # seconds


def compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF,
                win_baseline=WIN_BASELINE, sig_baseline=SIG_BASELINE):
    """Compute dF/F using Suite2p default 'maximin' baseline method.

    1. Neuropil correction: Fc = F - neucoeff * Fneu
    2. Maximin baseline: gaussian smooth -> running min -> running max
    3. dF/F = (Fc - F0) / F0
    """
    Fc = F - neucoeff * Fneu

    win = int(win_baseline * fs)  # 1800 frames
    sig = int(sig_baseline * fs)  # 300 frames

    # Maximin baseline (Suite2p default)
    Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)

    # Avoid division by zero
    Flow = np.maximum(Flow, 1e-6)
    dff = (Fc - Flow) / Flow

    return dff.astype(np.float32)


def interpolate_motion_energy(me, tstamps, n_frames):
    """Interpolate motion energy to match 2p imaging frames.

    When camera drops frames, motion energy has fewer samples than neural data.
    Interpolate to fill in missing frames.
    """
    if len(me) == n_frames:
        return me

    # Find where frames were dropped using interframe intervals
    ifi = np.diff(tstamps)
    median_ifi = np.median(ifi)

    # Build mapping: for each camera frame, determine which 2p frame it corresponds to
    # A gap of ~2x normal interval means one frame was dropped
    frame_indices = np.zeros(len(me), dtype=int)
    frame_indices[0] = 0
    cum_idx = 0
    for i in range(len(ifi)):
        n_skipped = int(np.round(ifi[i] / median_ifi))
        cum_idx += n_skipped
        frame_indices[i + 1] = cum_idx

    # Interpolate to all frames
    all_indices = np.arange(n_frames)
    me_interp = np.interp(all_indices, frame_indices, me)

    return me_interp


def bin_data(data, bin_size):
    """Average data in bins along the last axis. Truncate remainder."""
    n = data.shape[-1]
    n_bins = n // bin_size
    n_use = n_bins * bin_size

    if data.ndim == 1:
        return data[:n_use].reshape(n_bins, bin_size).mean(axis=1)
    else:
        return data[:, :n_use].reshape(data.shape[0], n_bins, bin_size).mean(axis=2)


def load_and_process_session(session_dir, n_2p_frames=None):
    """Load and process data for one session."""
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(session_dir, 'move_deve')

    # Load neural data
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))

    n_neurons, n_frames = F.shape

    # Compute dF/F
    dff = compute_dff(F, Fneu)

    # Load motion energy
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))

    # Interpolate motion energy if needed
    me = interpolate_motion_energy(me, tstamps, n_frames)

    # Bin data in 10-frame bins
    dff_binned = bin_data(dff, BIN_SIZE)
    me_binned = bin_data(me, BIN_SIZE)

    # Time in seconds for each bin (center of bin)
    bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS

    return dff_binned, me_binned, bin_centers


def split_into_trials(dff_binned, me_binned, time_bins, trial_duration_s=TRIAL_DURATION_S):
    """Split session data into 2-minute trials."""
    bins_per_trial = int(trial_duration_s * FS / BIN_SIZE)  # 120 * 30 / 10 = 360
    n_bins = dff_binned.shape[1]
    n_trials = n_bins // bins_per_trial

    neural_trials = []
    me_trials = []
    time_trials = []

    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(dff_binned[:, start:end])
        me_trials.append(me_binned[start:end])
        time_trials.append(time_bins[start:end])

    return neural_trials, me_trials, time_trials


def discretize_motion_energy(all_me_values, n_bins=N_BINS_OUTPUT):
    """Compute percentile thresholds for discretizing motion energy."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    thresholds = np.percentile(all_me_values, percentiles)
    return thresholds


def apply_discretization(me_values, thresholds):
    """Assign motion energy values to bins using precomputed thresholds."""
    n_bins = len(thresholds) - 1
    binned = np.digitize(me_values, thresholds[1:-1])  # values 0 to n_bins-1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned


def convert_data(data_dir=DATA_DIR, sample_mode=False):
    """Main conversion function."""
    subjects = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

    print(f"Found {len(subjects)} subjects: {subjects}")

    # First pass: load and process all sessions, collect all ME values for global discretization
    all_sessions_data = []
    all_me_flat = []

    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        sessions = sorted([d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d))])

        print(f"\n{subj}: {len(sessions)} sessions")

        for sess_name in sessions:
            sess_dir = os.path.join(subj_dir, sess_name)
            print(f"  Processing {sess_name}...", end=" ")

            dff_binned, me_binned, time_bins = load_and_process_session(sess_dir)
            neural_trials, me_trials, time_trials = split_into_trials(
                dff_binned, me_binned, time_bins
            )

            n_trials = len(neural_trials)
            n_neurons = neural_trials[0].shape[0]
            print(f"{n_neurons} neurons, {n_trials} trials, "
                  f"{neural_trials[0].shape[1]} bins/trial")

            all_sessions_data.append({
                'subject': subj,
                'session': sess_name,
                'neural_trials': neural_trials,
                'me_trials': me_trials,
                'time_trials': time_trials,
                'n_neurons': n_neurons,
            })

            for me_t in me_trials:
                all_me_flat.append(me_t)

    # Compute global percentile thresholds for motion energy discretization
    all_me_concat = np.concatenate(all_me_flat)
    thresholds = discretize_motion_energy(all_me_concat)
    print(f"\nMotion energy discretization thresholds (quintiles): {thresholds}")

    # Build output data structure
    neural = []
    input_data = []
    output_data = []
    subject_idx = []
    brain_region_idx_list = []

    for sess_data in all_sessions_data:
        subj = sess_data['subject']
        subj_i = subjects.index(subj)
        subject_idx.append(subj_i)

        # Neural: list of trials, each (n_neurons, n_timepoints)
        neural.append(sess_data['neural_trials'])

        # Input: time elapsed from beginning of session, shape (1, n_timepoints)
        sess_input = []
        for t_trial in sess_data['time_trials']:
            sess_input.append(t_trial.reshape(1, -1))
        input_data.append(sess_input)

        # Output: discretized motion energy, shape (1, n_timepoints)
        sess_output = []
        for me_trial in sess_data['me_trials']:
            me_disc = apply_discretization(me_trial, thresholds)
            sess_output.append(me_disc.reshape(1, -1))
        output_data.append(sess_output)

        # Brain region idx: all neurons from barrel cortex
        brain_region_idx_list.append(np.zeros(sess_data['n_neurons'], dtype=int))

    # Create output value labels
    output_values = [
        [f"bin_{i}" for i in range(N_BINS_OUTPUT)]
    ]

    data = {
        'neural': neural,
        'input': input_data,
        'output': output_data,

        'subjects': subjects,
        'subject_idx': np.array(subject_idx, dtype=int),

        'brain_regions': ['barrel_cortex'],
        'brain_region_idx': brain_region_idx_list,

        'input_names': ['time_s'],
        'output_names': ['motion_energy'],
        'output_values': output_values,

        'metadata': {
            'task_description': 'Decode motion energy from barrel cortex neural activity during spontaneous behavior in developing mice (P7-P14)',
            'time_bin_size': BIN_SIZE / FS * 1000,  # in ms = 333.33 ms
            'temporal_alignment_event': 'recording_onset',
            'off_start': 0.0,
            'off_end': float(TRIAL_DURATION_S),
            'imaging_rate_hz': FS,
            'bin_size_frames': BIN_SIZE,
            'trial_duration_s': TRIAL_DURATION_S,
            'n_percentile_bins': N_BINS_OUTPUT,
            'me_thresholds': thresholds.tolist(),
            'neuropil_coefficient': NEUCOEFF,
            'baseline_method': 'maximin',
            'win_baseline_s': WIN_BASELINE,
            'sig_baseline_s': SIG_BASELINE,
        }
    }

    return data


def print_sanity_checks(data):
    """Print sanity checks comparing to paper statistics."""
    print("\n" + "="*60)
    print("SANITY CHECKS")
    print("="*60)

    subjects = data['subjects']
    subject_idx = data['subject_idx']

    # Check neuron counts per mouse
    neuron_counts = {}
    for subj in subjects:
        neuron_counts[subj] = []

    for i, sess_neural in enumerate(data['neural']):
        subj = subjects[subject_idx[i]]
        n_neurons = sess_neural[0].shape[0]
        neuron_counts[subj].append(n_neurons)

    print("\nNeuron counts per mouse (paper: 526 +/- 190 std):")
    all_counts = []
    for subj in subjects:
        counts = neuron_counts[subj]
        print(f"  {subj}: {counts[0]} neurons (same across {len(counts)} sessions)")
        all_counts.append(counts[0])

    mean_n = np.mean(all_counts)
    std_n = np.std(all_counts)
    print(f"  Mean: {mean_n:.0f}, Std: {std_n:.0f}")

    # Check session counts per mouse
    print("\nSessions per mouse (paper: at least 6 consecutive days):")
    for subj in subjects:
        n_sess = sum(1 for idx in subject_idx if subjects[idx] == subj)
        print(f"  {subj}: {n_sess} sessions")

    # Check trial counts
    print("\nTrials per session:")
    for i, sess_neural in enumerate(data['neural']):
        subj = subjects[subject_idx[i]]
        n_trials = len(sess_neural)
        n_timepoints = sess_neural[0].shape[1]
        print(f"  {subj} session {i}: {n_trials} trials, {n_timepoints} bins/trial")

    # Check output distribution
    print("\nOutput class distribution (should be ~equal for percentile bins):")
    all_outputs = []
    for sess_out in data['output']:
        for trial_out in sess_out:
            all_outputs.append(trial_out.flatten())
    all_outputs = np.concatenate(all_outputs)
    for v in range(N_BINS_OUTPUT):
        frac = np.mean(all_outputs == v)
        print(f"  Bin {v}: {frac:.3f}")

    # Total dataset size
    n_sessions = len(data['neural'])
    n_trials_total = sum(len(s) for s in data['neural'])
    print(f"\nTotal: {n_sessions} sessions, {n_trials_total} trials")
    print(f"Time bin size: {data['metadata']['time_bin_size']:.2f} ms")


def create_sample(data, max_sessions=6):
    """Create a smaller sample dataset for quick testing."""
    sample = {k: v for k, v in data.items()}

    # Take first max_sessions sessions
    n = min(max_sessions, len(data['neural']))
    sample['neural'] = data['neural'][:n]
    sample['input'] = data['input'][:n]
    sample['output'] = data['output'][:n]
    sample['subject_idx'] = data['subject_idx'][:n]
    sample['brain_region_idx'] = data['brain_region_idx'][:n]

    return sample


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help='Create sample dataset only')
    parser.add_argument('--output', type=str, default='converted_data.pkl')
    args = parser.parse_args()

    data = convert_data()
    print_sanity_checks(data)

    # Save full dataset
    output_path = os.path.join('/app', args.output)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nSaved full dataset to {output_path}")

    # Also create sample
    sample = create_sample(data)
    sample_path = os.path.join('/app', 'sample_data.pkl')
    with open(sample_path, 'wb') as f:
        pickle.dump(sample, f)
    print(f"Saved sample dataset to {sample_path}")
