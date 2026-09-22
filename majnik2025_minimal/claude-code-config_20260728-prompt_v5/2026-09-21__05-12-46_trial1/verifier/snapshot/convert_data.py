"""
Convert Track2p longitudinal calcium imaging data to decoder format.

Data: Majnik et al. 2025 - barrel cortex development dataset
- 6 mice, 6-7 sessions each (daily recordings P7-P14)
- Neural: dF/F computed from Suite2p outputs (F.npy, Fneu.npy) using default Suite2p parameters
- Output: Motion energy discretized into 5 equal-percentile bins per session
- Input: Time elapsed from session start (seconds)

Processing decisions:
1. dF/F: Neuropil correction (Fc = F - 0.7*Fneu) + maximin baseline subtraction
   (Suite2p default parameters from ops.npy: neucoeff=0.7, baseline='maximin',
    sig_baseline=10, win_baseline=60s)
   Following track2p code (F_processing), baseline is subtracted not divided.
2. Binning: Average in bins of 10 consecutive frames (paper decoding methods),
   yielding 3 Hz effective rate (333.33 ms bins)
3. Motion energy: Truncated to match neural frame count when mismatched,
   binned by same 10-frame bins, discretized into 5 quintile bins per session
4. Trials: 60-second segments (180 time bins at 3 Hz)
5. Neurons: Already tracked and matched across sessions by Track2p pipeline;
   all pass iscell > 0.5 threshold
"""

import os
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d


DATA_DIR = '/app/data'
OUTPUT_PATH = '/app/converted_data.pkl'

MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
BIN_SIZE = 10       # frames to average per bin (paper: "bins of 10 consecutive timestamps")
FS = 30             # imaging rate in Hz
TRIAL_DURATION = 60 # seconds per trial
N_ME_BINS = 5       # number of motion energy quantile bins


def compute_dff(F, Fneu, fs, neucoeff=0.7, sig_baseline=10.0, win_baseline=60.0):
    """
    Compute baseline-corrected fluorescence (dF/F) using Suite2p default parameters.
    Following the track2p F_processing implementation:

    1. Neuropil correction: Fc = F - neucoeff * Fneu
    2. Baseline estimation: maximin filter (Gaussian smooth -> min filter -> max filter)
    3. dF/F = Fc - baseline  (subtraction, as in track2p code)
    """
    # Neuropil subtraction
    Fc = F - neucoeff * Fneu

    # Baseline: maximin filter
    win = int(win_baseline * fs)  # window in frames
    Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)

    # Baseline subtraction (as per track2p F_processing)
    dff = Fc - Flow

    return dff


def bin_data(data, bin_size):
    """
    Bin data by averaging consecutive frames.
    data: array of shape (..., n_frames)
    Returns: array of shape (..., n_frames // bin_size)
    """
    n_frames = data.shape[-1]
    n_bins = n_frames // bin_size
    # Truncate to exact multiple of bin_size
    data = data[..., :n_bins * bin_size]
    # Reshape and average
    new_shape = data.shape[:-1] + (n_bins, bin_size)
    return data.reshape(new_shape).mean(axis=-1)


def get_sessions(mouse_dir):
    """Get sorted session directories for a mouse."""
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
    sessions.sort()
    return sessions


def main():
    neural_all = []
    input_all = []
    output_all = []
    subjects = MICE
    subject_idx_list = []
    brain_region_idx_all = []

    for mouse_idx, mouse in enumerate(MICE):
        mouse_dir = os.path.join(DATA_DIR, mouse)
        sessions = get_sessions(mouse_dir)
        print(f"\n{mouse}: {len(sessions)} sessions")

        for session_name in sessions:
            session_dir = os.path.join(mouse_dir, session_name)
            s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
            me_dir = os.path.join(session_dir, 'move_deve')

            # Load neural data
            F = np.load(os.path.join(s2p_dir, 'F.npy'))
            Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
            ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
            fs = ops['fs']

            n_neurons, n_frames = F.shape

            # Load motion energy
            me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy')).astype(np.float64)

            # Truncate to common length (some sessions have fewer ME frames)
            common_len = min(n_frames, len(me))
            F = F[:, :common_len]
            Fneu = Fneu[:, :common_len]
            me = me[:common_len]

            # Compute dF/F
            dff = compute_dff(F, Fneu, fs,
                              neucoeff=ops.get('neucoeff', 0.7),
                              sig_baseline=ops.get('sig_baseline', 10.0),
                              win_baseline=ops.get('win_baseline', 60.0))

            # Bin neural data and motion energy (10 frames)
            dff_binned = bin_data(dff, BIN_SIZE)        # (n_neurons, n_bins)
            me_binned = bin_data(me[np.newaxis, :], BIN_SIZE)[0]  # (n_bins,)

            n_bins = dff_binned.shape[1]

            # Time vector (seconds from start of session, at bin centers)
            bin_duration = BIN_SIZE / fs  # seconds per bin
            time_vec = np.arange(n_bins) * bin_duration  # (n_bins,)

            # Discretize motion energy into 5 equal-percentile bins for this session
            # Use np.percentile to find bin edges at 20th, 40th, 60th, 80th percentiles
            percentiles = np.linspace(0, 100, N_ME_BINS + 1)
            bin_edges = np.percentile(me_binned, percentiles)
            # Make edges unique to handle ties
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            me_discrete = np.digitize(me_binned, bin_edges[1:])  # values 0 to N_ME_BINS-1
            me_discrete = np.clip(me_discrete, 0, N_ME_BINS - 1)

            # Split into 60-second trials
            bins_per_trial = int(TRIAL_DURATION / bin_duration)  # 180 bins per trial
            n_trials = n_bins // bins_per_trial

            session_neural = []
            session_input = []
            session_output = []

            for t in range(n_trials):
                start = t * bins_per_trial
                end = start + bins_per_trial

                # Neural: (n_neurons, T)
                session_neural.append(dff_binned[:, start:end].astype(np.float32))

                # Input: time elapsed in seconds, (1, T)
                session_input.append(time_vec[start:end][np.newaxis, :].astype(np.float32))

                # Output: discretized motion energy, (1, T)
                session_output.append(me_discrete[start:end][np.newaxis, :].astype(np.int64))

            neural_all.append(session_neural)
            input_all.append(session_input)
            output_all.append(session_output)
            subject_idx_list.append(mouse_idx)
            brain_region_idx_all.append(np.zeros(n_neurons, dtype=np.int64))

            print(f"  {session_name}: {n_neurons} neurons, {n_bins} bins, {n_trials} trials, "
                  f"ME bins distribution: {[np.sum(me_discrete==i) for i in range(N_ME_BINS)]}")

    # Build output dictionary
    time_bin_ms = (BIN_SIZE / FS) * 1000  # 333.33 ms

    # Output value names for 5 quintile bins
    output_values = [
        ['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)']
    ]

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': subjects,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
        'brain_regions': ['barrel_cortex'],
        'brain_region_idx': brain_region_idx_all,
        'input_names': ['time_elapsed_s'],
        'output_names': ['motion_energy'],
        'output_values': output_values,
        'metadata': {
            'task_description': 'Decode motion energy from barrel cortex calcium activity during spontaneous behavior in developing mice (P7-P14)',
            'time_bin_size': time_bin_ms,
            'temporal_alignment_event': 'Start of recording session',
            'off_start': 0.0,
            'off_end': float(TRIAL_DURATION),
            'imaging_rate_hz': FS,
            'bin_size_frames': BIN_SIZE,
            'effective_rate_hz': FS / BIN_SIZE,
            'trial_duration_s': TRIAL_DURATION,
            'n_motion_energy_bins': N_ME_BINS,
            'dff_method': 'Suite2p default (neucoeff=0.7, maximin baseline, sig_baseline=10, win_baseline=60s)',
        }
    }

    # Save
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Total sessions: {len(neural_all)}")
    print(f"Total trials: {sum(len(s) for s in neural_all)}")
    print(f"Subjects: {subjects}")
    print(f"Time bin size: {time_bin_ms:.2f} ms")


if __name__ == '__main__':
    main()
