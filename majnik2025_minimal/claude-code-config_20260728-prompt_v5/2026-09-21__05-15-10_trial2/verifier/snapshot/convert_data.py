"""
Convert Track2p longitudinal calcium imaging data for neural decoding.

Processing follows the paper (Majnik et al. 2025) and Suite2p defaults:
- dF/F: neuropil correction (F - 0.7*Fneu) + maximin baseline subtraction
  (Suite2p default parameters: sig_baseline=10, win_baseline=60, fs=30)
- Temporal binning: average every 10 frames (as in paper's decoding analysis)
- Motion energy: interpolated to neural frame count for sessions with dropped
  camera frames, then binned the same way
- Discretization: 5 equal-percentile bins per session
- Trials: 60-second segments (180 binned time points)
"""

import os
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter, minimum_filter1d, maximum_filter1d


DATA_DIR = '/app/data'
OUTPUT_PATH = '/app/converted_data.pkl'
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
FS = 30  # imaging frame rate (Hz)
BIN_SIZE = 10  # frames per bin (paper: "averaging in bins of 10 consecutive timestamps")
BIN_DUR = BIN_SIZE / FS  # duration of each bin in seconds
TRIAL_DUR = 60  # trial duration in seconds
TRIAL_BINS = int(TRIAL_DUR / BIN_DUR)  # 180 bins per 60s trial
N_BINS_OUTPUT = 5  # number of equal-percentile bins for motion energy

# Suite2p default parameters for dF/F
NEUCOEFF = 0.7
SIG_BASELINE = 10.0
WIN_BASELINE = 60.0


def compute_dff(F, Fneu, fs=FS):
    """Compute dF/F using Suite2p default parameters.

    1. Neuropil correction: Fc = F - 0.7 * Fneu
    2. Maximin baseline: gaussian smooth -> min filter -> max filter
    3. Baseline subtraction: dF/F = Fc - Flow
    """
    Fc = F - NEUCOEFF * Fneu
    win = int(WIN_BASELINE * fs)
    Flow = gaussian_filter(Fc.astype(np.float64), [0., SIG_BASELINE])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    dff = Fc - Flow
    return dff.astype(np.float32)


def bin_data(data, bin_size=BIN_SIZE):
    """Average data in non-overlapping bins along last axis.
    Truncates to a multiple of bin_size.
    """
    if data.ndim == 1:
        n = len(data) // bin_size * bin_size
        return data[:n].reshape(-1, bin_size).mean(axis=1)
    else:
        n = data.shape[1] // bin_size * bin_size
        return data[:, :n].reshape(data.shape[0], -1, bin_size).mean(axis=2)


def align_motion_energy(me, n_neural_frames, tstamps):
    """Align motion energy to neural frames by interpolation.

    When camera drops frames, ME has fewer samples than neural data.
    Use tstamps to interpolate ME to the full neural frame count.
    """
    if len(me) == n_neural_frames:
        return me

    # tstamps gives the time of each camera frame
    # Neural frames are evenly spaced
    # Interpolate ME to neural frame positions
    camera_pos = np.linspace(0, 1, len(me))
    neural_pos = np.linspace(0, 1, n_neural_frames)
    me_aligned = np.interp(neural_pos, camera_pos, me)
    return me_aligned


def process_session(session_dir):
    """Load and process a single session's data."""
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(session_dir, 'move_deve')

    # Load neural data
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))

    # Compute dF/F
    dff = compute_dff(F, Fneu)

    # Load motion energy and timestamps
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))

    # Align ME to neural frames (interpolate if camera dropped frames)
    n_neural = F.shape[1]
    me_aligned = align_motion_energy(me, n_neural, tstamps)

    # Bin both signals (average every 10 frames)
    dff_binned = bin_data(dff, BIN_SIZE)
    me_binned = bin_data(me_aligned, BIN_SIZE)

    return dff_binned, me_binned


def split_trials(neural, me_binned, trial_bins=TRIAL_BINS):
    """Split binned data into fixed-length trials.
    Discard any remainder that doesn't fill a complete trial.
    """
    n_bins = neural.shape[1]
    n_trials = n_bins // trial_bins

    neural_trials = []
    me_trials = []

    for t in range(n_trials):
        start = t * trial_bins
        end = start + trial_bins
        neural_trials.append(neural[:, start:end])
        me_trials.append(me_binned[start:end])

    return neural_trials, me_trials


def discretize_me(me_trials, n_bins=N_BINS_OUTPUT):
    """Discretize motion energy into equal-percentile bins per session.

    Compute percentile boundaries from all trials in the session,
    then assign each time point to a bin (0 to n_bins-1).
    """
    # Pool all ME values from this session
    all_me = np.concatenate(me_trials)

    # Compute percentile boundaries
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # inner boundaries
    boundaries = np.percentile(all_me, percentiles)

    # Digitize each trial
    discretized = []
    for me in me_trials:
        binned = np.digitize(me, boundaries)  # 0 to n_bins-1
        discretized.append(binned.astype(np.int64))

    return discretized


def main():
    all_neural = []
    all_input = []
    all_output = []
    subject_idx = []
    brain_region_idx = []

    for mouse_i, mouse in enumerate(MICE):
        mouse_dir = os.path.join(DATA_DIR, mouse)
        sessions = sorted([
            d for d in os.listdir(mouse_dir)
            if os.path.isdir(os.path.join(mouse_dir, d))
        ])

        for sess in sessions:
            session_dir = os.path.join(mouse_dir, sess)
            print(f"Processing {mouse}/{sess}...")

            # Process session
            dff_binned, me_binned = process_session(session_dir)
            n_neurons = dff_binned.shape[0]

            # Split into 60-second trials
            neural_trials, me_trials = split_trials(dff_binned, me_binned)

            # Discretize motion energy into 5 bins per session
            me_discrete = discretize_me(me_trials)

            # Create time input: elapsed time from session start in seconds
            input_trials = []
            for t in range(len(neural_trials)):
                # Time of each bin center relative to session start
                start_bin = t * TRIAL_BINS
                time_points = (np.arange(TRIAL_BINS) + start_bin + 0.5) * BIN_DUR
                input_trials.append(time_points.reshape(1, -1).astype(np.float32))

            # Store
            all_neural.append(neural_trials)
            all_input.append(input_trials)
            all_output.append([me.reshape(1, -1) for me in me_discrete])
            subject_idx.append(mouse_i)
            brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))

    # Build output value labels for 5 bins
    output_values = [[f'bin_{i}' for i in range(N_BINS_OUTPUT)]]

    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': MICE,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
        'brain_regions': ['barrel_cortex'],
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_in_session'],
        'output_names': ['motion_energy'],
        'output_values': output_values,
        'metadata': {
            'task_description': 'Decode motion energy (5 equal-percentile bins) from barrel cortex calcium activity during spontaneous behavior in developing mice',
            'time_bin_size': BIN_DUR * 1000,  # in ms: 333.33 ms
            'temporal_alignment_event': 'start of recording session',
            'off_start': 0.0,
            'off_end': None,
            'recording_rate_hz': FS,
            'bin_size_frames': BIN_SIZE,
            'trial_duration_s': TRIAL_DUR,
            'n_percentile_bins': N_BINS_OUTPUT,
            'dff_method': 'Suite2p default: F - 0.7*Fneu, maximin baseline subtraction (sig=10, win=60)',
        }
    }

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved converted data to {OUTPUT_PATH}")
    print(f"Sessions: {len(all_neural)}")
    print(f"Subjects: {len(MICE)}")
    for i, mouse in enumerate(MICE):
        n_sess = np.sum(np.array(subject_idx) == i)
        print(f"  {mouse}: {n_sess} sessions")


if __name__ == '__main__':
    main()
