"""
Convert Majnik et al. 2025 (Track2p) longitudinal 2-photon dataset into the
standard decoder format.

Decoder task: decode animal motion (motion energy, discretised into 5 per-session
equal-percentile bins) from neural activity of barrel cortex (S1) neurons.
Input: time elapsed from the beginning of the session (seconds).
Trials: consecutive 60 s blocks of each (continuous, spontaneous-activity) session.

Processing follows the paper / repository:
  - suite2p outputs provided by track2p contain only the neurons tracked across all
    days of a mouse; all of them have iscell==1 (curation already applied, ROIs with
    classifier probability > 0.5 were kept).
  - traces: dF/F = baseline-corrected fluorescence, computed as in
    track2p/gui/data_management.py:F_processing with the default suite2p parameters
    stored in ops (neuropil subtraction with neucoeff, 'maximin' baseline,
    sig_baseline=10 frames, win_baseline=60 s).
  - behaviour: 'motion_energy_glob.npy' (pixel-wise squared frame differences).
    Camera frames dropped during acquisition are recovered using 'interframe_int.npy'
    (gaps are integer multiples of the frame interval) and linearly interpolated.
  - both dF/F and behaviour are denoised by averaging in bins of 10 consecutive
    frames (30 Hz -> 3 Hz, 333.33 ms bins), exactly as for the decoding analysis of
    the paper.
"""

import os
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter1d, maximum_filter1d

DATA_DIR = '/app/data'
OUT_FILE = '/app/converted_data.pkl'

BIN_FRAMES = 10            # average 10 consecutive frames (paper's decoding analysis)
TRIAL_SEC = 60.0           # 60 s trials
N_ME_BINS = 5              # number of motion-energy percentile bins


def list_sessions(subject_dir):
    return sorted([f.path for f in os.scandir(subject_dir) if f.is_dir()])


def compute_dff(F, Fneu, ops):
    """Baseline-corrected fluorescence ('dF/F' of the paper), as in track2p's
    F_processing / suite2p preprocessing, using the parameters stored in ops."""
    neucoeff = float(ops.get('neucoeff', 0.7))
    sig_baseline = float(ops.get('sig_baseline', 10.0))
    win_baseline = float(ops.get('win_baseline', 60.0))
    fs = float(ops['fs'])
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter(Fc, [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    return Fc - Flow


def load_motion_energy(session_dir, nframes):
    """Motion energy on the 2-photon frame grid (camera was triggered by the
    microscope, so frames correspond 1:1 up to dropped camera frames)."""
    me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
    ifi = np.load(os.path.join(session_dir, 'move_deve', 'interframe_int.npy')).astype(np.float64)
    # the first sample is 0 by construction (no preceding frame to difference with)
    me[0] = me[1]
    if len(me) == nframes:
        return me
    # recover indices of the acquired camera frames on the full frame grid
    med = np.median(ifi)
    steps = np.round(ifi / med).astype(int)
    idx = np.concatenate([[0], np.cumsum(steps)])
    assert len(idx) == len(me)
    assert idx[-1] == nframes - 1, (session_dir, idx[-1], nframes)
    return np.interp(np.arange(nframes), idx, me)


def bin_average(x, binsize):
    """Average consecutive bins along the last axis (drops an incomplete last bin)."""
    x = np.asarray(x)
    n = (x.shape[-1] // binsize) * binsize
    x = x[..., :n]
    new_shape = x.shape[:-1] + (n // binsize, binsize)
    return x.reshape(new_shape).mean(axis=-1)


def main():
    subjects = sorted([d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, d))])

    neural_all, input_all, output_all = [], [], []
    subject_idx, brain_region_idx = [], []
    session_info = []

    for si, subject in enumerate(subjects):
        for session_dir in list_sessions(os.path.join(DATA_DIR, subject)):
            s2p = os.path.join(session_dir, 'suite2p', 'plane0')
            ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
            F = np.load(os.path.join(s2p, 'F.npy')).astype(np.float64)
            Fneu = np.load(os.path.join(s2p, 'Fneu.npy')).astype(np.float64)
            iscell = np.load(os.path.join(s2p, 'iscell.npy'))
            # only tracked cells are provided, and all pass the suite2p classifier
            keep = iscell[:, 0] > 0
            F, Fneu = F[keep], Fneu[keep]

            fs = float(ops['fs'])
            nframes = F.shape[1]

            dff = compute_dff(F, Fneu, ops)
            me = load_motion_energy(session_dir, nframes)

            # denoise: average bins of 10 frames -> 3 Hz
            dff_b = bin_average(dff, BIN_FRAMES)
            me_b = bin_average(me, BIN_FRAMES)
            bin_size_s = BIN_FRAMES / fs
            nbins = dff_b.shape[1]
            # time (s) at the centre of each bin, from the beginning of the session
            t_b = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs

            # discretise motion energy into 5 equal-percentile bins (per session)
            edges = np.percentile(me_b, np.linspace(0, 100, N_ME_BINS + 1)[1:-1])
            me_cat = np.digitize(me_b, edges).astype(np.int64)

            # cut into 60 s trials
            bins_per_trial = int(round(TRIAL_SEC * fs / BIN_FRAMES))
            ntrials = nbins // bins_per_trial
            neural_s, input_s, output_s = [], [], []
            for tr in range(ntrials):
                sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
                neural_s.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
                input_s.append(t_b[sl][None, :].astype(np.float32))
                output_s.append(me_cat[sl][None, :].astype(np.int64))

            neural_all.append(neural_s)
            input_all.append(input_s)
            output_all.append(output_s)
            subject_idx.append(si)
            brain_region_idx.append(np.zeros(F.shape[0], dtype=np.int64))
            session_info.append({
                'subject': subject,
                'session': os.path.basename(session_dir),
                'n_neurons': int(F.shape[0]),
                'n_frames': int(nframes),
                'n_trials': int(ntrials),
                'fs': fs,
            })
            print(subject, os.path.basename(session_dir), F.shape, 'trials', ntrials)

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': subjects,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
        'brain_regions': ['S1'],
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_session_start_s'],
        'output_names': ['motion_energy_quintile'],
        'output_values': [['q1_lowest', 'q2', 'q3', 'q4', 'q5_highest']],
        'metadata': {
            'task_description': (
                'Spontaneous activity (no sensory stimulation, dark, sensory-minimised) in '
                'layer 2/3 of mouse barrel cortex (S1) during the second postnatal week, '
                'head-fixed pups free to move on a non-motorised treadmill. Decode the '
                'animal motion energy (videography-based, discretised into 5 per-session '
                'equal-percentile bins) from the neural activity; the decoder additionally '
                'receives the time elapsed from the beginning of the session.'),
            'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,  # ms
            'temporal_alignment_event': (
                'start of each 60 s block of the continuous recording (trials are '
                'consecutive 60 s segments starting at the first imaging frame of the session)'),
            'off_start': 0.0,
            'off_end': TRIAL_SEC,
            'neural_data_type': (
                'baseline-corrected fluorescence (dF/F as defined in the paper): '
                'F - neucoeff*Fneu followed by suite2p maximin baseline subtraction '
                '(sig_baseline=10 frames, win_baseline=60 s), then averaged in bins of '
                '10 consecutive frames'),
            'imaging_rate_hz': 30.0,
            'neuron_tracking': (
                'only neurons tracked by Track2p across all days of a given mouse are '
                'provided; neuron order is matched across sessions of the same mouse'),
            'behavior_processing': (
                'global motion energy (sum of squared pixel-wise differences of consecutive '
                'video frames, camera triggered by the microscope at 30 Hz); dropped camera '
                'frames recovered from interframe_int.npy and linearly interpolated; averaged '
                'in bins of 10 frames; discretised into 5 equal-percentile bins per session'),
            'session_info': session_info,
        },
    }

    with open(OUT_FILE, 'wb') as f:
        pickle.dump(data, f)
    print('saved', OUT_FILE, 'sessions', len(neural_all))


if __name__ == '__main__':
    main()
