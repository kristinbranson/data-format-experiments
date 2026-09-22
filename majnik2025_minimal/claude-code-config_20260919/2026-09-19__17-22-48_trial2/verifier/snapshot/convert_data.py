"""
Convert the Majnik et al. 2025 (Track2p) longitudinal developmental barrel-cortex
dataset into the decoder dataset format.

Source data (/app/data) contains, for each mouse and each recording day:
  - suite2p/plane0/{F,Fneu,iscell,stat,ops}.npy : Suite2p output restricted by
    Track2p to the neurons that were successfully tracked across *all* days of
    that mouse (rows are matched across days within a mouse).
  - move_deve/{motion_energy_glob,tstamps,interframe_int}.npy : motion energy
    computed from the behaviour videography (sum of squared pixel-wise
    differences between consecutive frames), one value per *camera* frame.

Decoder task
  input  : time elapsed since the beginning of the session (seconds)
  output : motion energy discretised into 5 equal-percentile (quintile) bins,
           thresholds computed per session
Sessions are cut into consecutive 60 s trials.

Processing follows the paper's Methods ("Processing of calcium imaging data",
"Preprocessing videography", "Decoding") and the authors' own code
(track2p/gui/data_management.py :: F_processing):
  - neural signal = baseline-corrected fluorescence ("dF/F" in the paper) using
    the default Suite2p parameters: neuropil subtraction with neucoeff = 0.7
    followed by subtraction of a maximin baseline (gaussian sigma 10 frames,
    60 s min/max filter window).
  - both the neural traces and the behaviour trace are denoised by averaging in
    bins of 10 consecutive frames, exactly as done for all decoding analyses in
    the paper (30 Hz / 10 = 3 Hz, i.e. 333.33 ms bins).
  - each neuron's binned trace is then z-scored over the session.  This is the
    only step that is not in the paper: it is needed because this decoder, unlike
    the paper's ridge regression, initialises its per-session projection from a
    raw (un-centred, un-standardised) SVD of the neural matrix, so without it the
    projection is dominated by the few ROIs with the largest raw fluorescence
    amplitude (validation balanced accuracy 0.24 without vs 0.34 with).
    Z-scoring only rescales each neuron, it does not alter temporal structure or
    the alignment to behaviour.

Curation: the released data already contains only Track2p-tracked ROIs that
passed the Suite2p classifier threshold of 0.5 (verified: iscell == 1 for every
ROI of every session), so no further neuron selection is applied.  All 6 mice
and all 41 sessions have complete neural and behavioural data and are kept.
"""

import os
import glob
import pickle

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter1d, maximum_filter1d

DATA_DIR = '/app/data'
OUT_FILE = '/app/converted_data.pkl'

FS = 30.0                 # imaging frame rate (Hz), ops['fs'] for every session
BIN_FRAMES = 10           # denoising bin used for all decoding in the paper
TRIAL_SECONDS = 60.0      # requested trial length
BIN_SECONDS = BIN_FRAMES / FS                       # 0.3333 s
BINS_PER_TRIAL = int(round(TRIAL_SECONDS / BIN_SECONDS))   # 180
FRAMES_PER_TRIAL = BINS_PER_TRIAL * BIN_FRAMES             # 1800

NEUCOEFF = 0.7            # Suite2p default neuropil coefficient
WIN_BASELINE = 60.0       # Suite2p default maximin window (s)
SIG_BASELINE = 10.0       # Suite2p default gaussian smoothing (frames)

N_QUANTILES = 5           # motion energy -> quintiles


def compute_dff(F, Fneu, fs=FS):
    """Baseline-corrected fluorescence, as in track2p/gui/data_management.py.

    Fc = F - 0.7*Fneu ; F0 = maximin(Fc) ; dff = Fc - F0
    """
    Fc = F.astype(np.float64) - NEUCOEFF * Fneu.astype(np.float64)
    Flow = gaussian_filter(Fc, [0.0, SIG_BASELINE])
    win = int(WIN_BASELINE * fs)
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    return Fc - Flow


def load_motion_energy(session_dir, n_frames):
    """Motion energy resampled onto the imaging frame grid (length n_frames).

    The behaviour camera is hardware-triggered by the microscope, so camera
    frame i corresponds to imaging frame i as long as no camera frame was
    dropped.  When frames were dropped (len(motion_energy) < n_frames, README)
    the dropped frames are located from the camera timestamps: an inter-frame
    interval of k * dt means k-1 missing frames.  Reconstructing the frame index
    this way lands exactly on n_frames-1 for every affected session, and
    accounts for exactly the right number of missing frames.  Missing values are
    filled by linear interpolation.
    """
    me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
    # The first value is always 0: there is no preceding frame to difference
    # against.  Replace this edge artefact by the first real measurement.
    me[0] = me[1]

    if len(me) == n_frames:
        return me

    ts = np.load(os.path.join(session_dir, 'move_deve', 'tstamps.npy'))
    dts = np.diff(ts)
    steps = np.round(dts / np.median(dts)).astype(int)
    idx = np.concatenate([[0], np.cumsum(steps)])
    assert idx[-1] == n_frames - 1, (session_dir, idx[-1], n_frames)
    assert len(np.unique(idx)) == len(idx)

    full = np.full(n_frames, np.nan)
    full[idx] = me
    missing = np.isnan(full)
    full[missing] = np.interp(np.flatnonzero(missing), idx, me)
    return full


def bin_time(x, bin_frames=BIN_FRAMES):
    """Average consecutive bins of `bin_frames` samples along the last axis."""
    x = np.asarray(x)
    n = (x.shape[-1] // bin_frames) * bin_frames
    x = x[..., :n]
    return x.reshape(*x.shape[:-1], n // bin_frames, bin_frames).mean(axis=-1)


def main():
    subjects = sorted(d for d in os.listdir(DATA_DIR)
                      if os.path.isdir(os.path.join(DATA_DIR, d)))

    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': subjects, 'subject_idx': [],
        'brain_regions': ['S1'], 'brain_region_idx': [],
        'input_names': ['time_from_session_start_s'],
        'output_names': ['motion_energy_quintile'],
        'output_values': [[f'quintile_{i + 1}' for i in range(N_QUANTILES)]],
        'metadata': {},
    }
    session_info = []

    for subj_i, subj in enumerate(subjects):
        session_dirs = sorted(glob.glob(os.path.join(DATA_DIR, subj, '*_a')))
        for day_i, session_dir in enumerate(session_dirs):
            s2p = os.path.join(session_dir, 'suite2p', 'plane0')
            F = np.load(os.path.join(s2p, 'F.npy'))
            Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))
            iscell = np.load(os.path.join(s2p, 'iscell.npy'))
            ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
            fs = float(ops['fs'])
            assert fs == FS
            # Track2p output: all ROIs are already classifier-accepted cells
            # tracked across every day of this mouse.
            assert np.all(iscell[:, 0] == 1)

            n_neurons, n_frames = F.shape
            dff = compute_dff(F, Fneu, fs)
            me = load_motion_energy(session_dir, n_frames)

            # keep only the frames that fill a complete 60 s trial
            n_trials = n_frames // FRAMES_PER_TRIAL
            n_keep = n_trials * FRAMES_PER_TRIAL

            neural = bin_time(dff[:, :n_keep])                        # (n_neurons, n_bins)
            # z-score each neuron over the session (see module docstring)
            neural = (neural - neural.mean(axis=1, keepdims=True)) / \
                     (neural.std(axis=1, keepdims=True) + 1e-9)
            neural = neural.astype(np.float32)
            behav = bin_time(me[:n_keep])                             # (n_bins,)
            n_bins = behav.shape[0]

            # motion energy -> quintiles, thresholds computed per session
            thresholds = np.percentile(behav, np.arange(1, N_QUANTILES) * (100 / N_QUANTILES))
            labels = np.searchsorted(thresholds, behav, side='left').astype(np.int64)

            # time elapsed since the start of the session, at the centre of each bin
            t = (np.arange(n_bins) + 0.5) * BIN_SECONDS

            neural_trials, input_trials, output_trials = [], [], []
            for tr in range(n_trials):
                sl = slice(tr * BINS_PER_TRIAL, (tr + 1) * BINS_PER_TRIAL)
                neural_trials.append(neural[:, sl])
                input_trials.append(t[sl][None, :].astype(np.float32))
                output_trials.append(labels[sl][None, :])

            data['neural'].append(neural_trials)
            data['input'].append(input_trials)
            data['output'].append(output_trials)
            data['subject_idx'].append(subj_i)
            data['brain_region_idx'].append(np.zeros(n_neurons, dtype=np.int64))
            session_info.append({
                'subject': subj,
                'date': os.path.basename(session_dir).replace('_a', ''),
                'day_index': day_i,          # 0 = first recording day of this mouse
                'n_neurons': int(n_neurons),
                'n_frames': int(n_frames),
                'n_trials': int(n_trials),
                'frame_rate_hz': fs,
                'motion_energy_quintile_thresholds': thresholds.tolist(),
            })
            print(f'{subj} {os.path.basename(session_dir)}: '
                  f'{n_neurons} neurons, {n_trials} trials, {n_bins} bins')

    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    data['metadata'] = {
        'task_description': (
            'Spontaneous behaviour in the dark: head-fixed neonatal mice (P7-P14) are free '
            'to move/run on a non-motorised treadmill under sensory-minimised conditions, '
            'with no task or stimulus. Two-photon calcium imaging (GCaMP8m, 30 Hz, layer 2/3 '
            'of barrel cortex, 720x720 um FOV) of neurons tracked across days with Track2p, '
            'simultaneously with 30 Hz videography of the mouse. The decoder predicts the '
            'motion energy of the mouse (sum of squared pixel-wise differences between '
            'consecutive video frames, a proxy for arousal/movement state), discretised into '
            '5 equal-percentile bins with thresholds computed within each session, from the '
            'neural activity and the time elapsed since the start of the session.'),
        'time_bin_size': 1000.0 * BIN_SECONDS,   # ms (10 frames at 30 Hz)
        'temporal_alignment_event': (
            'start of the recording session (first imaging frame); each session is cut into '
            'consecutive non-overlapping 60 s trials'),
        'off_start': 0.0,
        'off_end': TRIAL_SECONDS,
        'neural_signal': (
            'baseline-corrected fluorescence (the paper\'s "dF/F"): F - 0.7*Fneu - F0, with F0 '
            'the Suite2p maximin baseline (gaussian sigma 10 frames, 60 s min/max window); '
            'averaged in bins of 10 consecutive frames as in the paper decoding analyses, then '
            'z-scored per neuron within each session'),
        'output_processing': (
            'motion energy averaged in the same 10-frame bins, then discretised into quintiles '
            'using the 20/40/60/80th percentiles of that session'),
        'behaviour_alignment': (
            'the behaviour camera is triggered by the microscope, so camera frame i = imaging '
            'frame i; sessions with dropped camera frames are realigned using the camera '
            'timestamps and the few missing frames are linearly interpolated'),
        'species': 'mouse (GAD67-Cre, GCaMP8m + tdTomato)',
        'brain_region_detail': 'primary somatosensory (barrel) cortex, layer 2/3, 100-200 um deep',
        'reference': ('Majnik et al. 2025, "Longitudinal tracking of neuronal activity from the '
                      'same cells in the developing brain using Track2p", eLife 14:RP107540'),
        'session_info': session_info,
    }

    with open(OUT_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nwrote {OUT_FILE}: {len(data["neural"])} sessions, '
          f'{sum(len(s) for s in data["neural"])} trials')


if __name__ == '__main__':
    main()
