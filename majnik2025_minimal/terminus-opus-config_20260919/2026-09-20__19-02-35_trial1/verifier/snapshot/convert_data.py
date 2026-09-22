"""
Convert the Track2p developmental barrel cortex dataset (Majnik et al. 2025, eLife)
into the standard decoder data format.

Decoder task: decode the animal's motion energy (arousal/movement proxy from
videography) from the calcium activity of neurons tracked with Track2p in mouse
barrel cortex (L2/3), using 60 s trials.

Processing follows the paper / track2p repository:
  - Neural data: Suite2p F.npy of the Track2p-tracked cells (already curated,
    iscell == 1 for all provided ROIs; we still apply the iscell > 0.5 criterion
    used in the paper).  dF/F = baseline-corrected fluorescence obtained with the
    Suite2p 'maximin' baseline with default parameters (win_baseline = 60 s,
    sig_baseline = 10 frames), exactly as implemented in
    track2p/gui/data_management.py :: F_processing (which is called without
    neuropil subtraction, neucoeff = 0).
  - Both neural and behavioural traces are denoised by averaging bins of 10
    consecutive frames (30 Hz -> 3 Hz), as done for all decoding analyses in the
    paper.
  - Motion energy: pixel-wise squared difference of consecutive videography
    frames.  The camera was triggered by the 2p microscope, so there is a 1:1
    correspondence between camera and imaging frames.  When camera frames are
    dropped (len(motion_energy) < n imaging frames) the dropped frames are
    located using the interframe intervals in tstamps.npy and linearly
    interpolated, as suggested in the dataset README.
"""

import os
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter1d, maximum_filter1d

DATA_DIR = '/app/data'
OUT_FILE = '/app/converted_data.pkl'

BIN_FRAMES = 10          # denoising bin (paper: average of 10 consecutive timestamps)
TRIAL_SEC = 60.0         # trial length in seconds
N_OUT_BINS = 5           # number of equal-percentile motion-energy bins


def maximin_dff(F, Fneu, ops):
    """Baseline-corrected fluorescence (dF/F), as in track2p F_processing /
    Suite2p defaults: maximin baseline."""
    fs = float(ops['fs'])
    neucoeff = 0.0  # track2p F_processing is called without neuropil subtraction
    sig_baseline = float(ops.get('sig_baseline', 10.0))
    win_baseline = float(ops.get('win_baseline', 60.0))
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter(Fc, [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    return Fc - Flow


def bin_average(x, binsize):
    """Average non-overlapping bins of binsize samples along the last axis."""
    x = np.asarray(x, dtype=np.float64)
    n = (x.shape[-1] // binsize) * binsize
    x = x[..., :n]
    newshape = x.shape[:-1] + (n // binsize, binsize)
    return x.reshape(newshape).mean(axis=-1)


def interp_nans(x):
    """Linearly interpolate NaNs (and extrapolate at edges with nearest value)."""
    x = np.asarray(x, dtype=np.float64).copy()
    nans = np.isnan(x)
    if nans.all():
        raise ValueError('all values missing')
    if nans.any():
        idx = np.arange(x.size)
        x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x


def load_motion_energy(session_dir, nframes):
    """Motion energy aligned to the imaging frames (length nframes)."""
    me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
    ts = np.load(os.path.join(session_dir, 'move_deve', 'tstamps.npy')).astype(np.float64)

    full = np.full(nframes, np.nan)
    if me.size == nframes:
        full[:] = me
    else:
        # locate dropped camera frames from the interframe intervals
        d = np.diff(ts)
        med = np.median(d)
        steps = np.round(d / med).astype(int)
        steps[steps < 1] = 1
        idx = np.concatenate([[0], np.cumsum(steps)])
        keep = idx < nframes
        full[idx[keep]] = me[keep]
    # first sample has no preceding frame -> not a valid motion energy value
    full[0] = np.nan
    return interp_nans(full)


def main():
    subjects = sorted([d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, d))])

    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': subjects, 'subject_idx': [],
        'brain_regions': ['S1 barrel cortex'], 'brain_region_idx': [],
        'input_names': ['time from session start (s)'],
        'output_names': ['motion energy'],
        'output_values': [['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']],
        'metadata': {},
    }
    session_info = []

    for si, subject in enumerate(subjects):
        subj_dir = os.path.join(DATA_DIR, subject)
        sessions = sorted([d for d in os.listdir(subj_dir)
                           if os.path.isdir(os.path.join(subj_dir, d))])
        for sess in sessions:
            sdir = os.path.join(subj_dir, sess)
            ops = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'ops.npy'),
                          allow_pickle=True).item()
            fs = float(ops['fs'])
            F = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'F.npy'))
            Fneu = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'Fneu.npy'))
            iscell = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'iscell.npy'))
            keep = iscell[:, 0] > 0.5
            F, Fneu = F[keep], Fneu[keep]
            nframes = F.shape[1]

            dff = maximin_dff(F.astype(np.float64), Fneu.astype(np.float64), ops)
            neural = bin_average(dff, BIN_FRAMES).astype(np.float32)  # (n_neurons, T)

            me = load_motion_energy(sdir, nframes)
            me_b = bin_average(me, BIN_FRAMES)

            T = min(neural.shape[1], me_b.size)
            neural, me_b = neural[:, :T], me_b[:T]

            # discretize motion energy into 5 equal-percentile bins (per session)
            edges = np.quantile(me_b, np.arange(1, N_OUT_BINS) / N_OUT_BINS)
            me_cat = np.digitize(me_b, edges).astype(np.int64)

            # time (s) at the centre of each bin, from the start of the session
            bin_size = BIN_FRAMES / fs
            tvec = (np.arange(T) + 0.5) * bin_size

            bins_per_trial = int(round(TRIAL_SEC / bin_size))
            ntrials = T // bins_per_trial

            neural_trials, input_trials, output_trials = [], [], []
            for tr in range(ntrials):
                sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
                neural_trials.append(np.ascontiguousarray(neural[:, sl]))
                input_trials.append(tvec[sl][None, :].astype(np.float32))
                output_trials.append(me_cat[sl][None, :])

            data['neural'].append(neural_trials)
            data['input'].append(input_trials)
            data['output'].append(output_trials)
            data['subject_idx'].append(si)
            data['brain_region_idx'].append(np.zeros(neural.shape[0], dtype=np.int64))
            session_info.append({'subject': subject, 'session': sess,
                                 'n_neurons': int(neural.shape[0]),
                                 'n_trials': int(ntrials),
                                 'fs_imaging_hz': fs})
            print(f'{subject} {sess}: {neural.shape[0]} neurons, {ntrials} trials '
                  f'of {bins_per_trial} bins')

    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    data['metadata'] = {
        'task_description': (
            'Spontaneous activity in L2/3 of mouse barrel cortex during the second '
            'postnatal week (P7-P14), recorded with 2-photon calcium imaging (GCaMP8m, '
            '30 Hz, 720x720 um FOV) while head-fixed pups were free to move on a '
            'non-motorised treadmill in the dark without sensory stimulation. Neurons '
            'were tracked across days with Track2p. The decoder predicts the animal\'s '
            'motion energy (videography-based movement/arousal proxy), discretized into '
            '5 equal-percentile (quintile) bins computed per session, from the neural '
            'activity and the time elapsed since the beginning of the session. '
            'Recordings are cut into consecutive 60 s trials.'),
        'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,  # ms (10 frames at 30 Hz)
        'temporal_alignment_event': 'start of the imaging session (first 2-photon frame)',
        'off_start': 0.0,
        'off_end': 60.0,
        'neural_data_type': ('baseline-corrected fluorescence (dF/F, Suite2p maximin '
                             'baseline, win_baseline=60 s, sig_baseline=10 frames), '
                             'averaged in bins of 10 consecutive frames (30 Hz -> 3 Hz)'),
        'output_discretization': ('motion energy binned in the same 10-frame bins and '
                                  'discretized into 5 equal-percentile bins using the '
                                  'quantiles of that session'),
        'trial_definition': ('recordings split into consecutive non-overlapping 60 s '
                             'trials; incomplete trials at the end are discarded'),
        'session_info': session_info,
        'paper': ('Majnik et al. 2025, Longitudinal tracking of neuronal activity from '
                  'the same cells in the developing brain using Track2p, eLife 14:RP107540'),
    }

    with open(OUT_FILE, 'wb') as f:
        pickle.dump(data, f)
    print('saved', OUT_FILE)


if __name__ == '__main__':
    main()
