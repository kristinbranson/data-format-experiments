"""
Convert the Majnik et al. 2025 (Track2p, eLife 14:RP107540) longitudinal
two-photon dataset into the decoder-ready dictionary format.

Dataset
-------
6 mice (jm031, jm032, jm038, jm039, jm040, jm046), each imaged daily for 6-7
consecutive days during the second postnatal week.  Layer 2/3 of barrel cortex
(S1), 720x720 um FOV, resonant scanner at ~30 Hz, sessions of 20 min (36000
frames) or 30 min (54000 frames).  The suite2p folders shipped with the dataset
are Track2p outputs: they already contain *only* the neurons that were
successfully tracked on every day of that mouse, row-matched across sessions,
and all of them already passed the suite2p iscell criterion (prob > 0.5).

Behaviour is "motion energy": the sum of squared pixel-wise differences between
consecutive frames of the behaviour video, which was hardware-triggered by the
microscope so that camera frame i == imaging frame i (up to dropped camera
frames, which are recovered from the camera timestamps).

Processing choices (follow the paper / the Track2p repo)
--------------------------------------------------------
* Neural signal: "baseline corrected fluorescence traces as our dF/F (using the
  default Suite2p parameters)".  Implemented exactly as
  ``track2p/gui/data_management.py::F_processing``: no neuropil subtraction
  (neucoeff=0), maximin baseline (gaussian sigma 10 frames, min/max filter over
  a 60 s window), dF = Fc - Flow.
* "For all decoding analysis we slightly denoised the dF/F as well as the
  behaviour traces by averaging in bins of 10 consecutive timestamps."  -> both
  neural and motion energy are averaged in non-overlapping bins of 10 imaging
  frames (~336 ms).
* No further neuron / session / mouse curation: the released data is already the
  curated set used for all analyses in the paper (Fig. 5 onwards).

Decoder-task specific choices
-----------------------------
* Sessions are cut into consecutive, non-overlapping 60 s trials = 1800 imaging
  frames = 180 binned timepoints (20 trials for a 20 min session, 30 for a
  30 min session).
* Input: time elapsed since the start of the session, in seconds (from the
  camera/microscope timestamps).
* Output: motion energy discretised into 5 equal-percentile (quintile) bins,
  with the bin edges computed separately for each session.

Writes /app/converted_data.pkl.
"""

import os
import pickle

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d

DATA_DIR = '/app/data'
OUT_FILE = '/app/converted_data.pkl'

FRAMES_PER_BIN = 10       # paper: denoise by averaging 10 consecutive timestamps
TRIAL_SECONDS = 60.0      # task specification: 60 s trials
NOMINAL_FS = 30.0         # Hz, resonant scanner (ops['fs'])
BINS_PER_TRIAL = int(round(TRIAL_SECONDS * NOMINAL_FS / FRAMES_PER_BIN))  # 180
N_OUTPUT_BINS = 5         # quintiles of motion energy

BRAIN_REGION = 'S1'       # barrel cortex, layer 2/3


def f_processing(F, Fneu, fs, neucoeff=0.0, sig_baseline=10.0, win_baseline=60.0):
    """dF/F as computed by Track2p (track2p/gui/data_management.py::F_processing),
    i.e. suite2p's default maximin baseline correction."""
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter(Fc, [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    return Fc - Flow


def camera_frame_index(tstamps, n_frames):
    """Map each recorded camera frame onto its imaging-frame index.

    The camera was triggered by the microscope, so nominally camera frame i is
    imaging frame i.  When the camera drops frames the timestamp trace shows
    gaps that are exact multiples of the frame interval; the number of dropped
    frames in each gap recovers the true index.
    """
    ncam = len(tstamps)
    if ncam == n_frames:
        # nothing dropped
        return np.arange(n_frames)
    ifi = np.diff(tstamps)
    dt = np.median(ifi)
    n_missing = np.round(ifi / dt).astype(int) - 1
    idx = np.concatenate([[0], np.cumsum(1 + n_missing)])
    assert idx[-1] == n_frames - 1, 'could not reconstruct dropped camera frames'
    assert n_missing.sum() == n_frames - ncam
    return idx


def interp_nan(x):
    """Linear interpolation over NaNs (edges filled with nearest valid value)."""
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x)
    if bad.all():
        raise ValueError('all values missing')
    t = np.arange(len(x))
    x[bad] = np.interp(t[bad], t[~bad], x[~bad])
    return x


def bin_mean(x, k):
    """Average non-overlapping bins of k samples along the last axis."""
    n = (x.shape[-1] // k) * k
    x = x[..., :n]
    return x.reshape(*x.shape[:-1], n // k, k).mean(axis=-1)


def load_session(session_dir):
    """Return binned dF/F (n_neurons, n_bins), bin time in s, binned motion energy."""
    s2p = os.path.join(session_dir, 'suite2p', 'plane0')
    mov = os.path.join(session_dir, 'move_deve')

    F = np.load(os.path.join(s2p, 'F.npy'))
    Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
    iscell = np.load(os.path.join(s2p, 'iscell.npy'))
    fs = float(ops['fs'])
    n_frames = F.shape[1]

    # The released Track2p output only contains tracked cells and they all pass
    # suite2p's default iscell threshold of 0.5; assert rather than filter.
    assert np.all(iscell[:, 0] == 1), 'unexpected non-cell ROI in Track2p output'

    dff = f_processing(F.astype(np.float64), Fneu.astype(np.float64), fs)

    # --- behaviour, put back onto the imaging frame grid -------------------
    me_cam = np.load(os.path.join(mov, 'motion_energy_glob.npy')).astype(np.float64)
    # tstamps are in units of 1000 s; convert to seconds
    ts_cam = np.load(os.path.join(mov, 'tstamps.npy')).astype(np.float64) * 1000.0
    assert len(me_cam) == len(ts_cam)

    idx = camera_frame_index(ts_cam, n_frames)

    me = np.full(n_frames, np.nan)
    me[idx] = me_cam
    # motion energy of the very first frame is 0 by construction (no preceding
    # frame to difference against): treat it as missing
    me[0] = np.nan
    me = interp_nan(me)

    tstamps = np.full(n_frames, np.nan)
    tstamps[idx] = ts_cam
    tstamps = interp_nan(tstamps)

    # --- denoise both streams by averaging 10 consecutive frames -----------
    dff_b = bin_mean(dff, FRAMES_PER_BIN)
    me_b = bin_mean(me, FRAMES_PER_BIN)
    t_b = bin_mean(tstamps, FRAMES_PER_BIN)  # bin-centre time, seconds

    return dff_b.astype(np.float32), t_b, me_b, float(np.median(np.diff(tstamps)))


def discretize_quintiles(x, nbins=N_OUTPUT_BINS):
    """Assign each sample to one of `nbins` equal-percentile bins of x."""
    edges = np.percentile(x, np.linspace(0, 100, nbins + 1)[1:-1])
    return np.digitize(x, edges).astype(np.int64)


def main():
    subjects = sorted(d for d in os.listdir(DATA_DIR)
                      if os.path.isdir(os.path.join(DATA_DIR, d)))

    neural, inputs, outputs = [], [], []
    subject_idx, brain_region_idx, session_info = [], [], []
    bin_durations = []

    for si, subj in enumerate(subjects):
        subj_dir = os.path.join(DATA_DIR, subj)
        sessions = sorted(d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d)))
        for di, sess in enumerate(sessions):
            dff, t, me, dt = load_session(os.path.join(subj_dir, sess))
            bin_durations.append(dt * FRAMES_PER_BIN)

            cls = discretize_quintiles(me)

            n_bins = dff.shape[1]
            n_trials = n_bins // BINS_PER_TRIAL
            assert n_trials >= 2

            ntr, ninp, nout = [], [], []
            for k in range(n_trials):
                sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
                ntr.append(np.ascontiguousarray(dff[:, sl]))
                ninp.append(t[sl][None, :].astype(np.float32))
                nout.append(cls[sl][None, :])

            neural.append(ntr)
            inputs.append(ninp)
            outputs.append(nout)
            subject_idx.append(si)
            brain_region_idx.append(np.zeros(dff.shape[0], dtype=np.int64))
            session_info.append({
                'subject': subj,
                'date': sess.split('_')[0],
                'day_index': di,              # 0 = first recording day of this mouse
                'n_neurons': int(dff.shape[0]),
                'n_trials': int(n_trials),
                'session_duration_s': float(t[-1]),
                'motion_energy_quintile_edges': [
                    float(v) for v in np.percentile(
                        me, np.linspace(0, 100, N_OUTPUT_BINS + 1)[1:-1])],
            })
            print(f'{subj} {sess}: {dff.shape[0]} neurons, {n_trials} trials')

    time_bin_size_ms = float(np.mean(bin_durations) * 1000.0)

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
        'brain_regions': [BRAIN_REGION],
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_session_start_s'],
        'output_names': ['motion_energy_quintile'],
        'output_values': [['q1 (0-20%)', 'q2 (20-40%)', 'q3 (40-60%)',
                           'q4 (60-80%)', 'q5 (80-100%)']],
        'metadata': {
            'task_description':
                'Head-fixed neonatal mice (P7-P14) run spontaneously on a '
                'non-motorised treadmill in the dark under sensory-minimised '
                'conditions; there is no imposed task or stimulus. Decode the '
                'animal\'s movement (motion energy from behavioural videography, '
                'discretised into five per-session equal-percentile bins) from '
                'the activity of the same Track2p-tracked layer 2/3 barrel '
                'cortex neurons.',
            'time_bin_size': time_bin_size_ms,
            'temporal_alignment_event':
                'start of the 60 s trial block; each session is cut into '
                'consecutive non-overlapping 1800-frame (60 s) blocks starting '
                'at imaging onset',
            'off_start': 0.0,
            'off_end': float(np.mean(bin_durations) * BINS_PER_TRIAL),
            'paper': 'Majnik et al. 2025, eLife 14:RP107540 '
                     '(Track2p longitudinal tracking)',
            'recording': 'two-photon calcium imaging (GCaMP8m), layer 2/3 of '
                         'barrel cortex, 720x720 um FOV, ~30 Hz resonant scanner',
            'neural_signal': 'baseline-corrected fluorescence (suite2p default '
                             'maximin baseline, sigma=10 frames, 60 s window, no '
                             'neuropil subtraction), averaged in bins of 10 frames',
            'behavior_signal': 'global motion energy (sum of squared pixel-wise '
                               'differences between consecutive video frames), '
                               'realigned to the imaging frame grid via the camera '
                               'timestamps, dropped frames linearly interpolated, '
                               'averaged in bins of 10 frames',
            'neuron_tracking': 'neurons are row-matched across all sessions of a '
                               'given mouse (Track2p output); session order within '
                               'a mouse is chronological',
            'session_info': session_info,
        },
    }

    with open(OUT_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nwrote {OUT_FILE}: {len(neural)} sessions, '
          f'{sum(len(s) for s in neural)} trials, '
          f'time_bin_size {time_bin_size_ms:.2f} ms')


if __name__ == '__main__':
    main()
