"""
Convert the Majnik et al. (2025) Track2p longitudinal barrel-cortex dataset into the
decoder format described in the task.

Dataset: daily 2-photon calcium imaging (30 Hz, layer 2/3 barrel cortex) of the same
Track2p-tracked neurons in 6 mice during the second postnatal week, together with
'motion energy' extracted from simultaneous videography of spontaneous behaviour.

Decoder task
    input : time elapsed from the beginning of the session (seconds), time-varying
    output: motion energy discretised into 5 equal-percentile (quintile) bins,
            computed per session, time-varying

Processing follows the paper / the track2p repository:
  * neural signal = 'dF/F' as defined in track2p (track2p/gui/data_management.py,
    F_processing): neuropil coefficient 0, maximin baseline correction with the
    Suite2p defaults (sig_baseline=10, win_baseline=60 s), i.e. F - F0.
  * for decoding, both the dF/F and the behaviour traces are denoised by averaging
    in bins of 10 consecutive frames (30 Hz -> 3 Hz, 333.33 ms bins).
  * all Track2p-tracked cells provided with the dataset are used (they are already
    curated: Suite2p iscell > 0.5 and matched across all days of a mouse).
"""

import os
import pickle

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d

DATA_DIR = '/app/data'
OUT_FILE = '/app/converted_data.pkl'

BIN_FRAMES = 10          # frames averaged together (as in the paper's decoding analysis)
TRIAL_SEC = 60.0         # trial ("block") length in seconds
N_OUTPUT_BINS = 5        # number of equal-percentile motion-energy bins


# ----------------------------------------------------------------------------------
# loading / processing
# ----------------------------------------------------------------------------------
def f_processing(F, Fneu, fs, neucoeff=0.0, baseline='maximin',
                 sig_baseline=10.0, win_baseline=60.0, prctile_baseline=8.0):
    """dF/F as computed by track2p (track2p/gui/data_management.py::F_processing).

    Baseline-corrected fluorescence with the default Suite2p baseline parameters.
    The track2p implementation uses neucoeff=0 (no neuropil subtraction), so the
    Fneu traces are loaded only for completeness.
    """
    Fc = F - neucoeff * Fneu

    win = int(win_baseline * fs)
    if baseline == 'maximin':
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    elif baseline == 'constant':
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = np.amin(Flow)
    elif baseline == 'constant_prctile':
        Flow = np.percentile(Fc, prctile_baseline, axis=1)
        Flow = np.expand_dims(Flow, axis=1)
    else:
        Flow = 0.

    return Fc - Flow


def load_neural(session_dir):
    """Load the Track2p-tracked traces of one session and return dF/F and ops."""
    plane = os.path.join(session_dir, 'suite2p', 'plane0')
    F = np.load(os.path.join(plane, 'F.npy')).astype(np.float64)
    Fneu = np.load(os.path.join(plane, 'Fneu.npy')).astype(np.float64)
    ops = np.load(os.path.join(plane, 'ops.npy'), allow_pickle=True).item()
    iscell = np.load(os.path.join(plane, 'iscell.npy'))

    # The distributed Track2p output only contains cells that passed the Suite2p
    # classifier (default threshold 0.5) on every day and were matched across days;
    # the check below makes that assumption explicit.
    assert np.all(iscell[:, 0] == 1), f'unexpected non-cell ROI in {session_dir}'

    dff = f_processing(F, Fneu, fs=ops['fs'])
    return dff, ops


def load_motion_energy(session_dir, nframes):
    """Motion energy resampled onto the imaging frame grid (length nframes).

    Videography was triggered by the 2-photon acquisition, so camera frames map
    one-to-one onto imaging frames. Some camera frames are dropped; these are
    located with the camera timestamps (gaps of ~k x the median inter-frame
    interval mean k-1 dropped frames) and linearly interpolated over, as
    suggested by the dataset README.
    """
    move_dir = os.path.join(session_dir, 'move_deve')
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
    tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))

    dt = np.median(np.diff(tstamps))
    n_missing = np.round(np.diff(tstamps) / dt).astype(int) - 1  # per inter-frame gap

    # index of each recorded camera frame on the (gap-filled) imaging frame grid
    idx = np.concatenate([[0], np.cumsum(1 + n_missing)])
    full = np.full(idx[-1] + 1, np.nan)
    full[idx] = me

    # linear interpolation over the dropped frames
    bad = np.isnan(full)
    if bad.any():
        full[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), full[~bad])

    # match the number of imaging frames (a few sessions have a handful of extra /
    # missing camera frames at the end of the recording)
    if len(full) > nframes:
        full = full[:nframes]
    elif len(full) < nframes:
        full = np.concatenate([full, np.full(nframes - len(full), full[-1])])

    return full


def bin_average(x, bin_frames):
    """Average consecutive bins of `bin_frames` samples along the last axis.

    Leftover frames at the end that do not fill a whole bin are dropped.
    """
    x = np.asarray(x)
    nbins = x.shape[-1] // bin_frames
    x = x[..., :nbins * bin_frames]
    return x.reshape(*x.shape[:-1], nbins, bin_frames).mean(axis=-1)


def discretize_percentile(x, nbins):
    """Discretise into `nbins` equal-percentile bins (values 0 .. nbins-1)."""
    edges = np.percentile(x, np.linspace(0, 100, nbins + 1)[1:-1])
    return np.digitize(x, edges).astype(np.int64)


# ----------------------------------------------------------------------------------
# main conversion
# ----------------------------------------------------------------------------------
def main():
    subjects = sorted(d for d in os.listdir(DATA_DIR)
                      if os.path.isdir(os.path.join(DATA_DIR, d)))

    neural, inputs, outputs = [], [], []
    subject_idx, brain_region_idx, session_info = [], [], []

    for si, subject in enumerate(subjects):
        subject_dir = os.path.join(DATA_DIR, subject)
        sessions = sorted(d for d in os.listdir(subject_dir)
                          if os.path.isdir(os.path.join(subject_dir, d)))

        for session in sessions:
            session_dir = os.path.join(subject_dir, session)

            dff, ops = load_neural(session_dir)
            fs = float(ops['fs'])
            nframes = dff.shape[1]
            assert nframes == ops['nframes']

            me = load_motion_energy(session_dir, nframes)

            # denoise by averaging in bins of 10 frames (30 Hz -> 3 Hz)
            dff_b = bin_average(dff, BIN_FRAMES)
            me_b = bin_average(me, BIN_FRAMES)
            nbins = dff_b.shape[1]

            # time of the centre of each bin, in seconds from the start of the session
            t = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs

            # motion energy -> quintile bins, thresholds chosen per session
            me_cat = discretize_percentile(me_b, N_OUTPUT_BINS)

            # split the session into consecutive 60 s trials
            bins_per_trial = int(round(TRIAL_SEC * fs / BIN_FRAMES))
            ntrials = nbins // bins_per_trial

            sess_neural, sess_input, sess_output = [], [], []
            for tr in range(ntrials):
                sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
                sess_neural.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
                sess_input.append(np.ascontiguousarray(t[None, sl], dtype=np.float32))
                sess_output.append(np.ascontiguousarray(me_cat[None, sl]))

            neural.append(sess_neural)
            inputs.append(sess_input)
            outputs.append(sess_output)
            subject_idx.append(si)
            brain_region_idx.append(np.zeros(dff.shape[0], dtype=np.int64))
            session_info.append({
                'subject': subject,
                'date': session.rstrip('_a').rstrip('_'),
                'session_dir': os.path.relpath(session_dir, DATA_DIR),
                'n_neurons': int(dff.shape[0]),
                'n_frames': int(nframes),
                'fs': fs,
                'duration_s': nframes / fs,
                'n_trials': ntrials,
                'motion_energy_bin_edges': np.percentile(
                    me_b, np.linspace(0, 100, N_OUTPUT_BINS + 1)[1:-1]).tolist(),
            })
            print(f'{subject} {session}: {dff.shape[0]} neurons, {ntrials} trials, '
                  f'{nbins} bins')

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
        'brain_regions': ['S1 barrel cortex'],
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_session_start_s'],
        'output_names': ['motion_energy_quintile'],
        'output_values': [[f'quintile_{i + 1}' for i in range(N_OUTPUT_BINS)]],
        'metadata': {
            'task_description': (
                'Head-fixed neonatal mice (P7-P14) run spontaneously on a non-motorised '
                'treadmill in the dark under sensory-minimised conditions; there is no '
                'imposed task. Layer 2/3 barrel-cortex neurons are imaged with 2-photon '
                'calcium imaging (GCaMP8m, 30 Hz) and the same cells are tracked across '
                'days with Track2p. Spontaneous movement is quantified from simultaneous '
                'videography as motion energy (summed squared pixel-wise difference '
                'between consecutive frames). The decoder predicts the motion energy, '
                'discretised into 5 equal-percentile bins (thresholds computed per '
                'session), from the neural activity and the time elapsed since the start '
                'of the session.'),
            'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,  # 333.33 ms
            'temporal_alignment_event': (
                'start of each 60 s block of the continuous recording; blocks tile the '
                'session from imaging onset (frame 0 of the 2-photon acquisition, which '
                'also triggers the behaviour camera)'),
            'off_start': 0.0,
            'off_end': TRIAL_SEC,
            'neural_signal': (
                "dF/F: baseline-corrected fluorescence as computed by track2p "
                "(F_processing), maximin baseline with the Suite2p defaults "
                "(neuropil coefficient 0, sig_baseline=10, win_baseline=60 s), then "
                "averaged in bins of 10 frames as in the paper's decoding analysis"),
            'behaviour_signal': (
                'global motion energy from videography, resampled onto the imaging frame '
                'grid (dropped camera frames interpolated using the camera timestamps) '
                'and averaged in bins of 10 frames'),
            'sampling_rate_hz': 30.0 / BIN_FRAMES,
            'acquisition_rate_hz': 30.0,
            'trial_length_s': TRIAL_SEC,
            'species': 'mouse (GAD67-Cre, P7-P14)',
            'recording_modality': '2-photon calcium imaging (GCaMP8m), layer 2/3',
            'neurons_tracked_across_sessions': (
                'within a subject, neuron i is the same Track2p-tracked cell in every '
                'session'),
            'reference': ('Majnik et al. (2025) Longitudinal tracking of neuronal '
                          'activity from the same cells in the developing brain using '
                          'Track2p. eLife 14:RP107540'),
            'session_info': session_info,
        },
    }

    with open(OUT_FILE, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nSaved {len(neural)} sessions to {OUT_FILE}')


if __name__ == '__main__':
    main()
