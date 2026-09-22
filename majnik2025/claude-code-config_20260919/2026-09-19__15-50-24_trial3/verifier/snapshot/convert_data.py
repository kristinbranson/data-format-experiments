#!/usr/bin/env python3
"""
Convert the Majnik et al. 2025 (Track2p, eLife 14:RP107540) longitudinal barrel-cortex
2-photon dataset into the decoder-compatible pickle format.

Usage
-----
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Options
-------
    --full            process all sessions (default)
    --sample          process only 2 sessions (for quick testing)
    --show-processing plot every processing step for up to 2 sessions to
                      processing_<session_id>.png

Decoder task
------------
    input  : time elapsed from the beginning of the session, in seconds (time-varying)
    output : motion energy discretised into 5 equal-percentile (quintile) bins,
             percentiles computed per session (time-varying)
    trials : consecutive, non-overlapping 60 s blocks of the session

Processing (matches the reference paper/code, see CONVERSION_NOTES.md)
----------------------------------------------------------------------
    neural  : F.npy (already restricted to Track2p-tracked, iscell>0.5 ROIs)
              -> dF/F = F - maximin_baseline(F)      [track2p/gui/data_management.py:185]
              -> non-overlapping mean of 10 frames   [Methods, "Decoding"]
    motion  : motion_energy_glob.npy mapped onto imaging frames (dropped camera frames
              become NaN) -> non-overlapping nan-mean of 10 frames -> per-session quintiles
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter1d, maximum_filter1d

# ----------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------
DATA_ROOT = '/app/data'

FS = 30.0                    # imaging / camera rate (Hz); ops['fs'] == 30 for every session
BIN_FRAMES = 10              # "averaging in bins of 10 consecutive timestamps" (Methods)
BIN_SIZE_S = BIN_FRAMES / FS         # 0.3333... s
BIN_SIZE_MS = BIN_SIZE_S * 1000.0    # 333.33 ms
TRIAL_LEN_S = 60.0                   # "Split sessions into 60-second trials"
BINS_PER_TRIAL = int(round(TRIAL_LEN_S / BIN_SIZE_S))   # 180

N_QUANTILES = 5              # "discretized into five equal-percentile bins"

# Suite2p default baseline parameters (track2p GUI F_processing defaults)
NEUCOEFF = 0.0               # reference code calls F_processing without neucoeff -> 0.0
SIG_BASELINE = 10.0          # frames
WIN_BASELINE = 60.0          # seconds

# Mouse id -> letter used in the paper (alphabetical order, see data/README.md) and the
# postnatal day of the first recording day (Fig. 5B of the paper).
SUBJECT_INFO = {
    'jm031': ('A', 7),
    'jm032': ('B', 7),
    'jm038': ('C', 8),
    'jm039': ('D', 8),
    'jm040': ('E', 9),
    'jm046': ('F', 8),
}

BRAIN_REGION = 'S1'          # barrel cortex, layer 2/3


# ----------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------
def list_sessions(data_root=DATA_ROOT):
    """Return [(subject, session_name, session_dir), ...] in subject/chronological order."""
    sessions = []
    for subject in sorted(SUBJECT_INFO.keys()):
        subj_dir = os.path.join(data_root, subject)
        for sess_dir in sorted(f.path for f in os.scandir(subj_dir) if f.is_dir()):
            sessions.append((subject, os.path.basename(sess_dir), sess_dir))
    return sessions


def load_traces(session_dir):
    """Raw fluorescence of the tracked cells, (n_neurons, n_frames).

    Same as `load_traces` in the dataset's own load_data.ipynb.
    """
    return np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))


def load_fs(session_dir):
    """Imaging rate from ops.npy."""
    ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'),
                  allow_pickle=True).item()
    return float(ops['fs'])


# ----------------------------------------------------------------------------------
# Neural processing
# ----------------------------------------------------------------------------------
def f_processing(F, Fneu=None, fs=FS, neucoeff=NEUCOEFF, baseline='maximin',
                 sig_baseline=SIG_BASELINE, win_baseline=WIN_BASELINE,
                 prctile_baseline=8.0, return_baseline=False):
    """dF/F as computed by the reference code.

    Verbatim port of `DataManagement.F_processing`
    (/app/code/track2p/gui/data_management.py:185), which is the reference
    implementation of the paper's "baseline corrected fluorescence traces ... (using
    the default Suite2p parameters)".
    """
    Fc = F.astype(np.float32, copy=True)
    if neucoeff:
        Fc = Fc - neucoeff * Fneu.astype(np.float32)

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

    dff = Fc - Flow
    if return_baseline:
        return dff, Flow
    return dff


def bin_mean(x, bin_frames=BIN_FRAMES, nan_aware=False):
    """Non-overlapping mean over the last axis in blocks of `bin_frames` samples.

    Trailing samples that do not fill a whole bin are dropped.
    """
    n = x.shape[-1] // bin_frames
    v = x[..., :n * bin_frames].reshape(*x.shape[:-1], n, bin_frames)
    if nan_aware:
        return np.nanmean(v, axis=-1)
    return v.mean(axis=-1)


# ----------------------------------------------------------------------------------
# Behaviour processing
# ----------------------------------------------------------------------------------
def motion_energy_on_imaging_frames(session_dir, n_frames):
    """Motion energy resampled onto the imaging frame grid, NaN where unavailable.

    The camera is hardware triggered by the 2-photon acquisition (Methods: "with the
    microscope acquisition acting as a trigger for camera frame acquisition"), so camera
    frame k corresponds to imaging frame k unless camera frames were dropped.  The
    dataset README states dropped frames can be recovered from tstamps.npy; a dropped
    frame shows up as an inter-frame interval of ~2x the median.

    Returns
    -------
    me   : (n_frames,) float64, NaN at imaging frames with no camera frame
    info : dict with diagnostics
    """
    me_raw = np.load(os.path.join(session_dir, 'move_deve',
                                  'motion_energy_glob.npy')).astype(np.float64)
    n_cam = len(me_raw)

    if n_cam > n_frames:
        # Never happens in this dataset; guard so that extra trailing camera frames
        # (acquisition stopped after the microscope) cannot shift the alignment.
        print(f'    WARNING: {session_dir}: {n_cam - n_frames} more camera frames than '
              f'imaging frames; truncating the trailing ones', flush=True)
        me_raw = me_raw[:n_frames]
        n_cam = n_frames

    if n_cam == n_frames:
        # No frames missing: one-to-one correspondence.
        frame_idx = np.arange(n_frames)
        n_missing = 0
        max_gap = 1
    else:
        ts = np.load(os.path.join(session_dir, 'move_deve', 'tstamps.npy'))
        d = np.diff(ts)
        step = np.median(d)
        # number of *extra* frame slots spanned by each inter-frame interval
        n_steps = np.round(d / step).astype(int)
        frame_idx = np.concatenate([[0], np.cumsum(n_steps)])
        n_missing = int(frame_idx[-1] + 1 - n_cam)
        max_gap = int(n_steps.max())
        if frame_idx[-1] + 1 != n_frames:
            raise ValueError(
                f'{session_dir}: reconstructed camera frame indices span '
                f'{frame_idx[-1] + 1} frames but there are {n_frames} imaging frames')

    me = np.full(n_frames, np.nan, dtype=np.float64)
    me[frame_idx] = me_raw

    # motion_energy_glob[0] is identically 0 in every session: there is no frame -1 to
    # difference against, so it is an artefact rather than a measurement.
    me[0] = np.nan

    info = {'n_camera_frames': n_cam, 'n_missing_frames': n_missing,
            'max_gap_frames': max_gap,
            'frac_valid': float(np.mean(~np.isnan(me)))}
    return me, info


def discretize_quantiles(x, n_quantiles=N_QUANTILES):
    """Assign each value of `x` to one of `n_quantiles` equal-percentile bins.

    Percentile edges are computed from `x` itself (i.e. per session).  Returns int labels
    in [0, n_quantiles-1] and the edges used.
    """
    edges = np.percentile(x, np.arange(1, n_quantiles) * (100.0 / n_quantiles))
    labels = np.searchsorted(edges, x, side='right').astype(np.int64)
    return labels, edges


# ----------------------------------------------------------------------------------
# Per-session conversion
# ----------------------------------------------------------------------------------
def convert_session(session_dir, keep_neurons=None, verbose=True):
    """Convert one session.

    Returns a dict with the per-trial neural/input/output lists and diagnostics.
    """
    t0 = time.time()

    F = load_traces(session_dir)
    fs = load_fs(session_dir)
    if fs != FS:
        raise ValueError(f'{session_dir}: unexpected fs={fs}')
    n_neurons_all, n_frames = F.shape

    if keep_neurons is not None:
        F = F[keep_neurons]
    n_neurons = F.shape[0]
    t_load = time.time() - t0

    # ---- neural: dF/F then 10-frame binning -------------------------------------
    t1 = time.time()
    dff, baseline = f_processing(F, fs=fs, return_baseline=True)
    dff_binned = bin_mean(dff).astype(np.float32)            # (n_neurons, n_bins)
    t_neural = time.time() - t1

    # ---- behaviour: motion energy on imaging frames, then 10-frame binning -------
    t2 = time.time()
    me_frames, me_info = motion_energy_on_imaging_frames(session_dir, n_frames)
    me_binned = bin_mean(me_frames, nan_aware=True)           # (n_bins,)
    if np.any(np.isnan(me_binned)):
        # A whole 10-frame bin without a single camera frame. Does not occur in this
        # dataset (all gaps are single frames) but handle it rather than crash.
        bad = np.isnan(me_binned)
        good = ~bad
        me_binned[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good),
                                   me_binned[good])
        me_info['n_interpolated_bins'] = int(bad.sum())
    else:
        me_info['n_interpolated_bins'] = 0

    me_labels, me_edges = discretize_quantiles(me_binned)
    t_behav = time.time() - t2

    # ---- trials: consecutive non-overlapping 60 s blocks -------------------------
    n_bins = dff_binned.shape[1]
    assert me_binned.shape[0] == n_bins
    n_trials = n_bins // BINS_PER_TRIAL
    n_dropped_bins = n_bins - n_trials * BINS_PER_TRIAL

    # time (s) elapsed from session start at the centre of each bin
    bin_centre_s = (np.arange(n_bins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs

    neural, inputs, outputs = [], [], []
    for k in range(n_trials):
        sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
        neural.append(np.ascontiguousarray(dff_binned[:, sl]))
        inputs.append(bin_centre_s[sl][None, :].astype(np.float32))
        outputs.append(me_labels[sl][None, :].astype(np.int64))

    out = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'n_neurons': n_neurons,
        'n_neurons_all': n_neurons_all,
        'n_frames': n_frames,
        'n_bins': n_bins,
        'n_trials': n_trials,
        'n_dropped_bins': n_dropped_bins,
        'me_edges': me_edges,
        'me_info': me_info,
        'class_fractions': np.bincount(me_labels, minlength=N_QUANTILES) / n_bins,
        'timing': {'load': t_load, 'neural': t_neural, 'behaviour': t_behav,
                   'total': time.time() - t0},
        # kept only for --show-processing
        '_raw': {'F': F, 'baseline': baseline, 'dff': dff, 'dff_binned': dff_binned,
                 'me_frames': me_frames, 'me_binned': me_binned,
                 'me_labels': me_labels, 'bin_centre_s': bin_centre_s},
    }
    if verbose:
        print(f'    {os.path.basename(session_dir)}: {n_neurons} neurons, '
              f'{n_frames} frames -> {n_bins} bins -> {n_trials} trials | '
              f'missing camera frames: {me_info["n_missing_frames"]} '
              f'({100 * (1 - me_info["frac_valid"]):.2f}% of frames) | '
              f'class fractions {np.round(out["class_fractions"], 3)} | '
              f'{out["timing"]["total"]:.2f}s', flush=True)
    return out


def find_bad_neurons(subject_sessions):
    """Boolean mask of tracked ROIs that are identically zero on at least one day.

    Such ROIs fell outside the imaged FOV on that day; Suite2p returns an all-zero trace.
    They are dropped for every session of the subject so that the tracked population
    stays matched across days.
    """
    bad = None
    for _, _, sess_dir in subject_sessions:
        F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')
        b = np.asarray(F).std(axis=1) == 0
        bad = b if bad is None else (bad | b)
    return bad


# ----------------------------------------------------------------------------------
# Plotting of the processing steps
# ----------------------------------------------------------------------------------
def plot_processing(session_id, res, outdir='/app'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    raw = res['_raw']
    F, base, dff = raw['F'], raw['baseline'], raw['dff']
    dffb, meb, lab = raw['dff_binned'], raw['me_binned'], raw['me_labels']
    tb = raw['bin_centre_s']
    tf = np.arange(F.shape[1]) / FS

    fig, ax = plt.subplots(7, 1, figsize=(17, 21))

    # 1) raw F + baseline for 3 example neurons
    ex = np.linspace(0, F.shape[0] - 1, 3).astype(int)
    for i, n in enumerate(ex):
        ax[0].plot(tf, F[n] + i * np.ptp(F[ex]), lw=0.4, color='C0',
                   label='raw F' if i == 0 else None)
        ax[0].plot(tf, base[n] + i * np.ptp(F[ex]), lw=1.0, color='C3',
                   label='maximin baseline' if i == 0 else None)
    ax[0].set_title(f'{session_id} — 1) raw fluorescence and Suite2p maximin baseline '
                    f'(neurons {ex})')
    ax[0].set_ylabel('F (a.u.)'); ax[0].legend(loc='upper right')

    # 2) dF/F for the same neurons
    for i, n in enumerate(ex):
        ax[1].plot(tf, dff[n] + i * np.ptp(dff[ex]), lw=0.4, color='C0',
                   label='dF/F (30 Hz)' if i == 0 else None)
        ax[1].plot(tb, dffb[n] + i * np.ptp(dff[ex]), lw=0.8, color='C1',
                   label='dF/F binned (3 Hz)' if i == 0 else None)
    ax[1].set_title('2) baseline-subtracted dF/F, before and after 10-frame binning')
    ax[1].set_ylabel('dF/F (a.u.)'); ax[1].legend(loc='upper right')

    # 3) binned raster + motion energy overlay
    ax[2].imshow(dffb, aspect='auto', cmap='gray_r',
                 vmin=0, vmax=np.percentile(dffb, 99),
                 extent=[tb[0], tb[-1], dffb.shape[0], 0])
    ax[2].set_title('3) binned dF/F raster (all neurons)')
    ax[2].set_ylabel('neuron')

    # 4) motion energy at frame rate and binned
    ax[3].plot(tf, raw['me_frames'], lw=0.3, color='0.6', label='motion energy (30 Hz)')
    ax[3].plot(tb, meb, lw=0.8, color='C2', label='motion energy binned (3 Hz)')
    ax[3].set_yscale('symlog')
    ax[3].set_title('4) motion energy on the imaging-frame grid (NaN = dropped camera frame)')
    ax[3].set_ylabel('motion energy'); ax[3].legend(loc='upper right')

    # 5) discretisation
    ax[4].plot(tb, meb, lw=0.8, color='C2', label='binned motion energy')
    for e in res['me_edges']:
        ax[4].axhline(e, color='k', ls='--', lw=0.8)
    ax[4].set_yscale('symlog')
    ax4b = ax[4].twinx()
    ax4b.plot(tb, lab, lw=0.6, color='C3', alpha=0.7)
    ax4b.set_ylabel('quintile label', color='C3')
    ax[4].set_title('5) quintile discretisation (dashed = 20/40/60/80th percentile of the session)')
    ax[4].set_ylabel('motion energy'); ax[4].legend(loc='upper right')

    # 6) class histogram
    ax[5].bar(np.arange(N_QUANTILES), res['class_fractions'])
    ax[5].axhline(1.0 / N_QUANTILES, color='k', ls='--')
    ax[5].set_title('6) class fractions (dashed = 1/5)')
    ax[5].set_xlabel('motion energy quintile'); ax[5].set_ylabel('fraction of bins')

    # 7) reassembled trials: check for temporal alignment / no gaps at trial borders
    t_cat = np.concatenate([res['input'][k][0] for k in range(res['n_trials'])])
    o_cat = np.concatenate([res['output'][k][0] for k in range(res['n_trials'])])
    n_cat = np.concatenate([res['neural'][k][ex[0]] for k in range(res['n_trials'])])
    ax[6].plot(t_cat, o_cat, lw=0.6, color='C3', label='output (from trials)')
    ax[6].plot(tb[:len(t_cat)], lab[:len(t_cat)] + 0.15, lw=0.6, color='k', ls=':',
               label='output (session-level, +0.15 offset)')
    ax6b = ax[6].twinx()
    ax6b.plot(t_cat, n_cat, lw=0.4, color='C0', alpha=0.6, label='neural (from trials)')
    ax6b.set_ylabel('dF/F neuron %d' % ex[0], color='C0')
    for k in range(res['n_trials']):
        ax[6].axvline(t_cat[k * BINS_PER_TRIAL], color='0.8', lw=0.5)
    ax[6].set_title('7) trials re-concatenated: time axis continuous across trial borders '
                    '(grey lines) and output identical to the session-level trace')
    ax[6].set_xlabel('time from session start (s)'); ax[6].set_ylabel('quintile')
    ax[6].legend(loc='upper right')

    for a in ax[:6]:
        a.set_xlabel('time from session start (s)')

    fig.tight_layout()
    path = os.path.join(outdir, f'processing_{session_id}.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f'    saved {path}', flush=True)


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile', type=str, help='output pickle path')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true', help='process all sessions (default)')
    g.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='plot every processing step for up to 2 sessions')
    args = ap.parse_args()

    t_start = time.time()
    all_sessions = list_sessions()
    print(f'Found {len(all_sessions)} sessions from '
          f'{len(set(s[0] for s in all_sessions))} subjects.', flush=True)

    if args.sample:
        # one 20-min session (jm031) and one 30-min session (jm039) so that both
        # session lengths and a session with dropped camera frames are exercised
        wanted = [('jm031', '2023-10-22_a'), ('jm039', '2024-05-06_a')]
        sessions = [s for s in all_sessions if (s[0], s[1]) in wanted]
        print(f'--sample: processing {len(sessions)} sessions: '
              f'{[s[0] + "/" + s[1] for s in sessions]}', flush=True)
    else:
        sessions = all_sessions

    # neurons to drop, computed per subject over ALL of that subject's sessions
    subjects_used = sorted(set(s[0] for s in sessions))
    keep_masks = {}
    for subject in subjects_used:
        subj_sessions = [s for s in all_sessions if s[0] == subject]
        bad = find_bad_neurons(subj_sessions)
        keep_masks[subject] = ~bad
        if bad.any():
            print(f'  {subject}: dropping {bad.sum()} of {len(bad)} tracked ROIs with an '
                  f'all-zero trace on >=1 day (indices {np.flatnonzero(bad).tolist()})',
                  flush=True)

    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': [], 'subject_idx': [],
        'brain_regions': [BRAIN_REGION], 'brain_region_idx': [],
        'input_names': ['time_from_session_start_s'],
        'output_names': ['motion_energy_quintile'],
        'output_values': [[f'q{i + 1}' for i in range(N_QUANTILES)]],
        'metadata': {},
    }
    subject_list = sorted(SUBJECT_INFO.keys())
    session_info = []
    n_plotted = 0

    for (subject, sess_name, sess_dir) in sessions:
        res = convert_session(sess_dir, keep_neurons=keep_masks[subject])

        data['neural'].append(res['neural'])
        data['input'].append(res['input'])
        data['output'].append(res['output'])
        data['subject_idx'].append(subject_list.index(subject))
        data['brain_region_idx'].append(np.zeros(res['n_neurons'], dtype=np.int64))

        # postnatal day: first session of the subject is SUBJECT_INFO[subject][1],
        # recordings are on consecutive days
        subj_sessions = [s[1] for s in all_sessions if s[0] == subject]
        day_index = subj_sessions.index(sess_name)
        session_info.append({
            'subject': subject,
            'paper_mouse': SUBJECT_INFO[subject][0],
            'session': sess_name,
            'date': sess_name[:10],
            'day_index': day_index,
            'postnatal_day': SUBJECT_INFO[subject][1] + day_index,
            'n_neurons': res['n_neurons'],
            'n_neurons_before_curation': res['n_neurons_all'],
            'n_frames': res['n_frames'],
            'n_bins': res['n_bins'],
            'n_trials': res['n_trials'],
            'duration_s': res['n_frames'] / FS,
            'motion_energy_quintile_edges': res['me_edges'].tolist(),
            'class_fractions': res['class_fractions'].tolist(),
            'n_missing_camera_frames': res['me_info']['n_missing_frames'],
            'frac_frames_with_behaviour': res['me_info']['frac_valid'],
        })

        if args.show_processing and n_plotted < 2:
            plot_processing(f'{subject}_{sess_name}', res)
            n_plotted += 1
        res.pop('_raw')

    data['subjects'] = subject_list
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)

    n_sessions = len(data['neural'])
    n_trials = sum(len(s) for s in data['neural'])
    data['metadata'] = {
        'task_description': (
            'Spontaneous activity in mouse barrel cortex (S1, layer 2/3) during the '
            'second postnatal week (P7-P14). Head-fixed pups sat on a non-motorised '
            'treadmill in the dark under sensory-minimised conditions; there was no task '
            'and no stimulus. Neural data are 2-photon GCaMP8m calcium traces (dF/F) of '
            'neurons tracked across all recording days of a mouse with Track2p. The '
            'decoder predicts the animal\'s behavioural state, quantified as the motion '
            'energy of simultaneous 30 Hz videography (sum of squared pixel-wise '
            'differences between consecutive video frames), discretised into five '
            'equal-percentile (quintile) bins computed separately for each session. The '
            'decoder additionally receives the time elapsed since the start of the '
            'session.'),
        'time_bin_size': BIN_SIZE_MS,
        'temporal_alignment_event': (
            'Start of the 60 s trial. Recordings are continuous spontaneous activity '
            'with no task events, so each session is simply cut into consecutive, '
            'non-overlapping 60 s blocks starting at the first imaging frame.'),
        'off_start': 0.0,
        'off_end': TRIAL_LEN_S,
        'sampling_rate_hz': FS,
        'bin_frames': BIN_FRAMES,
        'trial_duration_s': TRIAL_LEN_S,
        'bins_per_trial': BINS_PER_TRIAL,
        'neural_signal': (
            'dF/F = F - maximin baseline, computed exactly as in the reference code '
            '(track2p/gui/data_management.py:F_processing): no neuropil subtraction '
            '(neucoeff=0), gaussian_filter sigma=10 frames along time, minimum_filter1d '
            'then maximum_filter1d with a 60 s window; then averaged in non-overlapping '
            'bins of 10 frames (Methods: "averaging in bins of 10 consecutive '
            'timestamps").'),
        'neuron_curation': (
            'The released files already contain only the ROIs that Suite2p classified as '
            'cells (probability > 0.5) AND that Track2p tracked across all recording days '
            'of that mouse; rows are matched across days. Additionally, ROIs whose trace '
            'is identically zero on at least one day of the mouse (outside the imaged FOV '
            'that day) were removed from every session of that mouse.'),
        'input_units': ['s'],
        'input_description': [
            'Time elapsed from the first imaging frame of the session, in seconds, at the '
            'centre of each 333.33 ms bin. Continuous across trial boundaries.'],
        'output_description': [
            'Motion energy (videography), binned to 333.33 ms and discretised into five '
            'equal-percentile bins whose edges are the 20/40/60/80th percentiles of that '
            'session. 0 = least motion, 4 = most motion.'],
        'brain_region_description': {
            'S1': 'Primary somatosensory (barrel) cortex, layer 2/3, 100-200 um deep, '
                  '720x720 um FOV'},
        'species': 'Mus musculus (GAD67-Cre pups, P7-P14)',
        'paper': ('Majnik J, Mantez M, Zangila S, Bugeon S, Guignard L, Platel J-C, '
                  'Cossart R (2025) Longitudinal tracking of neuronal activity from the '
                  'same cells in the developing brain using Track2p. eLife 14:RP107540'),
        'n_sessions': n_sessions,
        'n_trials': n_trials,
        'session_info': session_info,
    }

    # ---------------------- sanity checks ----------------------------------------
    print('\nRunning sanity checks ...', flush=True)
    assert len(data['input']) == n_sessions and len(data['output']) == n_sessions
    assert len(data['subject_idx']) == n_sessions
    assert len(data['brain_region_idx']) == n_sessions
    for s in range(n_sessions):
        nt = len(data['neural'][s])
        assert nt >= 2, f'session {s} has only {nt} trials'
        assert len(data['input'][s]) == nt and len(data['output'][s]) == nt
        nneur = data['neural'][s][0].shape[0]
        assert len(data['brain_region_idx'][s]) == nneur
        for k in range(nt):
            assert data['neural'][s][k].shape == (nneur, BINS_PER_TRIAL)
            assert data['input'][s][k].shape == (1, BINS_PER_TRIAL)
            assert data['output'][s][k].shape == (1, BINS_PER_TRIAL)
            assert np.isfinite(data['neural'][s][k]).all()
            assert np.isfinite(data['input'][s][k]).all()
            assert data['output'][s][k].min() >= 0
            assert data['output'][s][k].max() < N_QUANTILES
        # time must be continuous and strictly increasing across the session
        t_cat = np.concatenate([data['input'][s][k][0] for k in range(nt)])
        dt = np.diff(t_cat)
        # atol accounts for float32 resolution (~1e-4 s at t ~ 1800 s)
        assert np.allclose(dt, BIN_SIZE_S, atol=1e-3), \
            f'session {s}: non-uniform time steps'
        assert np.isclose(t_cat[0], (BIN_FRAMES - 1) / 2.0 / FS)
    print('  all structural checks passed.', flush=True)

    per_subject = {}
    for s in range(n_sessions):
        per_subject[subject_list[data['subject_idx'][s]]] = data['neural'][s][0].shape[0]
    print(f'  subjects: {len(set(data["subject_idx"].tolist()))}, sessions: {n_sessions}, '
          f'trials: {n_trials}', flush=True)
    print(f'  neurons per subject: {per_subject} '
          f'(total {sum(per_subject.values())}, '
          f'mean {np.mean(list(per_subject.values())):.1f} +/- '
          f'{np.std(list(per_subject.values()), ddof=1):.1f})', flush=True)
    allout = np.concatenate([data['output'][s][k][0]
                             for s in range(n_sessions)
                             for k in range(len(data['output'][s]))])
    print(f'  overall output class fractions: '
          f'{np.round(np.bincount(allout, minlength=N_QUANTILES) / len(allout), 4)}',
          flush=True)
    allin = np.concatenate([data['input'][s][k][0]
                            for s in range(n_sessions)
                            for k in range(len(data['input'][s]))])
    print(f'  input range: [{allin.min():.4f}, {allin.max():.4f}] s', flush=True)

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(args.outfile) / 1e6
    print(f'\nWrote {args.outfile} ({size_mb:.1f} MB) in '
          f'{time.time() - t_start:.1f} s total.', flush=True)
    if n_sessions:
        print(f'Mean time per session: '
              f'{(time.time() - t_start) / n_sessions:.2f} s', flush=True)


if __name__ == '__main__':
    sys.exit(main())
