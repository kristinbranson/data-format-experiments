"""
Convert the Track2p longitudinal 2-photon dataset (Majnik et al. 2025, eLife 14:RP107540)
into the decoder-compatible pickle format.

Task
----
Decode the animal's motion (motion energy, discretised into 5 per-session quintile bins) from
neural activity recorded in L2/3 of mouse barrel cortex. Sessions are split into 60-second trials.
The decoder additionally receives the time elapsed from the beginning of the session (seconds).

Processing (matches the reference paper / code, see CONVERSION_NOTES.md)
-----------------------------------------------------------------------
* neural: `F.npy` of the Track2p-tracked cells (already curated: suite2p iscell prob > 0.5 and
  tracked on every day of that mouse) -> dF/F via the authors' own `F_processing()`
  (maximin baseline subtraction, suite2p default parameters, neucoeff = 0.0)
  -> averaged in bins of 10 consecutive frames (paper: "we slightly denoised the dF/F as well as
  the behaviour traces by averaging in bins of 10 consecutive timestamps") -> 333.33 ms bins.
* output: `motion_energy_glob.npy` placed back onto the 2-photon frame grid using the camera
  timestamps (the camera is triggered by the microscope, so camera frame i == imaging frame i
  except for dropped triggers), missing frames interpolated, averaged in the same 10-frame bins,
  then digitised into 5 equal-percentile bins using that session's 20/40/60/80th percentiles.
* input: bin-centre time from the first imaging frame, in seconds.
* trials: consecutive non-overlapping 60 s blocks (180 bins) from the start of the session.

Usage
-----
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter1d, maximum_filter1d

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

DATA_ROOT = '/app/data'

# Subjects, in the order given by the dataset README (alphabetical == mouse A ... mouse F).
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
MOUSE_LETTER = dict(zip(SUBJECTS, 'ABCDEF'))
# Postnatal day of the first recording day of each mouse, read from Fig. 5B of the paper.
# (A: P7-P13, B: P7-P13, C: P8-P14, D: P8-P14, E: P9-P14, F: P8-P14)
FIRST_PDAY = {'jm031': 7, 'jm032': 7, 'jm038': 8, 'jm039': 8, 'jm040': 9, 'jm046': 8}

FS = 30.0                 # nominal imaging rate (Hz); ops['fs'] == 30 for every session
BIN_FRAMES = 10           # paper: denoise by averaging 10 consecutive timestamps
BIN_SIZE_MS = 1000.0 * BIN_FRAMES / FS          # 333.333 ms
TRIAL_SECONDS = 60.0                            # task specification
BINS_PER_TRIAL = int(round(TRIAL_SECONDS * FS / BIN_FRAMES))   # 180 bins
FRAMES_PER_TRIAL = BINS_PER_TRIAL * BIN_FRAMES                 # 1800 frames

N_QUANTILES = 5           # task: motion energy in five equal-percentile bins

BRAIN_REGIONS = ['S1']    # barrel cortex (primary somatosensory), layer 2/3

# suite2p defaults, also stored in each session's ops.npy
WIN_BASELINE = 60.0
SIG_BASELINE = 10.0
PRCTILE_BASELINE = 8.0
# The reference implementation of the "dF/F0" trace (track2p GUI) calls F_processing() with its
# default neucoeff = 0.0, i.e. no neuropil subtraction. See CONVERSION_NOTES.md Step 4.
NEUCOEFF = 0.0


# --------------------------------------------------------------------------------------
# Reference code (copied verbatim from code/track2p/gui/data_management.py:185)
# --------------------------------------------------------------------------------------

def F_processing(F, Fneu, fs, neucoeff=0.0, baseline='maximin', sig_baseline=10.0,
                 win_baseline=60.0, prctile_baseline: float = 8):
    """dF/F computation used by the reference code base (track2p GUI, 'dF/F0' trace type).

    Verbatim copy of `DataManagement.F_processing` (prints removed). Mathematically identical
    to `suite2p.extraction.dcnv.preprocess` with the same parameters.
    """
    # neuropil substraction
    Fc = F - neucoeff * Fneu

    # baseline operation
    win = int(win_baseline * fs)
    if baseline == "maximin":
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    elif baseline == "constant":
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = np.amin(Flow)
    elif baseline == "constant_prctile":
        Flow = np.percentile(Fc, prctile_baseline, axis=1)
        Flow = np.expand_dims(Flow, axis=1)
    else:
        Flow = 0.

    F = Fc - Flow

    return F, Flow


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def find_sessions(data_root=DATA_ROOT):
    """Return a list of (subject, session_name, session_dir), chronologically per subject."""
    sessions = []
    for subject in SUBJECTS:
        subject_dir = os.path.join(data_root, subject)
        names = sorted(f.name for f in os.scandir(subject_dir) if f.is_dir())
        for name in names:
            sessions.append((subject, name, os.path.join(subject_dir, name)))
    return sessions


def load_session_neural(session_dir, neucoeff=NEUCOEFF, dff_mode='subtract'):
    """Load the tracked-cell fluorescence and convert it to dF/F.

    Returns (dff, Flow, F, ops, iscell). dff/Flow/F have shape (n_neurons, n_frames).
    """
    s2p = os.path.join(session_dir, 'suite2p', 'plane0')
    F = np.load(os.path.join(s2p, 'F.npy')).astype(np.float32)
    iscell = np.load(os.path.join(s2p, 'iscell.npy'))
    ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()

    # Sanity check: the released data is already curated (iscell prob > 0.5, tracked on all days).
    assert np.all(iscell[:, 0] == 1), f'{session_dir}: unexpected non-cell ROI'
    assert np.all(iscell[:, 1] > 0.5), f'{session_dir}: ROI below the 0.5 iscell threshold'
    assert F.shape[1] == ops['nframes'], f'{session_dir}: F/ops frame-count mismatch'
    assert ops['fs'] == FS, f'{session_dir}: unexpected frame rate {ops["fs"]}'

    if neucoeff != 0.0:
        Fneu = np.load(os.path.join(s2p, 'Fneu.npy')).astype(np.float32)
    else:
        Fneu = np.float32(0.0)   # not needed; avoids reading ~150 MB per session

    dff, Flow = F_processing(F, Fneu, fs=ops['fs'], neucoeff=neucoeff,
                             baseline=ops.get('baseline', 'maximin'),
                             sig_baseline=ops.get('sig_baseline', SIG_BASELINE),
                             win_baseline=ops.get('win_baseline', WIN_BASELINE),
                             prctile_baseline=ops.get('prctile_baseline', PRCTILE_BASELINE))

    # Alternative neural representations, used only for the sensitivity analysis of Step 12.
    if dff_mode == 'divide':          # literal dF/F0 = (F - F0) / F0
        dff = dff / np.maximum(Flow, 1e-3)
    elif dff_mode == 'zscore':        # per-neuron z-scored dF
        dff = (dff - dff.mean(axis=1, keepdims=True)) / (dff.std(axis=1, keepdims=True) + 1e-9)
    elif dff_mode != 'subtract':
        raise ValueError(dff_mode)

    # A handful of ROIs (8 of 20445 neuron-sessions) have an identically-zero fluorescence trace
    # although their neuropil trace is fine: signal extraction failed for that ROI on that day.
    # That is missing data, not silence, so those neurons are dropped -- from *every* session of
    # the mouse, to preserve the defining property of this dataset that row i is the same tracked
    # neuron on every recording day (see `drop_failed_neurons`).
    failed = np.all(F == 0, axis=1)

    return dff.astype(np.float32), Flow.astype(np.float32), F, ops, iscell, failed


def load_session_motion(session_dir, n_frames):
    """Load motion energy and place it on the 2-photon frame grid.

    The camera is hardware-triggered by the microscope, so camera sample i corresponds to
    imaging frame i unless the camera missed triggers. Missed triggers show up as interframe
    intervals that are integer multiples of the median interval (dataset README); the number of
    missed triggers before each sample is the cumulative sum of those multiples.

    Returns (motion, n_missing, raw, frame_index) where `motion` has length n_frames with
    missing values linearly interpolated.
    """
    md = os.path.join(session_dir, 'move_deve')
    raw = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
    ifi = np.load(os.path.join(md, 'interframe_int.npy')).astype(np.float64)
    assert len(ifi) == len(raw) - 1, f'{session_dir}: tstamps/motion length mismatch'

    # number of extra (missed) trigger intervals before each camera sample
    med = np.median(ifi)
    n_intervals = np.round(ifi / med).astype(np.int64)
    frame_index = np.concatenate([[0], np.cumsum(n_intervals)])

    motion = np.full(n_frames, np.nan)
    inside = frame_index < n_frames
    motion[frame_index[inside]] = raw[inside]

    # The first motion-energy sample is a boundary artefact (no preceding video frame): it is 0
    # in every session. Treat it as missing.
    motion[0] = np.nan

    n_missing = int(np.isnan(motion).sum())

    # linear interpolation over missing frames (sanctioned by the dataset README)
    idx = np.arange(n_frames)
    good = ~np.isnan(motion)
    motion = np.interp(idx, idx[good], motion[good])

    return motion, n_missing, raw, frame_index


# --------------------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------------------

def bin_time(x, bin_frames=BIN_FRAMES):
    """Average consecutive `bin_frames` samples along the last axis, dropping any remainder."""
    x = np.asarray(x)
    n = (x.shape[-1] // bin_frames) * bin_frames
    x = x[..., :n]
    new_shape = x.shape[:-1] + (n // bin_frames, bin_frames)
    return x.reshape(new_shape).mean(axis=-1)


def discretize_quantiles(x, n_bins=N_QUANTILES):
    """Digitise into `n_bins` equal-percentile bins; returns (labels int8, thresholds)."""
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1)[1:-1])
    labels = np.digitize(x, edges, right=False).astype(np.int8)
    return labels, edges


def process_session(subject, name, session_dir, neucoeff=NEUCOEFF, dff_mode='subtract',
                    verbose=True):
    """Full per-session conversion. Returns a dict with trials and bookkeeping info."""
    t0 = time.time()
    dff, Flow, F, ops, iscell, failed = load_session_neural(session_dir, neucoeff=neucoeff,
                                                            dff_mode=dff_mode)
    t_neural = time.time() - t0
    n_frames = dff.shape[1]

    t0 = time.time()
    motion, n_missing, motion_raw, frame_index = load_session_motion(session_dir, n_frames)
    t_motion = time.time() - t0

    # --- bin both streams identically (10 frames -> 333.33 ms) -------------------------
    t0 = time.time()
    dff_binned = bin_time(dff).astype(np.float32)          # (n_neurons, n_bins)
    motion_binned = bin_time(motion)                       # (n_bins,)
    n_bins = dff_binned.shape[1]
    assert motion_binned.shape[0] == n_bins

    # --- decoder output: per-session quintiles of the binned motion energy -------------
    labels, edges = discretize_quantiles(motion_binned)

    # --- decoder input: bin-centre time from the start of the session, in seconds ------
    bin_centre_time = (np.arange(n_bins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS

    # --- split into consecutive 60 s trials --------------------------------------------
    n_trials = n_bins // BINS_PER_TRIAL
    assert n_trials >= 2, f'{session_dir}: only {n_trials} complete 60 s trials'
    if n_trials * BINS_PER_TRIAL != n_bins and verbose:
        print(f'    note: dropping {n_bins - n_trials * BINS_PER_TRIAL} trailing bins '
              f'(incomplete 60 s trial)')

    neural_trials, input_trials, output_trials = [], [], []
    for k in range(n_trials):
        sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
        neural_trials.append(np.ascontiguousarray(dff_binned[:, sl]))
        input_trials.append(bin_centre_time[sl][None, :].astype(np.float32))
        output_trials.append(labels[sl][None, :].astype(np.int8))
    t_bin = time.time() - t0

    day_index = None  # filled by the caller
    info = {
        'subject': subject,
        'session_name': name,
        'date': name.rstrip('_a') if name.endswith('_a') else name,
        'n_neurons': int(dff.shape[0]),
        'n_frames': int(n_frames),
        'n_bins': int(n_bins),
        'n_trials': int(n_trials),
        'n_missing_camera_frames': int(n_missing),
        'n_failed_rois': int(failed.sum()),
        'quantile_edges': edges.tolist(),
        'session_duration_s': float(n_frames / FS),
        'day_index': day_index,
    }
    if verbose:
        print(f'    neurons={info["n_neurons"]:4d} frames={n_frames} bins={n_bins} '
              f'trials={n_trials} missing_cam={n_missing:3d} '
              f'[load+dff {t_neural:.2f}s, motion {t_motion:.2f}s, bin/split {t_bin:.2f}s]')

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'info': info,
        'failed': failed,
        # extras kept only for plotting / sanity checks
        '_raw': {'F': F, 'Flow': Flow, 'dff': dff, 'motion': motion, 'motion_raw': motion_raw,
                 'frame_index': frame_index, 'dff_binned': dff_binned,
                 'motion_binned': motion_binned, 'labels': labels, 'edges': edges,
                 'bin_centre_time': bin_centre_time},
    }


def drop_failed_neurons(data, session_info, session_subjects, session_names, failed_masks):
    """Remove ROIs whose signal extraction failed on at least one day of that mouse.

    The fluorescence trace of such an ROI is identically zero on the affected day while its
    neuropil trace is normal, i.e. the trace is missing rather than silent. Because row i is the
    same tracked neuron on every day of a mouse, the neuron is removed from all of that mouse's
    sessions, which keeps the cross-day correspondence intact.
    """
    bad_per_subject = {}
    processed_names = {}
    for subject, name, mask in zip(session_subjects, session_names, failed_masks):
        prev = bad_per_subject.get(subject)
        bad_per_subject[subject] = mask if prev is None else (prev | mask)
        processed_names.setdefault(subject, set()).add(name)

    # The rule is "failed on any day of that mouse", so days that were not processed in this run
    # (--sample) still have to be inspected; those are read lazily here.
    for subject in bad_per_subject:
        for subj, name, sdir in find_sessions():
            if subj != subject or name in processed_names[subject]:
                continue
            F = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')
            bad_per_subject[subject] |= np.all(np.asarray(F) == 0, axis=1)

    n_dropped = 0
    for isess, subject in enumerate(session_subjects):
        bad = bad_per_subject[subject]
        if not bad.any():
            continue
        keep = ~bad
        data['neural'][isess] = [t[keep] for t in data['neural'][isess]]
        data['brain_region_idx'][isess] = data['brain_region_idx'][isess][keep]
        session_info[isess]['n_neurons'] = int(keep.sum())
        session_info[isess]['n_neurons_dropped'] = int(bad.sum())
        n_dropped += int(bad.sum())
    for subject, bad in bad_per_subject.items():
        if bad.any():
            print(f'  {subject}: dropped {int(bad.sum())} neuron(s) with a failed (all-zero) '
                  f'fluorescence trace on at least one day: {np.where(bad)[0].tolist()}')
    return n_dropped


# --------------------------------------------------------------------------------------
# Visualisation of every processing step (--show-processing)
# --------------------------------------------------------------------------------------

def plot_processing(result, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r = result['_raw']
    info = result['info']
    F, Flow, dff = r['F'], r['Flow'], r['dff']
    n_frames = F.shape[1]
    t_frames = np.arange(n_frames) / FS
    tb = r['bin_centre_time']

    fig, axes = plt.subplots(8, 1, figsize=(18, 30))
    fig.suptitle(f'Processing steps: {info["subject"]} {info["session_name"]} '
                 f'({info["n_neurons"]} neurons, {info["n_trials"]} trials)', fontsize=15)

    # 1. raw F + maximin baseline for 3 example neurons
    ax = axes[0]
    ex = np.linspace(0, F.shape[0] - 1, 3).astype(int)
    for i, n in enumerate(ex):
        off = i * np.nanpercentile(F, 99)
        ax.plot(t_frames, F[n] + off, lw=0.4, color=f'C{i}', label=f'neuron {n} F')
        ax.plot(t_frames, Flow[n] + off, lw=1.0, color='k')
    ax.set_title('1) raw fluorescence F.npy (coloured) with the maximin baseline F0 (black)')
    ax.set_xlabel('time (s)'); ax.set_ylabel('F (a.u., offset)'); ax.legend(fontsize=7)

    # 2. dF/F for the same neurons
    ax = axes[1]
    for i, n in enumerate(ex):
        off = i * np.nanpercentile(dff, 99.9)
        ax.plot(t_frames, dff[n] + off, lw=0.4, color=f'C{i}')
    ax.set_title('2) dF/F = F - F0 (F_processing, neucoeff=%.1f)' % NEUCOEFF)
    ax.set_xlabel('time (s)'); ax.set_ylabel('dF/F (offset)')

    # 3. raw vs binned dF/F on a 60 s window (alignment check)
    ax = axes[2]
    n = ex[1]
    w = slice(0, FRAMES_PER_TRIAL)
    ax.plot(t_frames[w], dff[n][w], lw=0.6, color='0.6', label='dF/F @30 Hz')
    ax.plot(tb[:BINS_PER_TRIAL], r['dff_binned'][n][:BINS_PER_TRIAL], lw=1.2, color='C3',
            marker='.', ms=3, label='dF/F binned (10 frames, plotted at bin centres)')
    ax.set_title('3) temporal binning of the neural trace, first trial (no shift: binned curve '
                 'tracks the raw one)')
    ax.set_xlabel('time (s)'); ax.legend(fontsize=8)

    # 4. motion energy: raw camera samples, reconstruction, missing frames
    ax = axes[3]
    fi = r['frame_index']
    inside = fi < n_frames
    ax.plot(fi[inside] / FS, r['motion_raw'][inside], lw=0.4, color='0.5',
            label='raw camera samples (at their 2p frame index)')
    ax.plot(t_frames, r['motion'], lw=0.4, color='C0', alpha=0.7,
            label='on the 2p frame grid, missing values interpolated')
    miss = np.setdiff1d(np.arange(n_frames), fi[inside])
    miss = np.union1d(miss, [0])
    if len(miss):
        ax.plot(miss / FS, r['motion'][miss], 'rx', ms=4,
                label=f'interpolated frames (n={len(miss)})')
    ax.set_yscale('log')
    ax.set_title('4) motion energy placed on the imaging frame grid')
    ax.set_xlabel('time (s)'); ax.set_ylabel('motion energy'); ax.legend(fontsize=8)

    # 5. binned motion energy + quintile thresholds
    ax = axes[4]
    ax.plot(tb, r['motion_binned'], lw=0.6, color='C0', label='binned motion energy')
    for e in r['edges']:
        ax.axhline(e, color='r', lw=0.8, ls='--')
    ax.set_yscale('log')
    ax.set_title('5) binned motion energy with the session quintile thresholds (red dashed)')
    ax.set_xlabel('time (s)'); ax.legend(fontsize=8)

    # 6. discretised output + class histogram check
    ax = axes[5]
    ax.plot(tb, r['labels'], lw=0.6, color='C2', drawstyle='steps-mid')
    frac = np.bincount(r['labels'], minlength=N_QUANTILES) / len(r['labels'])
    ax.set_title('6) decoder output: motion-energy quintile, class fractions = '
                 + ', '.join(f'{f:.3f}' for f in frac))
    ax.set_xlabel('time (s)'); ax.set_ylabel('quintile (0-4)')

    # 7. session raster of the binned dF/F with motion energy underneath (cf. paper Fig. 5A)
    ax = axes[6]
    z = r['dff_binned']
    z = (z - z.mean(axis=1, keepdims=True)) / (z.std(axis=1, keepdims=True) + 1e-9)
    ax.imshow(z, aspect='auto', cmap='gray_r', vmin=0, vmax=3,
              extent=[tb[0], tb[-1], z.shape[0], 0], interpolation='nearest')
    axt = ax.twinx()
    axt.plot(tb, r['motion_binned'], color='C1', lw=0.7, alpha=0.8)
    axt.set_yscale('log'); axt.set_ylabel('motion energy', color='C1')
    for k in range(info['n_trials'] + 1):
        ax.axvline(k * TRIAL_SECONDS, color='C0', lw=0.5, alpha=0.5)
    ax.set_title('7) binned dF/F raster (z-scored for display) + motion energy; blue lines = '
                 '60 s trial boundaries')
    ax.set_xlabel('time (s)'); ax.set_ylabel('neuron')

    # 8. reassembled trials must reproduce the session arrays exactly
    ax = axes[7]
    neural_cat = np.concatenate(result['neural'], axis=1)
    input_cat = np.concatenate(result['input'], axis=1)[0]
    output_cat = np.concatenate(result['output'], axis=1)[0]
    nb = neural_cat.shape[1]
    ok_neural = np.allclose(neural_cat, r['dff_binned'][:, :nb])
    ok_out = np.array_equal(output_cat, r['labels'][:nb])
    ok_in = np.allclose(input_cat, tb[:nb])
    for k in range(min(4, info['n_trials'])):
        sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
        ax.plot(result['input'][k][0], result['output'][k][0], lw=1.0,
                label=f'trial {k}')
    ax.plot(tb[:4 * BINS_PER_TRIAL], r['labels'][:4 * BINS_PER_TRIAL], 'k:', lw=0.8,
            label='session-level output')
    ax.set_title(f'8) trials concatenate back to the session arrays: neural={ok_neural}, '
                 f'input={ok_in}, output={ok_out} (first 4 trials shown)')
    ax.set_xlabel('time from session start (s)'); ax.set_ylabel('quintile')
    ax.legend(fontsize=8, ncol=5)

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f'    wrote {out_png}')


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('outfile', type=str, help='output pickle file')
    parser.add_argument('--full', action='store_true', default=True,
                        help='process all sessions (default)')
    parser.add_argument('--sample', action='store_true',
                        help='process only 2 sessions (for testing)')
    parser.add_argument('--show-processing', action='store_true',
                        help='plot every processing step for up to 2 sessions')
    parser.add_argument('--neucoeff', type=float, default=NEUCOEFF,
                        help='neuropil coefficient for dF/F (reference code default: 0.0)')
    parser.add_argument('--dff-mode', type=str, default='subtract',
                        choices=['subtract', 'divide', 'zscore'],
                        help='neural representation; "subtract" (default) is the reference-code '
                             'baseline-corrected trace. The others are only for sensitivity tests.')
    args = parser.parse_args()

    sessions = find_sessions()
    if args.sample:
        # two sessions from two different mice, both with dropped camera frames, so that the
        # missing-data path is exercised
        wanted = [('jm031', '2023-10-22_a'), ('jm046', '2024-09-07_a')]
        sessions = [s for s in sessions if (s[0], s[1]) in wanted]
        assert len(sessions) == 2

    print(f'Converting {len(sessions)} sessions '
          f'({len(set(s[0] for s in sessions))} subjects), neucoeff={args.neucoeff}')

    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': SUBJECTS, 'subject_idx': [],
        'brain_regions': BRAIN_REGIONS, 'brain_region_idx': [],
        'input_names': ['time_in_session_s'],
        'output_names': ['motion_energy_quintile'],
        'output_values': [['q1 (lowest 20%)', 'q2', 'q3 (middle 20%)', 'q4',
                           'q5 (highest 20%)']],
        'metadata': {},
    }

    session_info = []
    session_subjects = []
    session_names = []
    failed_masks = []
    day_counter = {}
    t_start = time.time()
    for isess, (subject, name, session_dir) in enumerate(sessions):
        print(f'[{isess + 1}/{len(sessions)}] {subject} {name}')
        res = process_session(subject, name, session_dir, neucoeff=args.neucoeff,
                              dff_mode=args.dff_mode)

        day = day_counter.get(subject, 0)
        day_counter[subject] = day + 1
        res['info']['day_index'] = day
        res['info']['postnatal_day'] = FIRST_PDAY[subject] + day
        res['info']['mouse_letter'] = MOUSE_LETTER[subject]

        data['neural'].append(res['neural'])
        data['input'].append(res['input'])
        data['output'].append(res['output'])
        data['subject_idx'].append(SUBJECTS.index(subject))
        data['brain_region_idx'].append(np.zeros(res['info']['n_neurons'], dtype=np.int64))
        session_info.append(res['info'])
        session_subjects.append(subject)
        session_names.append(name)
        failed_masks.append(res['failed'])

        if args.show_processing and isess < 2:
            plot_processing(res, f'processing_{subject}_{name}.png')
        del res

    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)

    # curation of ROIs whose extraction failed (see drop_failed_neurons)
    n_dropped = drop_failed_neurons(data, session_info, session_subjects, session_names,
                                    failed_masks)
    print(f'  total neuron-sessions dropped: {n_dropped}')

    n_trials = sum(len(s) for s in data['neural'])
    n_neurons = [len(b) for b in data['brain_region_idx']]
    data['metadata'] = {
        'task_description': (
            'Spontaneous behaviour in head-fixed neonatal mice (P7-P14) free to move on a '
            'non-motorised treadmill, in darkness under sensory-minimised conditions (no task, no '
            'stimuli). Two-photon calcium imaging (GCaMP8m) of layer 2/3 of the barrel cortex, with '
            'the same neurons tracked across days by Track2p. Decoded variable: the animal\'s '
            'motion energy, computed from simultaneous 30 Hz videography (summed squared pixel-wise '
            'difference of consecutive video frames) and discretised into five equal-percentile '
            '(quintile) bins using the percentiles of that session.'),
        'time_bin_size': BIN_SIZE_MS,
        'temporal_alignment_event': (
            'start of the imaging session (first 2-photon frame); there is no trial structure in '
            'the experiment, so each session is cut into consecutive, non-overlapping 60 s trials'),
        'off_start': 0.0,
        'off_end': TRIAL_SECONDS,
        'bin_frames': BIN_FRAMES,
        'frame_rate_hz': FS,
        'measured_frame_interval_ms': 33.60,
        'neural_signal': (
            'dF/F = baseline-corrected fluorescence of the Track2p-tracked cells: '
            'F - maximin baseline (gaussian sigma=10 frames, 60 s min/max filter), '
            f'neuropil coefficient {args.neucoeff}, mode={args.dff_mode}, then averaged over '
            f'{BIN_FRAMES}-frame bins '
            '(reference implementation: track2p/gui/data_management.py F_processing)'),
        'neuron_curation': (
            'the released data already contains only ROIs with suite2p cell probability > 0.5 that '
            'were tracked by Track2p on every recording day of that mouse; in addition, the 8 ROIs '
            'whose fluorescence trace is identically zero on at least one day (failed signal '
            'extraction, i.e. missing data) were removed from every session of the affected mouse'),
        'trial_curation': (
            'every complete 60 s block of each session is kept; incomplete trailing blocks would be '
            'dropped (there are none: 36000 and 54000 frames are exact multiples of 1800)'),
        'missing_data_handling': (
            'camera frames dropped by the behaviour camera (<=0.4% of frames in 8/41 sessions) are '
            'located from the camera timestamps and linearly interpolated; the first motion-energy '
            'sample (always 0, a frame-difference boundary artefact) is interpolated as well'),
        'paper': ('Majnik J, Mantez M, Zangila S, Bugeon S, Guignard L, Platel J-C, Cossart R '
                  '(2025) eLife 14:RP107540, https://doi.org/10.7554/eLife.107540'),
        'n_sessions': len(sessions),
        'n_trials_total': n_trials,
        'n_neurons_total_over_sessions': int(np.sum(n_neurons)),
        'session_info': session_info,
    }

    t_conv = time.time() - t_start
    print(f'\nConversion done in {t_conv:.1f} s '
          f'({t_conv / max(len(sessions), 1):.1f} s/session)')
    print(f'  sessions: {len(sessions)}, subjects: {len(set(s[0] for s in sessions))}, '
          f'trials: {n_trials}, neurons/session: {n_neurons}')

    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  wrote {args.outfile} '
          f'({os.path.getsize(args.outfile) / 1e6:.1f} MB) in {time.time() - t0:.1f} s')

    # quick summary of the converted values
    all_out = np.concatenate([o[0] for sess in data['output'] for o in sess])
    frac = np.bincount(all_out, minlength=N_QUANTILES) / len(all_out)
    all_in = np.concatenate([i[0] for sess in data['input'] for i in sess])
    print(f'  output class fractions: {np.round(frac, 4).tolist()}')
    print(f'  input (time in session) range: [{all_in.min():.3f}, {all_in.max():.3f}] s')
    nz = np.concatenate([t.ravel()[:1000] for sess in data['neural'] for t in sess[:2]])
    print(f'  neural dF/F sample stats: mean={nz.mean():.2f} sd={nz.std():.2f} '
          f'min={nz.min():.2f} max={nz.max():.2f}')


if __name__ == '__main__':
    sys.exit(main())
