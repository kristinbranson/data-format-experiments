"""
Convert the Majnik et al. 2025 (Track2p) longitudinal barrel-cortex dataset into the
decoder-compatible pickle format.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing (see /app/CONVERSION_NOTES.md for justification):
  * neural  : suite2p F.npy of the track2p-tracked cells -> baseline-corrected dF
              (reference implementation `F_processing` from
               code/track2p/gui/data_management.py: neucoeff=0, maximin baseline,
               sig_baseline=10 frames, win_baseline=60 s) -> averaged in
               non-overlapping bins of 10 frames (as in the paper's decoding analyses)
  * input   : elapsed time from the start of the session (seconds), time-varying
  * output  : motion energy (videography), aligned to the imaging frames using the
              camera timestamps, binned identically, discretised into 5 equal-percentile
              bins (quintiles) computed per session
  * trials  : consecutive, non-overlapping 60 s blocks (1800 frames = 180 bins)
"""

import os
import sys
import glob
import time
import pickle
import argparse
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter1d, maximum_filter1d

DATA_ROOT = '/app/data'
BIN_FRAMES = 10          # frames averaged per time bin (paper: "bins of 10 consecutive timestamps")
TRIAL_SECONDS = 60.0     # trial length required by the decoder task
N_OUTPUT_BINS = 5        # quintiles of motion energy

# mouse id -> name used in the paper (alphabetical order of the folder names)
SUBJECT_LETTER = {'jm031': 'A', 'jm032': 'B', 'jm038': 'C',
                  'jm039': 'D', 'jm040': 'E', 'jm046': 'F'}


# ----------------------------------------------------------------------------- IO
def list_sessions():
    """Return list of (subject, session_name, session_dir), chronologically sorted."""
    sessions = []
    for subject in sorted(os.listdir(DATA_ROOT)):
        subj_dir = os.path.join(DATA_ROOT, subject)
        if not os.path.isdir(subj_dir) or not subject.startswith('jm'):
            continue
        for sdir in sorted(glob.glob(os.path.join(subj_dir, '*'))):
            if os.path.isdir(sdir) and os.path.isdir(os.path.join(sdir, 'suite2p')):
                sessions.append((subject, os.path.basename(sdir), sdir))
    return sessions


# ------------------------------------------------------------------- processing
def f_processing(F, Fneu=None, fs=30.0, neucoeff=0.0, baseline='maximin',
                 sig_baseline=10.0, win_baseline=60.0, prctile_baseline=8.0):
    """Baseline-corrected fluorescence ('dF/F' of the paper).

    Verbatim re-implementation of `DataManagement.F_processing` in
    code/track2p/gui/data_management.py (which is itself suite2p's dcnv.preprocess).
    Default arguments are the reference/suite2p defaults, and are identical to the
    values stored in ops.npy of every session (baseline='maximin', win_baseline=60,
    sig_baseline=10, fs=30).
    """
    # neuropil subtraction (neucoeff = 0 in the reference code -> no subtraction)
    Fc = F if (neucoeff == 0.0 or Fneu is None) else F - neucoeff * Fneu

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


def bin_trace(x, nframes_per_bin=BIN_FRAMES):
    """Average non-overlapping bins of `nframes_per_bin` along the last axis."""
    T = x.shape[-1] // nframes_per_bin * nframes_per_bin
    newshape = x.shape[:-1] + (T // nframes_per_bin, nframes_per_bin)
    return x[..., :T].reshape(newshape).mean(axis=-1)


def align_motion_energy(me, ts, nframes):
    """Put the camera motion-energy samples on the 2p imaging frame grid.

    The camera is hardware-triggered by the microscope, so camera frame i == imaging
    frame i unless camera frames were dropped (README of the dataset). Dropped frames
    are located with the camera timestamps (`tstamps.npy`, units of 1000 s): a gap of
    k median inter-frame intervals means k-1 missing frames. Missing samples (and the
    first sample, which is 0 by construction because there is no preceding video frame)
    are linearly interpolated.

    Returns (me_on_frame_grid (nframes,), n_interpolated)
    """
    me = np.asarray(me, dtype=np.float64)
    ncam = me.size
    if ncam == nframes:
        idx = np.arange(nframes)
    else:
        ifi = np.diff(np.asarray(ts, dtype=np.float64))
        med = np.median(ifi)
        steps = np.maximum(np.round(ifi / med).astype(np.int64), 1)
        idx = np.concatenate([[0], np.cumsum(steps)])

    full = np.full(nframes, np.nan)
    keep = idx < nframes
    full[idx[keep]] = me[keep]
    full[0] = np.nan            # motion energy of the very first frame is undefined (=0)

    nanmask = np.isnan(full)
    if nanmask.any():
        good = ~nanmask
        full[nanmask] = np.interp(np.flatnonzero(nanmask), np.flatnonzero(good), full[good])
    return full, int(nanmask.sum())


def discretize_quantiles(x, nbins=N_OUTPUT_BINS):
    """Discretise into `nbins` equal-percentile bins (edges from the data itself)."""
    edges = np.quantile(x, np.arange(1, nbins) / nbins)
    labels = np.searchsorted(edges, x, side='right').astype(np.int64)
    return labels, edges


def process_session(args):
    """Load + process a single session. Returns a dict with the converted arrays."""
    subject, sess_name, sess_dir, show_processing = args
    t0 = time.time()
    timings = {}

    p0 = os.path.join(sess_dir, 'suite2p', 'plane0')
    F = np.load(os.path.join(p0, 'F.npy'))
    ops = np.load(os.path.join(p0, 'ops.npy'), allow_pickle=True).item()
    iscell = np.load(os.path.join(p0, 'iscell.npy'))
    timings['load'] = time.time() - t0

    fs = float(ops['fs'])
    nframes = int(ops['nframes'])
    assert F.shape[1] == nframes, f'{sess_dir}: F has {F.shape[1]} frames, ops says {nframes}'
    # the released data contains only track2p-tracked cells that passed suite2p's
    # classifier (verified here as a sanity check)
    assert np.all((iscell[:, 0] == 1) | (iscell[:, 1] > 0.5)), f'{sess_dir}: uncurated ROIs present'
    assert iscell.shape[0] == F.shape[0]

    # ---- neural: baseline-corrected fluorescence, then 10-frame bin averaging
    t = time.time()
    dF = f_processing(F.astype(np.float32), fs=fs,
                      baseline=ops.get('baseline', 'maximin'),
                      sig_baseline=float(ops.get('sig_baseline', 10.0)),
                      win_baseline=float(ops.get('win_baseline', 60.0)))
    timings['dff'] = time.time() - t
    t = time.time()
    dF_binned = bin_trace(dF).astype(np.float32)          # (n_neurons, nbins)
    timings['bin_neural'] = time.time() - t

    # ---- behaviour: motion energy aligned to imaging frames, then binned
    t = time.time()
    me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
    ts = np.load(os.path.join(sess_dir, 'move_deve', 'tstamps.npy'))
    me_frames, n_interp = align_motion_energy(me_raw, ts, nframes)
    me_binned = bin_trace(me_frames)
    timings['behaviour'] = time.time() - t

    nbins = dF_binned.shape[1]
    assert me_binned.size == nbins

    # ---- discretise motion energy into per-session quintiles
    me_labels, quant_edges = discretize_quantiles(me_binned, N_OUTPUT_BINS)

    # ---- elapsed time (s) at the centre of each bin, from the start of the session
    bin_centre_frames = np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0
    time_s = (bin_centre_frames / fs).astype(np.float32)

    # ---- cut into consecutive 60 s trials
    bins_per_trial = int(round(TRIAL_SECONDS * fs / BIN_FRAMES))    # 180
    ntrials = nbins // bins_per_trial
    neural_trials, input_trials, output_trials = [], [], []
    for k in range(ntrials):
        sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
        neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
        input_trials.append(time_s[sl][None, :].copy())
        output_trials.append(me_labels[sl][None, :].copy())

    res = {
        'subject': subject,
        'session': sess_name,
        'session_dir': sess_dir,
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'n_neurons': int(F.shape[0]),
        'n_frames': nframes,
        'fs': fs,
        'n_bins': int(nbins),
        'n_trials': int(ntrials),
        'bins_per_trial': bins_per_trial,
        'n_interpolated_frames': n_interp,
        'n_camera_frames': int(me_raw.size),
        'quantile_edges': quant_edges.tolist(),
        'class_fractions': np.bincount(np.concatenate([o.ravel() for o in output_trials]),
                                       minlength=N_OUTPUT_BINS).tolist(),
        'timings': timings,
        'total_time': time.time() - t0,
    }

    if show_processing:
        plot_processing(res, F, dF, me_raw, me_frames, me_binned, me_labels, quant_edges, time_s)

    print(f"  [{subject}/{sess_name}] neurons={res['n_neurons']} frames={nframes} "
          f"bins={nbins} trials={ntrials} interp_frames={n_interp} "
          f"t={res['total_time']:.1f}s ({timings})", flush=True)
    return res


# -------------------------------------------------------------------- plotting
def plot_processing(res, F, dF, me_raw, me_frames, me_binned, me_labels, quant_edges, time_s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sid = f"{res['subject']}_{res['session']}"
    fs = res['fs']
    nframes = res['n_frames']
    tframe = np.arange(nframes) / fs
    nshow = min(5, F.shape[0])

    fig, ax = plt.subplots(8, 1, figsize=(16, 24))

    # 1) raw F for a few example neurons
    for i in range(nshow):
        ax[0].plot(tframe, F[i] + i * np.nanpercentile(F, 99), lw=0.4)
    ax[0].set_title(f'{sid}: 1) raw F.npy (first {nshow} tracked neurons)')
    ax[0].set_xlabel('time (s)')

    # 2) baseline-corrected dF for the same neurons
    for i in range(nshow):
        ax[1].plot(tframe, dF[i] + i * np.nanpercentile(dF, 99), lw=0.4)
    ax[1].set_title('2) baseline-corrected dF (reference F_processing: maximin, neucoeff=0)')
    ax[1].set_xlabel('time (s)')

    # 3) dF vs 10-frame binned dF for one neuron, zoomed on the first 60 s
    nb = res['n_bins']
    tbin = (np.arange(nb) * BIN_FRAMES + (BIN_FRAMES - 1) / 2) / fs
    m = tframe <= 60
    mb = tbin <= 60
    ax[2].plot(tframe[m], dF[0][m], lw=0.5, label='dF (30 Hz)')
    ax[2].plot(tbin[mb], res['neural'][0][0][:int(mb.sum())], lw=1.2, label='binned dF (3 Hz)')
    ax[2].legend(); ax[2].set_title('3) neuron 0, first trial: binning check (no temporal shift)')
    ax[2].set_xlabel('time (s)')

    # 4) binned raster
    z = (res['neural'][0] * 0 + 0)  # placeholder to keep flake quiet
    allneural = np.concatenate(res['neural'], axis=1)
    zs = (allneural - allneural.mean(axis=1, keepdims=True)) / (allneural.std(axis=1, keepdims=True) + 1e-9)
    ax[3].imshow(zs, aspect='auto', cmap='gray_r', vmin=0, vmax=3,
                 extent=[0, allneural.shape[1] / fs * BIN_FRAMES, allneural.shape[0], 0])
    ax[3].set_title('4) converted neural raster (z-scored for display), all trials concatenated')
    ax[3].set_xlabel('time (s)')

    # 5) raw vs aligned motion energy
    ax[4].plot(np.arange(me_raw.size) / fs, me_raw, lw=0.4, label='raw motion_energy_glob (camera frames)')
    ax[4].plot(tframe, me_frames, lw=0.4, alpha=0.7, label='aligned to imaging frames (+interpolation)')
    ax[4].legend(); ax[4].set_yscale('log')
    ax[4].set_title(f"5) motion energy alignment ({res['n_camera_frames']} camera frames -> "
                    f"{nframes} imaging frames, {res['n_interpolated_frames']} interpolated)")
    ax[4].set_xlabel('time (s)')

    # 6) binned motion energy with quintile edges
    ax[5].plot(tbin, me_binned, lw=0.5)
    for e in quant_edges:
        ax[5].axhline(e, color='r', ls='--', lw=0.8)
    ax[5].set_yscale('log')
    ax[5].set_title('6) binned motion energy with per-session quintile edges (red)')
    ax[5].set_xlabel('time (s)')

    # 7) discretised output
    ax[6].plot(tbin, me_labels, lw=0.5, drawstyle='steps-mid')
    out_all = np.concatenate([o.ravel() for o in res['output']])
    ax[6].plot(tbin[:out_all.size], out_all, lw=1.0, ls=':', color='k',
               label='output stored in pickle (trials concatenated)')
    ax[6].legend()
    ax[6].set_title('7) discretised output (0..4); dotted = values written to the pickle')
    ax[6].set_xlabel('time (s)')

    # 8) decoder input (elapsed time) and trial boundaries
    in_all = np.concatenate([i.ravel() for i in res['input']])
    ax[7].plot(in_all, lw=1.0)
    for k in range(res['n_trials'] + 1):
        ax[7].axvline(k * res['bins_per_trial'], color='r', lw=0.5, alpha=0.4)
    ax[7].set_title('8) decoder input: elapsed time (s) vs bin index, red = trial boundaries')
    ax[7].set_xlabel('bin index (trials concatenated)')

    fig.tight_layout()
    fig.savefig(f'/app/processing_{sid}.png', dpi=110)
    plt.close(fig)
    print(f'  saved /app/processing_{sid}.png', flush=True)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save per-step visualisations for up to 2 sessions')
    ap.add_argument('--nproc', type=int, default=6)
    args = ap.parse_args()

    t_start = time.time()
    sessions = list_sessions()
    print(f'Found {len(sessions)} sessions from '
          f'{len(set(s[0] for s in sessions))} subjects', flush=True)

    if args.sample:
        # one session from each of the first two subjects (different recording lengths)
        subs = sorted(set(s[0] for s in sessions))
        sessions = [next(s for s in sessions if s[0] == subs[0]),
                    next(s for s in sessions if s[0] == subs[2])]
        print(f'--sample: processing {[ (s[0],s[1]) for s in sessions ]}', flush=True)

    show = [args.show_processing and i < 2 for i in range(len(sessions))]
    jobs = [(s[0], s[1], s[2], show[i]) for i, s in enumerate(sessions)]

    nproc = min(args.nproc, len(jobs))
    if nproc > 1:
        with Pool(nproc) as pool:
            results = pool.map(process_session, jobs, chunksize=1)
    else:
        results = [process_session(j) for j in jobs]

    # ----------------------------------------------------------- assemble output
    subjects = sorted(set(r['subject'] for r in results))
    subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': ['S1'],
        'brain_region_idx': [np.zeros(r['n_neurons'], dtype=np.int64) for r in results],
        'input_names': ['time_from_session_start_s'],
        'output_names': ['motion_energy_quintile'],
        'output_values': [['q1_lowest', 'q2', 'q3', 'q4', 'q5_highest']],
        'metadata': {
            'task_description': (
                'Spontaneous behaviour during head-fixed two-photon calcium imaging of L2/3 '
                'barrel cortex (S1) in mouse pups (P7-P14), imaged daily with the same cells '
                'tracked across days by Track2p. No stimulus or task: the animal rests/moves '
                'freely on a non-motorised treadmill in the dark. The decoder predicts the '
                'animal motion energy (videography), discretised into 5 equal-percentile bins '
                'per session, from the population calcium activity; the decoder input is the '
                'elapsed time from the beginning of the session.'),
            'time_bin_size': float(BIN_FRAMES / results[0]['fs'] * 1000.0),  # ms
            'temporal_alignment_event': (
                'start of the recording session (first two-photon imaging frame); trials are '
                'consecutive non-overlapping 60 s blocks of the continuous recording'),
            'off_start': 0.0,
            'off_end': float(TRIAL_SECONDS),
            'neural_signal': (
                'baseline-corrected fluorescence (dF) of Track2p-tracked suite2p ROIs: '
                'F - F0 with the suite2p maximin baseline (gaussian filter sigma=10 frames, '
                '60 s minimum then maximum filter), neuropil coefficient 0, exactly as the '
                'reference implementation F_processing in track2p/gui/data_management.py; '
                'averaged in non-overlapping bins of 10 imaging frames as in the paper'),
            'output_processing': (
                'motion energy = sum of squared pixel-wise differences between consecutive '
                'videography frames (provided as move_deve/motion_energy_glob.npy), aligned to '
                'the imaging frames with the camera timestamps, missing camera frames linearly '
                'interpolated, averaged in the same 10-frame bins, then discretised into 5 '
                'equal-percentile bins with thresholds computed separately for each session'),
            'imaging_rate_hz': float(results[0]['fs']),
            'trial_duration_s': float(TRIAL_SECONDS),
            'bins_per_trial': int(results[0]['bins_per_trial']),
            'paper': ('Majnik et al. 2025, eLife 14:RP107540, "Longitudinal tracking of neuronal '
                      'activity from the same cells in the developing brain using Track2p"'),
            'subject_letters': {s: SUBJECT_LETTER.get(s, '?') for s in subjects},
            'session_info': [
                {'subject': r['subject'],
                 'subject_letter': SUBJECT_LETTER.get(r['subject'], '?'),
                 'session_date': r['session'],
                 'day_index_within_subject': sum(1 for q in results[:i]
                                                 if q['subject'] == r['subject']),
                 'n_neurons': r['n_neurons'],
                 'n_frames': r['n_frames'],
                 'fs': r['fs'],
                 'n_bins': r['n_bins'],
                 'n_trials': r['n_trials'],
                 'n_camera_frames': r['n_camera_frames'],
                 'n_interpolated_frames': r['n_interpolated_frames'],
                 'motion_energy_quintile_edges': r['quantile_edges'],
                 'class_counts': r['class_fractions']}
                for i, r in enumerate(results)],
        },
    }

    # ------------------------------------------------------------ sanity checks
    print('\n--- sanity checks ---', flush=True)
    ntrials_total = sum(len(s) for s in data['neural'])
    print(f'sessions: {len(results)}, subjects: {subjects}, trials: {ntrials_total}')
    for si, r in enumerate(results):
        for tr in range(len(r['neural'])):
            n, T = r['neural'][tr].shape
            assert n == r['n_neurons']
            assert T == r['bins_per_trial'], (si, tr, T)
            assert r['input'][tr].shape == (1, T)
            assert r['output'][tr].shape == (1, T)
            assert np.isfinite(r['neural'][tr]).all()
            assert np.isfinite(r['input'][tr]).all()
        o = np.concatenate([x.ravel() for x in r['output']])
        frac = np.bincount(o, minlength=N_OUTPUT_BINS) / o.size
        assert o.min() >= 0 and o.max() < N_OUTPUT_BINS
        if np.abs(frac - 0.2).max() > 0.02:
            print(f"  WARNING: session {r['subject']}/{r['session']} class fractions {frac}")
    print('all per-trial shape / finiteness / label-range checks passed', flush=True)

    neurons_per_subject = {s: max(r['n_neurons'] for r in results if r['subject'] == s)
                           for s in subjects}
    print('neurons per subject:', neurons_per_subject)
    vals = np.array(list(neurons_per_subject.values()), dtype=float)
    print(f'mean {vals.mean():.1f} +- {vals.std(ddof=1) if vals.size>1 else 0:.1f} '
          f'(paper: 526 +- 190)')

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nwrote {args.outfile} '
          f'({os.path.getsize(args.outfile)/1e6:.1f} MB) in {time.time()-t_start:.1f} s',
          flush=True)


if __name__ == '__main__':
    main()
