#!/usr/bin/env python3
"""Convert the Majnik et al. 2025 (Track2p) longitudinal calcium imaging dataset
into the decoder-compatible pickle format.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing (matches the reference paper/code, see CONVERSION_NOTES.md):
  neural  : suite2p F/Fneu -> dF/F (neuropil subtraction + maximin baseline,
            identical to track2p gui.data_management.F_processing and suite2p
            dcnv.preprocess with the ops parameters stored with the data)
            -> average in bins of 10 frames (paper: "denoised the dF/F ... by
            averaging in bins of 10 consecutive timestamps") -> z-score each
            neuron across the session (track2p raster preprocessing)
  input   : time elapsed from the beginning of the session (s), time-varying
  output  : motion energy (videography), camera-dropped frames repaired,
            averaged in bins of 10 frames, discretised into 5 equal-percentile
            bins per session
  trials  : consecutive, non-overlapping 60 s blocks (180 bins of 333.33 ms)
"""

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d
from scipy.stats import rankdata

DATA_ROOT = '/app/data'
BIN_SIZE = 10           # frames averaged together (paper: bins of 10 timestamps)
TRIAL_SEC = 60.0        # trial duration in seconds (task specification)
NCLASSES = 5            # motion energy quintiles (task specification)
SUBJECT_LETTER = {'jm031': 'A', 'jm032': 'B', 'jm038': 'C',
                  'jm039': 'D', 'jm040': 'E', 'jm046': 'F'}


# ----------------------------------------------------------------------------- loading
def list_sessions(subject):
    """Chronologically sorted session directories of one subject."""
    return sorted(p for p in glob.glob(os.path.join(DATA_ROOT, subject, '*'))
                  if os.path.isdir(p))


def load_ops(session_dir):
    return np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'),
                   allow_pickle=True).item()


def load_traces(session_dir):
    """Raw suite2p traces of the track2p-tracked cells (as in data/load_data.ipynb)."""
    p = os.path.join(session_dir, 'suite2p', 'plane0')
    F = np.load(os.path.join(p, 'F.npy'))
    Fneu = np.load(os.path.join(p, 'Fneu.npy'))
    iscell = np.load(os.path.join(p, 'iscell.npy'))
    return F, Fneu, iscell


# ----------------------------------------------------------------------------- processing
def compute_dff(F, Fneu, fs, neucoeff=0.7, baseline='maximin',
                sig_baseline=10.0, win_baseline=60.0, prctile_baseline=8.0):
    """dF/F = baseline corrected, neuropil subtracted fluorescence.

    Copied from track2p/gui/data_management.py::F_processing (identical to
    suite2p.extraction.dcnv.preprocess).  neucoeff is taken from the ops saved
    with the data (suite2p default 0.7).
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


def bin_average(x, bin_size=BIN_SIZE):
    """Average consecutive bin_size samples along the last axis (drops remainder)."""
    x = np.asarray(x, dtype=np.float64)
    n = (x.shape[-1] // bin_size) * bin_size
    newshape = x.shape[:-1] + (n // bin_size, bin_size)
    return x[..., :n].reshape(newshape).mean(axis=-1)


def zscore_rows(x):
    """z-score each neuron across the session (track2p raster preprocessing)."""
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)


def load_motion_energy(session_dir, nframes):
    """Motion energy, one value per imaging frame.

    The camera is hardware-triggered by the microscope, so video frame i
    corresponds to imaging frame i.  When camera frames were dropped
    (len(motion_energy) < nframes) their positions are recovered from the
    inter-frame intervals and the trace is linearly interpolated over them
    (as suggested in data/README.md).
    """
    md = os.path.join(session_dir, 'move_deve')
    me = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
    info = {'n_camera_samples': int(len(me)), 'n_missing': int(nframes - len(me))}
    if len(me) == nframes:
        info['gap_positions'] = []
        return me, info
    if len(me) > nframes:                      # more camera than imaging frames
        info['gap_positions'] = []
        return me[:nframes], info
    ifi = np.load(os.path.join(md, 'interframe_int.npy'))
    med = np.median(ifi)
    nmiss = np.maximum(np.round(ifi / med).astype(int) - 1, 0)
    # index of each acquired camera sample in the full imaging-frame timeline
    idx = np.arange(len(me)) + np.concatenate([[0], np.cumsum(nmiss)])
    if idx[-1] != nframes - 1:
        # inferred gaps do not exactly explain the deficit: fall back to a
        # uniform stretch of the acquired samples over the imaging timeline
        idx = np.linspace(0, nframes - 1, len(me))
        info['fallback_uniform'] = True
    idx = np.clip(idx, 0, nframes - 1)
    me_full = np.interp(np.arange(nframes), idx, me)
    info['gap_positions'] = np.where(nmiss > 0)[0].tolist()
    return me_full, info


def quantile_bin(x, nclasses=NCLASSES):
    """Discretise into nclasses equal-percentile bins (exactly equal counts).

    Rank based, so ties are split evenly and each class holds 1/nclasses of the
    samples of the session.
    """
    r = rankdata(x, method='ordinal')          # 1..n
    lab = ((r - 1) * nclasses) // len(x)
    return lab.astype(np.int64)


# ----------------------------------------------------------------------------- per session
def process_session(session_dir, drop_neurons=None, verbose=True):
    """Return dict with per-trial neural / input / output arrays for one session."""
    t0 = time.time()
    ops = load_ops(session_dir)
    fs = float(ops['fs'])
    F, Fneu, iscell = load_traces(session_dir)
    nframes = F.shape[1]
    t_load = time.time() - t0

    # ---- curation: all released ROIs are iscell==1 (suite2p prob > 0.5, applied
    # upstream by the authors).  Neurons with an identically-zero trace (failed
    # extraction on some day of this mouse) are dropped for all sessions.
    assert np.all(iscell[:, 0] == 1), 'unexpected non-cell ROI in released data'
    keep = np.ones(F.shape[0], dtype=bool)
    if drop_neurons is not None:
        keep[drop_neurons] = False
    F, Fneu = F[keep], Fneu[keep]

    # ---- neural: dF/F -> bin -> z-score
    t0 = time.time()
    dff = compute_dff(F, Fneu, fs,
                      neucoeff=float(ops.get('neucoeff', 0.7)),
                      baseline=str(ops.get('baseline', 'maximin')),
                      sig_baseline=float(ops.get('sig_baseline', 10.0)),
                      win_baseline=float(ops.get('win_baseline', 60.0)))
    dff_binned = bin_average(dff)
    neural = zscore_rows(dff_binned).astype(np.float32)
    t_neural = time.time() - t0

    # ---- behaviour: motion energy -> bin
    t0 = time.time()
    me, me_info = load_motion_energy(session_dir, nframes)
    me_binned = bin_average(me)
    t_beh = time.time() - t0

    nbins = neural.shape[1]
    assert me_binned.shape[0] == nbins

    # ---- trials: consecutive 60 s blocks
    bin_sec = BIN_SIZE / fs
    bins_per_trial = int(round(TRIAL_SEC / bin_sec))
    ntrials = nbins // bins_per_trial
    nused = ntrials * bins_per_trial

    # ---- output: 5 equal-percentile bins of motion energy, per session
    me_used = me_binned[:nused]
    labels = quantile_bin(me_used, NCLASSES)

    # ---- input: time elapsed from the beginning of the session (bin centres)
    tvec = (np.arange(nused) + 0.5) * bin_sec

    neural_trials, input_trials, output_trials = [], [], []
    for k in range(ntrials):
        sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
        neural_trials.append(np.ascontiguousarray(neural[:, sl]))
        input_trials.append(tvec[sl][None, :].astype(np.float32))
        output_trials.append(labels[sl][None, :].astype(np.int64))

    if verbose:
        print(f'    {os.path.basename(session_dir)}: {neural.shape[0]} neurons, '
              f'{nframes} frames, {nbins} bins, {ntrials} trials  '
              f'(load {t_load:.1f}s, neural {t_neural:.1f}s, beh {t_beh:.1f}s), '
              f'missing camera frames: {me_info["n_missing"]}')

    return dict(neural=neural_trials, input=input_trials, output=output_trials,
                nneurons=neural.shape[0], nframes=nframes, fs=fs,
                ntrials=ntrials, bins_per_trial=bins_per_trial,
                bin_sec=bin_sec, me_info=me_info,
                # kept for plotting / sanity checks
                _raw=dict(F=F, dff=dff, dff_binned=dff_binned, neural=neural,
                          me=me, me_binned=me_binned, labels=labels, tvec=tvec))


def find_zero_neurons(subject_sessions):
    """Neurons with an identically-zero F trace on any session of a mouse."""
    bad = set()
    for sd in subject_sessions:
        F = np.load(os.path.join(sd, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')
        z = np.where(~np.asarray(F).any(axis=1))[0]
        bad.update(z.tolist())
    return sorted(bad)


# ----------------------------------------------------------------------------- plotting
def plot_processing(sess, session_id, outdir='/app'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r = sess['_raw']
    fs = sess['fs']
    bin_sec = sess['bin_sec']
    n = 5  # example neurons
    tfr = np.arange(r['F'].shape[1]) / fs
    tbin = (np.arange(r['dff_binned'].shape[1]) + 0.5) * bin_sec

    fig, ax = plt.subplots(7, 1, figsize=(16, 20))
    ax[0].set_title(f'{session_id}: raw F (5 example neurons)')
    for i in range(n):
        ax[0].plot(tfr, r['F'][i] + i * 400, lw=0.4)
    ax[1].set_title('dF/F (neuropil subtracted, maximin baseline corrected)')
    for i in range(n):
        ax[1].plot(tfr, r['dff'][i] + i * 400, lw=0.4)
    ax[2].set_title(f'dF/F binned ({BIN_SIZE} frames = {bin_sec*1000:.1f} ms)')
    for i in range(n):
        ax[2].plot(tbin, r['dff_binned'][i] + i * 400, lw=0.4)
    ax[3].set_title('z-scored binned dF/F = neural output (raster, all neurons)')
    ax[3].imshow(r['neural'], aspect='auto', cmap='gray_r', vmin=0, vmax=3,
                 extent=[tbin[0], tbin[-1], r['neural'].shape[0], 0])
    ax[4].set_title('motion energy: raw (per imaging frame, after gap repair) and binned')
    ax[4].plot(tfr, r['me'], lw=0.3, color='grey', label='per frame')
    ax[4].plot(tbin, r['me_binned'], lw=0.6, color='C1', label='binned')
    ax[4].legend(loc='upper right')
    ax[5].set_title('motion energy quintiles (output) overlaid on binned motion energy')
    ax[5].plot(tbin, r['me_binned'], lw=0.6, color='grey')
    sc = ax[5].scatter(tbin[:len(r['labels'])], r['me_binned'][:len(r['labels'])],
                       c=r['labels'], cmap='viridis', s=3)
    ax[5].set_yscale('log')
    plt.colorbar(sc, ax=ax[5], label='quintile')
    ax[6].set_title('decoder input: time elapsed in session (s) with 60 s trial boundaries')
    ax[6].plot(r['tvec'], lw=1)
    for k in range(sess['ntrials'] + 1):
        ax[6].axvline(k * sess['bins_per_trial'], color='r', lw=0.4)
    ax[6].set_xlabel('bin index')
    for a in ax[:6]:
        a.set_xlabel('time in session (s)')
    fig.tight_layout()
    fn = os.path.join(outdir, f'processing_{session_id}.png')
    fig.savefig(fn, dpi=110)
    plt.close(fig)

    # second figure: trial-level check that neural / input / output are aligned
    fig, ax = plt.subplots(3, 3, figsize=(16, 8), sharex=True)
    for j, k in enumerate([0, sess['ntrials'] // 2, sess['ntrials'] - 1]):
        ax[0, j].imshow(sess['neural'][k], aspect='auto', cmap='gray_r', vmin=0, vmax=3)
        ax[0, j].set_title(f'trial {k}: neural')
        ax[1, j].plot(sess['input'][k][0])
        ax[1, j].set_title('input: time in session (s)')
        ax[2, j].plot(sess['output'][k][0], drawstyle='steps-mid')
        ax[2, j].set_title('output: motion energy quintile')
        ax[2, j].set_ylim(-0.5, NCLASSES - 0.5)
        ax[2, j].set_xlabel('bin within trial')
    fig.suptitle(f'{session_id}: example trials')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'processing_{session_id}_trials.png'), dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save plots of every processing step for up to 2 sessions')
    args = ap.parse_args()

    subjects = sorted(d for d in os.listdir(DATA_ROOT)
                      if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith('jm'))

    data = dict(neural=[], input=[], output=[], subjects=subjects,
                subject_idx=[], brain_regions=['S1'], brain_region_idx=[],
                input_names=['time_in_session_s'],
                output_names=['motion_energy_quintile'],
                output_values=[['very low (0-20%)', 'low (20-40%)', 'medium (40-60%)',
                                'high (60-80%)', 'very high (80-100%)']],
                metadata={})
    session_info = []
    nplotted = 0
    t_start = time.time()

    for si, subject in enumerate(subjects):
        sessions = list_sessions(subject)
        if args.sample:
            if si > 0:
                break
            sessions = sessions[:2]
        print(f'{subject}: {len(sessions)} sessions')
        drop = find_zero_neurons(sessions)
        if drop:
            print(f'  dropping {len(drop)} neuron(s) with all-zero F on some day: {drop}')
        for sd in sessions:
            sess = process_session(sd, drop_neurons=drop)
            data['neural'].append(sess['neural'])
            data['input'].append(sess['input'])
            data['output'].append(sess['output'])
            data['subject_idx'].append(si)
            data['brain_region_idx'].append(np.zeros(sess['nneurons'], dtype=np.int64))
            session_info.append(dict(
                subject=subject, mouse_letter=SUBJECT_LETTER.get(subject, '?'),
                session_dir=sd, date=os.path.basename(sd),
                day_index=sessions.index(sd), n_neurons=int(sess['nneurons']),
                n_frames=int(sess['nframes']), fs=sess['fs'],
                duration_s=float(sess['nframes'] / sess['fs']),
                n_trials=int(sess['ntrials']),
                n_dropped_camera_frames=int(sess['me_info']['n_missing']),
                n_dropped_neurons=int(len(drop))))
            if args.show_processing and nplotted < 2:
                plot_processing(sess, f'{subject}_{os.path.basename(sd)}')
                nplotted += 1
            del sess

    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    bin_ms = 1000.0 * BIN_SIZE / 30.0
    data['metadata'] = dict(
        task_description=(
            'Spontaneous activity in L2/3 of mouse barrel cortex (S1) during the second '
            'postnatal week (P7-P14), imaged daily with 2-photon calcium imaging (GCaMP8m, '
            '30 Hz) while the head-fixed pup could move spontaneously on a non-motorised '
            'treadmill in the dark. There is no stimulus or task. The decoder predicts the '
            "animal's motion energy (videography, sum of squared pixel-wise differences "
            'between consecutive video frames), discretised into five equal-percentile bins '
            'per session, from the population dF/F. Decoder input is the time elapsed from '
            'the beginning of the session.'),
        time_bin_size=bin_ms,
        temporal_alignment_event=('start of each 60 s block of the continuous recording '
                                  '(block k starts at k*60 s after session onset); '
                                  'recordings are continuous, there are no behavioural trials'),
        off_start=0.0,
        off_end=TRIAL_SEC,
        trial_duration_s=TRIAL_SEC,
        neural_data_type=('z-scored dF/F (suite2p neuropil-subtracted, maximin '
                          'baseline-corrected fluorescence), averaged in bins of 10 frames'),
        imaging_rate_hz=30.0,
        bin_size_frames=BIN_SIZE,
        neuron_curation=('cells released by the authors are already curated: suite2p '
                         'iscell probability > 0.5 and tracked across all days of the mouse '
                         'by Track2p; additionally neurons with an identically-zero F trace '
                         'on any day of a mouse were removed'),
        behaviour_processing=('motion energy per video frame; dropped camera frames '
                              'reinserted by linear interpolation using interframe_int; '
                              'averaged in bins of 10 frames; discretised into 5 '
                              'equal-percentile bins computed per session'),
        subject_letters=[SUBJECT_LETTER.get(s, '?') for s in subjects],
        reference=('Majnik et al. 2025, eLife 14:RP107540, '
                   'Longitudinal tracking of neuronal activity from the same cells in the '
                   'developing brain using Track2p'),
        session_info=session_info)

    # ---------------- sanity checks
    ns = len(data['neural'])
    ntr = sum(len(s) for s in data['neural'])
    print('\n--- sanity checks ---')
    assert len(data['input']) == ns and len(data['output']) == ns
    assert len(data['subject_idx']) == ns and len(data['brain_region_idx']) == ns
    for s in range(ns):
        assert len(data['input'][s]) == len(data['neural'][s]) == len(data['output'][s])
        nn = data['neural'][s][0].shape[0]
        for k in range(len(data['neural'][s])):
            nk, tk = data['neural'][s][k].shape
            assert nk == nn == len(data['brain_region_idx'][s])
            assert data['input'][s][k].shape[1] == tk
            assert data['output'][s][k].shape[1] == tk
            assert np.isfinite(data['neural'][s][k]).all()
            assert np.isfinite(data['input'][s][k]).all()
            assert data['output'][s][k].min() >= 0 and data['output'][s][k].max() < NCLASSES
    alllab = np.concatenate([o[0] for s in data['output'] for o in s])
    frac = [float(np.mean(alllab == c)) for c in range(NCLASSES)]
    allin = np.concatenate([i[0] for s in data['input'] for i in s])
    allneu = np.concatenate([n.ravel() for s in data['neural'] for n in s])
    print(f'sessions: {ns}, trials: {ntr}, '
          f'neurons total (neuron-sessions): {sum(n[0].shape[0] for n in data["neural"])}')
    print(f'trials/session: {ntr/ns:.1f}, timepoints/trial: {data["neural"][0][0].shape[1]}')
    print(f'output class fractions: {np.round(frac,4)}')
    print(f'input time range: [{allin.min():.3f}, {allin.max():.3f}] s')
    print(f'neural mean {allneu.mean():.4f}, std {allneu.std():.4f}, '
          f'min {allneu.min():.2f}, max {allneu.max():.2f}')

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nwrote {args.outfile} '
          f'({os.path.getsize(args.outfile)/1e6:.1f} MB) in {time.time()-t_start:.1f}s')


if __name__ == '__main__':
    main()
