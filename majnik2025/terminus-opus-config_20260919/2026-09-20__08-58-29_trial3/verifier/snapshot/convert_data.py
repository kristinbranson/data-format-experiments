"""
Convert the Majnik et al. 2025 (Track2p) longitudinal barrel-cortex dataset into the
decoder-compatible pickle format.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing summary (see /app/CONVERSION_NOTES.md for the full rationale):
  * neural : suite2p F.npy of the track2p-tracked cells -> baseline-corrected dF
             (reference implementation: track2p/gui/data_management.py F_processing,
              neucoeff=0.0, maximin baseline, sig_baseline=10, win_baseline=60 s)
             -> averaged in non-overlapping bins of 10 frames (paper's decoding
             preprocessing, 30 Hz / 10 = 3 Hz).
  * input  : elapsed time from session start (s), time-varying.
  * output : motion energy (videography) aligned to the imaging frames, averaged in
             the same 10-frame bins, discretized into 5 equal-percentile bins per session.
  * trials : sessions split into consecutive 60 s blocks (1800 frames = 180 bins).
"""

import os
import sys
import time
import pickle
import argparse

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter1d, maximum_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- config
DATA_ROOT = '/app/data'
FS = 30.0               # imaging (and camera) frame rate, Hz (ops['fs'])
BIN_FRAMES = 10         # paper: "averaging in bins of 10 consecutive timestamps"
TRIAL_SECONDS = 60.0    # decoder task: split sessions into 60 s trials
TRIAL_FRAMES = int(round(TRIAL_SECONDS * FS))          # 1800
TRIAL_BINS = TRIAL_FRAMES // BIN_FRAMES                # 180
NQUANTILES = 5          # decoder task: five equal-percentile bins
BRAIN_REGION = 'S1 barrel cortex (L2/3)'


# ------------------------------------------------------------------------- reference
def F_processing(F, Fneu=None, fs=FS, neucoeff=0.0, baseline='maximin',
                 sig_baseline=10.0, win_baseline=60.0, prctile_baseline=8.0):
    """Baseline-corrected fluorescence ('dF/F0') exactly as in the reference code
    (track2p/gui/data_management.py :: DataManagement.F_processing).

    With the reference default neucoeff=0.0 the neuropil trace is not used, so Fneu
    may be None (it is then not loaded, saving I/O).
    """
    if neucoeff:
        Fc = F - neucoeff * Fneu
    else:
        Fc = F.astype(np.float32, copy=True)

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


# ---------------------------------------------------------------------------- helpers
def list_sessions():
    """Return [(subject, session_name, session_dir), ...] sorted by subject then date."""
    out = []
    for subj in sorted(d for d in os.listdir(DATA_ROOT)
                       if os.path.isdir(os.path.join(DATA_ROOT, d))):
        subj_dir = os.path.join(DATA_ROOT, subj)
        for sess in sorted(d for d in os.listdir(subj_dir)
                           if os.path.isdir(os.path.join(subj_dir, d))):
            out.append((subj, sess, os.path.join(subj_dir, sess)))
    return out


def align_motion_energy(me, ifi, nframes):
    """Put the motion-energy samples on the imaging-frame grid.

    The camera is triggered by the microscope, so camera frame i == imaging frame i.
    When camera frames were dropped (len(me) < nframes, see data README) the true
    frame index of every camera sample is recovered from the inter-frame intervals
    (they are integer multiples of the median interval). Missing frames are then
    linearly interpolated. me[0] is always 0 (no preceding video frame to difference
    against) and is treated as missing as well.

    Returns (me_full (nframes,), n_missing, frame_index_of_samples)
    """
    me = np.asarray(me, dtype=np.float64)
    if len(me) == nframes:
        idx = np.arange(nframes)
    else:
        med = np.median(ifi)
        steps = np.round(ifi / med).astype(np.int64)
        idx = np.concatenate([[0], np.cumsum(steps)])
        assert idx[-1] == nframes - 1, (
            f'frame-index reconstruction failed: {idx[-1]} != {nframes - 1}')
        assert len(idx) == len(me)

    full = np.full(nframes, np.nan)
    full[idx] = me
    full[0] = np.nan                      # first-frame artifact (me[0] == 0)
    bad = np.isnan(full)
    good = ~bad
    full[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), full[good])
    return full, int(bad.sum()), idx


def bin_frames(x, nbin=BIN_FRAMES):
    """Average non-overlapping bins of nbin frames along the last axis."""
    x = np.asarray(x)
    T = x.shape[-1]
    nb = T // nbin
    x = x[..., :nb * nbin]
    return x.reshape(*x.shape[:-1], nb, nbin).mean(axis=-1)


def quantile_discretize(x, nq=NQUANTILES):
    """Discretize x into nq equal-percentile bins (thresholds from this session)."""
    edges = np.quantile(x, np.arange(1, nq) / nq)
    labels = np.searchsorted(edges, x, side='right').astype(np.int64)
    return labels, edges


# --------------------------------------------------------------------------- per session
def process_session(subj, sess, sdir, show_processing=False, verbose=True):
    t0 = time.time()
    plane = os.path.join(sdir, 'suite2p', 'plane0')
    F = np.load(os.path.join(plane, 'F.npy'))
    iscell = np.load(os.path.join(plane, 'iscell.npy'))
    ops = None
    nneurons, nframes = F.shape
    t_load = time.time() - t0

    # --- curation: the released data already contains only suite2p-classified cells
    #     (iscell[:,0]==1) that were tracked across all days by track2p. Assert this.
    assert iscell.shape[0] == nneurons
    assert np.all(iscell[:, 0] == 1), f'{subj}/{sess}: unexpected non-cell ROIs'

    # --- neural: baseline-corrected dF, then 10-frame binning
    t1 = time.time()
    dF = F_processing(F, None, fs=FS)
    t_dff = time.time() - t1
    t1 = time.time()
    dF_binned = bin_frames(dF).astype(np.float32)          # (nneurons, nbins)
    t_bin = time.time() - t1

    # --- behaviour: motion energy on the imaging grid, then the same binning
    md = os.path.join(sdir, 'move_deve')
    me_raw = np.load(os.path.join(md, 'motion_energy_glob.npy'))
    ifi = np.load(os.path.join(md, 'interframe_int.npy'))
    me_full, n_missing, sample_idx = align_motion_energy(me_raw, ifi, nframes)
    me_binned = bin_frames(me_full)                        # (nbins,)

    # --- discretize into 5 equal-percentile bins, per session
    me_labels, edges = quantile_discretize(me_binned)

    # --- time from session start (bin centres), seconds
    nbins = dF_binned.shape[1]
    t_bins = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS

    # --- split into 60 s trials
    ntrials = nbins // TRIAL_BINS
    assert ntrials * TRIAL_BINS == nbins, (
        f'{subj}/{sess}: {nbins} bins is not a multiple of {TRIAL_BINS}')
    neural_trials, input_trials, output_trials = [], [], []
    for k in range(ntrials):
        sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
        neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
        input_trials.append(t_bins[sl].astype(np.float32)[None, :])
        output_trials.append(me_labels[sl].astype(np.int64)[None, :])

    info = {
        'subject': subj,
        'session': sess,
        'n_neurons': int(nneurons),
        'n_frames': int(nframes),
        'n_bins': int(nbins),
        'n_trials': int(ntrials),
        'n_missing_camera_frames': int(n_missing),
        'motion_energy_quantile_edges': edges.tolist(),
        'duration_s': float(nframes / FS),
    }
    if verbose:
        frac = np.bincount(me_labels, minlength=NQUANTILES) / len(me_labels)
        print(f'  {subj}/{sess}: {nneurons} neurons, {nframes} frames -> {ntrials} trials '
              f'({nbins} bins), missing cam frames {n_missing}, '
              f'class fracs {np.round(frac, 3).tolist()}, '
              f'[load {t_load:.1f}s dff {t_dff:.1f}s bin {t_bin:.1f}s total {time.time()-t0:.1f}s]')

    if show_processing:
        plot_processing(subj, sess, F, dF, dF_binned, me_raw, me_full, me_binned,
                        me_labels, edges, t_bins, sample_idx, nframes,
                        neural_trials, input_trials, output_trials)

    return neural_trials, input_trials, output_trials, info


def plot_processing(subj, sess, F, dF, dF_binned, me_raw, me_full, me_binned,
                    me_labels, edges, t_bins, sample_idx, nframes,
                    neural_trials, input_trials, output_trials):
    """Visualise every processing step for one session."""
    tf = np.arange(nframes) / FS
    n_show = min(3, F.shape[0])
    fig, axes = plt.subplots(8, 1, figsize=(16, 22))

    ax = axes[0]
    for i in range(n_show):
        ax.plot(tf, F[i] + i * np.nanpercentile(F, 99), lw=.4)
    ax.set_title(f'{subj}/{sess} 1) raw F (suite2p, tracked cells), first {n_show} neurons')
    ax.set_ylabel('F (a.u.)')

    ax = axes[1]
    for i in range(n_show):
        ax.plot(tf, dF[i] + i * np.nanpercentile(dF, 99), lw=.4)
    ax.set_title('2) baseline-corrected dF (reference F_processing, maximin, 60 s window)')
    ax.set_ylabel('dF (a.u.)')

    ax = axes[2]
    for i in range(n_show):
        ax.plot(tf, dF[i], lw=.4, color='0.7')
        ax.plot(t_bins, dF_binned[i], lw=.8, color='C3')
    ax.set_xlim(0, 60)
    ax.set_title('3) 10-frame binning of dF (grey: 30 Hz, red: 3 Hz) - first 60 s')

    ax = axes[3]
    ax.imshow(dF_binned, aspect='auto', cmap='gray_r',
              vmin=0, vmax=np.percentile(dF_binned, 99),
              extent=[t_bins[0], t_bins[-1], dF_binned.shape[0], 0])
    ax.set_title('4) binned dF raster (neural output of the conversion)')
    ax.set_ylabel('neuron')

    ax = axes[4]
    ax.plot(sample_idx / FS, me_raw, lw=.4, color='0.6', label='raw motion energy (camera samples)')
    ax.plot(tf, me_full, lw=.4, color='C0', alpha=.7, label='aligned to imaging frames (gaps interpolated)')
    ax.legend(fontsize=7)
    ax.set_title('5) motion energy alignment')

    ax = axes[5]
    ax.plot(tf, me_full, lw=.4, color='0.7', label='30 Hz')
    ax.plot(t_bins, me_binned, lw=.8, color='C0', label='3 Hz binned')
    for e in edges:
        ax.axhline(e, color='C1', ls='--', lw=.7)
    ax.set_xlim(0, 120)
    ax.legend(fontsize=7)
    ax.set_title('6) binned motion energy with quintile edges (dashed) - first 120 s')

    ax = axes[6]
    ax.plot(t_bins, me_binned, lw=.6, color='0.5')
    ax2 = ax.twinx()
    ax2.step(t_bins, me_labels, where='mid', color='C2', lw=.8)
    ax2.set_ylabel('quintile label', color='C2')
    ax.set_xlim(0, 120)
    ax.set_title('7) discretization check: motion energy (grey) vs label 0-4 (green)')

    ax = axes[7]
    # verify the trial split reassembles the session and that time input is right
    cat_out = np.concatenate([o[0] for o in output_trials])
    cat_in = np.concatenate([i[0] for i in input_trials])
    cat_neu = np.concatenate([n[0] for n in neural_trials])
    ax.plot(cat_in, cat_out, lw=.6, color='C2', label='output (trials concatenated)')
    ax.plot(t_bins, me_labels, lw=.6, ls='--', color='k', label='output (session)')
    ax.plot(cat_in, cat_neu / max(np.abs(cat_neu).max(), 1e-9) * 4, lw=.4, color='C3',
            alpha=.5, label='neuron 0 dF (scaled)')
    ax.set_xlim(0, 180)
    ax.set_xlabel('time from session start (s) = decoder input')
    ax.legend(fontsize=7)
    ax.set_title('8) trials concatenated == session (alignment check), first 3 trials')

    fig.tight_layout()
    fname = f'processing_{subj}_{sess}.png'
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print(f'    saved {fname}')


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true',
                    help='process only 2 sessions (for testing)')
    ap.add_argument('--show-processing', action='store_true',
                    help='save per-step visualisations for up to 2 sessions')
    args = ap.parse_args()

    sessions = list_sessions()
    if args.sample:
        # one session from each of two different mice (different durations)
        sessions = [sessions[0], [s for s in sessions if s[0] == 'jm046'][0]]
    print(f'Processing {len(sessions)} sessions')

    subjects = sorted({s[0] for s in sessions})
    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': subjects,
        'subject_idx': [],
        'brain_regions': [BRAIN_REGION],
        'brain_region_idx': [],
        'input_names': ['time_from_session_start_s'],
        'output_names': ['motion_energy_quintile'],
        'output_values': [[f'quintile_{i+1}' for i in range(NQUANTILES)]],
        'metadata': {},
    }

    t_start = time.time()
    session_info = []
    for i, (subj, sess, sdir) in enumerate(sessions):
        show = args.show_processing and i < 2
        neural, inp, out, info = process_session(subj, sess, sdir, show_processing=show)
        data['neural'].append(neural)
        data['input'].append(inp)
        data['output'].append(out)
        data['subject_idx'].append(subjects.index(subj))
        data['brain_region_idx'].append(np.zeros(info['n_neurons'], dtype=np.int64))
        session_info.append(info)
    elapsed = time.time() - t_start
    print(f'Processed {len(sessions)} sessions in {elapsed:.1f} s '
          f'({elapsed/len(sessions):.1f} s/session)')

    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    data['metadata'] = {
        'task_description': (
            'Spontaneous (task-free) behaviour during chronic 2-photon calcium imaging of '
            'layer 2/3 barrel cortex in mouse pups (P7-P14; Majnik et al. 2025, Track2p). '
            'Mice were head-fixed on a non-motorised treadmill in the dark under '
            'sensory-minimised conditions. The decoder predicts the animal\'s motion energy '
            '(videography-based movement metric), discretized into 5 equal-percentile bins '
            'per session, from the population calcium activity (baseline-corrected dF) of '
            'neurons tracked across all days with Track2p.'),
        'time_bin_size': 1000.0 * BIN_FRAMES / FS,     # 333.33 ms
        'temporal_alignment_event': (
            'Start of the imaging session; recordings are continuous and are cut into '
            'consecutive non-overlapping 60 s trials.'),
        'off_start': 0.0,
        'off_end': TRIAL_SECONDS,
        'sampling_rate_hz': FS,
        'binning': f'{BIN_FRAMES} frames averaged (paper decoding preprocessing) -> {FS/BIN_FRAMES:.1f} Hz',
        'neural_signal': ('baseline-corrected fluorescence (dF) computed as in the track2p GUI '
                          '(F_processing: neucoeff=0.0, maximin baseline, sigma=10 frames, '
                          '60 s min/max window), averaged in 10-frame bins'),
        'neuron_curation': ('released data already contains only suite2p-classified cells '
                            '(classifier prob > 0.5) that were tracked across all days of the '
                            'mouse by Track2p; rows are matched across sessions of a mouse'),
        'output_discretization': ('motion energy binned to 3 Hz then discretized into 5 '
                                  'equal-percentile (quintile) bins using thresholds computed '
                                  'within each session'),
        'behaviour_alignment': ('camera is hardware-triggered by the microscope (frame i of the '
                                'video == imaging frame i); dropped camera frames are located '
                                'with move_deve/interframe_int.npy and linearly interpolated; '
                                'the first motion-energy sample (always 0, no preceding frame) '
                                'is interpolated as well'),
        'session_info': session_info,
        'reference': ('Majnik J, Mantez M, Zangila S, Bugeon S, Guignard L, Platel J-C, Cossart R '
                      '(2025) Longitudinal tracking of neuronal activity from the same cells in '
                      'the developing brain using Track2p. eLife 14:RP107540'),
    }

    # ------------------------------------------------------------------ sanity checks
    print('\nSanity checks:')
    ntrials = [len(t) for t in data['neural']]
    print(f'  sessions={len(data["neural"])}, subjects={len(subjects)}, total trials={sum(ntrials)}')
    for s in range(len(data['neural'])):
        nn = data['neural'][s][0].shape[0]
        for k in range(len(data['neural'][s])):
            assert data['neural'][s][k].shape == (nn, TRIAL_BINS)
            assert data['input'][s][k].shape == (1, TRIAL_BINS)
            assert data['output'][s][k].shape == (1, TRIAL_BINS)
            assert np.isfinite(data['neural'][s][k]).all()
            assert np.isfinite(data['input'][s][k]).all()
        assert len(data['brain_region_idx'][s]) == nn
    allout = np.concatenate([o[0] for s in data['output'] for o in s])
    fr = np.bincount(allout, minlength=NQUANTILES) / len(allout)
    print(f'  output class fractions (all sessions): {np.round(fr, 4).tolist()}')
    allin = np.concatenate([i[0] for s in data['input'] for i in s])
    print(f'  input range: [{allin.min():.3f}, {allin.max():.3f}] s')
    allneu = np.concatenate([n.ravel()[::101] for s in data['neural'] for n in s])
    print(f'  neural (subsampled) mean {allneu.mean():.3f}, std {allneu.std():.3f}, '
          f'min {allneu.min():.3f}, max {allneu.max():.3f}')

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nSaved {args.outfile} ({os.path.getsize(args.outfile)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
