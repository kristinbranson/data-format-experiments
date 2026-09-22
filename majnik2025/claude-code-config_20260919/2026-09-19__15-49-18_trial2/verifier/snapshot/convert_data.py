"""
Convert the Majnik et al. 2025 (eLife 107540, Track2p) longitudinal 2-photon dataset
into the decoder-compatible pickle format.

Dataset: 6 mice (jm031, jm032, jm038, jm039, jm040, jm046), 41 daily sessions of
spontaneous activity in L2/3 barrel cortex (P7-P14), GCaMP8m imaged at 30 Hz, with
simultaneous 30 Hz videography quantified as 'motion energy'.  The suite2p folders
released with the paper already contain only the Track2p-tracked, iscell-curated
neurons, row-matched across the days of a mouse.

Decoder task
    input  : time elapsed from the beginning of the session (s), time varying
    output : motion energy, discretised into 5 equal-percentile (quintile) bins,
             thresholds chosen per session, time varying
    trials : the continuous session is cut into consecutive 60 s trials

Processing (follows the paper / reference code, see CONVERSION_NOTES.md):
    * dF/F  = maximin baseline-corrected fluorescence, computed exactly as
      track2p/gui/data_management.py:F_processing with suite2p default parameters
      (neucoeff from ops = 0.7, sig_baseline = 10 frames, win_baseline = 60 s).
    * both dF/F and motion energy are averaged in bins of 10 consecutive frames
      (333.33 ms), as the paper does "for all decoding analysis".
    * motion energy is put back on the imaging frame grid using the camera
      inter-frame intervals (the camera is triggered by the microscope, so frame i
      of the camera == frame i of the imaging unless a trigger was dropped); dropped
      frames are linearly interpolated, as suggested by the dataset README.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d

DATA_DIR = '/app/data'

# --- fixed processing parameters -------------------------------------------------
BIN_FRAMES = 10            # paper: "averaging in bins of 10 consecutive timestamps"
TRIAL_SEC = 60.0           # task specification: 60 s trials
N_QUANTILES = 5            # task specification: 5 equal-percentile bins
SIG_BASELINE = 10.0        # suite2p default (frames)
WIN_BASELINE = 60.0        # suite2p default (s)

# subject -> (paper mouse label, postnatal day of the first session)
# Postnatal days are read from Fig. 5B of the paper (mice are named in alphabetically
# increasing order: jm031 = mouse A ... jm046 = mouse F).
SUBJECT_INFO = {
    'jm031': ('A', 7),
    'jm032': ('B', 7),
    'jm038': ('C', 8),
    'jm039': ('D', 8),
    'jm040': ('E', 9),
    'jm046': ('F', 8),
}
SUBJECTS = list(SUBJECT_INFO.keys())

BRAIN_REGION = 'S1 barrel cortex'
OUTPUT_VALUES = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']


# ---------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------
def list_sessions(subject):
    """Session directories of one subject, chronologically sorted."""
    d = os.path.join(DATA_DIR, subject)
    return sorted(f.path for f in os.scandir(d) if f.is_dir())


def load_session_raw(session_dir):
    """Load the raw arrays of one session.

    Returns F, Fneu (n_neurons, n_frames), fs, n_frames.
    """
    p = os.path.join(session_dir, 'suite2p', 'plane0')
    F = np.load(os.path.join(p, 'F.npy'))
    Fneu = np.load(os.path.join(p, 'Fneu.npy'))
    iscell = np.load(os.path.join(p, 'iscell.npy'))
    ops = np.load(os.path.join(p, 'ops.npy'), allow_pickle=True).item()
    # The released data contains only Track2p-tracked cells that already passed the
    # suite2p classifier; assert rather than filter so that any exception is visible.
    assert np.all(iscell[:, 0] == 1), f'{session_dir}: unexpected non-cell ROIs'
    assert F.shape == Fneu.shape
    assert F.shape[1] == ops['nframes'], f'{session_dir}: F length != ops["nframes"]'
    return F, Fneu, float(ops['fs']), int(ops['nframes']), float(ops['neucoeff'])


# ---------------------------------------------------------------------------------
# neural processing
# ---------------------------------------------------------------------------------
def f_processing(F, Fneu, fs, neucoeff, baseline='maximin',
                 sig_baseline=SIG_BASELINE, win_baseline=WIN_BASELINE):
    """dF/F as in track2p/gui/data_management.py:F_processing (suite2p 'maximin').

    Returns (dff, F0) where dff = (F - neucoeff*Fneu) - F0 and F0 is the maximin
    baseline.  Kept identical to the reference implementation apart from returning
    the baseline as well (used for the diagnostic plots).
    """
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    if baseline == 'maximin':
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    elif baseline == 'constant':
        Flow = np.amin(gaussian_filter(Fc, [0., sig_baseline]))
    else:
        Flow = 0.
    return Fc - Flow, Flow


def bin_average(x, k=None):
    """Average non-overlapping bins of k consecutive samples along the last axis.

    Trailing samples that do not fill a bin are dropped.
    """
    k = BIN_FRAMES if k is None else k
    n = (x.shape[-1] // k) * k
    return x[..., :n].reshape(*x.shape[:-1], n // k, k).mean(axis=-1)


# ---------------------------------------------------------------------------------
# behaviour processing
# ---------------------------------------------------------------------------------
def load_motion_energy(session_dir, n_frames):
    """Motion energy on the imaging frame grid.

    The camera is triggered by the 2-photon acquisition, so camera frame i
    corresponds to imaging frame i as long as no trigger was dropped.  Dropped
    triggers show up as inter-frame intervals that are an integer multiple of the
    nominal one; the cumulative sum of the rounded intervals therefore gives the
    imaging-frame index of every camera frame.  Missing values (dropped frames, and
    the first frame, for which no frame difference exists) are linearly interpolated,
    as suggested by the dataset README.

    Returns (motion_energy (n_frames,), invalid mask (n_frames,) bool).
    """
    md = os.path.join(session_dir, 'move_deve')
    me = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
    itv = np.load(os.path.join(md, 'interframe_int.npy'))
    assert len(itv) == len(me) - 1

    nominal = np.median(itv)
    steps = np.round(itv / nominal).astype(np.int64)
    idx = np.concatenate([[0], np.cumsum(steps)])

    out = np.full(n_frames, np.nan)
    keep = idx < n_frames                    # a few sessions record past the last 2p frame
    out[idx[keep]] = me[keep]
    out[0] = np.nan                          # me[0] is a placeholder (no previous frame)

    missing = np.isnan(out)
    if missing.all():
        raise ValueError(f'{session_dir}: no motion energy could be aligned')
    out[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(~missing),
                             out[~missing])
    return out, missing


def discretize_quantiles(x, n_bins=N_QUANTILES):
    """Assign each value of x to one of n_bins equal-percentile bins.

    Edges are the n_bins-1 interior quantiles of x itself, so the bins hold equal
    numbers of samples (up to ties).  Returns (labels int64, edges).
    """
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.percentile(x, qs)
    labels = np.searchsorted(edges, x, side='right').astype(np.int64)
    return labels, edges


# ---------------------------------------------------------------------------------
# per-session conversion
# ---------------------------------------------------------------------------------
def convert_session(session_dir, neural_scaling='center', verbose=True, neucoeff=None):
    """Convert one session to trial lists.

    neural_scaling: how the dF/F is conditioned for the decoder.  For a linear
        readout with a bias term (which is what the provided decoder fits on top of
        the per-session projection) subtracting a per-neuron constant and dividing
        the whole session by one scalar are information-preserving; they only change
        the conditioning of the optimisation and of the SVD initialisation.
        'none'   : dF/F as returned by f_processing (arbitrary fluorescence units)
        'global' : dF/F divided by one scalar per session (its pooled s.d.)
        'center' : (default) additionally subtract each neuron's session mean, so
                   that the projection is initialised with the SVD of the
                   *fluctuations* rather than of the (large, positive) mean offset
        'zscore' : each neuron's dF/F z-scored within the session; this also
                   re-weights neurons by 1/s.d., which is not information preserving
    """
    t0 = time.time()
    subject = os.path.basename(os.path.dirname(session_dir))
    sess_name = os.path.basename(session_dir)

    F, Fneu, fs, n_frames, ops_neucoeff = load_session_raw(session_dir)
    if neucoeff is None:
        neucoeff = ops_neucoeff
    t_load = time.time() - t0

    # --- neural: dF/F then 10-frame averaging ---
    t1 = time.time()
    dff, F0 = f_processing(F, Fneu, fs, neucoeff)
    dff_binned = bin_average(dff).astype(np.float32)
    t_dff = time.time() - t1

    # --- behaviour: motion energy on the imaging grid, then 10-frame averaging ---
    me, me_missing = load_motion_energy(session_dir, n_frames)
    me_binned = bin_average(me)
    # fraction of interpolated (dropped) camera frames contributing to each bin
    missing_binned = bin_average(me_missing.astype(np.float64))

    # --- cut into trials ---
    bins_per_trial = int(round(TRIAL_SEC * fs / BIN_FRAMES))
    n_trials = dff_binned.shape[1] // bins_per_trial
    n_used = n_trials * bins_per_trial
    dff_binned = dff_binned[:, :n_used]
    me_binned = me_binned[:n_used]
    missing_binned = missing_binned[:n_used]

    # --- neural scaling (see CONVERSION_NOTES.md Step 5) ---
    if neural_scaling == 'global':
        scale = float(dff_binned.std())
        neural_all = dff_binned / scale
    elif neural_scaling == 'center':
        scale = float(dff_binned.std())
        neural_all = (dff_binned - dff_binned.mean(axis=1, keepdims=True)) / scale
    elif neural_scaling == 'zscore':
        mu = dff_binned.mean(axis=1, keepdims=True)
        sd = dff_binned.std(axis=1, keepdims=True)
        neural_all = (dff_binned - mu) / np.maximum(sd, 1e-6)
        scale = None
    elif neural_scaling == 'none':
        neural_all = dff_binned
        scale = 1.0
    else:
        raise ValueError(neural_scaling)
    neural_all = neural_all.astype(np.float32)

    # --- output: quintiles of motion energy, thresholds per session ---
    labels_all, edges = discretize_quantiles(me_binned)

    # --- input: time from session start (s), at the centre of each bin ---
    bin_dt = BIN_FRAMES / fs
    time_all = (np.arange(n_used) + 0.5) * bin_dt

    neural, inputs, outputs = [], [], []
    for tr in range(n_trials):
        sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
        neural.append(np.ascontiguousarray(neural_all[:, sl]))
        inputs.append(time_all[sl][None, :].astype(np.float32))
        outputs.append(labels_all[sl][None, :])

    label, p0 = SUBJECT_INFO[subject]
    day_idx = list_sessions(subject).index(session_dir)
    info = {
        'subject': subject,
        'mouse_label': label,
        'session': sess_name,
        'date': sess_name[:10],
        'day_index': day_idx,
        'postnatal_day': p0 + day_idx,
        'n_neurons': int(neural_all.shape[0]),
        'n_frames': int(n_frames),
        'fs': fs,
        'neucoeff': neucoeff,
        'n_trials': int(n_trials),
        'session_duration_s': n_frames / fs,
        'frac_frames_interpolated': float(me_missing.mean()),
        'motion_energy_quintile_edges': edges.tolist(),
        'motion_energy_range': [float(me_binned.min()), float(me_binned.max())],
        'dff_scale_divisor': scale,
        'frac_bins_with_interpolated_frames': float((missing_binned > 0).mean()),
    }
    if verbose:
        print(f'  {subject}/{sess_name}: {info["n_neurons"]} neurons, {n_trials} trials, '
              f'P{info["postnatal_day"]}, dropped camera frames '
              f'{100 * info["frac_frames_interpolated"]:.3f}% '
              f'[load {t_load:.1f}s, dff {t_dff:.1f}s, total {time.time() - t0:.1f}s]',
              flush=True)

    diag = None
    if os.environ.get('T2P_KEEP_DIAG'):
        diag = dict(F=F, F0=F0, dff=dff, me=me, me_missing=me_missing,
                    dff_binned=dff_binned, me_binned=me_binned,
                    labels_all=labels_all, edges=edges, time_all=time_all,
                    bins_per_trial=bins_per_trial, fs=fs)
    return dict(neural=neural, input=inputs, output=outputs, info=info,
                brain_region_idx=np.zeros(neural_all.shape[0], dtype=np.int64)), diag


def _worker(args):
    session_dir, neural_scaling, neucoeff = args
    sess, _ = convert_session(session_dir, neural_scaling=neural_scaling,
                              neucoeff=neucoeff)
    return sess


# ---------------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------------
def plot_processing(session_dir, neural_scaling, outfile):
    """Diagnostic plot of every processing step for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.environ['T2P_KEEP_DIAG'] = '1'
    sess, d = convert_session(session_dir, neural_scaling=neural_scaling, verbose=False)
    del os.environ['T2P_KEEP_DIAG']

    fs, bpt = d['fs'], d['bins_per_trial']
    t_raw = np.arange(len(d['me'])) / fs
    t_bin = d['time_all']
    cell = int(np.argmax(d['dff'].std(axis=1)))   # a lively example neuron

    fig, ax = plt.subplots(7, 1, figsize=(20, 22))

    # 1. raw F and the maximin baseline
    ax[0].plot(t_raw, d['F'][cell], lw=.4, color='0.5', label='F (raw)')
    ax[0].plot(t_raw, d['F0'][cell], lw=1.2, color='r', label='F0 (maximin baseline)')
    ax[0].set_title(f'1) raw fluorescence and baseline, neuron {cell} — '
                    f'{os.path.basename(os.path.dirname(session_dir))}/'
                    f'{os.path.basename(session_dir)}')
    ax[0].legend(loc='upper right'); ax[0].set_ylabel('a.u.')

    # 2. dF/F before and after 10 frame binning
    ax[1].plot(t_raw, d['dff'][cell], lw=.4, color='0.6', label='dF/F (30 Hz)')
    ax[1].plot(t_bin, d['dff_binned'][cell], lw=.8, color='g',
               label='dF/F (10-frame average, 3 Hz)')
    ax[1].set_title('2) baseline-corrected dF/F and its 10-frame average')
    ax[1].legend(loc='upper right'); ax[1].set_ylabel('dF/F (a.u.)')

    # 3. motion energy: raw camera frames vs aligned/interpolated
    me_raw = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
    ax[2].plot(np.arange(len(me_raw)) / fs, me_raw, lw=.4, color='0.6',
               label='motion energy (raw camera order)')
    ax[2].plot(t_raw, d['me'], lw=.4, color='b', label='aligned to imaging frames')
    ax[2].plot(t_bin, d['me_binned'], lw=.8, color='k', label='10-frame average')
    nmiss = int(d['me_missing'].sum())
    if nmiss:
        ax[2].plot(np.flatnonzero(d['me_missing']) / fs,
                   d['me'][d['me_missing']], 'r.', ms=4,
                   label=f'interpolated ({nmiss} frames)')
    ax[2].set_title('3) motion energy alignment and binning')
    ax[2].legend(loc='upper right'); ax[2].set_ylabel('motion energy')

    # 4. discretisation into quintiles
    ax[3].plot(t_bin, d['me_binned'], lw=.8, color='k')
    for e in d['edges']:
        ax[3].axhline(e, color='r', ls='--', lw=.8)
    ax[3].set_yscale('log')
    ax[3].set_title('4) motion energy (log scale) with the 4 session quintile edges')
    ax[3].set_ylabel('motion energy')
    ax2 = ax[3].twinx()
    ax2.plot(t_bin, d['labels_all'], lw=.8, color='m', alpha=.6)
    ax2.set_ylabel('quintile label', color='m')

    # 5. neural raster (as saved) with motion energy overlaid
    neural_cat = np.concatenate(sess['neural'], axis=1)
    order = np.argsort(-neural_cat.std(axis=1))
    ax[4].imshow(neural_cat[order[:200]], aspect='auto', cmap='gray_r',
                 vmin=0, vmax=np.percentile(neural_cat, 99.5),
                 extent=[t_bin[0], t_bin[-1], 200, 0])
    ax4 = ax[4].twinx()
    ax4.plot(t_bin, d['me_binned'], lw=.7, color='b', alpha=.7)
    ax4.set_ylabel('motion energy', color='b')
    ax[4].set_title('5) saved neural data (200 most active neurons) with motion energy; '
                    'vertical lines = 60 s trial boundaries')
    for tr in range(1, len(sess['neural'])):
        ax[4].axvline(t_bin[tr * bpt], color='orange', lw=.6)
    ax[4].set_ylabel('neuron')

    # 6. round-trip check: concatenated trials must equal the continuous arrays
    inp_cat = np.concatenate(sess['input'], axis=1)[0]
    out_cat = np.concatenate(sess['output'], axis=1)[0]
    n_used = len(out_cat)
    ax[5].plot(t_bin, inp_cat, color='b', label='input[0]: time from session start (s)')
    ax[5].plot(t_bin, t_bin, 'r--', lw=.8, label='expected time')
    ax[5].set_title('6) decoder input: elapsed time (blue) vs expected (red dashed) — '
                    f'max abs error {np.abs(inp_cat - t_bin).max():.2e} s')
    ax[5].legend(loc='upper left'); ax[5].set_ylabel('s')

    # 7. per-trial zoom on the first trial: output labels vs motion energy
    sl = slice(0, bpt)
    ax[6].plot(t_bin[sl], d['me_binned'][sl], 'k-', label='motion energy (trial 0)')
    ax[6].set_yscale('log')
    for e in d['edges']:
        ax[6].axhline(e, color='r', ls='--', lw=.8)
    ax6 = ax[6].twinx()
    ax6.step(t_bin[sl], out_cat[sl], where='mid', color='m', label='saved output label')
    ax6.set_ylabel('quintile label', color='m')
    frac = [np.mean(out_cat == k) for k in range(N_QUANTILES)]
    ax[6].set_title('7) trial 0: motion energy vs saved labels — session label fractions '
                    + ', '.join(f'{f:.3f}' for f in frac))
    ax[6].set_xlabel('time from session start (s)')

    # round-trip: concatenating the trials must give back the continuous binned arrays
    assert np.allclose(out_cat, d['labels_all'][:n_used])
    assert np.allclose(inp_cat, d['time_all'][:n_used])
    if sess['info']['dff_scale_divisor'] is not None:
        expect = d['dff_binned'][:, :n_used]
        expect = expect - expect.mean(axis=1, keepdims=True)
        assert np.allclose(neural_cat * sess['info']['dff_scale_divisor'], expect,
                           atol=1e-3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=110)
    plt.close(fig)
    print(f'  wrote {outfile}', flush=True)


# ---------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------
def main():
    global BIN_FRAMES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true',
                    help='process only 2 sessions (for testing)')
    ap.add_argument('--show-processing', action='store_true',
                    help='plot every processing step for up to 2 sessions')
    ap.add_argument('--neural-scaling', type=str, default='center',
                    choices=['none', 'global', 'center', 'zscore'])
    ap.add_argument('--neucoeff', type=float, default=None,
                    help='neuropil coefficient; default: ops["neucoeff"] (= 0.7, the '
                         'suite2p default used to preprocess this dataset)')
    ap.add_argument('--bin-frames', type=int, default=BIN_FRAMES,
                    help='frames averaged per time bin; default 10, i.e. the '
                         '333 ms bins the paper uses for all decoding analyses')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    BIN_FRAMES = args.bin_frames

    t_start = time.time()

    session_dirs = []
    for s in SUBJECTS:
        session_dirs.extend(list_sessions(s))
    if args.sample:
        # one early 20 min session (jm031 = mouse A, P7, weak behaviour coupling) and
        # one late 30 min session (jm046 = mouse F, P13, strong behaviour coupling)
        session_dirs = [os.path.join(DATA_DIR, 'jm031', '2023-10-18_a'),
                        os.path.join(DATA_DIR, 'jm046', '2024-09-08_a')]
    print(f'Converting {len(session_dirs)} sessions '
          f'(neural scaling: {args.neural_scaling})', flush=True)

    if args.show_processing:
        for sd in session_dirs[:2]:
            sid = f'{os.path.basename(os.path.dirname(sd))}_{os.path.basename(sd)}'
            plot_processing(sd, args.neural_scaling, f'/app/processing_{sid}.png')

    t_conv = time.time()
    workers = min(args.workers, len(session_dirs))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            sessions = list(ex.map(_worker,
                                   [(sd, args.neural_scaling, args.neucoeff)
                                    for sd in session_dirs]))
    else:
        sessions = [_worker((sd, args.neural_scaling, args.neucoeff))
                    for sd in session_dirs]
    print(f'Conversion of {len(sessions)} sessions took {time.time() - t_conv:.1f}s '
          f'({(time.time() - t_conv) / len(sessions):.1f}s / session)', flush=True)

    data = {
        'neural': [s['neural'] for s in sessions],
        'input': [s['input'] for s in sessions],
        'output': [s['output'] for s in sessions],
        'subjects': SUBJECTS,
        'subject_idx': np.array([SUBJECTS.index(s['info']['subject'])
                                 for s in sessions], dtype=np.int64),
        'brain_regions': [BRAIN_REGION],
        'brain_region_idx': [s['brain_region_idx'] for s in sessions],
        'input_names': ['time from session start (s)'],
        'output_names': ['motion energy quintile'],
        'output_values': [OUTPUT_VALUES],
        'metadata': {
            'task_description':
                'Spontaneous behaviour during head-fixed 2-photon calcium imaging of '
                'layer 2/3 barrel cortex in mouse pups (P7-P14) freely moving on a '
                'non-motorised treadmill, in the dark under sensory-minimised '
                'conditions. There is no task and no stimulus: the decoder predicts '
                'the animal\'s movement (motion energy from videography, discretised '
                'into 5 equal-percentile bins per session) from the population '
                'calcium activity, given the elapsed time in the session.',
            'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,   # ms (10 frames at 30 Hz)
            'temporal_alignment_event':
                'start of the imaging session (first 2-photon frame); the continuous '
                'session is cut into consecutive non-overlapping 60 s trials, and each '
                'trial is aligned to its own start',
            'off_start': 0.0,
            'off_end': TRIAL_SEC,
            'neural_signal':
                'dF/F: neuropil-corrected fluorescence (F - neucoeff*Fneu, '
                'neucoeff = 0.7) minus the suite2p maximin baseline '
                '(gaussian sigma = 10 frames, 60 s min/max filter), averaged in bins '
                'of 10 consecutive frames'
                + {'global': ', divided by one scalar per session (the pooled s.d.)',
                   'center': ', mean-subtracted per neuron and divided by one scalar '
                             'per session (the pooled s.d.)',
                   'zscore': ', z-scored per neuron within the session',
                   'none': ''}[args.neural_scaling],
            'neural_scaling': args.neural_scaling,
            'sampling_rate_hz': 30.0 / BIN_FRAMES,
            'imaging_rate_hz': 30.0,
            'trial_duration_s': TRIAL_SEC,
            'bins_per_trial': int(round(TRIAL_SEC * 30.0 / BIN_FRAMES)),
            'behaviour':
                'motion energy = summed squared pixel-wise difference between '
                'consecutive frames of a 30 Hz infrared video of the mouse, '
                'sampled synchronously with the imaging (camera triggered by the '
                'microscope); dropped camera frames are linearly interpolated',
            'output_discretisation':
                'quintiles of the binned motion energy, thresholds computed '
                'separately for each session',
            'session_info': [s['info'] for s in sessions],
            'source':
                'Majnik et al. 2025, eLife 14:RP107540, '
                'https://doi.org/10.7554/eLife.107540.1 (Track2p dataset)',
        },
    }

    # ---- sanity checks on the assembled structure ----
    ns = len(data['neural'])
    assert len(data['input']) == len(data['output']) == ns
    assert len(data['subject_idx']) == ns and len(data['brain_region_idx']) == ns
    ntr = 0
    for i in range(ns):
        nn_i = data['neural'][i][0].shape[0]
        assert len(data['brain_region_idx'][i]) == nn_i
        assert len(data['input'][i]) == len(data['output'][i]) == len(data['neural'][i])
        for tr in range(len(data['neural'][i])):
            n, T = data['neural'][i][tr].shape
            assert n == nn_i
            assert T == data['metadata']['bins_per_trial']
            assert data['input'][i][tr].shape == (1, T)
            assert data['output'][i][tr].shape == (1, T)
            assert np.isfinite(data['neural'][i][tr]).all()
            assert data['output'][i][tr].min() >= 0 and data['output'][i][tr].max() < N_QUANTILES
            ntr += 1
    print(f'Structure checks passed: {ns} sessions, {ntr} trials, '
          f'{sum(d["neural"][0].shape[0] for d in sessions)} neurons total', flush=True)

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f)
    print(f'Wrote {args.outfile} '
          f'({os.path.getsize(args.outfile) / 1e6:.1f} MB) in '
          f'{time.time() - t_start:.1f}s total', flush=True)

    # ---- short summary ----
    allout = np.concatenate([o[0] for s in data['output'] for o in s])
    print('Output label fractions (pooled): ' +
          ', '.join(f'{k}:{np.mean(allout == k):.3f}' for k in range(N_QUANTILES)))
    alltime = np.concatenate([i[0] for s in data['input'] for i in s])
    print(f'Input (time in session) range: [{alltime.min():.3f}, {alltime.max():.3f}] s')
    allneu = np.concatenate([n.ravel() for s in data['neural'] for n in s])
    print(f'Neural range: [{allneu.min():.3f}, {allneu.max():.3f}], '
          f'mean {allneu.mean():.4f}, std {allneu.std():.4f}')


if __name__ == '__main__':
    main()
