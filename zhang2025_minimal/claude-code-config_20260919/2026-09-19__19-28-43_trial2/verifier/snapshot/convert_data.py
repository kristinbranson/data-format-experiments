"""Convert the IBL brain-wide map (BWM) dataset into the decoder data format.

The processing follows the pipeline of the methods paper ("Exploiting correlations
across trials and behavioral sessions to improve neural decoding", Zhang et al.) as
implemented in /app/code/code_zhang2025 (`src/0_data_caching.py`,
`src/utils/ibl_data_utils.py`), and the inclusion criteria of the data paper
("A brain-wide map of neural activity during complex behaviour").

Summary of the choices (see README-style notes next to each step below):

* Sessions      : the 459 sessions of the BWM release freeze (`data/bwm_release.csv`),
                  which already passed the data paper's session/insertion criteria.
                  Neurons of all probes of a session are merged, as in `prepare_data`.
* Neurons       : all spike-sorted units of the session (`qc=None` in
                  `load_spiking_data`), i.e. the methods paper's "we bin spike counts
                  using all neurons, sorted by Kilosort, from each session".
* Trials        : `load_trials_and_mask` criteria of the reference code
                  (0.08 s <= first movement - stimulus onset <= 2 s, no NaN in
                  stimOn_times / choice / feedback_times / probabilityLeft /
                  firstMovement_times / feedbackType, a choice was made, and
                  feedback_times - goCue_times <= 10 s), plus trials for which the
                  behavioural traces needed as decoder outputs are missing.
* Alignment     : stimulus onset, window (-0.5, 1.5) s, 20 ms bins -> 100 bins,
                  i.e. the `params` dict of `src/0_data_caching.py`.
* Behaviour     : wheel speed |velocity| and whisker motion energy, interpolated on the
                  same 20 ms grid as the spikes (`get_behavior_per_interval`), then
                  discretized into 3 classes at the within-session tertiles.

Usage:
    python convert_data.py [--out /app/converted_data.pkl] [--n-workers N]
                           [--limit N] [--eids EID [EID ...]]
"""

import argparse
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ----------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------

ONE_CACHE_DIR = '/app/data/one_cache'
BWM_RELEASE_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
DATALIMIT_SUBSET = '/app/data/DATALIMIT_SUBSET.csv'  # present only for the reduced dataset

# Trial setup, identical to `params` in code_zhang2025/src/0_data_caching.py.
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)   # seconds relative to the alignment event
BINSIZE = 0.02              # seconds
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100

# Trial exclusion criteria of load_trials_and_mask() (brainwidemap / zhang2025).
MIN_RT, MAX_RT, MAX_TRIAL_LEN = 0.08, 2.0, 10.0
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']

N_BEHAVIOR_CLASSES = 3      # wheel speed / whisker motion energy are split at tertiles
MIN_TRIALS_PER_SESSION = 2  # the decoder needs at least two trials per session


def get_one():
    """ONE client reading the staged cache (no network access to Alyx)."""
    from one.api import ONE
    return ONE(base_url='https://openalyx.internationalbrainlab.org',
               silent=True, cache_dir=ONE_CACHE_DIR)


# ----------------------------------------------------------------------------------
# Trials
# ----------------------------------------------------------------------------------

def load_trials_and_mask(sess_loader):
    """Trials table and inclusion mask.

    Re-implementation of `load_trials_and_mask` of the reference code
    (code_zhang2025/src/utils/ibl_data_utils.py, itself taken from brainwidemap) with
    the arguments used by its `prepare_data`: min_rt=0.08, max_rt=2.0,
    max_trial_len=10.0, the default NaN exclusions and exclude_nochoice=True.
    These are also the trial exclusions described in the data paper.
    """
    if sess_loader.trials.empty:
        sess_loader.load_trials()
    trials = sess_loader.trials

    query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
    query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
    query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
    for event in NAN_EXCLUDE:
        query += f' | {event}.isnull()'
    query += ' | (choice == 0)'   # exclude_nochoice

    mask = ~trials.eval(query)
    return trials, mask.to_numpy()


def trial_number_in_block(trials):
    """0-based index of each trial within its block of constant probabilityLeft.

    Computed on the full (unfiltered) trials table so that the count reflects the
    animal's actual position in the block, independent of the trial exclusions.
    """
    pleft = trials['probabilityLeft'].to_numpy()
    new_block = np.ones(len(pleft), dtype=bool)
    new_block[1:] = pleft[1:] != pleft[:-1]
    block_id = np.cumsum(new_block) - 1
    block_starts = np.flatnonzero(new_block)
    idx_in_block = np.arange(len(pleft)) - block_starts[block_id]
    return idx_in_block.astype(np.float64)


# ----------------------------------------------------------------------------------
# Spikes
# ----------------------------------------------------------------------------------

def load_spikes_and_clusters(one, eid, pids, probe_names):
    """Load and merge the spike sorting of every probe of a session.

    Mirrors `load_spiking_data` (with qc=None, i.e. all units) followed by
    `merge_probes` in the reference code.
    """
    from brainbox.io.one import SpikeSortingLoader

    spikes_list, clusters_list = [], []
    for pid, pname in zip(pids, probe_names):
        loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
        spikes, clusters, channels = loader.load_spike_sorting()
        if len(spikes) == 0 or len(clusters) == 0:
            raise RuntimeError(f'no spike sorting for probe {pname}')
        clusters_labeled = SpikeSortingLoader.merge_clusters(
            spikes, clusters, channels, compute_metrics=False).to_df()
        spikes_list.append(spikes)
        clusters_list.append(clusters_labeled)

    # merge_probes(): concatenate probes, re-indexing cluster ids, and sort by time
    merged_spikes, merged_clusters, cluster_max = [], [], 0
    for clusters, spikes in zip(clusters_list, spikes_list):
        spikes = dict(spikes)
        spikes['clusters'] = spikes['clusters'] + cluster_max
        cluster_max = clusters.index.max() + 1
        merged_spikes.append(spikes)
        merged_clusters.append(clusters)
    clusters = pd.concat(merged_clusters, ignore_index=True)
    spikes = {k: np.concatenate([s[k] for s in merged_spikes])
              for k in ('times', 'clusters')}
    sort_idx = np.argsort(spikes['times'], kind='stable')
    spikes = {k: v[sort_idx] for k, v in spikes.items()}
    return spikes, clusters


def bin_spikes(spike_times, spike_clusters, interval_begs, interval_ends):
    """Bin spikes of every trial into `N_BINS` non-overlapping bins.

    Equivalent to `bin_spiking_data` / `get_spike_data_per_interval` of the reference
    code (spikes in [t_beg, t_end), bin index floor((t - t_beg) / binsize)), but
    vectorized: only the units that fired at least once in the session are kept, in
    increasing cluster order, which is what `clusters_used_in_bins` returns there.

    Returns
    -------
    binned : (n_trials, n_units, N_BINS) uint16 array of spike counts
    unit_ids : (n_units,) cluster ids of axis 1 of `binned`
    """
    unit_ids = np.unique(spike_clusters)
    n_units = len(unit_ids)
    # map cluster id -> row index
    lookup = np.zeros(int(unit_ids.max()) + 1, dtype=np.int64)
    lookup[unit_ids] = np.arange(n_units)

    n_trials = len(interval_begs)
    binned = np.zeros((n_trials, n_units, N_BINS), dtype=np.uint16)
    i_beg = np.searchsorted(spike_times, interval_begs, side='left')
    i_end = np.searchsorted(spike_times, interval_ends, side='left')
    for trial in range(n_trials):
        sl = slice(i_beg[trial], i_end[trial])
        if sl.stop <= sl.start:
            continue
        bin_idx = np.floor((spike_times[sl] - interval_begs[trial]) / BINSIZE).astype(np.int64)
        keep = (bin_idx >= 0) & (bin_idx < N_BINS)   # guard against rounding at the edge
        rows = lookup[spike_clusters[sl][keep]]
        counts = np.bincount(rows * N_BINS + bin_idx[keep], minlength=n_units * N_BINS)
        binned[trial] = counts.reshape(n_units, N_BINS)
    return binned, unit_ids


# ----------------------------------------------------------------------------------
# Behaviour
# ----------------------------------------------------------------------------------

def load_behavior_trace(sess_loader, name):
    """Session-wide time series of a behavioural variable.

    Same sources as `load_target_behavior` in the reference code: wheel speed is the
    absolute value of the Gaussian-smoothed wheel velocity, whisker motion energy is
    taken from the left camera, falling back to the right camera when it is missing.
    """
    if name == 'wheel-speed':
        if sess_loader.wheel.empty:
            sess_loader.load_wheel()
        return (sess_loader.wheel['times'].to_numpy(),
                np.abs(sess_loader.wheel['velocity'].to_numpy()))
    if name == 'whisker-motion-energy':
        for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
            try:
                sess_loader.load_motion_energy(views=[view])
                me = sess_loader.motion_energy[key]
                return (me['times'].to_numpy(),
                        me['whiskerMotionEnergy'].to_numpy())
            except Exception:
                continue
        raise RuntimeError('no whisker motion energy for either camera')
    raise NotImplementedError(name)


def bin_behavior(target_times, target_vals, interval_begs, interval_ends):
    """Interpolate a behavioural trace onto the trial time bins.

    Follows `get_behavior_per_interval`: the trace is evaluated (linear
    interpolation) at the right edge of each of the `N_BINS` bins, a trial is
    discarded when the trace does not cover the trial window (starts more than one
    bin late or ends more than one bin early) or holds no sample at all.

    Unlike the reference code, which is called with allow_nans=True, trials whose
    interpolated values contain NaNs are discarded as well: the decoder data format
    does not allow NaNs, and these are trials where the video/wheel tracking failed.
    """
    n_trials = len(interval_begs)
    binned = np.zeros((n_trials, N_BINS), dtype=np.float64)
    good = np.zeros(n_trials, dtype=bool)

    idxs_beg = np.searchsorted(target_times, interval_begs, side='right')
    idxs_end = np.searchsorted(target_times, interval_ends, side='left')
    for trial in range(n_trials):
        t = target_times[idxs_beg[trial]:idxs_end[trial]]
        v = target_vals[idxs_beg[trial]:idxs_end[trial]]
        if len(v) == 0:
            continue
        if np.abs(interval_begs[trial] - t[0]) > BINSIZE:      # data starts too late
            continue
        if np.abs(interval_ends[trial] - t[-1]) > BINSIZE:     # data ends too early
            continue
        x = np.linspace(interval_begs[trial] + BINSIZE, interval_ends[trial], N_BINS)
        y = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
        if np.any(~np.isfinite(y)):
            continue
        binned[trial] = y
        good[trial] = True
    return binned, good


def discretize(values, n_classes=N_BEHAVIOR_CLASSES):
    """Discretize a continuous behavioural variable into `n_classes` classes.

    The class boundaries are the quantiles of all values of the session (every
    retained trial and time bin), so the classes are the within-session tertiles.
    Absolute values are not comparable across sessions -- whisker motion energy is
    in camera-specific units and depends on illumination and the position of the
    bounding box, and wheel-speed distributions differ between animals -- so
    session-wise boundaries keep the class labels comparable across the sessions
    that the decoder is trained on jointly, and keep the classes balanced.
    """
    quantiles = np.quantile(values, np.arange(1, n_classes) / n_classes)
    # ties (e.g. the many exactly-zero wheel-speed samples) all fall in the lowest
    # class, so make the boundaries strictly increasing to avoid empty classes
    edges = np.maximum.accumulate(quantiles)
    return np.digitize(values, edges, right=False).astype(np.int64), edges


# ----------------------------------------------------------------------------------
# One session
# ----------------------------------------------------------------------------------

def process_session(args):
    """Convert one session. Returns a dict, or None if the session is unusable."""
    eid, subject, lab, pids, probe_names = args
    from brainbox.io.one import SessionLoader

    t0 = time.time()
    one = get_one()

    # --- trials -------------------------------------------------------------------
    sess_loader = SessionLoader(one=one, eid=eid)
    trials, trials_mask = load_trials_and_mask(sess_loader)
    n_trials_total = len(trials)

    align_times = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
    # NaN alignment times are excluded by the mask; replace them so that the
    # searchsorted/interpolation below stay well defined.
    align_times_filled = np.where(np.isnan(align_times), 0.0, align_times)
    interval_begs = align_times_filled + TIME_WINDOW[0]
    interval_ends = align_times_filled + TIME_WINDOW[1]

    # --- behaviour ----------------------------------------------------------------
    wheel_times, wheel_vals = load_behavior_trace(sess_loader, 'wheel-speed')
    wheel, wheel_ok = bin_behavior(wheel_times, wheel_vals, interval_begs, interval_ends)
    whisker_times, whisker_vals = load_behavior_trace(sess_loader, 'whisker-motion-energy')
    whisker, whisker_ok = bin_behavior(whisker_times, whisker_vals,
                                       interval_begs, interval_ends)

    keep = trials_mask & wheel_ok & whisker_ok
    if keep.sum() < MIN_TRIALS_PER_SESSION:
        raise RuntimeError(f'only {keep.sum()} trials left after filtering')

    # --- spikes -------------------------------------------------------------------
    spikes, clusters = load_spikes_and_clusters(one, eid, pids, probe_names)
    binned, unit_ids = bin_spikes(spikes['times'], spikes['clusters'],
                                  interval_begs[keep], interval_ends[keep])
    del spikes

    from iblatlas.regions import BrainRegions
    acronyms = clusters['acronym'].to_numpy()[unit_ids]
    beryl = BrainRegions().acronym2acronym(acronyms, mapping='Beryl')

    # --- decoder inputs -----------------------------------------------------------
    # time since stimulus onset (bin centres) and trial number within the block
    bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)
    in_block = trial_number_in_block(trials)[keep]

    # --- decoder outputs ----------------------------------------------------------
    # choice: IBL codes +1 for a left choice (counter-clockwise wheel turn that
    # brings a left stimulus to the centre) and -1 for a right choice.
    choice = np.where(trials['choice'].to_numpy()[keep] > 0, 0, 1).astype(np.int64)
    pleft = trials['probabilityLeft'].to_numpy()[keep]
    prior = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                       np.isclose(pleft, 0.8)], [0, 1, 2], default=-1).astype(np.int64)
    if np.any(prior < 0):
        raise RuntimeError(f'unexpected probabilityLeft values: {np.unique(pleft)}')
    wheel_class, wheel_edges = discretize(wheel[keep])
    whisker_class, whisker_edges = discretize(whisker[keep])

    n_trials = int(keep.sum())
    neural, inputs, outputs = [], [], []
    for i in range(n_trials):
        neural.append(binned[i].astype(np.float32))
        inp = np.empty((2, N_BINS), dtype=np.float32)
        inp[0] = bin_centres
        inp[1] = in_block[i]
        inputs.append(inp)
        out = np.empty((4, N_BINS), dtype=np.int64)
        out[0] = choice[i]
        out[1] = prior[i]
        out[2] = wheel_class[i]
        out[3] = whisker_class[i]
        outputs.append(out)

    return {
        'eid': eid,
        'subject': subject,
        'lab': lab,
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'regions': list(beryl),
        'session_info': {
            'eid': eid,
            'subject': subject,
            'lab': lab,
            'n_probes': len(pids),
            'n_neurons': int(binned.shape[1]),
            'n_trials': n_trials,
            'n_trials_released': int(n_trials_total),
            'n_trials_passing_trial_qc': int(trials_mask.sum()),
            'wheel_speed_class_edges': [float(e) for e in wheel_edges],
            'whisker_motion_energy_class_edges': [float(e) for e in whisker_edges],
        },
        'runtime': time.time() - t0,
    }


def _safe_process_session(args):
    try:
        return process_session(args)
    except Exception as exc:  # noqa: BLE001 - a bad session must not stop the run
        return {'eid': args[0], 'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc()}


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------

def session_list(args):
    """(eid, subject, lab, pids, probe_names) for every session to convert."""
    bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
    if os.path.exists(DATALIMIT_SUBSET):
        subset = pd.read_csv(DATALIMIT_SUBSET)
        col = 'eid' if 'eid' in subset.columns else subset.columns[0]
        bwm_df = bwm_df[bwm_df.eid.isin(subset[col].astype(str))]
        print(f'Restricting to the {bwm_df.eid.nunique()} sessions of '
              f'{DATALIMIT_SUBSET}')
    sessions = []
    for eid, rows in bwm_df.groupby('eid', sort=False):
        sessions.append((eid, rows.subject.iloc[0], rows.lab.iloc[0],
                         list(rows.pid), list(rows.probe_name)))
    if args.eids:
        sessions = [s for s in sessions if s[0] in set(args.eids)]
    if args.limit:
        sessions = sessions[:args.limit]
    return sessions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=str, default='/app/converted_data.pkl')
    parser.add_argument('--n-workers', type=int, default=max(1, os.cpu_count() // 8))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--eids', type=str, nargs='+', default=None)
    args = parser.parse_args()

    sessions = session_list(args)
    print(f'Converting {len(sessions)} sessions with {args.n_workers} workers', flush=True)

    results, failures = [], []
    if args.n_workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=args.n_workers, maxtasksperchild=4) as pool:
            for i, res in enumerate(pool.imap_unordered(_safe_process_session, sessions)):
                _report(i, len(sessions), res, results, failures)
    else:
        for i, sess in enumerate(sessions):
            _report(i, len(sessions), _safe_process_session(sess), results, failures)

    results.sort(key=lambda r: r['eid'])
    data = assemble(results, failures)
    print(f'Writing {args.out}', flush=True)
    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {args.out} ({os.path.getsize(args.out) / 1e9:.1f} GB)')


def _report(i, n, res, results, failures):
    if 'error' in res:
        failures.append({'eid': res['eid'], 'error': res['error']})
        print(f'[{i + 1}/{n}] FAILED {res["eid"]}: {res["error"]}', flush=True)
    else:
        results.append(res)
        info = res['session_info']
        print(f'[{i + 1}/{n}] {res["eid"]} {info["subject"] if "subject" in info else ""} '
              f'{info["n_neurons"]} units, {info["n_trials"]} trials '
              f'({res["runtime"]:.0f} s)', flush=True)


def assemble(results, failures):
    """Build the output dictionary from the per-session results."""
    subjects = sorted({r['subject'] for r in results})
    subject_lookup = {s: i for i, s in enumerate(subjects)}
    brain_regions = sorted({reg for r in results for reg in r['regions']})
    region_lookup = {reg: i for i, reg in enumerate(brain_regions)}

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([subject_lookup[r['subject']] for r in results],
                                dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [np.array([region_lookup[reg] for reg in r['regions']],
                                      dtype=np.int64) for r in results],
        'input_names': ['time_from_stimulus_onset', 'trial_number_in_block'],
        'output_names': ['choice', 'prior_probability_left',
                         'wheel_speed', 'whisker_motion_energy'],
        'output_values': [
            ['left', 'right'],
            ['0.2', '0.5', '0.8'],
            ['low', 'medium', 'high'],
            ['low', 'medium', 'high'],
        ],
        'metadata': {
            'task_description': (
                'IBL decision-making task: a Gabor patch of one of five contrasts '
                '(100, 25, 12.5, 6, 0%) appears 35 deg to the left or right of centre '
                'and the mouse turns a wheel to bring it to the centre for a water '
                'reward. After 90 unbiased trials (p(left) = 0.5) the stimulus side is '
                'drawn in blocks of 20-100 trials with p(left) = 0.8 or 0.2, and the '
                'block switches are not cued. Decoded from the population spike counts '
                'are the choice (which side the mouse reported), the prior probability '
                'of a left stimulus of the current block, and two time-varying '
                'behaviours, wheel speed and whisker motion energy, each discretized '
                'into 3 classes at the within-session tertiles.'),
            'time_bin_size': BINSIZE * 1000.,
            'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
            'off_start': TIME_WINDOW[0],
            'off_end': TIME_WINDOW[1],
            'n_timepoints': N_BINS,
            'neural_data_description': (
                'spike counts of all spike-sorted units of the session (all probes '
                'merged, no single-unit quality-control selection, as in the methods '
                'paper) in non-overlapping 20 ms bins'),
            'input_description': {
                'time_from_stimulus_onset': (
                    'seconds from stimulus onset, centre of each 20 ms bin, from '
                    f'{TIME_WINDOW[0] + BINSIZE / 2} to {TIME_WINDOW[1] - BINSIZE / 2} s'),
                'trial_number_in_block': (
                    '0-based index of the trial within the current block of constant '
                    'probabilityLeft, counted on the full trials table'),
            },
            'output_description': {
                'choice': 'side the mouse reported (IBL choice +1 -> left, -1 -> right)',
                'prior_probability_left': 'trials.probabilityLeft of the block',
                'wheel_speed': (
                    'absolute wheel velocity, interpolated at the end of each 20 ms '
                    'bin, discretized at the within-session tertiles'),
                'whisker_motion_energy': (
                    'motion energy of the whisker pad of the left camera (right camera '
                    'when the left one is unavailable), interpolated at the end of '
                    'each 20 ms bin, discretized at the within-session tertiles'),
            },
            'inclusion_criteria': {
                'sessions': ('the 459 sessions of the brain-wide map release freeze, '
                             'which already satisfy the data paper session, insertion '
                             'and histology criteria'),
                'neurons': ('all units of the merged probes of a session that fired at '
                            'least one spike'),
                'trials': (f'{MIN_RT} s <= firstMovement_times - stimOn_times <= '
                           f'{MAX_RT} s, feedback_times - goCue_times <= '
                           f'{MAX_TRIAL_LEN} s, a choice was made, no NaN in '
                           f'{NAN_EXCLUDE}, and wheel and whisker traces covering the '
                           'whole trial window without NaNs'),
            },
            'dataset': 'IBL brain-wide map (Q3 2025 public release)',
            'session_info': [r['session_info'] for r in results],
            'failed_sessions': failures,
        },
    }
    return data


if __name__ == '__main__':
    sys.exit(main())
