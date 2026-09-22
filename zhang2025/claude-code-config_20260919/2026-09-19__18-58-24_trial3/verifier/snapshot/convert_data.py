#!/usr/bin/env python3
"""
Convert the IBL Brain-Wide Map (BWM) electrophysiology release into the decoder
format described in the task specification.

Processing follows the reference pipeline of Zhang et al., "Exploiting correlations
across trials and behavioral sessions to improve neural decoding"
(/app/code/code_zhang2025/src/0_data_caching.py and src/utils/ibl_data_utils.py) and
the inclusion criteria of IBL et al., "A brain-wide map of neural activity during
complex behaviour".

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""
import argparse
import os
import sys
import time
import pickle
import uuid
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = '/app/data/one_cache'
BWM_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
SESSIONS_PQT = f'{ROOT}/Brainwidemap/sessions.pqt'
DATASETS_PQT = f'{ROOT}/Brainwidemap/datasets.pqt'

# ---------------------------------------------------------------------------
# Trial / binning parameters.
#
# Identical to `params` in the reference 0_data_caching.py:
#     {'interval_len': 2, 'binsize': 0.02, 'single_region': False,
#      'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}
# and to the methods paper: "Recordings are split into 2-s trials, each divided into
# 20-ms bins, producing T = 100 time steps." / "For choice, we align trials to the
# stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset."
# ---------------------------------------------------------------------------
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100

# Cluster quality threshold. clusters['label'] is the fraction of the three RIGOR
# single-unit metrics passed (amplitude > 50 uV, noise cut-off < 20 uV, refractory
# period violation); label == 1 marks the "well-isolated neurons" of the BWM paper.
QC_LABEL = 1.0

N_OUTPUT_BINS = 3   # wheel speed / whisker motion energy are split into 3 classes

# Minimum well-isolated neurons for a session to be usable.  The BWM paper requires
# "at least five well-isolated neurons per session" for a region to enter its analyses;
# the decoding unit here is the whole session, so the same threshold is applied to the
# session's pooled neuron count.
MIN_NEURONS = 5

INPUT_NAMES = ['time_from_stim_on', 'trial_number_in_block']
OUTPUT_NAMES = ['choice', 'prior_prob_left', 'wheel_speed', 'whisker_motion_energy']
OUTPUT_VALUES = [
    ['left', 'right'],
    ['p_left=0.2', 'p_left=0.5', 'p_left=0.8'],
    ['low', 'medium', 'high'],
    ['low', 'medium', 'high'],
]


# ===========================================================================
# ONE setup
# ===========================================================================
def build_dataset_table(session_paths_map):
    """Build a ONE `datasets` cache table by scanning what is staged on disk.

    The shipped release tables are stale with respect to the staged files: every
    session's trials table only exists as revision ``#2025-03-03#`` on disk while the
    table lists the un-revised path, and the newest motion-energy revisions are split
    across two release tables. Loading through the shipped table therefore silently
    returns a one-column trials frame. Scanning the ``alf`` trees instead makes ONE
    resolve exactly the files that are present, including revisions.

    Args:
        session_paths_map: {eid (str): absolute session path}

    Returns:
        pandas.DataFrame indexed by (eid, id) with ONE's datasets-table columns.
    """
    rows = []
    for eid, sp in session_paths_map.items():
        eid_uuid = uuid.UUID(eid)
        for dirpath, _, filenames in os.walk(os.path.join(sp, 'alf')):
            rel = os.path.relpath(dirpath, sp)
            for f in filenames:
                rows.append((eid_uuid, uuid.uuid4(),
                             os.path.getsize(os.path.join(dirpath, f)),
                             None, True, 'NOT_SET', True, f'{rel}/{f}'))
    d = pd.DataFrame(rows, columns=['eid', 'id', 'file_size', 'hash',
                                    'default_revision', 'qc', 'exists', 'rel_path'])
    d = d.set_index(['eid', 'id']).sort_index()
    d['file_size'] = d['file_size'].astype('UInt64')
    d['qc'] = d['qc'].astype(pd.read_parquet(DATASETS_PQT)['qc'].dtype)
    return d


def get_session_table():
    """Return (bwm_df, sessions_df, {eid: session_path}) for the 699 released probes."""
    sess = pd.read_parquet(SESSIONS_PQT)
    sess.index = sess.index.astype(str)
    bwm = pd.read_csv(BWM_CSV, index_col=0)
    paths = {}
    for eid in bwm.eid.unique():
        r = sess.loc[eid]
        paths[eid] = (f"{ROOT}/{r['lab']}/Subjects/{r['subject']}/"
                      f"{str(r['date'])}/{int(r['number']):03d}")
    return bwm, sess, paths


def make_one(session_paths_map):
    from one.api import ONE
    one = ONE(base_url='https://openalyx.internationalbrainlab.org', mode='local')
    one.load_cache(tables_dir=f'{ROOT}/Brainwidemap')
    one._cache['datasets'] = build_dataset_table(session_paths_map)
    return one


# ===========================================================================
# Loading (mirrors ibl_data_utils.load_spiking_data / merge_probes)
# ===========================================================================
def load_spiking_data(one, pid, eid, pname, qc=QC_LABEL):
    """Load spikes and clusters for one probe insertion, keeping clusters with
    label >= qc.  Direct transcription of ibl_data_utils.load_spiking_data, minus
    the `raw_electrophysiology(...).fs` call, which needs a network connection and
    is only used for a metadata field."""
    from brainbox.io.one import SpikeSortingLoader
    from iblutil.numerical import ismember

    loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = loader.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
    n_units_total = len(clusters_labeled)
    if qc is None:
        return spikes, clusters_labeled, n_units_total
    iok = clusters_labeled['label'] >= qc
    selected_clusters = clusters_labeled[iok]
    spike_idx, ib = ismember(spikes['clusters'], selected_clusters.index)
    selected_clusters = selected_clusters.reset_index(drop=True)
    selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}
    selected_spikes['clusters'] = selected_clusters.index[ib].astype(np.int32)
    return selected_spikes, selected_clusters, n_units_total


def merge_probes(spikes_list, clusters_list):
    """Merge several probes of one session into a single spike train / cluster table.

    Follows ibl_data_utils.merge_probes, with one correction: the reference sets
    ``cluster_max = clusters.index.max() + 1`` inside the loop, which *replaces* the
    running offset with the current probe's cluster count instead of accumulating it.
    With two probes the two are identical, but from the third probe on the reference
    would map spikes onto the wrong rows of the concatenated cluster table, so the
    offset is accumulated here.
    """
    merged_spikes, merged_clusters, cluster_max = [], [], 0
    for clusters, spikes in zip(clusters_list, spikes_list):
        spikes = dict(spikes)
        spikes['clusters'] = spikes['clusters'] + cluster_max
        cluster_max += len(clusters)
        merged_spikes.append(spikes)
        merged_clusters.append(clusters)
    merged_clusters = pd.concat(merged_clusters, ignore_index=True)
    merged_spikes = {k: np.concatenate([s[k] for s in merged_spikes])
                     for k in merged_spikes[0].keys()}
    sort_idx = np.argsort(merged_spikes['times'], kind='stable')
    merged_spikes = {k: v[sort_idx] for k, v in merged_spikes.items()}
    return merged_spikes, merged_clusters


def load_trials_and_mask(sess_loader, min_rt=0.08, max_rt=2., nan_exclude='default',
                         min_trial_len=None, max_trial_len=10.0,
                         exclude_unbiased=False, exclude_nochoice=True):
    """Trials table plus the BWM trial-inclusion mask.

    Transcribed from ibl_data_utils.load_trials_and_mask; `max_trial_len=10.0` is the
    value the reference caching script passes.  The default `nan_exclude` list is
    exactly the BWM paper's: "trials were excluded if one of the following trial
    events could not be detected: choice, probabilityLeft, feedbackType, feedback
    times, stimOn times and firstMovement times.  Trials were further excluded if the
    time between stimulus onset and the first movement of the wheel were outside the
    range of 0.08-2.00 s."
    """
    if nan_exclude == 'default':
        nan_exclude = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
                       'firstMovement_times', 'feedbackType']
    if sess_loader.trials.empty:
        sess_loader.load_trials()

    query = f'(firstMovement_times - stimOn_times < {min_rt})' if min_rt is not None else ''
    if max_rt is not None:
        query += f' | (firstMovement_times - stimOn_times > {max_rt})'
    if min_trial_len is not None:
        query += f' | (feedback_times - goCue_times < {min_trial_len})'
    if max_trial_len is not None:
        query += f' | (feedback_times - goCue_times > {max_trial_len})'
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    if exclude_unbiased:
        query += ' | (probabilityLeft == 0.5)'
    if exclude_nochoice:
        query += ' | (choice == 0)'
    if min_rt is None:
        query = query[3:]

    mask = ~sess_loader.trials.eval(query)
    return sess_loader.trials, mask


def load_behavior_traces(sess_loader):
    """Load the two time-varying behaviours as {name: (times, values)}.

    Mirrors ibl_data_utils.load_target_behavior / bin_behaviors:
      * wheel speed  = |velocity| of the uniformly resampled, Butterworth-filtered
        wheel trace produced by SessionLoader.load_wheel.
      * whisker motion energy = leftCamera ROIMotionEnergy, falling back to the right
        camera when the left video is unavailable (the reference tries 'left' first
        and switches to 'right' on failure).
    """
    out = {}
    sess_loader.load_wheel()
    out['wheel-speed'] = (sess_loader.wheel['times'].to_numpy(),
                          np.abs(sess_loader.wheel['velocity'].to_numpy()))
    whisker = None
    for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sess_loader.load_motion_energy(views=[view])
            df = sess_loader.motion_energy[cam]
            whisker = (df['times'].to_numpy(), df['whiskerMotionEnergy'].to_numpy())
            out['whisker-camera'] = view
            break
        except Exception:
            continue
    if whisker is None:
        raise RuntimeError('no whisker motion energy available')
    out['whisker-motion-energy'] = whisker
    return out


# ===========================================================================
# Binning
# ===========================================================================
def bin_spiking_data(spike_times, spike_clusters, n_clusters, align_times):
    """Bin spikes into (ntrials, n_clusters, NBINS) spike counts.

    Vectorised equivalent of ibl_data_utils.get_spike_data_per_interval, which calls
    `bincount2D(times, clusters, xbin=binsize, xlim=[t_beg, t_end])` per trial after
    restricting to `t_beg <= t < t_end`, then keeps the first NBINS columns.
    bincount2D assigns bin index floor((t - t_beg) / binsize), i.e. left-closed
    20 ms bins starting at t_beg -- reproduced exactly here.

    Args:
        spike_times: sorted (nspikes,) spike times in seconds.
        spike_clusters: (nspikes,) cluster index in [0, n_clusters).
        n_clusters: number of clusters.
        align_times: (ntrials,) alignment event times.

    Returns:
        (ntrials, n_clusters, NBINS) float32 spike counts.
    """
    ntrials = len(align_times)
    out = np.zeros((ntrials, n_clusters, NBINS), dtype=np.float32)
    begs = align_times + TIME_WINDOW[0]
    ends = align_times + TIME_WINDOW[1]
    i0 = np.searchsorted(spike_times, begs, side='left')
    i1 = np.searchsorted(spike_times, ends, side='left')
    for k in range(ntrials):
        a, b = i0[k], i1[k]
        if b <= a:
            continue
        t = spike_times[a:b]
        c = spike_clusters[a:b]
        idx = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
        keep = (idx >= 0) & (idx < NBINS)
        if not np.any(keep):
            continue
        flat = c[keep].astype(np.int64) * NBINS + idx[keep]
        counts = np.bincount(flat, minlength=n_clusters * NBINS)
        out[k] = counts.reshape(n_clusters, NBINS)
    return out


def bin_behavior(target_times, target_values, align_times):
    """Resample a continuous behavioural trace onto the trial time base.

    Transcription of ibl_data_utils.get_behavior_per_interval: the trace is linearly
    interpolated at `np.linspace(t_beg + binsize, t_end, NBINS)` -- the right edge of
    each of the NBINS neural bins -- using only samples inside the interval, and a
    trial is rejected when the trace does not cover the interval (the reference's
    'target data starts too late' / 'ends too early' / 'nans in target data' /
    'target data not present' cases).

    `allow_nans` is effectively False here: the target format forbids NaN, so trials
    whose behavioural trace contains NaN inside the window are dropped rather than
    kept and mean-imputed later.

    Returns:
        values: (ntrials, NBINS) float64, NaN rows for rejected trials.
        good: (ntrials,) bool.
    """
    from scipy.interpolate import interp1d

    ntrials = len(align_times)
    values = np.full((ntrials, NBINS), np.nan)
    good = np.zeros(ntrials, dtype=bool)
    begs = align_times + TIME_WINDOW[0]
    ends = align_times + TIME_WINDOW[1]
    idxs_beg = np.searchsorted(target_times, begs, side='right')
    idxs_end = np.searchsorted(target_times, ends, side='left')

    for k in range(ntrials):
        tt = target_times[idxs_beg[k]:idxs_end[k]]
        vv = target_values[idxs_beg[k]:idxs_end[k]]
        if len(vv) == 0:
            continue
        if np.isnan(vv).any():
            continue
        if np.isnan(begs[k]) or np.isnan(ends[k]):
            continue
        if np.abs(begs[k] - tt[0]) > BINSIZE:
            continue
        if np.abs(ends[k] - tt[-1]) > BINSIZE:
            continue
        x = np.linspace(begs[k] + BINSIZE, ends[k], NBINS)
        values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
        good[k] = True
    good &= ~np.isnan(values).any(axis=1)
    return values, good


def discretize_tertiles(values):
    """Split a (ntrials, NBINS) continuous behaviour into 3 equal-occupancy classes.

    The spec asks for 3 bins but does not fix the edges.  Whisker motion energy is in
    arbitrary camera-dependent units (the left camera is 1280x1024 @ 60 Hz, the right
    640x512 @ 150 Hz), so no fixed threshold transfers across sessions; equal-occupancy
    (tertile) edges computed within a session give the same class semantics --
    low / medium / high movement for this animal in this session -- everywhere, and
    make chance performance exactly 1/3 for balanced accuracy.  Wheel speed is treated
    the same way for consistency and because its distribution is strongly
    zero-inflated, so fixed edges would leave near-empty classes in quiet sessions.

    Returns:
        labels: (ntrials, NBINS) int8 in {0, 1, 2}
        edges:  the two interior edges actually used
    """
    flat = values.ravel()
    edges = np.quantile(flat, [1. / 3., 2. / 3.])
    # Degenerate case: a mostly constant trace can give identical edges.  Nudge the
    # upper edge so np.digitize still produces monotone, well-defined classes.
    if not edges[1] > edges[0]:
        uniq = np.unique(flat)
        if len(uniq) >= 3:
            edges = np.quantile(uniq, [1. / 3., 2. / 3.])
        if not edges[1] > edges[0]:
            edges = np.array([edges[0], edges[0] + np.finfo(float).eps])
    labels = np.digitize(values, edges).astype(np.int8)
    return labels, edges


def trial_number_in_block(probability_left):
    """0-based index of each trial within its block of constant probabilityLeft.

    Computed on the full trials table before any trial exclusion, so the value is the
    animal's true position in the block rather than a position among surviving trials.
    """
    p = np.asarray(probability_left, dtype=float)
    newblock = np.ones(len(p), dtype=bool)
    if len(p) > 1:
        # NaN-safe change detection: a NaN probabilityLeft starts a new block.
        same = (p[1:] == p[:-1])
        newblock[1:] = ~same
    block_id = np.cumsum(newblock) - 1
    out = np.zeros(len(p), dtype=np.int64)
    for b in np.unique(block_id):
        m = block_id == b
        out[m] = np.arange(m.sum())
    return out, block_id


# ===========================================================================
# Per-session conversion
# ===========================================================================
def convert_session(eid, one, bwm, brain_regions, show_processing=False):
    """Convert a single session.  Returns a dict, or None if the session is dropped."""
    from brainbox.io.one import SessionLoader

    t_start = time.time()
    info = {'eid': eid, 'timings': {}}
    sub = bwm[bwm.eid == eid]
    info['subject'] = sub.subject.iloc[0]
    info['lab'] = sub.lab.iloc[0]

    # ---- trials -----------------------------------------------------------
    t0 = time.time()
    sl = SessionLoader(one=one, eid=eid)
    trials, mask = load_trials_and_mask(sl)
    info['timings']['trials'] = time.time() - t0
    info['n_trials_total'] = int(len(trials))
    mask = np.asarray(mask, dtype=bool)
    info['n_trials_after_trialmask'] = int(mask.sum())
    # Per-criterion counts, for comparison with the BWM paper (e.g. "22.8% of first
    # wheel-movement times occurred under 80 ms").
    rt = (trials['firstMovement_times'] - trials['stimOn_times']).to_numpy()
    info['n_rt_short'] = int(np.nansum(rt < 0.08))
    info['n_rt_long'] = int(np.nansum(rt > 2.0))
    info['n_rt_nan'] = int(np.isnan(rt).sum())
    info['n_nochoice'] = int((trials['choice'].to_numpy() == 0).sum())
    info['n_long_trial'] = int(np.nansum(
        (trials['feedback_times'] - trials['goCue_times']).to_numpy() > 10.0))
    fb = trials['feedbackType'].to_numpy()
    info['n_correct'] = int(np.nansum(fb == 1))
    info['n_incorrect'] = int(np.nansum(fb == -1))
    if mask.sum() < 2:
        info['skip_reason'] = 'fewer than 2 trials pass the trial mask'
        return None, info

    tnib_all, block_id_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
    trials_sel = trials[mask]
    align_times = trials_sel[ALIGN_TIME].to_numpy()
    tnib = tnib_all[mask]

    # ---- behaviour --------------------------------------------------------
    t0 = time.time()
    try:
        traces = load_behavior_traces(sl)
    except Exception as e:
        info['skip_reason'] = f'behaviour unavailable: {e}'
        return None, info
    info['whisker_camera'] = traces['whisker-camera']
    wheel_vals, wheel_good = bin_behavior(*traces['wheel-speed'], align_times)
    whisk_vals, whisk_good = bin_behavior(*traces['whisker-motion-energy'], align_times)
    info['timings']['behavior'] = time.time() - t0

    beh_good = wheel_good & whisk_good
    info['n_trials_after_behmask'] = int(beh_good.sum())
    if beh_good.sum() < 2:
        info['skip_reason'] = 'fewer than 2 trials with valid behaviour'
        return None, info

    # ---- spikes -----------------------------------------------------------
    t0 = time.time()
    spikes_list, clusters_list, n_units_total = [], [], 0
    for _, r in sub.iterrows():
        sp, cl, n_tot = load_spiking_data(one, r.pid, eid, r.probe_name)
        cl = cl.copy()
        cl['pid'] = r.pid
        spikes_list.append(sp)
        clusters_list.append(cl)
        n_units_total += n_tot
    spikes, clusters = merge_probes(spikes_list, clusters_list)
    info['n_units_total'] = int(n_units_total)
    info['timings']['spikes'] = time.time() - t0
    n_clusters = len(clusters)
    info['n_neurons'] = int(n_clusters)
    if n_clusters < MIN_NEURONS:
        info['skip_reason'] = (f'fewer than {MIN_NEURONS} well-isolated neurons '
                               f'({n_clusters})')
        return None, info

    # Spikes with NaN times (no sample-to-time solution) cannot be binned.
    finite = np.isfinite(spikes['times'])
    st = spikes['times'][finite]
    sc = spikes['clusters'][finite]
    order = np.argsort(st, kind='stable')
    st, sc = st[order], sc[order]

    t0 = time.time()
    binned = bin_spiking_data(st, sc, n_clusters, align_times)
    info['timings']['binning'] = time.time() - t0

    # ---- final trial selection -------------------------------------------
    # Trials in which the whole population is silent for the full 2 s window are
    # recording dropouts, not physiology: either the ephys stream ends before the
    # behavioural session does (the spike train of
    # 8c2f7f4d-7346-42a4-a715-4d37a5208535 stops 20 s before its last three trials) or
    # a chunk of the recording is missing (a 4.6 s gap in
    # b182b754-3c3e-4942-8144-6ee790926b58 swallows one trial).  This is the neural
    # counterpart of the reference's 'target data ends too early' behaviour check.
    neural_covered = binned.sum(axis=(1, 2)) > 0
    info['n_trials_no_spikes'] = int((~neural_covered).sum())
    keep = beh_good & neural_covered
    info['n_trials_after_neuralmask'] = int(keep.sum())
    if keep.sum() < 2:
        info['skip_reason'] = 'fewer than 2 trials with both behaviour and spikes'
        return None, info
    binned = binned[keep]
    wheel_vals = wheel_vals[keep]
    whisk_vals = whisk_vals[keep]
    tnib_keep = tnib[keep]
    trials_keep = trials_sel[keep]
    ntrials = binned.shape[0]
    info['n_trials'] = int(ntrials)

    # ---- outputs ----------------------------------------------------------
    # choice: IBL codes +1 when the mouse reported the stimulus on the LEFT and -1
    # when it reported the RIGHT (verified against feedbackType and contrastLeft /
    # contrastRight).  Spec asks for left = 0, right = 1.
    choice_raw = trials_keep['choice'].to_numpy()
    choice = (choice_raw < 0).astype(np.int8)

    # prior: probabilityLeft in {0.2, 0.5, 0.8} -> {0, 1, 2}
    pleft = trials_keep['probabilityLeft'].to_numpy()
    prior = np.full(ntrials, -1, dtype=np.int8)
    for val, lab in ((0.2, 0), (0.5, 1), (0.8, 2)):
        prior[np.isclose(pleft, val)] = lab
    if np.any(prior < 0):
        info['skip_reason'] = f'unexpected probabilityLeft values {np.unique(pleft)}'
        return None, info

    wheel_lab, wheel_edges = discretize_tertiles(wheel_vals)
    whisk_lab, whisk_edges = discretize_tertiles(whisk_vals)
    info['wheel_edges'] = wheel_edges.tolist()
    info['whisker_edges'] = whisk_edges.tolist()

    # ---- inputs -----------------------------------------------------------
    # Time from stimulus onset at the centre of each 20 ms neural bin.
    tvec = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)

    neural_list, input_list, output_list = [], [], []
    for k in range(ntrials):
        neural_list.append(binned[k])
        inp = np.empty((2, NBINS), dtype=np.float32)
        inp[0] = tvec
        inp[1] = tnib_keep[k]
        input_list.append(inp)
        out = np.empty((4, NBINS), dtype=np.int8)
        out[0] = choice[k]
        out[1] = prior[k]
        out[2] = wheel_lab[k]
        out[3] = whisk_lab[k]
        output_list.append(out)

    # ---- brain regions ----------------------------------------------------
    beryl = brain_regions.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
    beryl = np.asarray(beryl, dtype=object)

    info['timings']['total'] = time.time() - t_start
    result = {
        'eid': eid,
        'neural': neural_list,
        'input': input_list,
        'output': output_list,
        'subject': sub.subject.iloc[0],
        'lab': sub.lab.iloc[0],
        'beryl': beryl,
        'acronyms': clusters['acronym'].to_numpy(),
        'n_probes': int(len(sub)),
        'info': info,
    }
    if show_processing:
        result['_plotdata'] = {
            'trials': trials, 'mask': mask, 'keep': keep, 'align_times': align_times,
            'traces': traces, 'wheel_vals': wheel_vals, 'whisk_vals': whisk_vals,
            'wheel_lab': wheel_lab, 'whisk_lab': whisk_lab,
            'wheel_edges': wheel_edges, 'whisker_edges': whisk_edges,
            'binned': binned, 'st': st, 'sc': sc, 'tnib': tnib_keep,
            'choice': choice, 'prior': prior, 'tvec': tvec,
            'block_id': block_id_all, 'tnib_all': tnib_all,
        }
    return result, info


# ===========================================================================
# Diagnostic plots
# ===========================================================================
def plot_processing(result, outpath):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pd_ = result['_plotdata']
    tvec = pd_['tvec']
    edges = TIME_WINDOW[0] + BINSIZE * np.arange(NBINS + 1)
    fig, axs = plt.subplots(4, 2, figsize=(20, 18))

    # (0,0) raster vs binned spikes for one trial -- temporal alignment check
    k = min(5, len(pd_['align_times']) - 1)
    trial_ids = np.nonzero(pd_['keep'])[0]
    kk = trial_ids[min(5, len(trial_ids) - 1)]
    t0 = pd_['align_times'][kk]
    sel = (pd_['st'] >= t0 + TIME_WINDOW[0]) & (pd_['st'] < t0 + TIME_WINDOW[1])
    ax = axs[0, 0]
    ax.plot(pd_['st'][sel] - t0, pd_['sc'][sel], '|', ms=3, color='k')
    ax.set_title(f'raw spike raster, trial {kk} (t=0 at stimOn)')
    ax.set_xlabel('time from stimulus onset (s)'); ax.set_ylabel('neuron')
    ax.set_xlim(TIME_WINDOW); ax.axvline(0, color='r')

    ax = axs[0, 1]
    idx = np.nonzero(pd_['keep'])[0].tolist().index(kk) if kk in np.nonzero(pd_['keep'])[0] else 0
    im = ax.imshow(pd_['binned'][idx], aspect='auto', interpolation='nearest',
                   extent=[edges[0], edges[-1], pd_['binned'].shape[1], 0], cmap='Greys')
    ax.set_title('binned spike counts for the same trial (must match raster)')
    ax.set_xlabel('time from stimulus onset (s)'); ax.set_ylabel('neuron')
    ax.axvline(0, color='r'); plt.colorbar(im, ax=ax)

    # (1,0) PSTH across trials -- a stimulus-locked response confirms alignment
    ax = axs[1, 0]
    psth = pd_['binned'].mean(axis=(0, 1)) / BINSIZE
    ax.plot(tvec, psth)
    ax.axvline(0, color='r', label='stimulus onset')
    ax.set_title('population PSTH (mean firing rate over trials and neurons)')
    ax.set_xlabel('time from stimulus onset (s)'); ax.set_ylabel('sp/s'); ax.legend()

    # (1,1) raw wheel speed vs resampled, one trial
    ax = axs[1, 1]
    wt, wv = pd_['traces']['wheel-speed']
    sel = (wt >= t0 + TIME_WINDOW[0] - 0.1) & (wt <= t0 + TIME_WINDOW[1] + 0.1)
    ax.plot(wt[sel] - t0, wv[sel], color='0.6', label='raw |wheel velocity|')
    ax.plot(edges[1:], pd_['wheel_vals'][idx], 'o-', ms=3, label='binned (bin right edge)')
    for e in pd_['wheel_edges']:
        ax.axhline(e, ls='--', color='r')
    ax.set_title('wheel speed: raw vs resampled, with tertile edges (red)')
    ax.set_xlabel('time from stimulus onset (s)'); ax.legend()

    # (2,0) raw whisker ME vs resampled, one trial
    ax = axs[2, 0]
    mt, mv = pd_['traces']['whisker-motion-energy']
    sel = (mt >= t0 + TIME_WINDOW[0] - 0.1) & (mt <= t0 + TIME_WINDOW[1] + 0.1)
    ax.plot(mt[sel] - t0, mv[sel], color='0.6', label='raw whisker ME')
    ax.plot(edges[1:], pd_['whisk_vals'][idx], 'o-', ms=3, label='binned (bin right edge)')
    for e in pd_['whisker_edges']:
        ax.axhline(e, ls='--', color='r')
    ax.set_title(f"whisker motion energy ({pd_['traces']['whisker-camera']} camera): raw vs resampled")
    ax.set_xlabel('time from stimulus onset (s)'); ax.legend()

    # (2,1) discretisation check: continuous value coloured by assigned class
    ax = axs[2, 1]
    flat_v = pd_['wheel_vals'].ravel(); flat_l = pd_['wheel_lab'].ravel()
    sub = np.random.default_rng(0).choice(len(flat_v), size=min(20000, len(flat_v)), replace=False)
    for c, col in zip(range(3), ['tab:blue', 'tab:orange', 'tab:green']):
        m = flat_l[sub] == c
        ax.plot(sub[m], flat_v[sub][m], '.', ms=2, color=col, label=f'class {c}')
    for e in pd_['wheel_edges']:
        ax.axhline(e, ls='--', color='r')
    ax.set_yscale('symlog', linthresh=1e-3)
    ax.set_title('wheel speed discretisation (classes must not overlap in value)')
    ax.legend(markerscale=6)

    # (3,0) outputs / inputs over trials
    ax = axs[3, 0]
    ax.plot(pd_['tnib'], label='trial number in block (input 1)')
    ax.plot(pd_['prior'] * 10, label='prior class x10 (output 1)')
    ax.plot(pd_['choice'] * 5, '.', ms=2, label='choice (output 0) x5')
    ax.set_xlabel('retained trial'); ax.legend()
    ax.set_title('per-trial input/output variables')

    # (3,1) class occupancy
    ax = axs[3, 1]
    names = ['choice', 'prior', 'wheel', 'whisker']
    data = [np.bincount(pd_['choice'], minlength=2) / len(pd_['choice']),
            np.bincount(pd_['prior'], minlength=3) / len(pd_['prior']),
            np.bincount(pd_['wheel_lab'].ravel(), minlength=3) / pd_['wheel_lab'].size,
            np.bincount(pd_['whisk_lab'].ravel(), minlength=3) / pd_['whisk_lab'].size]
    for i, (n, d) in enumerate(zip(names, data)):
        ax.bar(np.arange(len(d)) + i * 4, d)
        for j, v in enumerate(d):
            ax.text(j + i * 4, v, f'{v:.2f}', ha='center', fontsize=8)
    ax.set_xticks([i * 4 + 1 for i in range(4)]); ax.set_xticklabels(names)
    ax.set_title('output class fractions')

    fig.suptitle(f"processing checks: {result['eid']} ({result['subject']}), "
                 f"{len(result['neural'])} trials, {result['neural'][0].shape[0]} neurons")
    fig.tight_layout()
    fig.savefig(outpath, dpi=110)
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
_GLOBALS = {}


def _worker(eid):
    try:
        res, info = convert_session(eid, _GLOBALS['one'], _GLOBALS['bwm'],
                                    _GLOBALS['br'], show_processing=False)
        if res is not None:
            res.pop('_plotdata', None)
        return res, info
    except Exception as e:
        import traceback
        return None, {'eid': eid, 'skip_reason': f'{type(e).__name__}: {e}',
                      'traceback': traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--n-workers', type=int, default=24)
    args = ap.parse_args()

    t_all = time.time()
    bwm, sess, paths = get_session_table()
    eids = list(bwm.eid.unique())
    if args.sample:
        eids = eids[:2]
    print(f'Converting {len(eids)} sessions ({bwm[bwm.eid.isin(eids)].pid.nunique()} probes, '
          f'{bwm[bwm.eid.isin(eids)].subject.nunique()} subjects)')

    t0 = time.time()
    one = make_one({e: paths[e] for e in eids})
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    print(f'ONE cache built in {time.time() - t0:.1f}s '
          f'({len(one._cache["datasets"])} dataset records)')

    _GLOBALS['one'] = one
    _GLOBALS['bwm'] = bwm
    _GLOBALS['br'] = br

    results, infos = [], []
    if args.show_processing:
        for eid in eids[:2]:
            res, info = convert_session(eid, one, bwm, br, show_processing=True)
            if res is not None:
                plot_processing(res, f'processing_{eid}.png')
                print(f'  wrote processing_{eid}.png')
                res.pop('_plotdata', None)
            results.append(res)
            infos.append(info)
        rest = eids[2:]
    else:
        rest = eids

    if rest:
        from concurrent.futures import ProcessPoolExecutor
        nw = min(args.n_workers, max(1, len(rest)))
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=nw) as ex:
            for res, info in ex.map(_worker, rest, chunksize=1):
                results.append(res)
                infos.append(info)
                done += 1
                if done % 20 == 0 or done == len(rest):
                    el = time.time() - t0
                    print(f'  {done}/{len(rest)} sessions  ({el:.0f}s elapsed, '
                          f'{el / done:.1f}s/session, ETA {el / done * (len(rest) - done):.0f}s)',
                          flush=True)

    # ---- assemble ---------------------------------------------------------
    kept = [r for r in results if r is not None]
    dropped = [i for i in infos if 'skip_reason' in i]
    print(f'\nKept {len(kept)} sessions, dropped {len(dropped)}')
    reasons = defaultdict(int)
    for d in dropped:
        reasons[d['skip_reason'].split(':')[0]] += 1
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f'  dropped ({v}): {k}')
    for d in dropped[:5]:
        if 'traceback' in d:
            print(d['eid'], d['traceback'][-800:])

    subjects = sorted({r['subject'] for r in kept})
    subj_idx = np.array([subjects.index(r['subject']) for r in kept], dtype=np.int64)
    all_regions = sorted({a for r in kept for a in r['beryl']})
    reg_lookup = {a: i for i, a in enumerate(all_regions)}
    brain_region_idx = [np.array([reg_lookup[a] for a in r['beryl']], dtype=np.int64)
                        for r in kept]

    session_info = []
    for r in kept:
        i = r['info']
        session_info.append({
            'eid': r['eid'], 'subject': r['subject'], 'lab': r['lab'],
            'n_probes': r['n_probes'],
            'n_units_total': i['n_units_total'],
            'n_neurons': i['n_neurons'],
            'n_trials_total': i['n_trials_total'],
            'n_trials_after_trialmask': i['n_trials_after_trialmask'],
            'n_trials_after_behmask': i['n_trials_after_behmask'],
            'n_trials_no_spikes': i['n_trials_no_spikes'],
            'n_trials': i['n_trials'],
            'n_rt_short': i['n_rt_short'], 'n_rt_long': i['n_rt_long'],
            'n_rt_nan': i['n_rt_nan'], 'n_nochoice': i['n_nochoice'],
            'n_long_trial': i['n_long_trial'],
            'n_correct': i['n_correct'], 'n_incorrect': i['n_incorrect'],
            'whisker_camera': i.get('whisker_camera'),
            'wheel_speed_tertile_edges': i.get('wheel_edges'),
            'whisker_me_tertile_edges': i.get('whisker_edges'),
        })

    data = {
        'neural': [r['neural'] for r in kept],
        'input': [r['input'] for r in kept],
        'output': [r['output'] for r in kept],
        'subjects': subjects,
        'subject_idx': subj_idx,
        'brain_regions': all_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': (
                'IBL decision-making task. A Gabor patch of one of 5 contrasts '
                '(100, 25, 12.5, 6.25, 0%) appears 35 deg to the left or right; the '
                'mouse turns a wheel to bring it to the centre for a water reward. '
                'After 90 unbiased trials (p(left)=0.5) the stimulus side is drawn '
                'from uncued blocks with p(left)=0.2 or 0.8, lasting 20-100 trials. '
                'Decoded outputs: the mouse\'s choice (left/right), the block prior '
                'p(left) (0.2/0.5/0.8), wheel speed in 3 within-session tertiles and '
                'whisker-pad motion energy in 3 within-session tertiles. Decoder '
                'inputs: time from stimulus onset and the trial index within the '
                'current block.'),
            'time_bin_size': BINSIZE * 1000.,
            'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
            'off_start': TIME_WINDOW[0],
            'off_end': TIME_WINDOW[1],
            'n_timepoints': NBINS,
            'neural_units': 'spike counts per 20 ms bin',
            'neural_curation': (
                'Spike-sorted with pykilosort (revision 2024-05-06); only '
                'well-isolated neurons kept (clusters.label == 1, i.e. passing all '
                'three RIGOR single-unit metrics: amplitude > 50 uV, noise cut-off '
                '< 20 uV, refractory-period violation). Probes of the same session '
                'are merged.'),
            'trial_curation': (
                'BWM trial mask: no NaN in stimOn_times, choice, feedback_times, '
                'probabilityLeft, firstMovement_times, feedbackType; '
                '0.08 s <= firstMovement_times - stimOn_times <= 2 s; '
                'feedback_times - goCue_times <= 10 s; choice != 0. Trials whose '
                'wheel or whisker trace does not cover the full window, or contains '
                'NaN in it, are also dropped.'),
            'brain_region_mapping': 'Allen CCF acronyms mapped to the Beryl atlas',
            'input_descriptions': {
                'time_from_stim_on': 'seconds from stimulus onset at the centre of each 20 ms bin (-0.49 .. 1.49)',
                'trial_number_in_block': '0-based index of the trial within its block of constant probabilityLeft, counted over all trials of the session',
            },
            'output_descriptions': {
                'choice': 'reported stimulus side; IBL choice=+1 -> left (0), choice=-1 -> right (1)',
                'prior_prob_left': 'trials.probabilityLeft: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2',
                'wheel_speed': '|wheel velocity| resampled at 20 ms, split at within-session tertiles',
                'whisker_motion_energy': 'whisker-pad ROI motion energy (left camera, right if absent) resampled at 20 ms, split at within-session tertiles',
            },
            'source': ('IBL Brain-Wide Map public release '
                       '(https://doi.org/10.6084/m9.figshare.21400815); processing '
                       'follows Zhang et al. 2025 src/0_data_caching.py'),
            'session_info': session_info,
        },
    }

    ntr = [len(n) for n in data['neural']]
    nneu = [n[0].shape[0] for n in data['neural']]
    print(f'\nTotals: {len(kept)} sessions, {len(subjects)} subjects, '
          f'{sum(ntr)} trials, {sum(nneu)} neurons, {len(all_regions)} brain regions')
    print(f'  trials/session  mean {np.mean(ntr):.1f} median {np.median(ntr):.0f} '
          f'min {np.min(ntr)} max {np.max(ntr)}')
    print(f'  neurons/session mean {np.mean(nneu):.1f} median {np.median(nneu):.0f} '
          f'min {np.min(nneu)} max {np.max(nneu)}')

    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'Wrote {args.outfile} ({os.path.getsize(args.outfile) / 1e9:.2f} GB) '
          f'in {time.time() - t0:.1f}s')
    print(f'Total run time {time.time() - t_all:.1f}s')


if __name__ == '__main__':
    main()
