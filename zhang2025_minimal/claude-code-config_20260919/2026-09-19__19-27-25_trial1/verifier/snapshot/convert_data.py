"""Convert the IBL brain-wide map (BWM) release into the decoder's dictionary format.

The processing follows the reference code that accompanies "Exploiting correlations
across trials and behavioral sessions to improve neural decoding"
(``code/code_zhang2025/src/0_data_caching.py`` and ``src/utils/ibl_data_utils.py``) and
the inclusion criteria of "A brain-wide map of neural activity during complex behaviour".

Trials are aligned to stimulus onset and span -0.5 s to +1.5 s around it, binned into
100 non-overlapping 20 ms bins -- the ``params`` block of ``0_data_caching.py``.

Decoder inputs (2 x 100 per trial)
    0. time since stimulus onset (s), bin centres, -0.49 .. 1.49
    1. trial number within the current block (constant within a trial)

Decoder outputs (4 x 100 per trial, integer class labels)
    0. choice                  left=0, right=1                (constant within a trial)
    1. prior probability left  0.2->0, 0.5->1, 0.8->2         (constant within a trial)
    2. wheel speed             low/medium/high tertile        (time varying)
    3. whisker motion energy   low/medium/high tertile        (time varying)

Usage:  python convert_data.py [--out /app/converted_data.pkl] [--n-sessions N]
"""
import argparse
import datetime
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

REPO = Path('/app/code/code_zhang2025')
BWM_RELEASE = REPO / 'data' / 'bwm_release.csv'
CACHE_DIR = Path('/app/data/one_cache')


# --------------------------------------------------------------------------------------
# ONE cache repair
#
# The cache tables shipped with this dataset are stale with respect to the file tree:
# the released table lists e.g. ``alf/_ibl_trials.table.pqt`` while every session on disk
# holds that file under a revision folder (``alf/#2025-03-03#/...``).  ONE in local mode
# resolves a dataset by the exact ``rel_path`` in the table, so without this repair the
# trials table, the wheel position and the camera motion energy fail to load and most
# sessions come out empty.
#
# The three shipped tables are merged and every row is re-pointed at the newest revision
# of that (collection, filename) present on disk, keeping the released session/dataset
# UUIDs so that eid lookups against bwm_release.csv still work.
# --------------------------------------------------------------------------------------

TABLE_DIRS = ['Brainwidemap', '2022_Q4_IBL_et_al_BWM', '2025_Q3_IBL_et_al_BWM']
PATCHED_DIR = CACHE_DIR / 'patched_tables'


def _split_rel_path(rel_path):
    """(collection, revision, filename) for an ALF relative path."""
    parts = rel_path.split('/')
    filename = parts[-1]
    if len(parts) > 1 and parts[-2].startswith('#') and parts[-2].endswith('#'):
        return '/'.join(parts[:-2]), parts[-2], filename
    return '/'.join(parts[:-1]), '', filename


def _session_file_index(session_dir):
    """Map (collection, filename) -> list of (revision, rel_path) for one session."""
    index = {}
    for root, _dirs, files in os.walk(session_dir):
        rel_root = os.path.relpath(root, session_dir)
        rel_root = '' if rel_root == '.' else rel_root
        collection, revision, _ = _split_rel_path(rel_root + '/_')
        for f in files:
            rel_path = f'{rel_root}/{f}' if rel_root else f
            index.setdefault((collection, f), []).append((revision, rel_path))
    return index


def build_patched_tables(force=False, verbose=True):
    """Write merged, disk-verified sessions.pqt/datasets.pqt and return their directory."""
    if PATCHED_DIR.exists() and not force:
        return PATCHED_DIR

    sessions = pd.concat(
        [pd.read_parquet(CACHE_DIR / d / 'sessions.pqt') for d in TABLE_DIRS]
    )
    sessions = sessions[~sessions.index.duplicated(keep='first')]

    datasets = pd.concat(
        [pd.read_parquet(CACHE_DIR / d / 'datasets.pqt') for d in TABLE_DIRS]
    )
    datasets = datasets[~datasets.index.duplicated(keep='first')].reset_index()
    datasets['rel_path'] = datasets['rel_path'].astype(str)

    # Session directory for every eid, from the released sessions table.
    session_dirs = {
        eid: CACHE_DIR / rec['lab'] / 'Subjects' / rec['subject'] /
        str(rec['date']) / f"{int(rec['number']):03d}"
        for eid, rec in sessions.iterrows()
    }

    keep_rows, new_paths = [], []
    for eid, group in datasets.groupby('eid', sort=False):
        session_dir = session_dirs.get(eid)
        if session_dir is None or not session_dir.exists():
            continue
        index = _session_file_index(session_dir)
        # One row per (collection, filename): the newest revision present on disk.
        # Revision folders are ISO dates, so a plain string max is chronological.
        by_key = {}
        for row, rel_path in zip(group.index, group['rel_path']):
            collection, _revision, filename = _split_rel_path(rel_path)
            by_key.setdefault((collection, filename), []).append((row, rel_path))
        for key, rows in by_key.items():
            candidates = index.get(key)
            if not candidates:
                continue
            best = max(candidates)[1]
            # Prefer the table row that describes the file actually on disk: when a
            # dataset was superseded by a revision, the old row is the one carrying the
            # CRITICAL qc flag that had it replaced, and its metadata must not be
            # attached to the replacement.
            match = [row for row, rel_path in rows if rel_path == best]
            keep_rows.append(match[0] if match else rows[0][0])
            new_paths.append(best)

    datasets = datasets.loc[keep_rows].copy()
    datasets['rel_path'] = new_paths
    datasets['exists'] = True
    # The released size/hash belong to the superseded revision, so ONE would report a
    # mismatch for every file it opens. Re-stat and drop the hash instead.
    datasets['file_size'] = [
        (session_dirs[eid] / rel).stat().st_size
        for eid, rel in zip(datasets['eid'], datasets['rel_path'])]
    datasets['hash'] = None
    # Every surviving row is now the unique, on-disk version of its dataset, so mark it
    # default: otherwise ONE emits a 'no default revision' warning per dataset per load.
    datasets['default_revision'] = True
    datasets = datasets.set_index(['eid', 'id']).sort_index()

    PATCHED_DIR.mkdir(parents=True, exist_ok=True)
    # ONE only accepts a table whose parquet metadata carries 'date_created', and it
    # re-casts a string index back to UUIDs on load, so write via its own helper with
    # the released table's metadata.
    from iblutil.io import parquet
    meta = {'date_created': datetime.datetime.now().isoformat(sep=' ', timespec='minutes'),
            'origin': 'patched-from-filesystem',
            'min_api_version': '2.10'}
    sessions.index = sessions.index.map(str)
    datasets.index = pd.MultiIndex.from_arrays(
        [datasets.index.get_level_values(i).map(str) for i in range(2)],
        names=datasets.index.names)
    parquet.save(PATCHED_DIR / 'sessions.pqt', sessions, meta)
    parquet.save(PATCHED_DIR / 'datasets.pqt', datasets, meta)
    if verbose:
        print(f'Patched cache tables: {len(sessions)} sessions, {len(datasets)} datasets '
              f'-> {PATCHED_DIR}')
    return PATCHED_DIR


def get_one(force_rebuild=False):
    """A local-mode ONE pointed at the patched tables."""
    from one.api import ONE
    tables_dir = build_patched_tables(force=force_rebuild)
    return ONE(
        base_url='https://openalyx.internationalbrainlab.org',
        silent=True,
        cache_dir=str(CACHE_DIR),
        tables_dir=str(tables_dir),
        mode='local',
    )


# Trial setup, copied from src/0_data_caching.py.
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100

# Trial exclusions, from ibl_data_utils.load_trials_and_mask (called with
# max_trial_len=10.0 by the reference caching script).
MIN_RT, MAX_RT = 0.08, 2.0
MAX_TRIAL_LEN = 10.0
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']

# "Neurons ... were excluded if they failed one of the three criteria ... amplitude,
# noise cut-off and refractory period violation" -- clusters.label is the fraction of
# those three that passed, so a well-isolated neuron has label == 1.
QC_LABEL = 1.0
# "restricted to regions that were designated grey matter ... contained at least five
# well-isolated neurons per session and were recorded from in at least two such
# sessions".  Beryl maps everything outside grey matter to 'root' or 'void'.
NON_GREY = {'root', 'void'}
MIN_NEURONS_PER_REGION = 5
MIN_SESSIONS_PER_REGION = 2

N_OUTPUT_BINS = 3  # tertiles for the two continuous behaviours
PRIOR_VALUES = np.array([0.2, 0.5, 0.8])


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def trials_mask(trials):
    """Reference load_trials_and_mask criteria, as a boolean array.

    True for trials to keep.  The unbiased 50:50 block at the start of the session is
    kept (``exclude_unbiased=False`` in the reference), which is what makes the
    prior a three-valued target.
    """
    query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
    query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
    query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
    for event in NAN_EXCLUDE:
        query += f' | {event}.isnull()'
    query += ' | (choice == 0)'
    return (~trials.eval(query)).to_numpy()


def trial_number_in_block(probability_left):
    """Index of each trial within its block of constant probabilityLeft (0-based).

    Computed on the full trials table so that the count is unaffected by trials that
    are later excluded -- the block the mouse experienced does not skip trials.
    """
    p = np.asarray(probability_left, dtype=float)
    new_block = np.ones(len(p), dtype=bool)
    new_block[1:] = p[1:] != p[:-1]
    block_start = np.maximum.accumulate(np.where(new_block, np.arange(len(p)), 0))
    return np.arange(len(p)) - block_start


def bin_spikes(spike_times, spike_clusters, cluster_ids, align_times):
    """Spike counts in (n_trials, n_clusters, N_BINS).

    Bin i of a trial covers [align + TIME_WINDOW[0] + i*BINSIZE, ... + BINSIZE), the
    same half-open bins that ``bincount2D`` produces inside the reference
    ``get_spike_data_per_interval``.
    """
    n_clusters = len(cluster_ids)
    # Position of each cluster id in the output matrix; ids are the row index in the
    # merged clusters table, so a direct lookup array is enough.
    size = int(max(cluster_ids.max(), spike_clusters.max())) + 1
    row_of_cluster = np.full(size, -1, dtype=np.int64)
    row_of_cluster[cluster_ids] = np.arange(n_clusters)

    begs = align_times + TIME_WINDOW[0]
    ends = align_times + TIME_WINDOW[1]
    i0 = np.searchsorted(spike_times, begs, side='left')
    i1 = np.searchsorted(spike_times, ends, side='left')

    out = np.zeros((len(align_times), n_clusters, N_BINS), dtype=np.float32)
    for k in range(len(align_times)):
        if i1[k] <= i0[k]:
            continue
        t = spike_times[i0[k]:i1[k]]
        c = row_of_cluster[spike_clusters[i0[k]:i1[k]]]
        keep = c >= 0
        if not keep.any():
            continue
        t, c = t[keep], c[keep]
        b = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
        np.clip(b, 0, N_BINS - 1, out=b)
        counts = np.bincount(c * N_BINS + b, minlength=n_clusters * N_BINS)
        out[k] = counts.reshape(n_clusters, N_BINS)
    return out


def bin_behavior(times, values, align_times):
    """Sample a continuous behavioural trace on the trial time grid.

    Returns (n_trials, N_BINS) values and a boolean mask of usable trials.

    The grid is the reference one from ``get_behavior_per_interval``: N_BINS points
    running from ``interval_start + BINSIZE`` to ``interval_end``, i.e. the right edge
    of each bin.  A trial is rejected when the trace does not cover the interval to
    within one bin at either end -- the same check the reference applies.  NaN samples
    are dropped before the coverage check rather than tolerated as the reference's
    ``allow_nans=True`` does, because the decoder needs a finite class label in every
    bin; a trial whose interval holds no finite sample therefore fails the check.
    """
    begs = align_times + TIME_WINDOW[0]
    ends = align_times + TIME_WINDOW[1]
    n_trials = len(align_times)
    grid = begs[:, None] + (np.arange(1, N_BINS + 1) * BINSIZE)[None, :]

    finite = np.isfinite(values)
    times, values = times[finite], values[finite]
    order = np.argsort(times, kind='stable')
    times, values = times[order], values[order]

    # Coverage check against the samples that fall strictly inside the interval.
    i0 = np.searchsorted(times, begs, side='right')
    i1 = np.searchsorted(times, ends, side='left')
    mask = i1 > i0
    first = times[np.clip(i0, 0, len(times) - 1)]
    last = times[np.clip(i1 - 1, 0, len(times) - 1)]
    mask &= np.abs(begs - first) <= BINSIZE
    mask &= np.abs(ends - last) <= BINSIZE
    mask &= np.isfinite(align_times)

    # Interpolated from the full trace rather than the within-interval slice, so the
    # edge bins are interpolated instead of extrapolated.  Only trials that already
    # passed the coverage check above are used.
    binned = np.interp(grid.ravel(), times, values).reshape(n_trials, N_BINS)
    mask &= np.isfinite(binned).all(axis=1)
    return binned, mask


def tertile_labels(values):
    """Discretise into N_OUTPUT_BINS equal-occupancy bins from this session's own values.

    Session-level quantiles rather than global ones: whisker motion energy is in
    camera-dependent arbitrary units and the wheel is used at very different rates by
    different mice, so a common threshold would mean 'low'/'high' referred to different
    behaviour in different sessions, while the decoder's output heads are shared across
    sessions.
    """
    edges = np.quantile(values, np.arange(1, N_OUTPUT_BINS) / N_OUTPUT_BINS)
    return np.searchsorted(edges, values, side='right').astype(np.int8)


# --------------------------------------------------------------------------------------
# per-session conversion
# --------------------------------------------------------------------------------------
def process_session(args):
    eid, pids, probe_names, subject, lab = args
    try:
        return _process_session(eid, pids, probe_names, subject, lab)
    except Exception as e:  # a session that cannot be loaded is reported and skipped
        return {'eid': eid, 'skip': f'{type(e).__name__}: {e}',
                'traceback': traceback.format_exc()}


def _process_session(eid, pids, probe_names, subject, lab):
    from brainbox.io.one import SessionLoader, SpikeSortingLoader
    from iblatlas.regions import BrainRegions
    one = get_one()

    # ---- trials -----------------------------------------------------------------
    sess_loader = SessionLoader(one=one, eid=eid)
    sess_loader.load_trials()
    trials = sess_loader.trials
    keep = trials_mask(trials)
    if keep.sum() < 2:
        return {'eid': eid, 'skip': f'only {int(keep.sum())} trials pass the criteria'}

    block_idx = trial_number_in_block(trials['probabilityLeft'].to_numpy())
    trials = trials[keep]
    block_idx = block_idx[keep]
    align_times = trials[ALIGN_TIME].to_numpy()

    # ---- spikes -----------------------------------------------------------------
    # Probes from the same session share the same behaviour and are not statistically
    # independent, so the reference merges them into one population.
    spikes_list, clusters_list = [], []
    for pid, pname in zip(pids, probe_names):
        ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
        spikes, clusters, channels = ssl.load_spike_sorting()
        if len(spikes) == 0:
            continue
        clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
        spikes_list.append(spikes)
        clusters_list.append(clusters)
    if not spikes_list:
        return {'eid': eid, 'skip': 'no spike sorting available'}

    offset = 0
    merged_clusters, merged_times, merged_ids = [], [], []
    for spikes, clusters in zip(spikes_list, clusters_list):
        merged_times.append(spikes['times'])
        merged_ids.append(spikes['clusters'].astype(np.int64) + offset)
        offset += int(clusters.index.max()) + 1
        merged_clusters.append(clusters)
    clusters = pd.concat(merged_clusters, ignore_index=True)
    spike_times = np.concatenate(merged_times)
    spike_clusters = np.concatenate(merged_ids)
    order = np.argsort(spike_times, kind='stable')
    spike_times, spike_clusters = spike_times[order], spike_clusters[order]
    finite = np.isfinite(spike_times)
    spike_times, spike_clusters = spike_times[finite], spike_clusters[finite]

    # ---- neuron inclusion criteria ----------------------------------------------
    beryl = BrainRegions().acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
    beryl = np.asarray(beryl, dtype=object)
    good = (clusters['label'].to_numpy() >= QC_LABEL) & ~np.isin(beryl, list(NON_GREY))
    # at least five well-isolated neurons per region in this session
    regions, counts = np.unique(beryl[good], return_counts=True)
    enough = set(regions[counts >= MIN_NEURONS_PER_REGION])
    good &= np.array([r in enough for r in beryl])
    cluster_ids = np.nonzero(good)[0]
    if len(cluster_ids) == 0:
        return {'eid': eid, 'skip': 'no neurons pass the inclusion criteria'}

    binned_spikes = bin_spikes(spike_times, spike_clusters, cluster_ids, align_times)

    # ---- behaviour ---------------------------------------------------------------
    sess_loader.load_wheel()
    wheel_speed, wheel_mask = bin_behavior(
        sess_loader.wheel['times'].to_numpy(),
        np.abs(sess_loader.wheel['velocity'].to_numpy()), align_times)

    # The reference prefers the left camera and falls back to the right one.
    whisker, whisker_mask = None, None
    for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sess_loader.load_motion_energy(views=[view])
            me = sess_loader.motion_energy[cam]
            whisker, whisker_mask = bin_behavior(
                me['times'].to_numpy(), me['whiskerMotionEnergy'].to_numpy(), align_times)
            break
        except Exception:
            continue
    if whisker is None:
        return {'eid': eid, 'skip': 'no whisker motion energy available'}

    # ---- align the streams --------------------------------------------------------
    ok = wheel_mask & whisker_mask
    if ok.sum() < 2:
        return {'eid': eid, 'skip': f'only {int(ok.sum())} trials have complete behaviour'}
    binned_spikes = binned_spikes[ok]
    wheel_speed, whisker = wheel_speed[ok], whisker[ok]
    trials, block_idx = trials[ok], block_idx[ok]
    align_times = align_times[ok]
    n_trials = int(ok.sum())

    # ---- decoder inputs ------------------------------------------------------------
    bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
    inputs = np.empty((n_trials, 2, N_BINS), dtype=np.float32)
    inputs[:, 0, :] = bin_centres
    inputs[:, 1, :] = block_idx[:, None]

    # ---- decoder outputs -----------------------------------------------------------
    # trials.choice is +1 when the mouse reported the left side and -1 for the right.
    choice = ((1 - trials['choice'].to_numpy()) / 2).astype(np.int8)
    prior = np.abs(trials['probabilityLeft'].to_numpy()[:, None]
                   - PRIOR_VALUES[None, :]).argmin(axis=1).astype(np.int8)
    outputs = np.empty((n_trials, 4, N_BINS), dtype=np.int8)
    outputs[:, 0, :] = choice[:, None]
    outputs[:, 1, :] = prior[:, None]
    outputs[:, 2, :] = tertile_labels(wheel_speed)
    outputs[:, 3, :] = tertile_labels(whisker)

    return {
        'eid': eid, 'subject': subject, 'lab': lab, 'skip': None,
        'neural': binned_spikes,
        'input': inputs,
        'output': outputs,
        'regions': beryl[cluster_ids].astype(str),
        'n_probes': len(spikes_list),
        'n_trials_total': int(len(keep)),
        'n_trials_pass_criteria': int(keep.sum()),
    }


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/app/converted_data.pkl')
    ap.add_argument('--n-sessions', type=int, default=None)
    ap.add_argument('--n-workers', type=int, default=16)
    args = ap.parse_args()

    build_patched_tables()

    bwm_df = pd.read_csv(BWM_RELEASE, index_col=0)
    # Optional subset marker shipped with reduced copies of the dataset.
    subset_file = Path('/app/data/DATALIMIT_SUBSET.csv')
    if subset_file.exists():
        subset = pd.read_csv(subset_file)
        col = 'eid' if 'eid' in subset.columns else subset.columns[0]
        bwm_df = bwm_df[bwm_df.eid.isin(subset[col].astype(str))]
        print(f'DATALIMIT_SUBSET.csv present: restricted to {bwm_df.eid.nunique()} sessions')

    jobs = []
    for eid, grp in bwm_df.groupby('eid', sort=False):
        jobs.append((eid, list(grp.pid), list(grp.probe_name),
                     grp.subject.iloc[0], grp.lab.iloc[0]))
    jobs.sort(key=lambda j: j[0])
    if args.n_sessions:
        jobs = jobs[:args.n_sessions]
    print(f'Processing {len(jobs)} sessions with {args.n_workers} workers', flush=True)

    results, skipped = [], []
    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=args.n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_session, jobs)):
            if res.get('skip'):
                skipped.append((res['eid'], res['skip']))
                print(f"[{i + 1}/{len(jobs)}] skipped {res['eid']}: {res['skip']}", flush=True)
            else:
                results.append(res)
                print(f"[{i + 1}/{len(jobs)}] {res['eid']} "
                      f"{res['neural'].shape[0]} trials x {res['neural'].shape[1]} neurons "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print(f'Loaded {len(results)} sessions, skipped {len(skipped)} in {time.time() - t0:.0f}s')

    results.sort(key=lambda r: r['eid'])
    assemble_and_save(results, skipped, args.out)


def assemble_and_save(results, skipped, out_path):
    # "recorded from in at least two such sessions": applied across the whole dataset,
    # after every session has been reduced to its qualifying regions.
    n_sessions_with_region = {}
    for res in results:
        for r in set(res['regions'].tolist()):
            n_sessions_with_region[r] = n_sessions_with_region.get(r, 0) + 1
    keep_regions = sorted(r for r, n in n_sessions_with_region.items()
                          if n >= MIN_SESSIONS_PER_REGION)
    region_index = {r: i for i, r in enumerate(keep_regions)}

    neural, inputs, outputs = [], [], []
    subjects, subject_idx, brain_region_idx, session_info = [], [], [], []
    for res in results:
        keep = np.array([r in region_index for r in res['regions']])
        if keep.sum() == 0:
            skipped.append((res['eid'], 'no neurons in regions recorded in >= 2 sessions'))
            continue
        spikes = res['neural'][:, keep, :]
        neural.append([spikes[k] for k in range(spikes.shape[0])])
        inputs.append([res['input'][k] for k in range(res['input'].shape[0])])
        outputs.append([res['output'][k] for k in range(res['output'].shape[0])])
        if res['subject'] not in subjects:
            subjects.append(res['subject'])
        subject_idx.append(subjects.index(res['subject']))
        brain_region_idx.append(
            np.array([region_index[r] for r in res['regions'][keep]], dtype=np.int64))
        session_info.append({
            'eid': res['eid'], 'subject': res['subject'], 'lab': res['lab'],
            'n_probes': res['n_probes'], 'n_trials': int(spikes.shape[0]),
            'n_neurons': int(spikes.shape[1]),
            'n_trials_in_session': res['n_trials_total'],
        })

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
        'brain_regions': keep_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_stimulus_onset', 'trial_number_in_block'],
        'output_names': ['choice', 'prior_probability_left', 'wheel_speed',
                         'whisker_motion_energy'],
        'output_values': [
            ['left', 'right'],
            ['0.2', '0.5', '0.8'],
            ['low', 'medium', 'high'],
            ['low', 'medium', 'high'],
        ],
        'metadata': {
            'task_description': (
                'IBL decision-making task: a Gabor patch appears at +-35 deg azimuth with '
                'one of five contrasts and the mouse turns a wheel to bring it to the '
                'centre. After 90 unbiased trials the prior probability of a left stimulus '
                'alternates between 0.2 and 0.8 in uncued blocks of 20-100 trials. '
                'Decode from the population spike counts: the choice (left/right), the '
                'block prior probability of left, and the tertile of wheel speed and of '
                'whisker motion energy in each 20 ms bin.'),
            'time_bin_size': BINSIZE * 1000,
            'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
            'off_start': TIME_WINDOW[0],
            'off_end': TIME_WINDOW[1],
            'n_timepoints': N_BINS,
            'neural_data': 'spike counts per 20 ms bin, Kilosort 2.5 / pykilosort units',
            'dataset': 'IBL brain-wide map public release (bwm_release.csv)',
            'input_descriptions': [
                'signed time of the bin centre relative to stimulus onset, in seconds',
                'index of the trial within its block of constant probabilityLeft, 0-based',
            ],
            'output_descriptions': [
                'reported stimulus side: left=0, right=1 (trials.choice +1/-1)',
                'block prior probability that the stimulus appears on the left: '
                '0.2 -> 0, 0.5 -> 1, 0.8 -> 2',
                'absolute wheel velocity, discretised into per-session tertiles',
                'whisker pad motion energy (left camera, right camera as fallback), '
                'discretised into per-session tertiles',
            ],
            'neuron_inclusion': (
                'well-isolated units (clusters.label == 1: amplitude, noise cut-off and '
                'refractory-period criteria all passed), in Allen grey-matter regions '
                '(Beryl mapping, excluding root/void), in regions with at least 5 such '
                'units in the session and recorded in at least 2 sessions; probes from '
                'the same session merged'),
            'trial_inclusion': (
                'trials with a detected choice, stimOn_times, feedback_times, '
                'probabilityLeft, firstMovement_times and feedbackType, reaction time '
                '(first wheel movement - stimulus onset) in [0.08, 2.0] s, '
                'feedback within 10 s of the go cue, and wheel and whisker traces '
                'covering the full 2 s window'),
            'session_info': session_info,
            'skipped_sessions': skipped,
        },
    }

    print(f'Writing {out_path} ...', flush=True)
    with open(out_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    total_trials = sum(len(s) for s in neural)
    print(f'{len(neural)} sessions, {total_trials} trials, '
          f'{sum(len(b) for b in brain_region_idx)} neurons, '
          f'{len(subjects)} subjects, {len(keep_regions)} brain regions')
    print(f'File size: {os.path.getsize(out_path) / 1e9:.2f} GB')


if __name__ == '__main__':
    sys.exit(main())
