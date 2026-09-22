#!/usr/bin/env python3
"""Convert the IBL brain-wide map data for stimulus-aligned neural decoding.

Processing follows code_zhang2025: all Kilosort 2.5 clusters, probes merged by
session, Beryl regions, 20-ms spike counts from -0.5 to 1.5 s around stimulus
onset, paper trial QC, smoothed absolute wheel velocity, and whisker ROI motion
energy (left camera, with right fallback).  The paper repository's deterministic
seed-42 BWM subject/session sample is used (10 sessions by default).
"""
import os, sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/app')
SRC = ROOT/'code/code_zhang2025/src'
sys.path.insert(0, str(SRC))
from one.api import ONE
from one.alf.cache import load_tables
from iblatlas.regions import BrainRegions
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from utils.ibl_data_utils import merge_probes, bin_spiking_data

BIN = 0.02
OFF0, OFF1 = -0.5, 1.5
PARAMS = dict(interval_len=2, binsize=BIN, single_region=False,
              align_time='stimOn_times', time_window=(OFF0, OFF1))
RELEASE = ROOT/'code/code_zhang2025/data/bwm_release.csv'
CACHE = ROOT/'data/one_cache'
WORK = ROOT/'data/converted_sessions'
OUT = ROOT/'converted_data.pkl'
N_DEFAULT = 10


def selected_sessions(n=None):
    """Return all release sessions, or the mounted data-limit subset when supplied."""
    bwm = pd.read_csv(RELEASE, index_col=0)
    subset_file = ROOT/'data/DATALIMIT_SUBSET.csv'
    if subset_file.exists():
        subset = pd.read_csv(subset_file)
        values = set(subset.astype(str).to_numpy().ravel())
        bwm = bwm[bwm.eid.astype(str).isin(values) | bwm.pid.astype(str).isin(values)]
    rows = []
    for eid, group in bwm.groupby('eid', sort=False):
        rows.append((str(group.subject.iloc[0]), str(eid),
                     list(group.pid.astype(str)), list(group.probe_name.astype(str))))
    # Explicit override is useful for smoke tests only; production defaults to all.
    if n is not None:
        rows = rows[:n]
    return rows


def trial_number_in_block(prob):
    out = np.empty(len(prob), dtype=np.float32)
    k = 0
    previous = None
    for i, value in enumerate(prob):
        if i == 0 or value != previous:
            k = 1
        else:
            k += 1
        out[i] = k
        previous = value
    return out


def local_spiking_data(one, eid, pid, pname):
    loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = loader.load_spike_sorting()
    # merge_probes offsets cluster IDs in place; local arrays may be read-only memmaps.
    spikes['clusters'] = np.array(spikes['clusters'], dtype=np.int32, copy=True)
    clusters = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=False).to_df()
    return spikes, clusters


def load_local_trials(one, eid):
    """Load the newest consolidated trial table (avoids obsolete ALF attributes)."""
    path = Path(one.eid2path(eid))/'alf'
    files = sorted(path.glob('#*#/_ibl_trials.table.pqt'))
    if not files:
        files = sorted(path.glob('_ibl_trials.table.pqt'))
    if not files:
        raise FileNotFoundError('no consolidated trials table')
    return pd.read_parquet(files[-1])


def paper_trial_mask(trials):
    required = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
                'firstMovement_times', 'feedbackType']
    valid = trials[required].notna().all(axis=1).to_numpy(dtype=bool, copy=True)
    rt = trials.firstMovement_times.to_numpy() - trials.stimOn_times.to_numpy()
    duration = trials.feedback_times.to_numpy() - trials.goCue_times.to_numpy()
    valid &= (rt >= 0.08) & (rt <= 2.0)
    valid &= duration <= 10.0
    valid &= trials.choice.to_numpy() != 0
    return valid


def local_continuous_behaviors(one, eid, trials):
    """Reference wheel processing and direct revision-safe whisker ROI loading."""
    sl = SessionLoader(one=one, eid=eid)
    sl.load_wheel()
    wheel_t = sl.wheel.times.to_numpy()
    wheel_v = np.abs(sl.wheel.velocity.to_numpy())
    alf = Path(one.eid2path(eid))/'alf'
    motion_t = motion_v = None
    for view in ('left', 'right'):
        tf = sorted(alf.glob(f'#*#/_ibl_{view}Camera.times.npy'))
        if not tf:
            tf = sorted(alf.glob(f'_ibl_{view}Camera.times.npy'))
        mf = sorted(alf.glob(f'#*#/{view}Camera.ROIMotionEnergy.npy'))
        if not mf:
            mf = sorted(alf.glob(f'{view}Camera.ROIMotionEnergy.npy'))
        for tfile in reversed(tf):
            t = np.load(tfile)
            match = None
            for mfile in reversed(mf):
                m = np.load(mfile)
                if len(m) == len(t):
                    match = m
                    break
            if match is not None:
                motion_t, motion_v = t, match
                break
        if motion_t is not None:
            break
    if motion_t is None:
        raise FileNotFoundError('no matching whisker motion energy and camera times')

    def interpolate(times, values):
        out, good = [], np.zeros(len(trials), bool)
        nbin = int(np.ceil((OFF1-OFF0)/BIN))
        for i, onset in enumerate(trials.stimOn_times.to_numpy()):
            beg, end = onset + OFF0, onset + OFF1
            ib = np.searchsorted(times, beg, side='right')
            ie = np.searchsorted(times, end, side='left')
            tx, vx = times[ib:ie], values[ib:ie]
            if len(vx) == 0 or not np.all(np.isfinite(vx)):
                out.append(None); continue
            if abs(beg-tx[0]) > BIN or abs(end-tx[-1]) > BIN:
                out.append(None); continue
            xi = np.linspace(beg+BIN, end, nbin)
            out.append(np.interp(xi, tx, vx).astype(np.float32))
            good[i] = True
        return out, good
    wheel, wheel_good = interpolate(wheel_t, wheel_v)
    whisk, whisk_good = interpolate(motion_t, motion_v)
    return wheel, whisk, wheel_good, whisk_good


def process_session(one, subject, eid, pids, probe_names):
    print(f'Processing {subject} {eid} ({len(pids)} probes)', flush=True)
    spikes_list, clusters_list = [], []
    for pid, pname in zip(pids, probe_names):
        spikes, clusters = local_spiking_data(one, eid, pid, pname)
        clusters['pid'] = pid
        spikes_list.append(spikes); clusters_list.append(clusters)
    spikes, clusters = merge_probes(spikes_list, clusters_list)
    trials = load_local_trials(one, eid)
    paper_mask = paper_trial_mask(trials)

    cluster_ids = np.arange(len(clusters), dtype=np.int64)
    neural_df = {'spike_times': spikes['times'], 'spike_clusters': spikes['clusters']}
    binned, used = bin_spiking_data(cluster_ids, neural_df, trials_df=trials,
                                    n_workers=4, **PARAMS)
    wheel_all, whisk_all, wheel_good, whisk_good = local_continuous_behaviors(one, eid, trials)
    valid = paper_mask & wheel_good & whisk_good
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        raise RuntimeError('fewer than two valid trials')

    neural = [np.asarray(binned[i].T, dtype=np.int16) for i in idx]
    wheel = [wheel_all[i] for i in idx]
    whisk = [whisk_all[i] for i in idx]
    choice_raw = trials.choice.to_numpy()[idx]
    if not np.all(np.isin(choice_raw, [-1, 1])):
        raise ValueError(f'unexpected choices {np.unique(choice_raw)}')
    choice = (choice_raw == 1).astype(np.int8)
    pleft_raw = trials.probabilityLeft.to_numpy()[idx]
    pmap = {0.2: 0, 0.5: 1, 0.8: 2}
    prior = np.array([pmap[round(float(x), 1)] for x in pleft_raw], np.int8)
    block_trial = trial_number_in_block(trials.probabilityLeft.to_numpy())[idx]
    beryl = BrainRegions().acronym2acronym(clusters.acronym.to_numpy(), mapping='Beryl')
    beryl = np.asarray(beryl)[np.asarray(used, dtype=int)]
    return dict(subject=subject, eid=eid, neural=neural, wheel=wheel,
                whisk=whisk, choice=choice, prior=prior, block_trial=block_trial,
                regions=beryl, source_trial_idx=idx)

def category_edges(sessions, key):
    values = np.concatenate([np.concatenate(s[key]) for s in sessions])
    edges = np.quantile(values, [1/3, 2/3])
    if not edges[0] < edges[1]:
        edges = np.array([np.nanpercentile(values, 33.333),
                          np.nanpercentile(values, 66.667)])
    return edges.astype(float)


def main():
    n_env = os.environ.get('IBL_N_SESSIONS')
    n = int(n_env) if n_env else None
    WORK.mkdir(parents=True, exist_ok=True)
    one = ONE(cache_dir=CACHE, mode='local', silent=True)
    one._cache = load_tables(CACHE/'Brainwidemap')
    sessions = []
    failures = []
    rows = selected_sessions(n)
    shard_count = int(os.environ.get('IBL_SHARD_COUNT', '1'))
    shard_index = int(os.environ.get('IBL_SHARD_INDEX', '0'))
    rows = [row for i, row in enumerate(rows) if i % shard_count == shard_index]
    for subject, eid, pids, pnames in rows:
        cache_file = WORK/f'{eid}.pkl'
        fail_file = WORK/f'{eid}.failed'
        try:
            if fail_file.exists():
                failures.append((eid, fail_file.read_text().strip()))
                print(f'SKIP {eid}: cached failure: {fail_file.read_text().strip()}', flush=True)
                continue
            if cache_file.exists():
                print('Loading intermediate', cache_file, flush=True)
                with open(cache_file, 'rb') as f: sess = pickle.load(f)
            else:
                sess = process_session(one, subject, eid, pids, pnames)
                tmp_file = cache_file.with_suffix('.pkl.tmp')
                with open(tmp_file, 'wb') as f:
                    pickle.dump(sess, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_file, cache_file)
            sessions.append(sess)
            if os.environ.get('IBL_CACHE_ONLY') == '1':
                sessions.pop()
                del sess
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f'SKIP {eid}: {type(exc).__name__}: {exc}', flush=True)
            failures.append((eid, repr(exc)))
            fail_file.write_text(f'{type(exc).__name__}: {exc}\n')
    if os.environ.get('IBL_CACHE_ONLY') == '1':
        print(f'Cache shard {shard_index}/{shard_count} complete; failures={len(failures)}')
        return
    if not sessions:
        raise RuntimeError('No sessions converted')

    wheel_edges = category_edges(sessions, 'wheel')
    whisk_edges = category_edges(sessions, 'whisk')
    time = np.arange(1, int(round((OFF1-OFF0)/BIN))+1, dtype=np.float32)*BIN + OFF0

    all_region_names = sorted(set(np.concatenate([s['regions'] for s in sessions]).tolist()))
    rmap = {r:i for i,r in enumerate(all_region_names)}
    subjects = sorted(set(s['subject'] for s in sessions))
    smap = {s:i for i,s in enumerate(subjects)}
    neural_all, input_all, output_all, region_idx = [], [], [], []
    session_info = []
    for s in sessions:
        ins, outs = [], []
        for j in range(len(s['neural'])):
            T = s['neural'][j].shape[1]
            inp = np.vstack((time[:T], np.full(T, s['block_trial'][j], np.float32)))
            wb = np.digitize(s['wheel'][j], wheel_edges).astype(np.int8)
            mb = np.digitize(s['whisk'][j], whisk_edges).astype(np.int8)
            out = np.vstack((np.full(T, s['choice'][j], np.int8),
                             np.full(T, s['prior'][j], np.int8), wb, mb))
            ins.append(inp); outs.append(out)
        neural_all.append(s['neural']); input_all.append(ins); output_all.append(outs)
        region_idx.append(np.array([rmap[x] for x in s['regions']], np.int32))
        session_info.append(dict(eid=s['eid'], subject=s['subject'],
            n_trials=len(s['neural']), n_neurons=len(s['regions']),
            source_trial_indices=s['source_trial_idx'].tolist()))

    data = dict(
        neural=neural_all, input=input_all, output=output_all,
        subjects=subjects,
        subject_idx=np.array([smap[s['subject']] for s in sessions], np.int32),
        brain_regions=all_region_names, brain_region_idx=region_idx,
        input_names=['time since stimulus onset', 'trial number in block'],
        output_names=['choice', 'prior probability of left', 'wheel speed',
                      'whisker motion energy'],
        output_values=[['left', 'right'], ['0.2', '0.5', '0.8'],
                       ['low', 'medium', 'high'], ['low', 'medium', 'high']],
        metadata=dict(
            task_description=('IBL visual decision task; decode choice, block prior, '
                'wheel speed and whisker motion energy from brain-wide activity.'),
            time_bin_size=20.0,
            temporal_alignment_event='visual stimulus onset (stimOn_times)',
            off_start=-0.5, off_end=1.5,
            neural_representation='Kilosort 2.5 spike counts in non-overlapping 20 ms bins',
            trial_filter=('required trial events; first movement minus stimulus onset '
                '0.08-2.00 s; go cue to feedback <=10 s; choice nonzero'),
            neuron_filter='all clusters; probes in the same session merged',
            brain_region_mapping='IBL Beryl ontology',
            behavior_processing=('wheel speed is absolute Gaussian-smoothed wheel velocity; '
                'whisker ROI motion energy uses left camera with right fallback; linear '
                'interpolation to bin right edges'),
            discretization='global pooled tertiles over retained samples',
            wheel_speed_bin_edges=wheel_edges.tolist(),
            whisker_motion_energy_bin_edges=whisk_edges.tolist(),
            session_selection=('all unique sessions in bundled BWM release (or DATALIMIT_SUBSET when present)'),
            session_info=session_info, failed_sessions=failures))
    with open(OUT, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {OUT}: {len(sessions)} sessions, '
          f'{sum(map(len, neural_all))} trials, {OUT.stat().st_size/1e6:.1f} MB')

if __name__ == '__main__':
    main()
