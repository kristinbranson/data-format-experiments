#!/usr/bin/env python3
"""Convert the frozen IBL BWM release to the decoder-compatible pickle format.

Usage: python -u /app/convert_data.py OUTPUT [--full|--sample] [--show-processing]
Reads the staged, read-only ONE cache at /mnt/dataset/one_cache.  No network is used.
"""
from __future__ import annotations
import argparse, gc, pickle, time, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from brainbox.behavior import wheel as wheel_utils
from iblatlas.regions import BrainRegions

SEED = 42
BIN = 0.02
OFF_START, OFF_END = -0.5, 1.5
N_BINS = 100
EDGES_REL = np.linspace(OFF_START, OFF_END, N_BINS + 1)
# Reference behavior interpolation predicts the value at each spike bin's right edge.
TIME_REL = EDGES_REL[1:].astype(np.float32)
SOURCE = Path('/mnt/dataset/one_cache')
BWM_CSV = Path('/app/code/code_zhang2025/data/bwm_release.csv')
SESSIONS_PQT = SOURCE / '2025_Q3_IBL_et_al_BWM' / 'sessions.pqt'
DATASETS_PQT = SOURCE / 'Brainwidemap' / 'datasets.pqt'


def choose_file(root: Path, basename: str, parent_contains: str | None = None) -> Path:
    """Choose the newest revisioned source file, falling back to unrevisioned."""
    candidates = [p for p in root.rglob(basename)
                  if parent_contains is None or parent_contains in p.as_posix()]
    if not candidates:
        raise FileNotFoundError(f'{basename} under {root}')
    # A revision component #YYYY-MM-DD# sorts after an empty revision; newest wins.
    def key(p):
        revisions = [q.strip('#') for q in p.parts if q.startswith('#') and q.endswith('#')]
        return (max(revisions, default=''), p.as_posix())
    return sorted(candidates, key=key)[-1]


def session_dir(row: pd.Series) -> Path:
    return (SOURCE / str(row.lab) / 'Subjects' / str(row.subject) /
            str(row.date) / f'{int(row.session_number):03d}')


def select_reference_sessions(sample: bool) -> list[dict]:
    """Match 0_data_caching.py: seeded subject permutation, first CSV EID/subject."""
    bwm = pd.read_csv(BWM_CSV, index_col=0)
    subjects = np.unique(bwm.subject)
    rng = np.random.RandomState(SEED)
    selected = rng.choice(subjects, len(subjects), replace=False)
    by_subject = bwm.groupby('subject', sort=True)
    out = []
    for subject in selected:
        first_idx = by_subject.groups[subject][0]
        row = bwm.loc[first_idx]
        out.append({'eid': str(row.eid), 'subject': str(subject), 'row': row})
    # Sample means first two successfully converted sessions, so retain candidates here.
    return out


def load_trials(sdir: Path):
    p = choose_file(sdir / 'alf', '_ibl_trials.table.pqt')
    return pd.read_parquet(p), p


def load_behavior(sdir: Path, trials: pd.DataFrame):
    """Return reference wheel speed and left-first whisker stream plus trial samples."""
    alf = sdir / 'alf'
    wt = np.load(choose_file(alf, '_ibl_wheel.timestamps.npy'), mmap_mode='r')
    wp = np.load(choose_file(alf, '_ibl_wheel.position.npy'), mmap_mode='r')
    if len(wt) != len(wp) or len(wt) < 20:
        raise ValueError('invalid wheel timestamps/position lengths')
    # Robustly mirror loader behavior for rare duplicated timestamps: retain the first
    # finite sample at each time.  (One frozen session has one exact duplicate.)
    wf = np.isfinite(wt) & np.isfinite(wp)
    wt0, wp0 = np.asarray(wt[wf]), np.asarray(wp[wf])
    _, unique_idx = np.unique(wt0, return_index=True)
    unique_idx.sort(); wt0, wp0 = wt0[unique_idx], wp0[unique_idx]
    if len(wt0) < 20 or np.any(np.diff(wt0) <= 0):
        raise ValueError('wheel timestamps cannot be made strictly increasing')
    # Exact IBL reference primitives: 1 kHz linear position then filtered derivative.
    pos_i, time_i = wheel_utils.interpolate_position(wt0, wp0, freq=1000, kind='linear')
    vel_i, _ = wheel_utils.velocity_filtered(pos_i, fs=1000)
    speed_i = np.abs(vel_i)

    camera = None
    for side in ('left', 'right'):
        try:
            ct = np.load(choose_file(alf, f'_ibl_{side}Camera.times.npy'), mmap_mode='r')
            cv = np.load(choose_file(alf, f'{side}Camera.ROIMotionEnergy.npy'), mmap_mode='r')
            if len(ct) == len(cv) and len(ct) > 20:
                cf = np.isfinite(ct) & np.isfinite(cv)
                ct0, cv0 = np.asarray(ct[cf]), np.asarray(cv[cf])
                _, ci = np.unique(ct0, return_index=True)
                ci.sort(); ct0, cv0 = ct0[ci], cv0[ci]
                if len(ct0) > 20 and np.all(np.diff(ct0) > 0):
                    ct, cv = ct0, cv0
                    camera = side
                    break
        except FileNotFoundError:
            continue
    if camera is None:
        raise ValueError('no complete left or right whisker motion-energy stream')

    stim = trials.stimOn_times.to_numpy(float)
    targets = stim[:, None] + TIME_REL[None, :]
    # np.interp is linear and only used for rows whose full endpoints are covered.
    coverage = ((targets[:, 0] >= time_i[0]) & (targets[:, -1] <= time_i[-1]) &
                (targets[:, 0] >= ct[0]) & (targets[:, -1] <= ct[-1]))
    wheel_vals = np.full((len(trials), N_BINS), np.nan, dtype=np.float32)
    whisk_vals = np.full_like(wheel_vals, np.nan)
    for i in np.flatnonzero(coverage & np.isfinite(stim)):
        wheel_vals[i] = np.interp(targets[i], time_i, speed_i).astype(np.float32)
        whisk_vals[i] = np.interp(targets[i], ct, cv).astype(np.float32)
    coverage &= np.isfinite(wheel_vals).all(1) & np.isfinite(whisk_vals).all(1)
    return wheel_vals, whisk_vals, coverage, camera


def trial_mask(trials: pd.DataFrame, behavior_coverage: np.ndarray):
    required = ['stimOn_times', 'firstMovement_times', 'feedback_times', 'choice',
                'probabilityLeft', 'intervals_0', 'intervals_1']
    missing = [c for c in required if c not in trials]
    if missing:
        raise ValueError(f'missing trial columns {missing}')
    a = trials[required].to_numpy(float)
    finite = np.isfinite(a).all(1)
    stim = trials.stimOn_times.to_numpy(float)
    move = trials.firstMovement_times.to_numpy(float)
    feedback = trials.feedback_times.to_numpy(float)
    rt = move - stim
    duration = trials.intervals_1.to_numpy(float) - trials.intervals_0.to_numpy(float)
    choice = trials.choice.to_numpy(float)
    prior = trials.probabilityLeft.to_numpy(float)
    valid_prior = np.isclose(prior[:, None], [0.2, 0.5, 0.8], atol=1e-6).any(1)
    mask = (finite & behavior_coverage & (rt >= 0.08) & (rt <= 2.0) &
            (feedback >= stim) & (duration > 0) & (duration <= 10.0) &
            np.isin(choice, [-1.0, 1.0]) & valid_prior)
    reasons = {
        'nonfinite_required': int((~finite).sum()),
        'reaction_time_outside_0.08_2.0': int((finite & ~((rt >= .08) & (rt <= 2))).sum()),
        'invalid_duration_or_order': int((finite & ~((feedback >= stim) & (duration > 0) & (duration <= 10))).sum()),
        'invalid_choice': int((~np.isin(choice, [-1., 1.])).sum()),
        'invalid_prior': int((~valid_prior).sum()),
        'behavior_window_uncovered': int((~behavior_coverage).sum()),
    }
    return mask, reasons


def load_probe(probe_root: Path, br: BrainRegions):
    """Load one probe's good clusters and memory-mapped spike arrays."""
    # Select a coherent latest pykilosort revision from metrics, then sibling arrays.
    metrics_p = choose_file(probe_root, 'clusters.metrics.pqt', 'pykilosort')
    parent = metrics_p.parent
    metrics = pd.read_parquet(metrics_p)
    ch = np.load(parent / 'clusters.channels.npy', mmap_mode='r')
    st = np.load(parent / 'spikes.times.npy', mmap_mode='r')
    sc = np.load(parent / 'spikes.clusters.npy', mmap_mode='r')
    if len(st) != len(sc) or len(ch) != len(metrics):
        raise ValueError(f'inconsistent spike/cluster files in {parent}')
    channel_ids = np.load(parent / 'channels.brainLocationIds_ccf_2017.npy', mmap_mode='r')
    cids = metrics.cluster_id.to_numpy(int) if 'cluster_id' in metrics else np.arange(len(metrics))
    labels = metrics.label.to_numpy(float)
    channels = np.asarray(ch, int)
    anatomical_ok = (channels >= 0) & (channels < len(channel_ids))
    atlas_ids = np.zeros(len(channels), dtype=np.int64)
    atlas_ids[anatomical_ok] = np.asarray(channel_ids)[channels[anatomical_ok]]
    regions = np.asarray(br.id2acronym(atlas_ids, mapping='Beryl')).astype(str)
    bad_names = np.isin(np.char.lower(regions), ['root', 'void', 'nan', 'none', ''])
    good = (labels >= 1.0) & anatomical_ok & (~bad_names)
    good_cids = cids[good]
    good_regions = regions[good]
    # Cluster IDs index the lookup; tolerate sparse IDs.
    lookup = np.full(max(int(np.max(cids)), int(np.max(sc[:min(len(sc), 1000000)]))) + 2,
                     -1, dtype=np.int32)
    lookup[good_cids] = np.arange(len(good_cids), dtype=np.int32)
    uuids_p = parent / 'clusters.uuids.csv'
    if uuids_p.exists():
        u = pd.read_csv(uuids_p, header=None).iloc[:, -1].astype(str).to_numpy()
        uuids = u[good] if len(u) == len(good) else np.array([f'{parent}:{x}' for x in good_cids])
    else:
        uuids = np.array([f'{parent}:{x}' for x in good_cids])
    return {'times': st, 'clusters': sc, 'lookup': lookup, 'regions': good_regions,
            'uuids': uuids, 'n_raw': len(metrics), 'n_good': int(good.sum()),
            'source': str(parent)}


def bin_spikes(probes, stim_times):
    n_neurons = sum(p['n_good'] for p in probes)
    out = []
    offsets = np.cumsum([0] + [p['n_good'] for p in probes[:-1]])
    for k, stim in enumerate(stim_times):
        mat = np.zeros((n_neurons, N_BINS), dtype=np.float32)
        abs_edges = stim + EDGES_REL
        for p, off in zip(probes, offsets):
            ts, cs = p['times'], p['clusters']
            lo = int(np.searchsorted(ts, abs_edges[0], side='left'))
            hi = int(np.searchsorted(ts, abs_edges[-1], side='left'))
            if hi <= lo or p['n_good'] == 0:
                continue
            t = np.asarray(ts[lo:hi])
            c = np.asarray(cs[lo:hi], dtype=np.int64)
            in_lookup = c < len(p['lookup'])
            local = np.full(len(c), -1, dtype=np.int32)
            local[in_lookup] = p['lookup'][c[in_lookup]]
            tb = np.floor((t - abs_edges[0]) / BIN + 1e-10).astype(np.int32)
            ok = (local >= 0) & (tb >= 0) & (tb < N_BINS)
            flat = (local[ok].astype(np.int64) * N_BINS + tb[ok])
            counts = np.bincount(flat, minlength=p['n_good'] * N_BINS)
            mat[off:off+p['n_good']] += counts.reshape(p['n_good'], N_BINS).astype(np.float32)
        out.append(mat)
    return out


def process_session(item, br):
    t0 = time.time(); row = item['row']; eid = item['eid']; sdir = session_dir(row)
    if not sdir.exists():
        raise FileNotFoundError(sdir)
    trials, trial_path = load_trials(sdir)
    wheel, whisk, coverage, camera = load_behavior(sdir, trials)
    mask, reasons = trial_mask(trials, coverage)
    trial_idx = np.flatnonzero(mask)
    if len(trial_idx) < 2:
        raise ValueError(f'only {len(trial_idx)} valid trials')

    # Use exactly the probe insertions listed in the frozen release for this EID.
    bwm = pd.read_csv(BWM_CSV, index_col=0)
    probe_names = bwm.loc[bwm.eid.astype(str) == eid, 'probe_name'].astype(str).tolist()
    probes = []
    for name in probe_names:
        root = sdir / 'alf' / name
        if root.exists():
            p = load_probe(root, br)
            if p['n_good']:
                probes.append(p)
    if not probes:
        raise ValueError('no good grey-matter neurons')
    regions = np.concatenate([p['regions'] for p in probes])
    uuids = np.concatenate([p['uuids'] for p in probes])
    if len(set(uuids)) != len(uuids):
        raise ValueError('duplicate neuron UUIDs after probe merge')

    selected = trials.iloc[trial_idx]
    neural = bin_spikes(probes, selected.stimOn_times.to_numpy(float))
    inputs, scalar = [], []
    prior_levels = np.array([0.2, 0.5, 0.8])
    for raw_i, (_, tr) in zip(trial_idx, selected.iterrows()):
        inp = np.vstack([TIME_REL, np.full(N_BINS, raw_i, dtype=np.float32)]).astype(np.float32)
        choice = 0 if float(tr.choice) == -1 else 1
        prior = int(np.argmin(np.abs(prior_levels - float(tr.probabilityLeft))))
        inputs.append(inp)
        scalar.append((choice, prior))
    info = {
        'eid': eid, 'subject': item['subject'], 'session_path': str(sdir),
        'trial_source': str(trial_path), 'camera_side': camera,
        'n_trials_raw': len(trials), 'trial_indices': trial_idx.astype(np.int32),
        'n_trials_retained': len(trial_idx), 'trial_exclusion_counts': reasons,
        'n_units_raw': int(sum(p['n_raw'] for p in probes)),
        'n_units_retained': len(regions), 'probe_sources': [p['source'] for p in probes],
        'neuron_uuids': uuids.tolist(), 'elapsed_s': time.time()-t0,
    }
    return {'neural': neural, 'input': inputs, 'scalar': scalar,
            'wheel': wheel[trial_idx], 'whisk': whisk[trial_idx],
            'regions': regions, 'info': info}


def plot_processing(sess, thresholds, outpath):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wthr, qthr = thresholds
    nshow = min(3, len(sess['neural']))
    fig, ax = plt.subplots(4, nshow, figsize=(5*nshow, 12), squeeze=False)
    for j in range(nshow):
        pop = sess['neural'][j].sum(0)
        ax[0,j].plot(TIME_REL, pop); ax[0,j].axvline(0,color='r'); ax[0,j].set_title(f"trial {sess['info']['trial_indices'][j]} spikes")
        ax[1,j].plot(TIME_REL, sess['wheel'][j]); ax[1,j].axhline(wthr[0],ls='--'); ax[1,j].axhline(wthr[1],ls='--'); ax[1,j].set_title('wheel speed + tertiles')
        ax[2,j].plot(TIME_REL, sess['whisk'][j]); ax[2,j].axhline(qthr[0],ls='--'); ax[2,j].axhline(qthr[1],ls='--'); ax[2,j].set_title(f"{sess['info']['camera_side']} whisker ME + tertiles")
        ax[3,j].imshow(sess['neural'][j][:min(80,len(sess['neural'][j]))], aspect='auto', extent=[OFF_START,OFF_END,min(80,len(sess['neural'][j])),0]); ax[3,j].set_title('spike counts')
        for i in range(4): ax[i,j].set_xlabel('s from stimulus onset')
    fig.tight_layout(); fig.savefig(outpath, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='process full reference cohort (default)')
    mode.add_argument('--sample', action='store_true', help='process first two eligible sessions')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()
    sample = args.sample
    print(f'IBL conversion mode={"sample" if sample else "full"}; source={SOURCE}', flush=True)
    print(f'grid: {N_BINS} bins, {BIN}s, [{OFF_START},{OFF_END}) aligned to stimulus onset', flush=True)
    br = BrainRegions()
    candidates = select_reference_sessions(sample)
    sessions, skipped = [], []
    target_n = 2 if sample else len(candidates)
    for i, item in enumerate(candidates):
        if sample and len(sessions) >= 2:
            break
        print(f'[{i+1}/{len(candidates)}] {item["eid"]} subject={item["subject"]}', flush=True)
        try:
            z = process_session(item, br)
            sessions.append(z)
            print(f"  retained trials={z['info']['n_trials_retained']}/{z['info']['n_trials_raw']} neurons={z['info']['n_units_retained']} camera={z['info']['camera_side']} time={z['info']['elapsed_s']:.1f}s", flush=True)
        except Exception as e:
            skipped.append({'eid': item['eid'], 'subject': item['subject'], 'reason': f'{type(e).__name__}: {e}'})
            print('  SKIP', skipped[-1]['reason'], flush=True)
        gc.collect()
    if len(sessions) < (2 if sample else 1):
        raise RuntimeError(f'insufficient converted sessions: {len(sessions)}')

    # Pooled aligned-sample tertiles; deterministic and independent of output labels.
    wheel_all = np.concatenate([s['wheel'].ravel() for s in sessions])
    whisk_all = np.concatenate([s['whisk'].ravel() for s in sessions])
    wthr = np.quantile(wheel_all, [1/3, 2/3]).astype(float)
    qthr = np.quantile(whisk_all, [1/3, 2/3]).astype(float)
    print('wheel tertiles',wthr,'whisker tertiles',qthr,flush=True)

    region_names = sorted(set(np.concatenate([s['regions'] for s in sessions]).tolist()))
    region_map = {r:i for i,r in enumerate(region_names)}
    subjects = sorted(set(s['info']['subject'] for s in sessions))
    subject_map = {v:i for i,v in enumerate(subjects)}
    outputs=[]
    for s in sessions:
        so=[]
        for (choice,prior), w, q in zip(s['scalar'],s['wheel'],s['whisk']):
            out=np.empty((4,N_BINS),dtype=np.int8)
            out[0]=choice; out[1]=prior
            out[2]=np.searchsorted(wthr,w,side='right').astype(np.int8)
            out[3]=np.searchsorted(qthr,q,side='right').astype(np.int8)
            so.append(out)
        outputs.append(so)

    data={
      'neural':[s['neural'] for s in sessions],
      'input':[s['input'] for s in sessions],
      'output':outputs,
      'subjects':subjects,
      'subject_idx':np.array([subject_map[s['info']['subject']] for s in sessions],dtype=np.int32),
      'brain_regions':region_names,
      'brain_region_idx':[np.array([region_map[x] for x in s['regions']],dtype=np.int32) for s in sessions],
      'input_names':['time_since_stimulus_onset_s','trial_number_in_session'],
      'output_names':['choice','prior_probability_left','wheel_speed_bin','whisker_motion_energy_bin'],
      'output_values':[['left','right'],['0.2','0.5','0.8'],['low','medium','high'],['low','medium','high']],
      'metadata':{
        'task_description':'Visual two-alternative forced-choice task; decode choice, block prior, wheel speed and whisker motion energy from stimulus-aligned spike counts.',
        'time_bin_size':20.0,
        'temporal_alignment_event':'visual stimulus onset (trials.stimOn_times)',
        'off_start':OFF_START,'off_end':OFF_END,
        'source_release':'2025_Q3_IBL_et_al_BWM; insertions from code_zhang2025/data/bwm_release.csv',
        'session_selection':'one first-listed EID per subject; subjects permuted with NumPy RandomState seed 42',
        'bin_semantics':'neural counts in [edge_i,edge_i+1); time/behavior sampled at right bin edge',
        'neuron_filter':'IBL merged cluster label >= 1 (all RIGOR metrics pass) and valid Beryl grey-matter acronym',
        'trial_filter':'finite ordered events, 0.08<=firstMovement-stimOn<=2.0 s, duration<=10 s, valid choice/prior, full behavior window',
        'choice_mapping':{'-1':0,'1':1},
        'prior_mapping':{'0.2':0,'0.5':1,'0.8':2},
        'wheel_speed_tertiles':wthr.tolist(),'whisker_motion_energy_tertiles':qthr.tolist(),
        'discretization':'pooled empirical tertiles over retained aligned bins; np.searchsorted(side=right)',
        'time_bin_right_edges_s':TIME_REL.tolist(),
        'session_info':[s['info'] for s in sessions],
        'skipped_sessions':skipped,
      }}
    # Internal structural checks before writing.
    for si in range(len(sessions)):
        assert len(data['neural'][si])==len(data['input'][si])==len(data['output'][si])>=2
        assert data['brain_region_idx'][si].shape==(data['neural'][si][0].shape[0],)
        for n,x,y in zip(data['neural'][si],data['input'][si],data['output'][si]):
            assert n.shape[1]==x.shape[1]==y.shape[1]==N_BINS
            assert x.shape[0]==2 and y.shape[0]==4 and np.isfinite(n).all() and np.isfinite(x).all()
    with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    if args.show_processing:
        for s in sessions[:2]:
            plot_processing(s,(wthr,qthr),f"processing_{s['info']['eid']}.png")
    nt=sum(len(x) for x in data['neural']); nn=sum(x[0].shape[0] for x in data['neural'])
    print(f'WROTE {args.outpicklefile}: sessions={len(sessions)} subjects={len(subjects)} trials={nt} summed_session_neurons={nn} regions={len(region_names)} skipped={len(skipped)}',flush=True)
    print('choice counts',Counter(int(y[0,0]) for sess in outputs for y in sess),flush=True)
    print('prior counts',Counter(int(y[1,0]) for sess in outputs for y in sess),flush=True)
    print('wheel counts',Counter(np.concatenate([y[2] for sess in outputs for y in sess]).tolist()),flush=True)
    print('whisker counts',Counter(np.concatenate([y[3] for sess in outputs for y in sess]).tolist()),flush=True)

if __name__=='__main__': main()
