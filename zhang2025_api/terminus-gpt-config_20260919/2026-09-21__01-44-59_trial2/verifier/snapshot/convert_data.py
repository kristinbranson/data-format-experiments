#!/usr/bin/env python3
"""Convert staged IBL BWM data to the decoder pickle format.

All neuroscience arrays are loaded through ONE and brainbox.  The only filesystem
operation on the source cache is ONE's own make_parquet_db indexer.
"""
from __future__ import annotations
import argparse, gc, pickle, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

from one.api import ONE
from one.alf.cache import make_parquet_db
from brainbox.io.one import SpikeSortingLoader
from iblatlas.regions import BrainRegions

CACHE_ROOT = Path('/mnt/dataset/one_cache')
INDEX_DIR = Path('/app/cache/local_one_index')
TARGET_EIDS = Path('/app/code/code_zhang2025/data/repro_ephys_release.txt')
DT = 0.020
OFF_START, OFF_END = -0.5, 1.5
EDGES_REL = np.arange(OFF_START, OFF_END + DT / 2, DT)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
N_TIME = len(CENTERS_REL)
BR = BrainRegions()


def get_one():
    """Return local ONE backed by an index built by ONE itself."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    ses, dat = INDEX_DIR / 'sessions.pqt', INDEX_DIR / 'datasets.pqt'
    if not (ses.exists() and dat.exists() and ses.stat().st_size > 100):
        print('Building local ONE index...', flush=True)
        make_parquet_db(CACHE_ROOT, out_dir=INDEX_DIR, hash_ids=True, hash_files=False)
    one = ONE(cache_dir=CACHE_ROOT, mode='local')
    one.load_cache(tables_dir=INDEX_DIR)
    return one


def dataset_eids(one, filename):
    d = one._cache['datasets']
    names = d.rel_path.astype(str).str.rsplit('/', n=1).str[-1]
    return set(d.index.get_level_values('eid')[names.eq(filename)])


def candidate_eids(one):
    """Physically indexed sessions containing every required modality."""
    core = (dataset_eids(one, '_ibl_trials.table.pqt') &
            dataset_eids(one, '_ibl_wheel.timestamps.npy') &
            dataset_eids(one, '_ibl_wheel.position.npy') &
            dataset_eids(one, 'spikes.times.npy'))
    left = dataset_eids(one, '_ibl_leftCamera.times.npy') & dataset_eids(one, 'leftCamera.ROIMotionEnergy.npy')
    right = dataset_eids(one, '_ibl_rightCamera.times.npy') & dataset_eids(one, 'rightCamera.ROIMotionEnergy.npy')
    complete = core & (left | right)
    # The local index uses deterministic path-hash EIDs. Map the methods-paper
    # Alyx EIDs to these local IDs through session identity in TWO ONE tables.
    # This preserves the paper cohort without ever traversing/reading ALF files directly.
    if TARGET_EIDS.exists():
        targets = [x.strip() for x in TARGET_EIDS.read_text().splitlines() if x.strip()]
        release = ONE(cache_dir='/app/data/one_cache', mode='local')
        release.load_cache(tag='Brainwidemap')
        rs = release._cache['sessions']; ls = one._cache['sessions']
        target_rows = rs.loc[[i for i in rs.index if str(i) in set(targets)]]
        def key(row):
            return (str(row.get('lab')), str(row.get('subject')), str(row.get('date')), str(row.get('number')))
        local_by_key = {key(row): eid for eid, row in ls.iterrows() if eid in complete}
        selected = [local_by_key[key(row)] for _, row in target_rows.iterrows() if key(row) in local_by_key]
        if selected:
            print(f'Cohort: mapped {len(selected)}/{len(targets)} methods-paper EIDs to complete local BWM sessions')
            return selected
    print('WARNING: methods cohort unavailable; using all complete BWM sessions')
    return sorted(complete, key=str)


def trial_frame(tr):
    """Normalize an ALF trials object to a DataFrame."""
    if 'table' in tr and isinstance(tr['table'], pd.DataFrame):
        df = tr['table'].copy()
        for k, v in tr.items():
            if k != 'table' and np.asarray(v).ndim == 1 and len(v) == len(df):
                df[k] = v
        return df
    cols = {}
    for k, v in tr.items():
        a = np.asarray(v)
        if a.ndim == 1:
            cols[k] = a
        elif k == 'intervals' and a.ndim == 2 and a.shape[1] == 2:
            cols['intervals_0'], cols['intervals_1'] = a[:, 0], a[:, 1]
    return pd.DataFrame(cols)


def valid_trials(df, wheel_t, cam_t):
    n = len(df)
    mask = np.ones(n, dtype=bool)
    required = ['choice', 'probabilityLeft', 'feedbackType', 'feedback_times',
                'stimOn_times', 'firstMovement_times']
    for c in required:
        if c not in df:
            raise KeyError(f'missing trial column {c}')
        mask &= np.isfinite(df[c].to_numpy(float))
    choice = df.choice.to_numpy(float)
    prior = df.probabilityLeft.to_numpy(float)
    mask &= np.isin(choice, [-1, 1])
    mask &= np.isclose(prior[:, None], [0.2, 0.5, 0.8], atol=1e-6).any(axis=1)
    if {'intervals_0', 'intervals_1'} <= set(df):
        duration = df.intervals_1.to_numpy(float) - df.intervals_0.to_numpy(float)
        mask &= np.isfinite(duration) & (duration <= 10) & (duration > 0)
    stim = df.stimOn_times.to_numpy(float)
    starts, ends = stim + OFF_START, stim + OFF_END
    mask &= (starts >= wheel_t[0]) & (ends <= wheel_t[-1])
    mask &= (starts >= cam_t[0]) & (ends <= cam_t[-1])
    return mask


def clean_timeseries(t, x):
    t, x = np.asarray(t, float), np.asarray(x, float)
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    order = np.argsort(t, kind='stable'); t, x = t[order], x[order]
    keep = np.r_[True, np.diff(t) > 0]
    return t[keep], x[keep]


def wheel_speed(t, pos):
    t, pos = clean_timeseries(t, pos)
    # Timestamp-aware central differences; native position is angular radians.
    speed = np.abs(np.gradient(pos, t))
    speed[~np.isfinite(speed)] = np.nan
    return t, speed


def sample_trials(t, x, stim):
    """Linear interpolation at common bin centers (reference behavior method)."""
    q = stim[:, None] + CENTERS_REL[None, :]
    y = np.interp(q.ravel(), t, x, left=np.nan, right=np.nan).reshape(len(stim), N_TIME)
    return y.astype(np.float32)


def discretize_tertiles(values):
    finite = np.isfinite(values)
    q1, q2 = np.nanquantile(values, [1/3, 2/3])
    method = 'quantile'
    if not q2 > q1:
        # Preserve equal values in one class; derive thresholds from distinct values.
        uniq = np.unique(values[finite])
        if len(uniq) >= 3:
            q1, q2 = np.quantile(uniq, [1/3, 2/3]); method = 'unique_value_quantile'
        elif len(uniq) == 2:
            q1, q2 = uniq[0], uniq[0]; method = 'binary_degenerate'
        else:
            q1 = q2 = uniq[0]; method = 'constant'
    out = np.zeros(values.shape, dtype=np.int64)
    out[values > q1] = 1
    out[values > q2] = 2
    out[~finite] = -1
    return out, (float(q1), float(q2)), method


def load_camera(one, eid):
    errors = []
    for side in ('left', 'right'):
        try:
            cam = one.load_object(eid, f'{side}Camera', collection='alf')
            t, me = clean_timeseries(cam['times'], cam['ROIMotionEnergy'])
            if len(t) > 100 and len(t) == len(me):
                return side, t, me
        except Exception as exc:
            errors.append(f'{side}:{type(exc).__name__}')
    raise RuntimeError('no valid whisker motion stream (' + ','.join(errors) + ')')


def load_neural(one, eid):
    """Load, QC, anatomically annotate, and merge all probes in a session."""
    probes = sorted({c.split('/')[1] for c in one.list_collections(eid)
                     if c.startswith('alf/probe') and '/pykilosort' in c})
    all_t, all_c, all_reg = [], [], []
    offset = 0; raw_units = 0
    for probe in probes:
        sl = SpikeSortingLoader(eid=eid, pname=probe, one=one)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
        cid = np.asarray(clusters['cluster_id']).astype(int)
        raw_units += len(cid)
        label = np.asarray(clusters.get('label', np.zeros(len(cid))), float)
        atlas = np.asarray(clusters.get('atlas_id', np.zeros(len(cid))), int)
        good = (label >= 1) & (atlas > 0)
        good_ids = cid[good]
        if not len(good_ids):
            continue
        # Compact IDs allow fast bincount/histogramming.
        lut = {int(c): i + offset for i, c in enumerate(good_ids)}
        sc = np.asarray(spikes['clusters']).astype(int)
        keep = np.isin(sc, good_ids)
        mapped = np.fromiter((lut[int(c)] for c in sc[keep]), dtype=np.int32, count=int(keep.sum()))
        all_t.append(np.asarray(spikes['times'])[keep].astype(np.float64))
        all_c.append(mapped)
        beryl_ids = BR.remap(atlas[good], source_map='Allen', target_map='Beryl')
        all_reg.extend(BR.id2acronym(beryl_ids).tolist())
        offset += len(good_ids)
        del spikes, clusters, channels, sc, keep
        gc.collect()
    if offset == 0:
        raise RuntimeError('no QC-passing units')
    t = np.concatenate(all_t); c = np.concatenate(all_c)
    order = np.argsort(t, kind='stable')
    return t[order], c[order], np.asarray(all_reg, dtype=object), raw_units, probes


def bin_spikes(spike_t, spike_c, stim, n_units):
    out = np.zeros((len(stim), n_units, N_TIME), dtype=np.float32)
    for i, st in enumerate(stim):
        lo, hi = st + OFF_START, st + OFF_END
        a, b = np.searchsorted(spike_t, [lo, hi])
        relbin = np.floor((spike_t[a:b] - lo) / DT).astype(np.int32)
        clu = spike_c[a:b]
        ok = (relbin >= 0) & (relbin < N_TIME) & (clu >= 0) & (clu < n_units)
        flat = clu[ok] * N_TIME + relbin[ok]
        out[i] = np.bincount(flat, minlength=n_units * N_TIME).reshape(n_units, N_TIME)
    return out


def trial_number_in_block(prior):
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = 0 if not np.isclose(prior[i], prior[i-1]) else out[i-1] + 1
    return out


def session_details(one, eid):
    row = one._cache['sessions'].loc[eid]
    return {k: (str(row[k]) if k in row and pd.notna(row[k]) else None)
            for k in ('subject', 'date', 'number', 'lab')}


def process_session(one, eid, plot=False):
    t0 = time.time()
    tr = trial_frame(one.load_object(eid, 'trials'))
    wheel = one.load_object(eid, 'wheel')
    wt, ws = wheel_speed(wheel['timestamps'], wheel['position'])
    camera, ct, me = load_camera(one, eid)
    mask = valid_trials(tr, wt, ct)
    idx = np.flatnonzero(mask)
    if len(idx) < 2: raise RuntimeError(f'only {len(idx)} valid trials')
    stim_all = tr.stimOn_times.to_numpy(float)
    stim = stim_all[idx]
    wheel_cont = sample_trials(wt, ws, stim)
    whisk_cont = sample_trials(ct, me, stim)
    finite = np.isfinite(wheel_cont).all(1) & np.isfinite(whisk_cont).all(1)
    idx, stim = idx[finite], stim[finite]
    wheel_cont, whisk_cont = wheel_cont[finite], whisk_cont[finite]
    if len(idx) < 2: raise RuntimeError('fewer than two finite behavior trials')

    spike_t, spike_c, regions, raw_units, probes = load_neural(one, eid)
    neural_arr = bin_spikes(spike_t, spike_c, stim, len(regions))
    wheel_cls, wheel_thr, wheel_method = discretize_tertiles(wheel_cont)
    whisk_cls, whisk_thr, whisk_method = discretize_tertiles(whisk_cont)

    prior_all = tr.probabilityLeft.to_numpy(float)
    block_no_all = trial_number_in_block(prior_all)
    prior = prior_all[idx]
    choice = tr.choice.to_numpy(float)[idx]
    choice_cls = (choice == 1).astype(np.int64)  # -1 left, +1 right
    prior_cls = np.argmin(np.abs(prior[:, None] - np.array([.2, .5, .8])), axis=1).astype(np.int64)

    inputs=[]; outputs=[]; neural=[]
    for j in range(len(idx)):
        inp = np.vstack((CENTERS_REL.astype(np.float32),
                         np.full(N_TIME, block_no_all[idx[j]], np.float32)))
        out = np.vstack((np.full(N_TIME, choice_cls[j], np.int64),
                         np.full(N_TIME, prior_cls[j], np.int64),
                         wheel_cls[j], whisk_cls[j])).astype(np.int64)
        neural.append(neural_arr[j]); inputs.append(inp); outputs.append(out)

    info = session_details(one, eid)
    info.update(dict(eid=str(eid), probes=probes, motion_camera=camera,
                     raw_trials=len(tr), initially_valid_trials=int(mask.sum()), retained_trials=len(idx),
                     raw_units=int(raw_units), retained_units=len(regions),
                     wheel_tertiles=wheel_thr, wheel_discretization=wheel_method,
                     whisker_tertiles=whisk_thr, whisker_discretization=whisk_method,
                     source_trial_indices=idx.tolist(), processing_seconds=round(time.time()-t0, 3)))
    continuous = (wheel_cont, whisk_cont) if plot else None
    return neural, inputs, outputs, regions, info, continuous


def make_plot(eid, neural, outputs, continuous, info):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wheel, whisk = continuous
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    ax[0].imshow(neural[0], aspect='auto', interpolation='nearest', extent=[OFF_START,OFF_END,len(neural[0]),0])
    ax[0].set_ylabel('QC units'); ax[0].set_title(f'{eid} trial 0; stimulus at 0 s')
    ax[1].plot(CENTERS_REL,wheel[0]); ax[1].step(CENTERS_REL,outputs[0][2],where='mid'); ax[1].set_ylabel('wheel speed/bin')
    ax[2].plot(CENTERS_REL,whisk[0]); ax[2].step(CENTERS_REL,outputs[0][3],where='mid'); ax[2].set_ylabel('whisker ME/bin')
    ax[3].plot(CENTERS_REL,outputs[0][0],label='choice'); ax[3].plot(CENTERS_REL,outputs[0][1],label='prior'); ax[3].legend(); ax[3].set_xlabel('seconds from stimulus onset')
    for a in ax: a.axvline(0,color='r',ls='--',lw=.8)
    fig.tight_layout(); fig.savefig(f'/app/processing_{eid}.png',dpi=140); plt.close(fig)


def validate(data):
    assert len(data['neural']) == len(data['input']) == len(data['output']) == len(data['brain_region_idx'])
    assert len(data['subject_idx']) == len(data['neural'])
    for s in range(len(data['neural'])):
        assert len(data['neural'][s]) >= 2
        n = data['neural'][s][0].shape[0]
        assert len(data['brain_region_idx'][s]) == n
        for x,i,o in zip(data['neural'][s],data['input'][s],data['output'][s]):
            assert x.shape == (n,N_TIME) and i.shape == (2,N_TIME) and o.shape == (4,N_TIME)
            assert np.isfinite(x).all() and np.isfinite(i).all()
            assert set(np.unique(o[0])) <= {0,1}; assert set(np.unique(o[1])) <= {0,1,2}
            assert set(np.unique(o[2])) <= {0,1,2}; assert set(np.unique(o[3])) <= {0,1,2}


def main():
    ap=argparse.ArgumentParser(); g=ap.add_mutually_exclusive_group()
    g.add_argument('--full',action='store_true'); g.add_argument('--sample',action='store_true')
    ap.add_argument('--show-processing',action='store_true'); ap.add_argument('outpicklefile')
    args=ap.parse_args(); start=time.time()
    warnings.filterwarnings('ignore'); import logging; logging.getLogger('one').setLevel(logging.ERROR)
    one=get_one(); eids=candidate_eids(one)
    if args.sample: eids=eids[:2]
    print(f'Processing {len(eids)} sessions',flush=True)
    neural=[]; inputs=[]; outputs=[]; region_names=[]; infos=[]; subjects=[]; failures=[]
    for k,eid in enumerate(eids,1):
        try:
            print(f'[{k}/{len(eids)}] {eid}',flush=True)
            n,i,o,r,info,cont=process_session(one,eid,plot=args.show_processing and k<=2)
            neural.append(n); inputs.append(i); outputs.append(o); region_names.append(r); infos.append(info); subjects.append(info['subject'] or 'unknown')
            if cont is not None: make_plot(str(eid),n,o,cont,info)
            print(f"  kept trials={len(n)} units={len(r)} camera={info['motion_camera']} time={info['processing_seconds']}s",flush=True)
        except Exception as exc:
            failures.append({'eid':str(eid),'error':f'{type(exc).__name__}: {exc}'})
            print('  EXCLUDED:',failures[-1]['error'],flush=True)
        gc.collect()
    if not neural: raise RuntimeError('No sessions converted; failures='+repr(failures))
    subject_list=sorted(set(subjects)); subject_idx=np.array([subject_list.index(x) for x in subjects],dtype=np.int64)
    brain_regions=sorted(set(x for r in region_names for x in r.tolist()))
    region_idx=[np.array([brain_regions.index(x) for x in r],dtype=np.int64) for r in region_names]
    data=dict(neural=neural,input=inputs,output=outputs,subjects=subject_list,subject_idx=subject_idx,
              brain_regions=brain_regions,brain_region_idx=region_idx,
              input_names=['time_since_stimulus_onset','trial_number_in_block'],
              output_names=['choice','prior_probability_left','wheel_speed_bin','whisker_motion_energy_bin'],
              output_values=[['left','right'],['0.2','0.5','0.8'],['low','medium','high'],['low','medium','high']],
              metadata=dict(task_description='Stimulus-aligned IBL decision-task neural decoding of choice, prior, wheel speed and whisker motion energy.',
                time_bin_size=20.0,temporal_alignment_event='visual stimulus onset (stimOn_times)',off_start=OFF_START,off_end=OFF_END,
                n_timepoints=N_TIME,trial_number_in_block='zero-based; resets when probabilityLeft changes',
                neural_representation='QC-filtered spike counts per 20 ms bin',unit_qc='merged cluster label >= 1 with resolved atlas_id',
                dynamic_discretization='within-session behavioral tertiles; thresholds in session_info',session_info=infos,
                excluded_sessions=failures,source='IBL Brain-Wide Map ONE cache; methods processing from Zhang et al. 2025'))
    validate(data)
    with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    print(f'Saved {args.outpicklefile}: sessions={len(neural)} trials={sum(map(len,neural))} units={sum(x[0].shape[0] for x in neural)} size={Path(args.outpicklefile).stat().st_size/1e6:.1f}MB elapsed={time.time()-start:.1f}s')
    print(f'Excluded sessions: {len(failures)}')

if __name__=='__main__': main()
