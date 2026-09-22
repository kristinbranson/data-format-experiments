#!/usr/bin/env python3
"""Convert the local IBL Brain-Wide Map ONE cache to decoder format.

All scientific data are loaded with one.api.ONE and brainbox loaders.  The only
filesystem operations concern output files and ONE cache path/revision metadata.
"""
from __future__ import annotations
import argparse, pickle, re, time, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from brainbox.singlecell import bin_spikes2D
from brainbox.behavior import wheel as wheellib
from iblatlas.regions import BrainRegions

CACHE = Path('/app/data/one_cache')
BASE_TABLES = CACHE / 'Brainwidemap'
UPDATE_TABLES = CACHE / '2025_Q3_IBL_et_al_BWM'
DT = 0.020
OFF_START, OFF_END = -0.5, 1.5
N_BINS = 100
BIN_CENTERS = (OFF_START + DT / 2 + np.arange(N_BINS) * DT).astype(np.float32)
TRIAL_REV_DEFAULT = '2025-03-03'
BAD_REGIONS = {'void', 'root', 'nan', 'None', ''}


def make_ones():
    return (ONE(cache_dir=CACHE, tables_dir=BASE_TABLES, mode='local'),
            ONE(cache_dir=CACHE, tables_dir=UPDATE_TABLES, mode='local'))


def revision_from_path(path):
    m = re.search(r'/#([^#]+)#/', str(path))
    return m.group(1) if m else None


def load_trial_table(one, eid):
    """Load full aggregate table through ONE with cache-revision reconciliation."""
    rows = one.list_datasets(eid, filename='*trials.table*', details=True)
    if len(rows) != 1:
        raise RuntimeError(f'expected one trial table, found {len(rows)}')
    uid = one.to_eid(eid)
    # Cache release updates may leave stale/stripped revisions in table records.
    # Inspect path metadata only, then perform the scientific read through ONE.
    alf = one.eid2path(eid) / 'alf'
    matches = sorted(alf.glob('#*#/_ibl_trials.table.pqt'))
    if not matches:
        raise RuntimeError('no cached full trial table')
    physical = matches[-1]
    rev = physical.parent.name.strip('#')
    d = one._cache['datasets']
    mask = ((d.index.get_level_values('eid') == uid) &
            d.rel_path.astype(str).str.contains('_ibl_trials.table.pqt', regex=False))
    if mask.sum() != 1:
        raise RuntimeError(f'cannot identify trial-table cache record ({mask.sum()})')
    d.loc[mask, 'rel_path'] = f'alf/#{rev}#/_ibl_trials.table.pqt'
    return one.load_dataset(eid, '_ibl_trials.table.pqt', collection='alf',
                            revision=rev, check_hash=False)


def load_motion_energy(one_update, eid, camera):
    pattern = f'*{camera}Camera.ROIMotionEnergy*'
    rows = one_update.list_datasets(eid, filename=pattern, details=True)
    if len(rows) != 1:
        raise RuntimeError(f'{camera} motion energy rows={len(rows)}')
    rev = revision_from_path(rows.iloc[0].rel_path)
    if not rev:
        raise RuntimeError(f'{camera} motion energy revision unavailable')
    return one_update.load_dataset(eid, f'{camera}Camera.ROIMotionEnergy.npy',
                                   collection='alf', revision=rev, check_hash=False)


def candidate_eids(base, update):
    """Core ephys sessions with wheel and at least one paired camera stream."""
    bd, ud = base._cache['datasets'], update._cache['datasets']
    bp, up = bd.rel_path.astype(str), ud.rel_path.astype(str)
    def es(d, p, text):
        return set(d[p.str.contains(text, regex=False)].index.get_level_values('eid'))
    core = es(bd, bp, 'spikes.times') & es(bd, bp, '_ibl_wheel.timestamps.npy')
    left = es(ud, up, 'leftCamera.ROIMotionEnergy.npy') & es(bd, bp, '_ibl_leftCamera.times.npy')
    right = es(ud, up, 'rightCamera.ROIMotionEnergy.npy') & es(bd, bp, '_ibl_rightCamera.times.npy')
    return sorted(core & (left | right), key=str), left, right


def trial_number_in_block(prior):
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = 0 if prior[i] != prior[i - 1] else out[i - 1] + 1
    return out


def trial_mask(trials):
    required = ['choice', 'probabilityLeft', 'feedbackType', 'feedback_times',
                'stimOn_times', 'firstMovement_times']
    missing = [x for x in required if x not in trials]
    if missing:
        raise RuntimeError(f'missing trial columns {missing}')
    m = np.ones(len(trials), dtype=bool)
    for c in required:
        m &= np.isfinite(trials[c].to_numpy(dtype=float))
    choice = trials.choice.to_numpy(dtype=float)
    prior = trials.probabilityLeft.to_numpy(dtype=float)
    rt = trials.firstMovement_times.to_numpy(dtype=float) - trials.stimOn_times.to_numpy(dtype=float)
    m &= np.isin(choice, [-1., 1.])
    m &= np.isin(prior, [.2, .5, .8])
    m &= (rt >= .08) & (rt <= 2.0)
    return m


def interp_trials(times, values, align):
    """Linear interpolation at common bin centers; no extrapolation."""
    times = np.asarray(times, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    ok = np.isfinite(times) & np.isfinite(values)
    times, values = times[ok], values[ok]
    if len(times) < 2:
        return np.full((len(align), N_BINS), np.nan, np.float32)
    order = np.argsort(times, kind='stable')
    times, values = times[order], values[order]
    # Drop duplicate timestamps, retaining first (np.interp requires monotonic x).
    keep = np.r_[True, np.diff(times) > 0]
    times, values = times[keep], values[keep]
    query = align[:, None] + BIN_CENTERS[None, :]
    out = np.interp(query.ravel(), times, values, left=np.nan, right=np.nan)
    return out.reshape(len(align), N_BINS).astype(np.float32)


def load_wheel_speed(base, eid, align):
    w = base.load_object(eid, 'wheel', collection='alf')
    ts = np.asarray(w['timestamps'], float)
    pos = np.asarray(w['position'], float)
    # Reference brainbox processing: 1 kHz interpolation then filtered derivative.
    ipos, its = wheellib.interpolate_position(ts, pos, freq=1000)
    vel, _ = wheellib.velocity_filtered(ipos, fs=1000)
    return interp_trials(its, np.abs(vel), align), (ts, pos, its, vel)


def load_whisker(base, update, eid, align, left_set, right_set):
    """Load preferred left camera, falling back to right on absence/load failure."""
    uid = base.to_eid(eid)
    errors = []
    for camera, available in [('left', left_set), ('right', right_set)]:
        if uid not in available:
            continue
        try:
            me = load_motion_energy(update, eid, camera)
            times = base.load_dataset(eid, f'_ibl_{camera}Camera.times.npy', collection='alf')
            if len(me) != len(times):
                raise RuntimeError(f'{camera} motion/timestamp length mismatch {len(me)} != {len(times)}')
            return interp_trials(times, me, align), camera, (times, me)
        except Exception as exc:
            errors.append(f'{camera}: {type(exc).__name__}: {exc}')
    raise RuntimeError('no loadable paired motion-energy camera; ' + ' | '.join(errors))


def load_binned_neural(base, eid, align):
    """Load each probe through brainbox, curate units, bin, and concatenate neurons."""
    collections = sorted(c for c in base.list_collections(eid)
                         if c.startswith('alf/probe') and c.endswith('/pykilosort'))
    if not collections:
        raise RuntimeError('no pykilosort probe collections')
    arrays, regions, uuids = [], [], []
    spike_total = 0
    for coll in collections:
        pname = coll.split('/')[1]
        ssl = SpikeSortingLoader(eid=eid, pname=pname, one=base)
        spikes, clusters, channels = ssl.load_spike_sorting(revision='2024-05-06')
        merged = ssl.merge_clusters(spikes, clusters, channels).to_df()
        label = merged['label'].to_numpy() if 'label' in merged else np.zeros(len(merged))
        allen_acr = merged['acronym'].astype(str).to_numpy() if 'acronym' in merged else np.array(['void'] * len(merged))
        # Match the reference code: collapse fine Allen labels to Beryl, where
        # fiber tracts and non-gray labels map to root/void.
        acr = BrainRegions().acronym2acronym(allen_acr, mapping='Beryl').astype(str)
        good = (label >= 1) & ~np.isin(acr, list(BAD_REGIONS))
        ids = merged.index.to_numpy()[good]
        if not len(ids):
            continue
        spike_clusters = np.asarray(spikes['clusters'])
        selected_spikes = np.isin(spike_clusters, ids)
        binned, tscale = bin_spikes2D(np.asarray(spikes['times'])[selected_spikes],
                                    spike_clusters[selected_spikes], ids, align,
                                    pre_time=.5, post_time=1.5, bin_size=DT)
        # brainbox returns trials x clusters x bins.
        arrays.append(np.asarray(binned))
        regions.extend(acr[good].tolist())
        if 'uuids' in merged:
            uuids.extend(merged.loc[good, 'uuids'].astype(str).tolist())
        else:
            uuids.extend([f'{pname}:{x}' for x in ids])
        spike_total += len(spikes['times'])
    if not arrays:
        raise RuntimeError('zero curated neurons')
    x = np.concatenate(arrays, axis=1)
    if x.shape != (len(align), len(regions), N_BINS):
        raise RuntimeError(f'unexpected neural shape {x.shape}')
    # The validator and trainer expect float32. Counts remain exact because
    # observed per-bin integers are far below float32's exact integer limit.
    return x.astype(np.float32), regions, uuids, spike_total


def process_session(base, update, eid, left_set, right_set, show=False):
    t0 = time.time()
    trials = load_trial_table(base, eid)
    block_num = trial_number_in_block(trials.probabilityLeft.to_numpy(float))
    mask = trial_mask(trials)
    original_idx = np.flatnonzero(mask)
    align = trials.stimOn_times.to_numpy(float)[mask]
    choice_raw = trials.choice.to_numpy(float)[mask]
    prior_raw = trials.probabilityLeft.to_numpy(float)[mask]
    block_num = block_num[mask]
    wheel, wheel_raw = load_wheel_speed(base, eid, align)
    whisk, camera, whisk_raw = load_whisker(base, update, eid, align, left_set, right_set)
    complete = np.isfinite(wheel).all(1) & np.isfinite(whisk).all(1)
    original_idx, align = original_idx[complete], align[complete]
    choice_raw, prior_raw, block_num = choice_raw[complete], prior_raw[complete], block_num[complete]
    wheel, whisk = wheel[complete], whisk[complete]
    if len(align) < 2:
        raise RuntimeError(f'fewer than two complete valid trials ({len(align)})')
    neural, regions, uuids, nspikes = load_binned_neural(base, eid, align)
    # Trials with no spikes from any curated unit contain no neural observation
    # and are rejected by the supplied validator. Remove them synchronously.
    neural_valid = np.any(neural != 0, axis=(1, 2))
    n_zero_neural = int((~neural_valid).sum())
    if n_zero_neural:
        original_idx, align = original_idx[neural_valid], align[neural_valid]
        choice_raw, prior_raw, block_num = choice_raw[neural_valid], prior_raw[neural_valid], block_num[neural_valid]
        wheel, whisk = wheel[neural_valid], whisk[neural_valid]
        neural = neural[neural_valid]
    if len(align) < 2:
        raise RuntimeError(f'fewer than two neurally informative trials ({len(align)})')
    choice = (choice_raw == 1).astype(np.uint8)
    prior = np.select([prior_raw == .2, prior_raw == .5, prior_raw == .8], [0, 1, 2]).astype(np.uint8)
    ref = base.eid2ref(eid)
    out = dict(eid=str(eid), subject=str(ref.subject), neural=neural,
               regions=regions, uuids=uuids, choice=choice, prior=prior,
               trial_number=block_num.astype(np.float32), wheel=wheel, whisker=whisk,
               trial_indices=original_idx.astype(np.int32), camera=camera,
               n_trials_raw=len(trials), n_trials_valid=len(align), n_zero_neural=n_zero_neural, nspikes=nspikes)
    print(f"SESSION {eid} subject={out['subject']} trials={len(trials)}->{len(align)} "
          f"neurons={neural.shape[1]} camera={camera} spikes={nspikes} time={time.time()-t0:.2f}s", flush=True)
    return out, (trials, wheel_raw, whisk_raw) if show else None


def make_plot(session, raw, thresholds, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    trials, wheel_raw, whisk_raw = raw
    n = min(3, session['n_trials_valid'])
    fig, axes = plt.subplots(4, n, figsize=(5*n, 11), squeeze=False)
    for j in range(n):
        idx = session['trial_indices'][j]
        stim = float(trials.stimOn_times.iloc[idx])
        its, ipos, uts, vel = wheel_raw
        wt, me = whisk_raw
        axes[0,j].imshow(session['neural'][j], aspect='auto', interpolation='nearest',
                         extent=[OFF_START, OFF_END, session['neural'].shape[1], 0])
        axes[0,j].axvline(0,color='r'); axes[0,j].set_title(f'trial {idx} neural counts')
        m=(uts>=stim+OFF_START)&(uts<stim+OFF_END)
        axes[1,j].plot(uts[m]-stim,np.abs(vel[m]),alpha=.5,label='raw/filter')
        axes[1,j].plot(BIN_CENTERS,session['wheel'][j],label='20 ms samples'); axes[1,j].legend()
        m=(wt>=stim+OFF_START)&(wt<stim+OFF_END)
        axes[2,j].plot(wt[m]-stim,me[m],'.-',alpha=.5); axes[2,j].plot(BIN_CENTERS,session['whisker'][j])
        axes[3,j].step(BIN_CENTERS,np.digitize(session['wheel'][j],thresholds['wheel']),where='mid',label='wheel class')
        axes[3,j].step(BIN_CENTERS,np.digitize(session['whisker'][j],thresholds['whisker']),where='mid',label='whisk class'); axes[3,j].legend()
        for ax in axes[:,j]: ax.axvline(0,color='r',lw=.7); ax.set_xlim(OFF_START,OFF_END)
    fig.suptitle(f"{session['eid']} ({session['subject']}); camera={session['camera']}")
    fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)


def build_output(sessions, show_raw=None):
    wheel_vals = np.concatenate([s['wheel'].ravel() for s in sessions])
    whisk_vals = np.concatenate([s['whisker'].ravel() for s in sessions])
    thresholds = {'wheel': np.quantile(wheel_vals, [1/3, 2/3]).astype(float),
                  'whisker': np.quantile(whisk_vals, [1/3, 2/3]).astype(float)}
    for k,v in thresholds.items():
        if not np.isfinite(v).all() or v[0] >= v[1]:
            raise RuntimeError(f'invalid {k} tertiles {v}')
    subjects = sorted({s['subject'] for s in sessions}); smap={x:i for i,x in enumerate(subjects)}
    brain_regions = sorted({r for s in sessions for r in s['regions']}); rmap={x:i for i,x in enumerate(brain_regions)}
    neural, inputs, outputs, bridx = [], [], [], []
    session_info=[]
    for s in sessions:
        wc=np.digitize(s['wheel'],thresholds['wheel']).astype(np.uint8)
        mc=np.digitize(s['whisker'],thresholds['whisker']).astype(np.uint8)
        ns, ins, outs=[],[],[]
        for j in range(s['n_trials_valid']):
            ns.append(s['neural'][j])
            ins.append(np.vstack((BIN_CENTERS, np.full(N_BINS,s['trial_number'][j],np.float32))))
            outs.append(np.vstack((np.full(N_BINS,s['choice'][j],np.uint8),
                                   np.full(N_BINS,s['prior'][j],np.uint8),wc[j],mc[j])))
        neural.append(ns); inputs.append(ins); outputs.append(outs)
        bridx.append(np.array([rmap[r] for r in s['regions']],dtype=np.int32))
        session_info.append({k:s[k] for k in ['eid','subject','camera','n_trials_raw','n_trials_valid','n_zero_neural','trial_indices','uuids']})
    data={'neural':neural,'input':inputs,'output':outputs,'subjects':subjects,
          'subject_idx':np.array([smap[s['subject']] for s in sessions],dtype=np.int32),
          'brain_regions':brain_regions,'brain_region_idx':bridx,
          'input_names':['time_since_stimulus_onset_s','trial_number_in_block'],
          'output_names':['choice','prior_probability_left','wheel_speed_tertile','whisker_motion_energy_tertile'],
          'output_values':[['left','right'],['0.2','0.5','0.8'],['low','medium','high'],['low','medium','high']],
          'metadata':{'task_description':'Decode binary wheel choice, prior block, wheel-speed tertile, and whisker-motion-energy tertile from stimulus-aligned Neuropixels spike counts.',
                      'time_bin_size':20.0,'temporal_alignment_event':'visual stimulus onset (stimOn_times)',
                      'off_start':OFF_START,'off_end':OFF_END,'neural_representation':'spike counts per 20 ms bin',
                      'source_release':'IBL Brain-Wide Map public release (459 sessions, 699 insertions)',
                      'neuron_filter':'well-isolated clusters: label >= 1; non-void atlas acronym',
                      'trial_filter':'finite required events; choice +/-1; prior in {0.2,0.5,0.8}; stimulus-to-first-movement 0.08-2.00 s; complete behavior coverage',
                      'wheel_processing':'brainbox 1 kHz position interpolation, 20 Hz low-pass filtered derivative, absolute speed sampled at bin centers',
                      'whisker_processing':'left camera ROI motion energy (right fallback), sampled at bin centers',
                      'discretization':'global tertiles over retained finite time bins',
                      'wheel_speed_thresholds':thresholds['wheel'].tolist(),
                      'whisker_motion_energy_thresholds':thresholds['whisker'].tolist(),
                      'session_info':session_info}}
    return data, thresholds


def validate(data):
    assert len(data['neural'])==len(data['input'])==len(data['output'])==len(data['subject_idx'])
    for si,(ns,ins,outs,ri) in enumerate(zip(data['neural'],data['input'],data['output'],data['brain_region_idx'])):
        assert len(ns)==len(ins)==len(outs)>=2
        assert len(ri)==ns[0].shape[0]
        for n,x,y in zip(ns,ins,outs):
            assert n.ndim==2 and n.shape[1]==N_BINS and n.shape[0]==len(ri)
            assert x.shape==(2,N_BINS) and y.shape==(4,N_BINS)
            assert np.isfinite(n).all() and np.isfinite(x).all() and np.isfinite(y).all()
            assert np.allclose(x[0],BIN_CENTERS)
            assert set(np.unique(y[0])).issubset({0,1})
            for k in [1,2,3]: assert set(np.unique(y[k])).issubset({0,1,2})
    print('INTERNAL VALIDATION PASSED',flush=True)


def _process_worker(eid_str):
    """Independent process worker; ONE instances are intentionally not pickled."""
    base, update = make_ones()
    eids, left, right = candidate_eids(base, update)
    return process_session(base, update, base.to_eid(eid_str), left, right, False)[0]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('outpicklefile')
    g=ap.add_mutually_exclusive_group(); g.add_argument('--full',action='store_true'); g.add_argument('--sample',action='store_true')
    ap.add_argument('--show-processing',action='store_true'); args=ap.parse_args()
    sample=args.sample
    base,update=make_ones(); eids,left,right=candidate_eids(base,update)
    print(f'CANDIDATES {len(eids)} mode={"sample" if sample else "full"}',flush=True)
    sessions=[]; raws={}; failures=[]; target=2 if sample else len(eids); t0=time.time()
    if sample:
        for eid in eids:
            if len(sessions)>=target: break
            try:
                sess,raw=process_session(base,update,eid,left,right,args.show_processing and len(sessions)<2)
                sessions.append(sess)
                if raw is not None: raws[sess['eid']]=raw
            except Exception as exc:
                failures.append((str(eid),type(exc).__name__,str(exc)))
                print(f'SKIP {eid} {type(exc).__name__}: {exc}',flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os
        workers=min(8, os.cpu_count() or 1)
        print(f'PARALLEL workers={workers}',flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs={ex.submit(_process_worker,str(eid)):str(eid) for eid in eids}
            for done,fut in enumerate(as_completed(futs),1):
                eid=futs[fut]
                try: sessions.append(fut.result())
                except Exception as exc:
                    failures.append((eid,type(exc).__name__,str(exc)))
                    print(f'SKIP {eid} {type(exc).__name__}: {exc}',flush=True)
                if done % 20 == 0: print(f'PROGRESS {done}/{len(futs)} elapsed={time.time()-t0:.1f}s',flush=True)
        sessions.sort(key=lambda x:x['eid'])
    if len(sessions)<2: raise RuntimeError(f'only {len(sessions)} sessions converted')
    data,thresholds=build_output(sessions); validate(data)
    out=Path(args.outpicklefile); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as f: pickle.dump(data,f,protocol=5)
    if args.show_processing:
        for s in sessions[:2]:
            if s['eid'] in raws: make_plot(s,raws[s['eid']],thresholds,Path('/app')/f"processing_{s['eid']}.png")
    print('TERTILES', {k:v.tolist() for k,v in thresholds.items()},flush=True)
    print('SUMMARY sessions',len(sessions),'subjects',len(data['subjects']),'trials',sum(map(len,data['neural'])),
          'neurons',sum(x[0].shape[0] for x in data['neural']),'regions',len(data['brain_regions']),flush=True)
    print('FAILURES',len(failures),Counter(x[1] for x in failures),flush=True)
    for x in failures[:30]: print('FAILURE_DETAIL',x,flush=True)
    print(f'SAVED {out} bytes={out.stat().st_size} elapsed={time.time()-t0:.2f}s',flush=True)

if __name__=='__main__': main()
