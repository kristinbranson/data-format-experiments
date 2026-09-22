#!/usr/bin/env python3
"""Convert the IBL Brainwidemap release to the requested neural-decoder pickle.

Scientific arrays are loaded exclusively with ONE and brainbox loaders.  The methods-paper
freeze CSV supplies only the curated EID/PID cohort and probe names.
"""
from __future__ import annotations
import argparse, pickle, re, sys, time, traceback, warnings, uuid
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from one.api import ONE
from brainbox.io.one import SessionLoader, SpikeSortingLoader

BIN = 0.02
OFF_START, OFF_END = -0.5, 1.5
EDGES_REL = np.arange(OFF_START, OFF_END + BIN / 2, BIN, dtype=np.float64)
CENTERS_REL = ((EDGES_REL[:-1] + EDGES_REL[1:]) / 2).astype(np.float32)
assert EDGES_REL.size == 101 and CENTERS_REL.size == 100
FREEZE = Path('/app/code/code_zhang2025/data/bwm_release.csv')
INVALID_REGIONS = {'', 'void', 'root', 'nan', 'none'}


def init_one() -> ONE:
    """Open the staged release and repair its canonical trials-table default metadata."""
    one = ONE(silent=True)
    one.load_cache(tag='Brainwidemap', clobber=True)
    # The frozen cache marks the canonical table non-default, causing load_object to return
    # only supplemental columns.  Repair metadata in memory; no source file is touched.
    ds = one._cache['datasets']
    mask = ds.rel_path.str.endswith('_ibl_trials.table.pqt')
    ds.loc[mask, 'default_revision'] = True
    return one


def cohort(sample: bool) -> tuple[pd.DataFrame, list[str]]:
    f = pd.read_csv(FREEZE).drop(columns=['Unnamed: 0'], errors='ignore')
    # Preserve freeze order (paper/reference order), grouping all probes from each EID.
    eids = list(dict.fromkeys(f.eid.astype(str)))
    if sample:
        eids = eids[:2]
    return f[f.eid.astype(str).isin(eids)].copy(), eids


def trial_number_in_block(prior: np.ndarray) -> np.ndarray:
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = out[i - 1] + 1 if np.isfinite(prior[i]) and prior[i] == prior[i - 1] else 0
    return out


def interpolate_trials(times, values, stim, centers_rel=CENTERS_REL):
    """Linear interpolation onto common trial bin centers; NaN outside stream support."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    good = np.isfinite(times) & np.isfinite(values)
    times, values = times[good], values[good]
    if times.size < 2:
        return np.full((len(stim), len(centers_rel)), np.nan, dtype=np.float32)
    order = np.argsort(times)
    times, values = times[order], values[order]
    # Remove duplicate timestamps, retaining first; np.interp requires increasing x.
    keep = np.r_[True, np.diff(times) > 0]
    times, values = times[keep], values[keep]
    q = np.asarray(stim)[:, None] + centers_rel[None, :]
    flat = np.interp(q.ravel(), times, values, left=np.nan, right=np.nan)
    return flat.reshape(q.shape).astype(np.float32)


def load_behavior(one: ONE, eid: str, stim: np.ndarray):
    sl = SessionLoader(one=one, eid=eid)
    sl.load_wheel()
    w = sl.wheel
    speed = interpolate_trials(w['times'].to_numpy(), np.abs(w['velocity'].to_numpy()), stim)
    errors = []
    for side in ('left', 'right'):
        try:
            sl.load_motion_energy(views=[side])
            key = f'{side}Camera'
            me = sl.motion_energy[key]
            vals = me['whiskerMotionEnergy'].to_numpy()
            whisk = interpolate_trials(me['times'].to_numpy(), vals, stim)
            if np.isfinite(whisk).any():
                return speed, whisk, side, errors
        except Exception as ex:
            errors.append(f'{side}:{type(ex).__name__}:{ex}')
    raise RuntimeError('no usable whisker motion energy; ' + '; '.join(errors))


def cluster_fields(clusters):
    """Return cluster IDs, labels and acronyms from merged SpikeSortingLoader output."""
    n = len(clusters.get('channels', clusters.get('depths', [])))
    ids = np.asarray(clusters.get('cluster_id', np.arange(n)), dtype=np.int64)
    labels = np.asarray(clusters.get('label', np.zeros(n)), dtype=float)
    acr = np.asarray(clusters.get('acronym', np.repeat('', n))).astype(str)
    if not (len(ids) == len(labels) == len(acr)):
        raise ValueError(f'cluster metadata length mismatch {len(ids)}, {len(labels)}, {len(acr)}')
    return ids, labels, acr


def bin_probe(spike_times, spike_clusters, good_ids, stim):
    """Vectorized counts for selected units, returning trials x units x time."""
    spike_times = np.asarray(spike_times, dtype=float)
    spike_clusters = np.asarray(spike_clusters, dtype=np.int64)
    good_ids = np.asarray(good_ids, dtype=np.int64)
    out = np.zeros((len(stim), len(good_ids), 100), dtype=np.uint16)
    if not len(good_ids):
        return out
    # Dense mapping is fast for Kilosort nonnegative IDs.
    mapper = np.full(max(int(spike_clusters.max(initial=0)), int(good_ids.max(initial=0))) + 1, -1, dtype=np.int32)
    mapper[good_ids] = np.arange(len(good_ids), dtype=np.int32)
    for ti, st in enumerate(stim):
        lo, hi = np.searchsorted(spike_times, (st + OFF_START, st + OFF_END))
        ts = spike_times[lo:hi]
        cs = spike_clusters[lo:hi]
        valid_id = (cs >= 0) & (cs < mapper.size)
        ui = np.full(cs.shape, -1, dtype=np.int32)
        ui[valid_id] = mapper[cs[valid_id]]
        keep = ui >= 0
        if not np.any(keep):
            continue
        bi = np.floor((ts[keep] - (st + OFF_START)) / BIN).astype(np.int32)
        valid = (bi >= 0) & (bi < 100)
        flat = ui[keep][valid].astype(np.int64) * 100 + bi[valid]
        cnt = np.bincount(flat, minlength=len(good_ids) * 100).reshape(len(good_ids), 100)
        if cnt.max(initial=0) > np.iinfo(np.uint16).max:
            raise OverflowError('spike count exceeds uint16')
        out[ti] = cnt.astype(np.uint16)
    return out


def load_neural(one: ONE, probe_rows: pd.DataFrame, stim: np.ndarray):
    mats, regions, probe_info, failures = [], [], [], []
    for row in probe_rows.itertuples(index=False):
        try:
            loader = SpikeSortingLoader(pid=str(row.pid), one=one)
            spikes, clusters, channels = loader.load_spike_sorting()
            clusters = loader.merge_clusters(spikes, clusters, channels)
            ids, labels, acr = cluster_fields(clusters)
            region_ok = np.array([x.strip().lower() not in INVALID_REGIONS for x in acr])
            keep = (labels >= 1) & region_ok
            good_ids, good_acr = ids[keep], acr[keep]
            if not len(good_ids):
                probe_info.append({'pid': str(row.pid), 'probe': row.probe_name, 'units': 0})
                continue
            pm = bin_probe(spikes['times'], spikes['clusters'], good_ids, stim)
            mats.append(pm); regions.extend(good_acr.tolist())
            probe_info.append({'pid': str(row.pid), 'probe': row.probe_name,
                               'units': int(len(good_ids)), 'clusters': int(len(ids))})
        except Exception as ex:
            failures.append({'pid': str(row.pid), 'probe': row.probe_name,
                             'error': f'{type(ex).__name__}: {ex}'})
            print(f'  WARNING probe {row.probe_name}/{row.pid} failed: {type(ex).__name__}: {ex}', flush=True)
    if not mats:
        raise RuntimeError('no probe yielded qualified units')
    return np.concatenate(mats, axis=1), regions, probe_info, failures


def quantile_classes(x: np.ndarray, valid_trials: np.ndarray):
    vals = x[valid_trials]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError('no finite values for discretization')
    q = np.quantile(vals, [1/3, 2/3]).astype(float)
    if q[1] <= q[0]:
        q[1] = np.nextafter(q[0], np.inf)
    cls = np.digitize(x, q, right=False).astype(np.uint8)
    return cls, q.tolist()


def process_session(one, freeze, eid):
    t0 = time.time()
    sl = SessionLoader(one=one, eid=eid)
    sl.load_trials()
    tr = sl.trials
    required = ['choice','probabilityLeft','feedbackType','feedback_times','stimOn_times',
                'firstMovement_times','intervals_0','intervals_1']
    missing = [x for x in required if x not in tr]
    if missing:
        raise KeyError(f'missing trial columns {missing}; got {list(tr.columns)}')
    n_native = len(tr)
    stim = tr.stimOn_times.to_numpy(float)
    prior = tr.probabilityLeft.to_numpy(float)
    choice = tr.choice.to_numpy(float)
    block_num = trial_number_in_block(prior)
    vals = tr[required].to_numpy(float)
    valid = np.all(np.isfinite(vals), axis=1)
    valid &= np.isin(choice, [-1, 1]) & np.isin(prior, [0.2, 0.5, 0.8])
    valid &= (tr.intervals_1.to_numpy() - tr.intervals_0.to_numpy() <= 10.0)
    latency = tr.firstMovement_times.to_numpy() - stim
    valid &= (latency >= 0.08) & (latency <= 2.0)

    speed, whisk, camera, behavior_errors = load_behavior(one, eid, stim)
    valid &= np.all(np.isfinite(speed), axis=1) & np.all(np.isfinite(whisk), axis=1)
    if valid.sum() < 2:
        raise RuntimeError(f'only {valid.sum()} valid trials')

    # Bin neural only after behavioral/trial QC to avoid unnecessary work.
    idx = np.flatnonzero(valid)
    neural_all, regions, probes, probe_failures = load_neural(one, freeze, stim[idx])
    speed_cls, speed_q = quantile_classes(speed, valid)
    whisk_cls, whisk_q = quantile_classes(whisk, valid)

    inputs, outputs, neural = [], [], []
    prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
    for j, raw_i in enumerate(idx):
        inp = np.vstack((CENTERS_REL, np.full(100, block_num[raw_i], np.float32)))
        out = np.empty((4, 100), dtype=np.uint8)
        out[0] = 0 if choice[raw_i] == -1 else 1
        out[1] = prior_map[float(prior[raw_i])]
        out[2] = speed_cls[raw_i]
        out[3] = whisk_cls[raw_i]
        inputs.append(inp); outputs.append(out); neural.append(neural_all[j].copy())
        # neural_all is trials x units x time; each target matrix is units x time
    assert all(a.shape == (len(regions), 100) for a in neural), 'neural orientation/region mismatch'
    assert all(a.shape == (2, 100) for a in inputs), 'input shape mismatch'
    assert all(a.shape == (4, 100) for a in outputs), 'output shape mismatch'
    try:
        details = one._cache['sessions'].loc[uuid.UUID(str(eid))]
        subject, lab, date = str(details.subject), str(details.lab), str(details.date)
    except (KeyError, ValueError):
        # Freeze metadata are identifiers/provenance only; scientific arrays remain ONE-loaded.
        fr = freeze.iloc[0]
        subject, lab, date = str(fr.subject), str(fr.lab), str(fr.date)
    info = {'eid': eid, 'subject': subject, 'lab': lab,
            'date': date, 'native_trials': n_native, 'retained_trials': len(idx),
            'raw_trial_indices': idx.tolist(), 'n_neurons': len(regions), 'camera_side': camera,
            'wheel_tertiles': speed_q, 'whisker_tertiles': whisk_q, 'probes': probes,
            'probe_failures': probe_failures, 'behavior_load_notes': behavior_errors,
            'choice_counts': dict(Counter(int(outputs[k][0,0]) for k in range(len(outputs)))),
            'prior_counts': dict(Counter(int(outputs[k][1,0]) for k in range(len(outputs)))),
            'wheel_class_counts': np.bincount(np.concatenate([x[2] for x in outputs]), minlength=3).tolist(),
            'whisker_class_counts': np.bincount(np.concatenate([x[3] for x in outputs]), minlength=3).tolist(),
            'elapsed_s': time.time() - t0}
    print(f"  retained {len(idx)}/{n_native} trials, {len(regions)} units, camera={camera}, {info['elapsed_s']:.1f}s", flush=True)
    return neural, inputs, outputs, regions, info, speed[idx], whisk[idx]


def processing_plot(eid, neural, outputs, speed, whisk):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = min(25, len(neural)); x = CENTERS_REL
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    pop = np.stack([a.sum(0) for a in neural[:n]])
    ax[0].imshow(pop, aspect='auto', extent=[OFF_START, OFF_END, n, 0]); ax[0].set_ylabel('trial'); ax[0].set_title(f'{eid}: population spike counts')
    ax[1].plot(x, speed[:n].T, alpha=.15, color='k'); ax[1].set_ylabel('wheel speed'); ax[1].set_title('continuous wheel processing')
    ax[2].imshow(np.stack([o[2] for o in outputs[:n]]), aspect='auto', vmin=0, vmax=2, extent=[OFF_START,OFF_END,n,0]); ax[2].set_ylabel('trial'); ax[2].set_title('wheel classes (0/1/2)')
    ax[3].imshow(np.stack([o[3] for o in outputs[:n]]), aspect='auto', vmin=0, vmax=2, extent=[OFF_START,OFF_END,n,0]); ax[3].set_ylabel('trial'); ax[3].set_title('whisker classes (0/1/2)'); ax[3].set_xlabel('time from stimulus onset (s)')
    for a in ax: a.axvline(0,color='r',lw=1)
    fig.savefig(f'/app/processing_{eid}.png', dpi=140); plt.close(fig)


_WORKER_ONE = None

def _worker_init():
    global _WORKER_ONE
    _WORKER_ONE = init_one()

def _worker_run(payload):
    eid, records = payload
    try:
        fr = pd.DataFrame.from_records(records)
        return ('ok', eid, process_session(_WORKER_ONE, fr, eid))
    except Exception as ex:
        return ('error', eid, f'{type(ex).__name__}: {ex}', traceback.format_exc(limit=3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode = ap.add_mutually_exclusive_group(); mode.add_argument('--full', action='store_true'); mode.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--workers', type=int, default=4, help='full-mode session workers (default 4)')
    args = ap.parse_args()
    sample = args.sample
    freeze, eids = cohort(sample)
    print(f'mode={"sample" if sample else "full"}; candidate sessions={len(eids)}, probes={len(freeze)}', flush=True)
    one = init_one()
    neural_s=[]; input_s=[]; output_s=[]; region_names_s=[]; infos=[]; excluded=[]
    t0=time.time()
    def accept(result, i):
        status, eid, *rest = result
        print(f'[{i}/{len(eids)}] completed {eid}', flush=True)
        if status == 'error':
            err, tb = rest
            excluded.append({'eid':eid,'error':err})
            print(f'  EXCLUDED {err}\n{tb}', flush=True)
            return
        n,x,y,r,info,speed,whisk = rest[0]
        neural_s.append(n); input_s.append(x); output_s.append(y); region_names_s.append(r); infos.append(info)
        if args.show_processing and len(infos)<=2: processing_plot(eid,n,y,speed,whisk)

    if sample or args.workers <= 1:
        for i,eid in enumerate(eids,1):
            print(f'[{i}/{len(eids)}] {eid}', flush=True)
            try:
                result=('ok',eid,process_session(one,freeze[freeze.eid.astype(str)==eid],eid))
            except Exception as ex:
                result=('error',eid,f'{type(ex).__name__}: {ex}',traceback.format_exc(limit=3))
            accept(result,i)
    else:
        payloads=[(eid,freeze[freeze.eid.astype(str)==eid].to_dict('records')) for eid in eids]
        print(f'parallel full conversion with {args.workers} workers',flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
            for i,result in enumerate(pool.map(_worker_run,payloads,chunksize=1),1):
                accept(result,i)
    if not neural_s: raise RuntimeError('no sessions converted')
    subjects=sorted({x['subject'] for x in infos}); subject_idx=np.array([subjects.index(x['subject']) for x in infos],dtype=np.int32)
    brain_regions=sorted(set(z for r in region_names_s for z in r)); lut={x:i for i,x in enumerate(brain_regions)}
    region_idx=[np.array([lut[z] for z in r],dtype=np.int32) for r in region_names_s]
    data={'neural':neural_s,'input':input_s,'output':output_s,'subjects':subjects,'subject_idx':subject_idx,
          'brain_regions':brain_regions,'brain_region_idx':region_idx,
          'input_names':['time_since_stimulus_onset','trial_number_in_block'],
          'output_names':['choice','prior_probability_left','wheel_speed_bin','whisker_motion_energy_bin'],
          'output_values':[['left','right'],['0.2','0.5','0.8'],['low','medium','high'],['low','medium','high']],
          'metadata':{'task_description':'IBL visual two-alternative choice task; decode choice, block prior, wheel speed and whisker motion energy from stimulus-aligned spikes.',
                      'time_bin_size':20.0,'temporal_alignment_event':'visual stimulus onset (trials.stimOn_times)',
                      'off_start':OFF_START,'off_end':OFF_END,'n_time_bins':100,
                      'neural_representation':'uint16 spike counts per 20 ms bin; well-isolated clusters label>=1',
                      'trial_filter':'finite required events; binary choice; valid prior; duration<=10 s; first movement latency 0.08-2.00 s; complete behavior windows',
                      'behavior_discretization':'per-session tertiles fitted on retained finite samples',
                      'session_info':infos,'excluded_sessions':excluded,'candidate_sessions':len(eids),
                      'source_release':'IBL Brainwidemap; code_zhang2025 bwm_release.csv',
                      'conversion_elapsed_s':time.time()-t0}}
    out=Path(args.outpicklefile); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(f'SAVED {out} size={out.stat().st_size/1e6:.1f} MB sessions={len(neural_s)} trials={sum(map(len,neural_s))} neurons={sum(x.shape[0] for x in region_idx)} excluded={len(excluded)} elapsed={time.time()-t0:.1f}s',flush=True)

if __name__=='__main__': main()
