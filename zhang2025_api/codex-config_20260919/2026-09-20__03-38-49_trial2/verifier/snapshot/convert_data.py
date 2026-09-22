#!/usr/bin/env python3
"""Convert the IBL BWM ONE cache to the decoder pickle format."""
from __future__ import annotations

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from iblatlas.atlas import BrainRegions
from one.api import ONE

CACHE = Path('/app/data/one_cache')
RELEASE = Path('/app/code/code_zhang2025/data/bwm_release.csv')
SUBSET = Path('/app/data/DATALIMIT_SUBSET.csv')
BIN = 0.02
OFF_START, OFF_END = -0.5, 1.5
N_TIME = 100
TIME = np.arange(1, N_TIME + 1, dtype=np.float32) * BIN + OFF_START
NEEDED = {'stimOn_times', 'choice', 'probabilityLeft', 'firstMovement_times',
          'feedback_times', 'feedbackType', 'goCue_times'}


def get_one():
    """Construct a local ONE client over the aggregate release tables."""
    return ONE(cache_dir=CACHE, tables_dir=CACHE / 'Brainwidemap', mode='local')


def release_rows():
    rows = pd.read_csv(RELEASE).drop(columns=['Unnamed: 0'], errors='ignore')
    if SUBSET.exists():
        subset = pd.read_csv(SUBSET)
        key = next((x for x in ('eid', 'session', 'session_id') if x in subset), None)
        if key:
            rows = rows[rows.eid.astype(str).isin(subset[key].astype(str))]
    return rows


def trial_number_in_block(prob):
    out = np.zeros(len(prob), dtype=np.float32)
    for i in range(1, len(prob)):
        out[i] = out[i - 1] + 1 if prob[i] == prob[i - 1] else 0
    return out


def load_trials(one, eid):
    """Load trials through brainbox SessionLoader and construct reference QC mask."""
    sl = SessionLoader(one=one, eid=eid)
    sl.load_trials()
    tr = sl.trials.copy()
    missing = NEEDED - set(tr.columns)
    if missing:
        raise RuntimeError(f'trials object missing columns {sorted(missing)}')
    block_number = trial_number_in_block(tr.probabilityLeft.to_numpy())
    finite = np.ones(len(tr), dtype=bool)
    for col in ['stimOn_times', 'choice', 'probabilityLeft', 'firstMovement_times',
                'feedback_times', 'feedbackType']:
        finite &= np.isfinite(tr[col].to_numpy(dtype=float))
    rt = tr.firstMovement_times.to_numpy() - tr.stimOn_times.to_numpy()
    duration = tr.feedback_times.to_numpy() - tr.goCue_times.to_numpy()
    mask = finite & (rt >= .08) & (rt <= 2.) & (duration <= 10.) & (tr.choice.to_numpy() != 0)
    return sl, tr, mask, block_number


def load_behavior(sl):
    sl.load_wheel()
    wt = sl.wheel.times.to_numpy(dtype=float)
    ws = np.abs(sl.wheel.velocity.to_numpy(dtype=float))
    last_error = None
    for view in ('left', 'right'):
        try:
            sl.load_motion_energy(views=[view])
            df = sl.motion_energy[f'{view}Camera']
            return wt, ws, df.times.to_numpy(dtype=float), df.whiskerMotionEnergy.to_numpy(dtype=float), view
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f'no whisker motion-energy view: {last_error}')


def interpolate_trials(times, values, onsets):
    query = onsets[:, None] + TIME[None, :]
    good = ((query[:, 0] >= times[0]) & (query[:, -1] <= times[-1]))
    out = np.full(query.shape, np.nan, dtype=np.float32)
    if np.any(good):
        out[good] = np.interp(query[good].ravel(), times, values).reshape((-1, N_TIME))
    return out, good & np.isfinite(out).all(axis=1)


def load_spikes(one, probe_rows):
    all_times, all_clusters, all_regions = [], [], []
    offset = 0
    for row in probe_rows.itertuples():
        ssl = SpikeSortingLoader(pid=str(row.pid), eid=str(row.eid), pname=row.probe_name, one=one)
        spikes, clusters, channels = ssl.load_spike_sorting()
        clusters = ssl.merge_clusters(spikes, clusters, channels, compute_metrics=False)
        n = len(clusters['channels'])
        all_times.append(np.asarray(spikes['times'], dtype=float))
        all_clusters.append(np.asarray(spikes['clusters'], dtype=np.int64) + offset)
        all_regions.extend(np.asarray(clusters['acronym']).astype(str).tolist())
        offset += n
    if not all_times:
        raise RuntimeError('no probes')
    times = np.concatenate(all_times); clu = np.concatenate(all_clusters)
    order = np.argsort(times, kind='stable')
    return times[order], clu[order], np.asarray(all_regions, dtype=str)


def bin_spikes(times, clusters, onsets, n_neurons):
    result = []
    for onset in onsets:
        beg, end = onset + OFF_START, onset + OFF_END
        lo, hi = np.searchsorted(times, [beg, end])
        relbin = np.floor((times[lo:hi] - beg) / BIN).astype(np.int64)
        ok = (relbin >= 0) & (relbin < N_TIME)
        flat = clusters[lo:hi][ok] * N_TIME + relbin[ok]
        counts = np.bincount(flat, minlength=n_neurons * N_TIME).reshape(n_neurons, N_TIME)
        result.append(counts.astype(np.float32))
    return result


def tertiles(x):
    edges = np.quantile(x, [1 / 3, 2 / 3])
    if edges[0] < edges[1]:
        return np.digitize(x, edges, right=False).astype(np.int64), edges
    # Deterministic rank fallback for degenerate signals.
    order = np.argsort(x.ravel(), kind='stable'); labels = np.empty(order.size, dtype=np.int64)
    labels[order] = np.minimum(2, np.arange(order.size) * 3 // order.size)
    return labels.reshape(x.shape), edges


def process_session(one, eid, rows, show=False):
    t0 = time.perf_counter()
    sl, tr, mask, block_no = load_trials(one, eid)
    wt, ws, mt, me, view = load_behavior(sl)
    onsets = tr.stimOn_times.to_numpy(dtype=float)
    wheel, wheel_good = interpolate_trials(wt, ws, onsets)
    whisk, whisk_good = interpolate_trials(mt, me, onsets)
    valid = mask & wheel_good & whisk_good
    idx = np.flatnonzero(valid)
    if idx.size < 2:
        raise RuntimeError(f'only {idx.size} jointly valid trials')
    st, sc, regions = load_spikes(one, rows)
    neural = bin_spikes(st, sc, onsets[idx], len(regions))
    wheel_cat, wheel_edges = tertiles(wheel[idx])
    whisk_cat, whisk_edges = tertiles(whisk[idx])
    inputs, outputs = [], []
    prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
    for j, raw_i in enumerate(idx):
        inp = np.vstack([TIME, np.full(N_TIME, block_no[raw_i], dtype=np.float32)])
        choice = 0 if tr.choice.iloc[raw_i] == -1 else 1
        prior = prior_map[round(float(tr.probabilityLeft.iloc[raw_i]), 1)]
        out = np.vstack([np.full(N_TIME, choice), np.full(N_TIME, prior),
                         wheel_cat[j], whisk_cat[j]]).astype(np.int64)
        inputs.append(inp); outputs.append(out)
    if show:
        fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        ax[0].imshow(np.stack(neural[:min(20, len(neural))]).sum(1), aspect='auto', extent=[-.5, 1.5, min(20,len(neural)),0]); ax[0].set_ylabel('trial'); ax[0].set_title('population spike counts')
        ax[1].plot(TIME, wheel[idx[0]], label='interpolated'); ax[1].step(TIME, wheel_cat[0], where='mid', label='tertile'); ax[1].legend(); ax[1].set_ylabel('wheel')
        ax[2].plot(TIME, whisk[idx[0]], label=f'{view} whisker ME'); ax[2].step(TIME, whisk_cat[0], where='mid', label='tertile'); ax[2].legend(); ax[2].set_ylabel('motion')
        ax[3].plot(TIME, inputs[0][0]); ax[3].axvline(0, color='r'); ax[3].set_ylabel('time input'); ax[3].set_xlabel('seconds from stimulus onset')
        fig.tight_layout(); fig.savefig(f'/app/processing_{eid}.png', dpi=140); plt.close(fig)
    elapsed = time.perf_counter() - t0
    info = dict(eid=str(eid), subject=str(rows.subject.iloc[0]), n_trials=len(idx),
                n_neurons=len(regions), raw_trials=len(tr), whisker_view=view,
                wheel_edges=wheel_edges.tolist(), whisker_edges=whisk_edges.tolist(), seconds=elapsed)
    return neural, inputs, outputs, regions, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='process all sessions (default)')
    mode.add_argument('--sample', action='store_true', help='retain the first two processable sessions')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()
    rows = release_rows(); one = get_one()
    grouped = list(rows.groupby('eid', sort=False))
    target = 2 if args.sample else None
    neural_all, input_all, output_all, region_names_all, infos, failures = [], [], [], [], [], []
    for k, (eid, erows) in enumerate(grouped, 1):
        try:
            vals = process_session(one, str(eid), erows, args.show_processing and len(infos) < 2)
            neural, inp, out, regs, info = vals
            neural_all.append(neural); input_all.append(inp); output_all.append(out)
            region_names_all.append(regs); infos.append(info)
            print(f"[{k}/{len(grouped)}] {eid}: {info['n_trials']} trials, {info['n_neurons']} neurons, {info['seconds']:.2f}s", flush=True)
            if target and len(infos) >= target:
                break
        except Exception as exc:
            failures.append((str(eid), f'{type(exc).__name__}: {exc}'))
            print(f'[{k}/{len(grouped)}] SKIP {eid}: {failures[-1][1]}', flush=True)
    if not infos:
        raise RuntimeError(f'no sessions converted; first failures: {failures[:5]}')
    br = BrainRegions()
    mapped = [br.acronym2acronym(x, mapping='Beryl') for x in region_names_all]
    vocabulary = sorted(set(np.concatenate(mapped).astype(str)))
    lookup = {r: i for i, r in enumerate(vocabulary)}
    region_idx = [np.asarray([lookup[str(x)] for x in sess], dtype=np.int64) for sess in mapped]
    subjects = sorted({x['subject'] for x in infos}); subject_lookup = {s:i for i,s in enumerate(subjects)}
    data = {
        'neural': neural_all, 'input': input_all, 'output': output_all,
        'subjects': subjects,
        'subject_idx': np.asarray([subject_lookup[x['subject']] for x in infos], dtype=np.int64),
        'brain_regions': vocabulary, 'brain_region_idx': region_idx,
        'input_names': ['time_since_stimulus_onset', 'trial_number_in_block'],
        'output_names': ['choice', 'prior_probability_left', 'wheel_speed', 'whisker_motion_energy'],
        'output_values': [['left', 'right'], ['0.2', '0.5', '0.8'], ['low', 'medium', 'high'], ['low', 'medium', 'high']],
        'metadata': {
            'task_description': 'IBL visual decision task; predict choice, block prior, wheel speed tertile, and whisker motion-energy tertile from stimulus-aligned spikes.',
            'time_bin_size': 20.0, 'temporal_alignment_event': 'visual stimulus onset (stimOn_times)',
            'off_start': OFF_START, 'off_end': OFF_END, 'bin_coordinate': 'right edge',
            'neural_measure': 'spike counts per 20-ms bin', 'cluster_filter': 'all spike-sorted clusters, matching Zhang et al. caching code',
            'trial_filter': 'finite required events; 0.08<=firstMovement-stimOn<=2 s; choice!=0; goCue-to-feedback<=10 s; full wheel/whisker coverage',
            'dynamic_discretization': 'within-session tertiles over all retained trial-time samples',
            'session_info': infos, 'failed_sessions': failures,
            'source_release_candidates': len(grouped), 'source_release_probes': len(rows),
        }
    }
    with open(args.outpicklefile, 'wb') as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {args.outpicklefile}: {len(infos)} sessions; {sum(x["n_trials"] for x in infos)} trials; {sum(x["n_neurons"] for x in infos)} session-neurons; {len(failures)} skipped', flush=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='Multiple revisions')
    main()
