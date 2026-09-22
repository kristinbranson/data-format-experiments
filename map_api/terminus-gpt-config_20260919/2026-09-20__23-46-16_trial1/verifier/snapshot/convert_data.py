#!/usr/bin/env python3
"""Convert MAP NWB sessions to the decoder-compatible pickle format.

All source data access uses pynwb. Neural rates are 50-ms non-overlapping bins
aligned to go cue over [-2.5, 1.5) seconds.
"""
from __future__ import annotations
import argparse, pickle, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO

OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
N_TIME = 80
VISIBILITY_THRESHOLD = 0.9


def as_strings(column):
    return np.asarray(column.data[:]).astype(str)


def choose_tone_onsets(trial_starts, go_times, sample_starts, sample_stops,
                       delay_starts, delay_stops):
    """Choose the auditory sample state that leads into the final pre-go delay.

    Early-lick trials can restart sample states several times. The trial stimulus is
    the final 0.65-s sample state nearest the delay state whose stop is the go cue.
    """
    out = np.empty(len(go_times), dtype=np.float64)
    for i, (start, go) in enumerate(zip(trial_starts, go_times)):
        didx = np.flatnonzero((delay_starts >= start) & (delay_starts <= go))
        sidx = np.flatnonzero((sample_starts >= start) & (sample_starts < go))
        if len(didx) == 0 or len(sidx) == 0:
            raise ValueError(f"trial {i} lacks sample or delay state before go")
        dk = didx[np.argmin(np.abs(delay_stops[didx] - go))]
        sk = sidx[np.argmin(np.abs(sample_stops[sidx] - delay_starts[dk]))]
        out[i] = sample_starts[sk]
    return out


def bin_spikes(spike_times, absolute_edges):
    """Vectorized-across-trials exact histogram for each neuron, returning Hz."""
    n_trials, n_edges = absolute_edges.shape
    flat_edges = absolute_edges.ravel()
    if np.any(np.diff(flat_edges) < 0):
        raise ValueError("trial windows overlap or are not chronological")
    rates = np.empty((n_trials, len(spike_times), n_edges - 1), dtype=np.float32)
    for j, spikes in enumerate(spike_times):
        spikes = np.asarray(spikes, dtype=np.float64)
        cumulative = np.searchsorted(spikes, flat_edges, side='left').reshape(n_trials, n_edges)
        rates[:, j, :] = np.diff(cumulative, axis=1) / BIN_S
    return rates


def nearest_indices(sorted_times, query):
    idx = np.searchsorted(sorted_times, query)
    idx = np.clip(idx, 1, len(sorted_times) - 1)
    left = idx - 1
    use_left = np.abs(query - sorted_times[left]) <= np.abs(sorted_times[idx] - query)
    return np.where(use_left, left, idx)


def trial_choice(instruction, outcome):
    if outcome == 'ignore':
        return 2
    if outcome == 'hit':
        return 0 if instruction == 'left' else 1
    if outcome == 'miss':
        return 1 if instruction == 'left' else 0
    raise ValueError(f"unknown outcome {outcome!r}")


def plot_session(session_id, rates, inputs, outputs, tongue_y, tongue_lk, q40, q60):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nshow = min(6, len(rates))
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    pop = np.stack([x.mean(axis=0) for x in rates[:nshow]])
    axes[0].imshow(pop, aspect='auto', extent=[OFF_START, OFF_END, nshow, 0])
    axes[0].set_ylabel('trial'); axes[0].set_title(f'{session_id}: population firing rate (Hz)')
    axes[1].plot(CENTERS_REL, inputs[0][0], label='time from tone'); axes[1].plot(CENTERS_REL, inputs[0][1], label='photostim')
    axes[1].legend(); axes[1].set_ylabel('input')
    axes[2].plot(CENTERS_REL, tongue_y[0], label='nearest-frame y'); axes[2].axhline(q40, ls='--'); axes[2].axhline(q60, ls='--')
    axes[2].legend(); axes[2].set_ylabel('tongue y')
    axes[3].plot(CENTERS_REL, tongue_lk[0]); axes[3].axhline(VISIBILITY_THRESHOLD, ls='--'); axes[3].set_ylabel('likelihood')
    axes[4].step(CENTERS_REL, outputs[0][3], where='mid'); axes[4].set_yticks([0,1,2,3]); axes[4].set_ylabel('tongue class'); axes[4].set_xlabel('time from go (s)')
    fig.tight_layout(); fig.savefig(f'/app/processing_{session_id}.png', dpi=140); plt.close(fig)


def process_file(path, make_plot=False):
    t0 = time.perf_counter()
    with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
        nwb = io.read(); trials = nwb.trials; units = nwb.units
        session_id = nwb.identifier
        classes = as_strings(units['classification'])
        neuron_idx = np.flatnonzero(classes == 'good')
        if len(neuron_idx) == 0:
            print(f'SKIP {session_id}: zero classifier-good units', flush=True)
            return None

        auto = as_strings(trials['auto_water']) == '1'
        free = as_strings(trials['free_water']) == '1'
        trial_keep = ~(auto | free)
        # A trial is retained only if every retained classifier-good unit is valid.
        # Short validity vectors correspond to a contiguous ephys recording block,
        # which can start late as well as end early; map them using obs_intervals.
        all_go_for_validity = np.asarray(
            nwb.acquisition['BehavioralEvents'].time_series['go_start_times'].timestamps[:],
            dtype=np.float64)
        for ui in neuron_idx:
            stored = np.asarray(units['is_good_trials'][ui], dtype=bool)
            if len(stored) > len(trials):
                raise ValueError(f'{session_id} unit {ui}: is_good_trials longer than trial table')
            if len(stored) == len(trials):
                valid = stored
            else:
                represented = np.zeros(len(trials), dtype=bool)
                obs = np.asarray(units['obs_intervals'][ui], dtype=np.float64).reshape(-1, 2)
                for a, b in obs:
                    represented |= (all_go_for_validity >= a) & (all_go_for_validity <= b)
                represented_idx = np.flatnonzero(represented)
                if len(represented_idx) != len(stored):
                    raise ValueError(
                        f'{session_id} unit {ui}: cannot map {len(stored)} validity flags '
                        f'to {len(represented_idx)} observation-covered trials')
                valid = np.zeros(len(trials), dtype=bool)
                valid[represented_idx] = stored
            trial_keep &= valid
        trial_idx = np.flatnonzero(trial_keep)
        if len(trial_idx) < 2:
            print(f'SKIP {session_id}: fewer than two valid standard trials', flush=True)
            return None

        ev = nwb.acquisition['BehavioralEvents'].time_series
        all_go = np.asarray(ev['go_start_times'].timestamps[:], dtype=np.float64)
        if len(all_go) != len(trials):
            raise ValueError(f'{session_id}: go/trial count mismatch')
        starts = np.asarray(trials['start_time'].data[:], dtype=np.float64)
        sample_starts = np.asarray(ev['sample_start_times'].timestamps[:], dtype=np.float64)
        sample_stops = np.asarray(ev['sample_stop_times'].timestamps[:], dtype=np.float64)
        delay_starts = np.asarray(ev['delay_start_times'].timestamps[:], dtype=np.float64)
        delay_stops = np.asarray(ev['delay_stop_times'].timestamps[:], dtype=np.float64)
        all_tone = choose_tone_onsets(starts, all_go, sample_starts, sample_stops,
                                      delay_starts, delay_stops)
        go = all_go[trial_idx]; tone = all_tone[trial_idx]
        abs_edges = go[:, None] + EDGES_REL[None, :]
        abs_centers = go[:, None] + CENTERS_REL[None, :]

        spikes = [np.asarray(units['spike_times'][int(i)], dtype=np.float64) for i in neuron_idx]
        rate_cube = bin_spikes(spikes, abs_edges)
        # A whole-session-neuron matrix of zeros is effectively outside usable ephys
        # coverage (127 such boundary/misaligned trials were detected in review).
        nonzero_neural = np.any(rate_cube != 0, axis=(1, 2))
        if not np.all(nonzero_neural):
            trial_idx = trial_idx[nonzero_neural]
            go = go[nonzero_neural]
            tone = tone[nonzero_neural]
            abs_edges = abs_edges[nonzero_neural]
            abs_centers = abs_centers[nonzero_neural]
            rate_cube = rate_cube[nonzero_neural]
        if len(trial_idx) < 2:
            print(f'SKIP {session_id}: fewer than two nonzero-neural trials', flush=True)
            return None
        neural = [np.ascontiguousarray(rate_cube[i]) for i in range(len(trial_idx))]

        # Inputs: continuous time since tone, binary photostimulation at bin center.
        tone_time = abs_centers - tone[:, None]
        stim = np.zeros((len(trial_idx), N_TIME), dtype=np.float32)
        ps = np.asarray(ev['photostim_start_times'].timestamps[:], dtype=np.float64)
        pe = np.asarray(ev['photostim_stop_times'].timestamps[:], dtype=np.float64)
        for k, src_i in enumerate(trial_idx):
            a, b = starts[src_i], float(trials['stop_time'][src_i])
            hits = np.flatnonzero((ps >= a) & (ps <= b))
            if len(hits) > 1:
                raise ValueError(f'{session_id} trial {src_i}: multiple photostim intervals')
            if len(hits) == 1:
                h = hits[0]
                stim[k] = ((abs_centers[k] >= ps[h]) & (abs_centers[k] < pe[h])).astype(np.float32)
        inputs = [np.stack((tone_time[k].astype(np.float32), stim[k])) for k in range(len(trial_idx))]

        # Tongue frame assignment and visible-session percentile thresholds.
        tongue = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
        video_t = np.asarray(tongue.timestamps[:], dtype=np.float64)
        video_d = np.asarray(tongue.data[:], dtype=np.float64)
        visible_session = video_d[:, 2] >= VISIBILITY_THRESHOLD
        if not np.any(visible_session):
            raise ValueError(f'{session_id}: no visible tongue frames')
        q40, q60 = np.percentile(video_d[visible_session, 1], [40, 60])
        frame_idx = nearest_indices(video_t, abs_centers.ravel()).reshape(len(trial_idx), N_TIME)
        ty = video_d[frame_idx, 1]; tl = video_d[frame_idx, 2]
        tongue_class = np.full((len(trial_idx), N_TIME), 3, dtype=np.int64)
        vis = tl >= VISIBILITY_THRESHOLD
        tongue_class[vis & (ty < q40)] = 0
        tongue_class[vis & (ty >= q40) & (ty <= q60)] = 1
        tongue_class[vis & (ty > q60)] = 2

        instruction = as_strings(trials['trial_instruction'])
        outcome = as_strings(trials['outcome'])
        early = as_strings(trials['early_lick'])
        omap = {'ignore': 0, 'miss': 1, 'hit': 2}
        outputs = []
        for k, src_i in enumerate(trial_idx):
            arr = np.empty((4, N_TIME), dtype=np.int64)
            arr[0] = trial_choice(instruction[src_i], outcome[src_i])
            arr[1] = omap[outcome[src_i]]
            arr[2] = 0 if early[src_i] == 'no early' else 1
            arr[3] = tongue_class[k]
            outputs.append(arr)

        regions = as_strings(units['anno_name'])[neuron_idx].tolist()
        if any(x in ('', 'nan') for x in regions):
            raise ValueError(f'{session_id}: classifier-good unit lacks anatomy')
        if make_plot:
            plot_session(session_id, neural, inputs, outputs, ty, tl, q40, q60)
        elapsed = time.perf_counter() - t0
        print(f'{session_id}: {len(trial_idx)}/{len(trials)} trials, {len(neuron_idx)} units, '
              f'visible={visible_session.mean():.3f}, q=({q40:.2f},{q60:.2f}), {elapsed:.2f}s', flush=True)
        return dict(neural=neural, input=inputs, output=outputs,
                    subject=str(nwb.subject.subject_id), regions=regions,
                    session_id=session_id, source_file=str(path), source_trial_idx=trial_idx.tolist(),
                    q40=float(q40), q60=float(q60), elapsed=elapsed)


def validate(data):
    ns = len(data['neural'])
    assert ns == len(data['input']) == len(data['output']) == len(data['subject_idx']) == len(data['brain_region_idx'])
    for s in range(ns):
        assert len(data['neural'][s]) >= 2
        assert len(data['neural'][s]) == len(data['input'][s]) == len(data['output'][s])
        nn = len(data['brain_region_idx'][s])
        for n, x, y in zip(data['neural'][s], data['input'][s], data['output'][s]):
            assert n.shape == (nn, N_TIME) and n.dtype == np.float32 and np.isfinite(n).all()
            assert x.shape == (2, N_TIME) and x.dtype == np.float32 and np.isfinite(x).all()
            assert y.shape == (4, N_TIME) and np.issubdtype(y.dtype, np.integer)
            assert set(np.unique(y[0])).issubset({0,1,2})
            assert set(np.unique(y[1])).issubset({0,1,2})
            assert set(np.unique(y[2])).issubset({0,1})
            assert set(np.unique(y[3])).issubset({0,1,2,3})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='process all sessions (default)')
    mode.add_argument('--sample', action='store_true', help='process first two usable sessions')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()
    files = sorted(Path('/app/data').rglob('*.nwb'))
    if not files: raise FileNotFoundError('no NWB files under /app/data')
    target = 2 if args.sample else None
    results=[]; wall=time.perf_counter()
    for path in files:
        r=process_file(path, make_plot=args.show_processing and len(results)<2)
        if r is not None: results.append(r)
        if target is not None and len(results)>=target: break
    if not results: raise RuntimeError('no usable sessions')
    subjects=sorted({r['subject'] for r in results}); subject_map={x:i for i,x in enumerate(subjects)}
    brain_regions=sorted({x for r in results for x in r['regions']}); region_map={x:i for i,x in enumerate(brain_regions)}
    data={
      'neural':[r['neural'] for r in results],
      'input':[r['input'] for r in results],
      'output':[r['output'] for r in results],
      'subjects':subjects,
      'subject_idx':np.asarray([subject_map[r['subject']] for r in results],dtype=np.int64),
      'brain_regions':brain_regions,
      'brain_region_idx':[np.asarray([region_map[x] for x in r['regions']],dtype=np.int64) for r in results],
      'input_names':['time from tone onset (s)','photostimulation on'],
      'output_names':['lick direction choice','outcome','early lick','tongue y-position'],
      'output_values':[['left','right','no lick'],['ignore','miss','hit'],['no','yes'],['below 40th percentile','40th to 60th percentile','above 60th percentile','not visible']],
      'metadata':{
        'task_description':'Head-fixed auditory delayed-response task; decode choice, outcome, early lick, and discretized tongue y-position from go-aligned neural firing rates.',
        'time_bin_size':50.0,
        'temporal_alignment_event':'go cue onset (BehavioralEvents/go_start_times)',
        'off_start':OFF_START,'off_end':OFF_END,
        'interval_convention':'[-2.5, 1.5) s; values at 50-ms bin centers',
        'neural_unit':'spikes/s (Hz)',
        'unit_filter':"NWB units classification == 'good' (published region-specific classifier QC)",
        'trial_filter':'exclude auto-water, free-water, and any trial invalid for any retained unit; retain early-lick, ignore, miss, hit, and photostimulation trials',
        'tongue_visibility_threshold':VISIBILITY_THRESHOLD,
        'tongue_percentiles':'per session over all Camera0 tongue frames with likelihood >= 0.9',
        'session_info':[{k:r[k] for k in ['session_id','source_file','source_trial_idx','q40','q60','elapsed']} for r in results],
      }
    }
    validate(data)
    print(f'Validated {len(results)} sessions, {sum(map(len,data["neural"]))} trials, '
          f'{sum(len(x) for x in data["brain_region_idx"])} session-neurons',flush=True)
    out=Path(args.outpicklefile); out.parent.mkdir(parents=True,exist_ok=True)
    t=time.perf_counter()
    with out.open('wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {out} ({out.stat().st_size/2**30:.3f} GiB) in {time.perf_counter()-t:.2f}s; total {time.perf_counter()-wall:.2f}s',flush=True)

if __name__=='__main__': main()
