#!/usr/bin/env python3
"""Convert the MAP NWB release to the decoder's session/trial representation.

Decisions follow the supplied paper and preprocessing repository:
* classifier-labelled good units are used; as in preprocessing_DJ_2022Aug.py,
  units must also have histology (a nonempty CCF annotation);
* spikes are aligned to go-cue onset and represented as firing rates;
* bins are nonoverlapping, left-closed/right-open 50-ms intervals;
* all released behavioral sessions and trials are retained;
* DLC tongue points with likelihood < .9 are treated as not visible.
"""
from pathlib import Path
import json, pickle, gc
import numpy as np
from pynwb import NWBHDF5IO

DATA_ROOT = Path('/app/data')
OUT = Path('/app/converted_data.pkl')
START, END, DT = -2.5, 1.5, 0.05
EDGES = np.linspace(START, END, 81, dtype=np.float64)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
DLC_THRESHOLD = 0.9


def first_event_in_trial(times, starts, stops):
    """First event in each NWB trial, NaN if absent."""
    times = np.asarray(times, dtype=np.float64)
    ans = np.full(len(starts), np.nan)
    j = 0
    for i, (a, b) in enumerate(zip(starts, stops)):
        j = np.searchsorted(times, a, side='left')
        if j < len(times) and times[j] < b:
            ans[i] = times[j]
    return ans


def bin_spikes(spike_times, go):
    # Histogram absolute spike times with the same [left,right) convention as
    # the reference sliding_histogram, then convert counts to Hz.
    out = np.empty((len(spike_times), 80), dtype=np.float32)
    abs_edges = go + EDGES
    for u, st in enumerate(spike_times):
        st = np.asarray(st)
        # NWB spike trains are sorted. Edge insertion indices give the same
        # [left,right) counts as the reference code without rescanning the
        # full session spike train once per bin.
        out[u] = np.diff(np.searchsorted(st, abs_edges, side='left')) / DT
    return out


def main():
    files = sorted(DATA_ROOT.rglob('*.nwb'))
    subjects = sorted({p.name.split('_')[0].replace('sub-', '') for p in files})
    subject_to_idx = {s:i for i,s in enumerate(subjects)}
    neural, inputs, outputs = [], [], []
    subject_idx, region_names_by_session, session_info = [], [], []

    for si, path in enumerate(files):
        print(f'[{si+1}/{len(files)}] {path.name}', flush=True)
        with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
            nwb = io.read()
            sid = str(nwb.subject.subject_id)
            tr = nwb.trials
            starts = np.asarray(tr['start_time'][:], float)
            stops = np.asarray(tr['stop_time'][:], float)
            ntr = len(starts)

            # One go onset is expected in every retained trial. Assign by trial
            # interval rather than relying on array position.
            events = nwb.acquisition['BehavioralEvents'].time_series
            go = first_event_in_trial(events['go_start_times'].timestamps[:], starts, stops)
            if np.any(~np.isfinite(go)):
                raise ValueError(f'{path}: {np.sum(~np.isfinite(go))} trials lack go cue')
            tone = first_event_in_trial(events['sample_start_times'].timestamps[:], starts, stops)
            # Audio-delay task sessions should all contain a sample/tone onset.
            if np.any(~np.isfinite(tone)):
                raise ValueError(f'{path}: {np.sum(~np.isfinite(tone))} trials lack tone onset')

            quality = np.asarray(nwb.units['unit_quality'][:], dtype=str)
            annotation = np.char.strip(np.asarray(nwb.units['anno_name'][:], dtype=str))
            keep = (quality == 'good') & (np.char.str_len(annotation) > 0) & (np.char.lower(annotation) != 'nan')
            kept_idx = np.flatnonzero(keep)
            if len(kept_idx) == 0:
                print('  skipped: no good annotated units', flush=True)
                continue
            regions = annotation[keep].tolist()
            # Bulk-load NWB's ragged spike vector and offsets once.
            spike_col = nwb.units['spike_times']
            spike_data = np.asarray(spike_col.target.data[:], dtype=np.float64)
            spike_ends = np.asarray(spike_col.data[:], dtype=np.int64)
            spike_starts = np.r_[0, spike_ends[:-1]]
            spikes = [spike_data[spike_starts[i]:spike_ends[i]] for i in kept_idx]

            # Photostimulation is represented continuously from the recorded
            # onset to offset, sampled at each neural-bin center.
            ps = np.asarray(events['photostim_start_times'].timestamps[:], float)
            pe = np.asarray(events['photostim_stop_times'].timestamps[:], float)
            if len(ps) != len(pe):
                raise ValueError(f'{path}: unequal photostim onset/offset counts')

            # Session-wide DLC thresholds use only confidently visible tongue
            # samples. Each neural bin receives the nearest camera sample.
            tts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
            cam_t = np.asarray(tts.timestamps[:], dtype=float)
            cam = np.asarray(tts.data[:], dtype=float)
            visible_all = np.isfinite(cam[:,1]) & np.isfinite(cam[:,2]) & (cam[:,2] >= DLC_THRESHOLD)
            if not np.any(visible_all):
                q40 = q60 = np.nan
            else:
                q40, q60 = np.percentile(cam[visible_all,1], [40, 60])

            instruction = np.asarray(tr['trial_instruction'][:], dtype=str)
            outcome = np.asarray(tr['outcome'][:], dtype=str)
            early = np.asarray(tr['early_lick'][:], dtype=str)
            out_map = {'ignore':0, 'miss':1, 'hit':2}

            sess_n, sess_i, sess_o = [], [], []
            for ti in range(ntr):
                bt = go[ti] + CENTERS
                sess_n.append(bin_spikes(spikes, go[ti]))

                time_from_tone = (bt - tone[ti]).astype(np.float32)
                # Interval membership, [on,off), across all stim epochs.
                photo = np.zeros(80, dtype=np.float32)
                if len(ps):
                    photo[:] = np.any((bt[:,None] >= ps[None,:]) & (bt[:,None] < pe[None,:]), axis=1)
                sess_i.append(np.vstack((time_from_tone, photo)).astype(np.float32, copy=False))

                if outcome[ti] == 'ignore':
                    choice = 2                         # no lick/no response
                elif outcome[ti] == 'hit':
                    choice = 0 if instruction[ti] == 'left' else 1
                else:                                  # miss: wrong direction
                    choice = 1 if instruction[ti] == 'left' else 0

                idx = np.searchsorted(cam_t, bt)
                idx = np.clip(idx, 1, len(cam_t)-1)
                prev = idx - 1
                idx = np.where(np.abs(cam_t[prev]-bt) <= np.abs(cam_t[idx]-bt), prev, idx)
                y = cam[idx,1]
                vis = np.isfinite(y) & np.isfinite(cam[idx,2]) & (cam[idx,2] >= DLC_THRESHOLD)
                tongue_cat = np.full(80, 3, dtype=np.int8)
                tongue_cat[vis & (y < q40)] = 0
                tongue_cat[vis & (y >= q40) & (y <= q60)] = 1
                tongue_cat[vis & (y > q60)] = 2

                # First three outputs are per-trial; tongue position is
                # time-varying. To keep a rectangular categorical output, the
                # trial labels are repeated over time, as permitted by format.
                o = np.empty((4,80), dtype=np.int8)
                o[0] = choice
                o[1] = out_map[outcome[ti]]
                o[2] = 1 if early[ti] == 'early' else 0
                o[3] = tongue_cat
                sess_o.append(o)

            neural.append(sess_n); inputs.append(sess_i); outputs.append(sess_o)
            subject_idx.append(subject_to_idx[sid]); region_names_by_session.append(regions)
            session_info.append({'file': path.name, 'subject': sid, 'n_trials': ntr,
                                 'n_neurons': len(kept_idx), 'tongue_q40': float(q40),
                                 'tongue_q60': float(q60), 'dlc_likelihood_threshold': DLC_THRESHOLD})
        gc.collect()

    brain_regions = sorted(set(r for rr in region_names_by_session for r in rr))
    rmap = {r:i for i,r in enumerate(brain_regions)}
    brain_region_idx = [np.asarray([rmap[r] for r in rr], dtype=np.int32) for rr in region_names_by_session]
    data = {
        'neural': neural, 'input': inputs, 'output': outputs,
        'subjects': subjects, 'subject_idx': np.asarray(subject_idx, dtype=np.int32),
        'brain_regions': brain_regions, 'brain_region_idx': brain_region_idx,
        'input_names': ['time from tone onset', 'photostimulation on'],
        'output_names': ['lick direction choice', 'outcome', 'early lick', 'tongue y-position'],
        'output_values': [['left','right','no lick'], ['ignore','miss','hit'],
                          ['no','yes'], ['below 40th percentile','40th to 60th percentile',
                                         'above 60th percentile','not visible']],
        'metadata': {
            'task_description': 'Memory-guided auditory discrimination; decode choice, outcome, early licking, and tongue y-position from brain-wide activity.',
            'time_bin_size': 50.0,
            'temporal_alignment_event': 'go cue onset',
            'off_start': START, 'off_end': END,
            'neural_measure': 'firing rate (Hz)',
            'bin_convention': '80 nonoverlapping [left,right) bins; centers -2.475 to 1.475 s',
            'unit_filter': "unit_quality == 'good' and valid nonempty histological anno_name",
            'trial_filter': 'all trials with go cue and tone onset',
            'tongue_visibility': 'DLC likelihood >= 0.9; nearest camera sample to bin center',
            'session_info': session_info,
            'source': 'MAP dandiset: Brain-wide neural activity underlying memory-guided movement'
        }
    }
    print(f'Writing {OUT} ({len(neural)} sessions, {sum(map(len, neural))} trials, {sum(x[0].shape[0] for x in neural)} session-neurons)', flush=True)
    with open(OUT, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done', flush=True)

if __name__ == '__main__':
    main()
