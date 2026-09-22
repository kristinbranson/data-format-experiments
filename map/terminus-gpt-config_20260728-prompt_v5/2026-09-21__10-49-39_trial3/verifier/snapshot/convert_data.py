#!/usr/bin/env python3
import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt


BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
BIN_EDGES = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2


TRIAL_CANDIDATES = {
    'go_cue': ['go_cue_start_time', 'goCue_start_time', 'go_start_time', 'goCue_times', 'go_cue_time'],
    'early_lick': ['early_lick', 'early_licks', 'early_lick_flag'],
    'outcome': ['outcome', 'trial_outcome', 'result'],
    'choice': ['choice', 'lick_flag', 'trial_instruction', 'response_side'],
    'photostim': ['photostim', 'photostimulation', 'laser_on_trial'],
}


def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def decode_arr(arr):
    if arr is None:
        return None
    if hasattr(arr, 'to_numpy'):
        arr = arr.to_numpy()
    arr = np.asarray(arr)
    if arr.dtype.kind == 'S':
        return np.array([x.decode('utf-8') for x in arr], dtype=object)
    return arr


def safe_float(x):
    try:
        if isinstance(x, (bytes, np.bytes_)):
            x = x.decode('utf-8')
        s = str(x).strip()
        if s.lower() in {'n/a', 'na', 'none', 'nan', ''}:
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def get_table_dict(group):
    out = {}
    for k in group.keys():
        obj = group[k]
        if isinstance(obj, h5py.Dataset):
            try:
                out[k] = obj[()]
            except Exception:
                pass
    return out


def load_trial_table(f):
    g = f['intervals/trials']
    table = {k: decode_arr(g[k][()]) for k in g.keys() if isinstance(g[k], h5py.Dataset)}
    return table


def load_units_table(f):
    g = f['units']
    table = {k: decode_arr(g[k][()]) for k in g.keys() if isinstance(g[k], h5py.Dataset)}
    return table


def choose_good_units(units):
    n = len(units['id'])
    # Prefer explicit curated labels when available
    for key in ['classification', 'unit_quality', 'quality', 'label', 'cluster_quality']:
        if key in units:
            vals = units[key]
            sval = np.array([str(v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else v).strip().lower() for v in vals], dtype=object)
            good_words = {'good', 'single', 'single_unit', 'single unit'}
            if np.isin(sval, list(good_words)).any():
                return np.isin(sval, list(good_words))
    # Fallback to boolean flags if present
    for key in ['good', 'is_good', 'is_good_trials']:
        if key in units:
            vals = np.asarray(units[key])
            if vals.dtype.kind in 'biu' or set(np.unique(vals)).issubset({0, 1}):
                return vals.astype(bool)
    # Metric-based fallback approximating standard QC thresholds
    mask = np.ones(n, dtype=bool)
    if 'presence_ratio' in units:
        mask &= np.asarray(units['presence_ratio'], dtype=float) >= 0.9
    if 'amplitude_cutoff' in units:
        mask &= np.asarray(units['amplitude_cutoff'], dtype=float) <= 0.1
    if 'isi_violation' in units:
        mask &= np.asarray(units['isi_violation'], dtype=float) <= 0.5
    elif 'isi_violations' in units:
        mask &= np.asarray(units['isi_violations'], dtype=float) <= 0.5
    if 'nn_hit_rate' in units:
        mask &= np.asarray(units['nn_hit_rate'], dtype=float) >= 0.9
    if mask.sum() == 0:
        mask = np.ones(n, dtype=bool)
    return mask


def get_brain_regions(units, n_units, f=None):
    if f is not None and 'general/extracellular_ephys/electrodes/location' in f and 'electrodes' in units:
        import json
        eloc = f['general/extracellular_ephys/electrodes/location'][()]
        eidx = np.asarray(units['electrodes']).astype(int)
        out = []
        for idx in eidx:
            try:
                raw = eloc[idx]
                s = raw.decode('utf-8') if isinstance(raw, (bytes, np.bytes_)) else str(raw)
                d = json.loads(s)
                out.append(d.get('brain_regions', 'unknown'))
            except Exception:
                out.append('unknown')
        vals = np.array(out, dtype=object)
        if np.any(vals != 'unknown'):
            return vals
    for key in ['location', 'brain_region', 'region', 'structure_acronym', 'ccf_acronym', 'anno_name']:
        if key in units:
            vals = np.array([str(v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else v) for v in units[key]], dtype=object)
            if np.any(vals != 'unknown'):
                return vals
    return np.array(['unknown'] * n_units, dtype=object)


def get_spike_times_for_unit(units_group, idx):
    st = units_group['spike_times']
    if 'spike_times_index' in units_group:
        ind = units_group['spike_times_index'][:]
        end = ind[idx]
        start = 0 if idx == 0 else ind[idx - 1]
        return st[start:end]
    return np.asarray(st[idx])


def infer_go_cue_times(table, f=None):
    if f is not None and 'acquisition/BehavioralEvents/go_start_times/timestamps' in f:
        return np.asarray(f['acquisition/BehavioralEvents/go_start_times/timestamps'][:], dtype=float)
    cols = set(table.keys())
    key = first_existing(cols, TRIAL_CANDIDATES['go_cue'])
    if key is not None:
        return np.asarray(table[key], dtype=float)
    start = np.asarray(table['start_time'], dtype=float)
    stop = np.asarray(table['stop_time'], dtype=float) if 'stop_time' in table else start + 4.0
    return start + np.minimum(2.0, (stop - start) / 2)


def infer_choice_outcome_early(table, left_licks, right_licks, go_times):
    n = len(go_times)
    choice = np.zeros(n, dtype=np.int64)
    outcome = np.zeros(n, dtype=np.int64)
    early = np.zeros(n, dtype=np.int64)
    start = np.asarray(table['start_time'], dtype=float)
    stop = np.asarray(table['stop_time'], dtype=float) if 'stop_time' in table else go_times + 3.0

    if 'early_lick' in table:
        ev = np.asarray(table['early_lick'])
        early = np.array([
            1 if str(v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else v).strip().lower() == 'early' else 0
            for v in ev
        ], dtype=np.int64)

    left_licks = np.asarray(left_licks)
    right_licks = np.asarray(right_licks)
    for i in range(n):
        l_post = np.any((left_licks >= go_times[i]) & (left_licks < min(stop[i], go_times[i] + 1.5)))
        r_post = np.any((right_licks >= go_times[i]) & (right_licks < min(stop[i], go_times[i] + 1.5)))
        if l_post and not r_post:
            choice[i] = 0
        elif r_post and not l_post:
            choice[i] = 1
        else:
            choice[i] = 2

    if 'outcome' in table:
        ov = np.array([str(v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else v).strip().lower() for v in table['outcome']], dtype=object)
        for i, s in enumerate(ov):
            if s == 'hit' or 'correct' in s:
                outcome[i] = 2
            elif s == 'miss' or 'error' in s or 'incorrect' in s:
                outcome[i] = 1
            elif s == 'ignore' or 'no' in s:
                outcome[i] = 0
            else:
                outcome[i] = 0 if choice[i] == 2 else 2
    else:
        outcome[:] = np.where(choice == 2, 0, 2)
    return choice, outcome, early

def build_photostim_vector(starts, stops, go_time):
    x = np.zeros(N_BINS, dtype=np.float32)
    for s, e in zip(starts, stops):
        rs = s - go_time
        re = e - go_time
        on = (BIN_CENTERS >= rs) & (BIN_CENTERS < re)
        x[on] = 1.0
    return x


def build_time_from_tone_vector(sample_start, go_time):
    return (BIN_CENTERS - (sample_start - go_time)).astype(np.float32)[None, :]


def discretize_tongue_y(data_xy):
    y = np.asarray(data_xy[:, 1], dtype=float)
    visible = np.isfinite(y)
    if visible.sum() == 0:
        return np.full(len(y), 3, dtype=np.int64), (np.nan, np.nan)
    q40, q60 = np.nanpercentile(y[visible], [40, 60])
    out = np.full(len(y), 3, dtype=np.int64)
    out[visible & (y < q40)] = 0
    out[visible & (y >= q40) & (y <= q60)] = 1
    out[visible & (y > q60)] = 2
    return out, (q40, q60)


def bin_tongue_for_trial(timestamps, labels, go_time):
    out = np.full(N_BINS, 3, dtype=np.int64)
    rel = timestamps - go_time
    for b in range(N_BINS):
        m = (rel >= BIN_EDGES[b]) & (rel < BIN_EDGES[b + 1])
        if np.any(m):
            vals = labels[m]
            vals = vals[vals != 3]
            out[b] = 3 if len(vals) == 0 else np.bincount(vals, minlength=3).argmax()
    return out


def process_session(path, show_processing=False):
    t0 = time.time()
    with h5py.File(path, 'r') as f:
        trial_table = load_trial_table(f)
        units = load_units_table(f)
        good_mask = choose_good_units(units)
        region_names = get_brain_regions(units, len(units['id']), f)
        print(f'loading {path.name}: total_units={len(units["id"])} good_units={int(good_mask.sum())} trials={len(trial_table["start_time"])}', flush=True)
        go_times = infer_go_cue_times(trial_table, f)
        start_times = np.asarray(trial_table['start_time'], dtype=float)
        stop_times = np.asarray(trial_table['stop_time'], dtype=float) if 'stop_time' in trial_table else go_times + 3.0
        sample_starts = np.asarray(trial_table['start_time'], dtype=float)
        if 'acquisition/BehavioralEvents/sample_start_times/timestamps' in f:
            sample_event_times = np.asarray(f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:], dtype=float)
            if len(sample_event_times) == len(go_times):
                sample_starts = sample_event_times.copy()
            else:
                for i in range(len(go_times)):
                    cand = sample_event_times[(sample_event_times >= start_times[i]) & (sample_event_times <= stop_times[i])]
                    if len(cand):
                        sample_starts[i] = cand[0]
        left_licks = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:] if 'acquisition/BehavioralEvents/left_lick_times/timestamps' in f else np.array([])
        right_licks = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:] if 'acquisition/BehavioralEvents/right_lick_times/timestamps' in f else np.array([])
        choice, outcome, early = infer_choice_outcome_early(trial_table, left_licks, right_licks, go_times)
        ps_starts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps'][:] if 'acquisition/BehavioralEvents/photostim_start_times/timestamps' in f else np.array([])
        ps_stops = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:] if 'acquisition/BehavioralEvents/photostim_stop_times/timestamps' in f else np.array([])
        tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:] if 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps' in f else np.array([])
        tongue_xy = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:] if 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data' in f else np.zeros((0,2))
        tongue_disc, tongue_thr = discretize_tongue_y(tongue_xy) if len(tongue_ts) else (np.array([], dtype=np.int64), (np.nan, np.nan))

        good_inds = np.flatnonzero(good_mask)
        session_neural = []
        session_input = []
        session_output = []
        # Pre-bin each good unit once over the full session timeline for speed
        session_start = float(np.min(start_times) + T_START - 0.5)
        session_stop = float(np.max(stop_times) + T_END + 0.5)
        session_edges = np.arange(session_start, session_stop + BIN_SIZE, BIN_SIZE)
        session_fr = np.zeros((len(good_inds), len(session_edges) - 1), dtype=np.float32)
        for j, u in enumerate(good_inds):
            st = get_spike_times_for_unit(f['units'], int(u))
            m = (st >= session_start) & (st < session_stop)
            counts, _ = np.histogram(st[m], bins=session_edges)
            session_fr[j] = counts.astype(np.float32) / BIN_SIZE
        for i, go in enumerate(go_times):
            start_idx = int(np.round((go + T_START - session_start) / BIN_SIZE))
            end_idx = start_idx + N_BINS
            if start_idx < 0 or end_idx > session_fr.shape[1]:
                continue
            trial_mats = session_fr[:, start_idx:end_idx].copy()
            inp0 = build_time_from_tone_vector(sample_starts[i], go)
            if 'photostim_onset' in trial_table and 'photostim_duration' in trial_table:
                onset = safe_float(trial_table['photostim_onset'][i])
                dur = safe_float(trial_table['photostim_duration'][i])
                if np.isfinite(onset) and np.isfinite(dur) and dur > 0:
                    abs_on = start_times[i] + onset
                    pst = np.array([abs_on], dtype=float)
                    pen = np.array([abs_on + dur], dtype=float)
                else:
                    pst = np.array([], dtype=float)
                    pen = np.array([], dtype=float)
            else:
                pst = ps_starts[(ps_starts >= start_times[i]) & (ps_starts <= stop_times[i] + 0.1)]
                pen = ps_stops[(ps_stops >= start_times[i]) & (ps_stops <= stop_times[i] + 0.5)]
                if len(pen) < len(pst):
                    pen = np.concatenate([pen, np.full(len(pst)-len(pen), go)])
            inp1 = build_photostim_vector(pst, pen, go)[None, :]
            inp = np.vstack([inp0, inp1]).astype(np.float32)
            if len(tongue_ts):
                m = (tongue_ts >= go + T_START) & (tongue_ts < go + T_END)
                tongue_trial = bin_tongue_for_trial(tongue_ts[m], tongue_disc[m], go)
            else:
                tongue_trial = np.full(N_BINS, 3, dtype=np.int64)
            out = np.vstack([
                np.full(N_BINS, choice[i], dtype=np.int64),
                np.full(N_BINS, outcome[i], dtype=np.int64),
                np.full(N_BINS, early[i], dtype=np.int64),
                tongue_trial.astype(np.int64),
            ])
            if np.allclose(trial_mats, 0):
                continue
            if np.allclose(trial_mats, 0):
                continue
            session_neural.append(trial_mats)
            session_input.append(inp)
            session_output.append(out)

        subj = path.parent.name
        regions = region_names[good_inds]
        if show_processing:
            fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            ax[0].imshow(session_neural[0], aspect='auto', interpolation='nearest')
            ax[0].set_title(f'{path.stem} neural trial0')
            ax[1].plot(BIN_CENTERS, session_input[0][0], label='time_from_tone')
            ax[1].plot(BIN_CENTERS, session_input[0][1], label='photostim')
            ax[1].legend()
            ax[2].plot(BIN_CENTERS, session_output[0][3], label='tongue_y_disc')
            ax[2].legend()
            fig.tight_layout()
            fig.savefig(f'/app/processing_{path.stem}.png', dpi=150)
            plt.close(fig)
    print(f'processed {path.name} in {time.time()-t0:.2f}s trials={len(session_neural)} good_units={len(good_inds)}')
    return session_neural, session_input, session_output, subj, regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true')
    g.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()

    files = sorted(Path('/app/data').rglob('*.nwb'))
    if args.sample:
        files = files[:2]
    subjects = []
    subject_idx = []
    brain_regions = []
    brain_region_idx = []
    neural = []
    input_data = []
    output_data = []

    for f in files:
        sn, si, so, subj, regions = process_session(f, show_processing=args.show_processing)
        if len(sn) < 2:
            continue
        if subj not in subjects:
            subjects.append(subj)
        subject_idx.append(subjects.index(subj))
        neural.append(sn)
        input_data.append(si)
        output_data.append(so)
        reg_idx = []
        for r in regions:
            r = str(r)
            if r not in brain_regions:
                brain_regions.append(r)
            reg_idx.append(brain_regions.index(r))
        brain_region_idx.append(np.asarray(reg_idx, dtype=np.int64))

    data = {
        'neural': neural,
        'input': input_data,
        'output': output_data,
        'subjects': subjects,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_tone_onset_sec', 'photostimulation_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right', 'no lick'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['lt_40th', '40th_to_60th', 'gt_60th', 'not_visible'],
        ],
        'metadata': {
            'task_description': 'Auditory delayed response task with go-cue alignment; decode choice, outcome, early lick, and tongue y-position from neural activity.',
            'time_bin_size': 50.0,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': T_START,
            'off_end': T_END,
            'n_timepoints': N_BINS,
            'session_files': [str(f) for f in files],
        }
    }
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {args.outpicklefile}')


if __name__ == '__main__':
    main()
