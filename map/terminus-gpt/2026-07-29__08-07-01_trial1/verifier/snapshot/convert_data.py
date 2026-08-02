#!/usr/bin/env python3
import argparse
import json
import pickle
import time
from pathlib import Path

import h5py
import numpy as np

BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
BIN_EDGES = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2

CHOICE_MAP = {'left': 0, 'right': 1}
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
EARLY_MAP = {'no early': 0, 'early': 1}
MAJOR_REGIONS = {'left ALM','right ALM','left Striatum','right Striatum','left Thalamus','right Thalamus','left Midbrain','right Midbrain','left Medulla','right Medulla'}


def decode_arr(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode('utf-8', 'ignore'))
        else:
            out.append(str(x))
    return np.array(out)


def parse_region_strings(locs):
    regs = []
    for s in locs:
        if isinstance(s, bytes):
            s = s.decode('utf-8', 'ignore')
        else:
            s = str(s)
        try:
            regs.append(json.loads(s).get('brain_regions', 'UNKNOWN'))
        except Exception:
            regs.append(s)
    return np.array(regs)


def get_subject_id(nwb_path):
    return nwb_path.parent.name


def get_session_id(nwb_path):
    return nwb_path.stem


def choose_sessions(all_files, sample=False):
    all_files = sorted(all_files)
    return all_files[:2] if sample else all_files


def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    valid = np.isfinite(track_y) & np.isfinite(rel_t)
    if valid.sum() < 2:
        return np.full(N_BINS, np.nan, dtype=np.float32)
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], left=np.nan, right=np.nan).astype(np.float32)


def build_photostim_series(start_times, stop_times, go_time):
    x = np.zeros(N_BINS, dtype=np.float32)
    for s, e in zip(start_times, stop_times):
        rs = s - go_time
        re = e - go_time
        overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
        x[overlap] = 1.0
    return x


def build_time_from_tone(sample_time, go_time):
    tone_rel = sample_time - go_time
    rel = BIN_CENTERS - tone_rel
    rel[rel < 0] = 0.0
    return rel.astype(np.float32)


def find_last_event_within_trial(event_times, t_start, t_stop, before_time=None):
    mask = (event_times >= t_start) & (event_times <= t_stop)
    if before_time is not None:
        mask &= (event_times <= before_time)
    vals = event_times[mask]
    if len(vals) == 0:
        return np.nan
    return float(vals[-1])


def bin_unit_spikes_fast(spike_times, go_time):
    lo = go_time + T_START
    hi = go_time + T_END
    a = np.searchsorted(spike_times, lo, side='left')
    b = np.searchsorted(spike_times, hi, side='left')
    if b <= a:
        return np.zeros(N_BINS, dtype=np.float32)
    rel = spike_times[a:b] - lo
    bins = np.floor(rel / BIN_SIZE).astype(np.int64)
    bins = bins[(bins >= 0) & (bins < N_BINS)]
    counts = np.bincount(bins, minlength=N_BINS).astype(np.float32)
    return counts / BIN_SIZE


def load_session(nwb_path, brain_regions):
    with h5py.File(nwb_path, 'r') as f:
        trials = f['intervals/trials']
        n_trials = trials['id'].shape[0]

        unit_quality = decode_arr(f['units/unit_quality'][:])
        unit_electrodes = f['units/electrodes'][:]
        elec_regions = parse_region_strings(f['general/extracellular_ephys/electrodes/location'][:])
        unit_regions = elec_regions[unit_electrodes]
        keep_units = (unit_quality == 'good') & np.isin(unit_regions, list(MAJOR_REGIONS))
        kept_idx = np.where(keep_units)[0]

        spikes_flat = f['units/spike_times'][:]
        spikes_index = f['units/spike_times_index'][:]
        starts = np.concatenate([[0], spikes_index[:-1]])
        kept_spikes = [spikes_flat[starts[u]:spikes_index[u]] for u in kept_idx]

        region_idx = []
        for r in unit_regions[kept_idx]:
            if r not in brain_regions:
                brain_regions.append(r)
            region_idx.append(brain_regions.index(r))
        region_idx = np.array(region_idx, dtype=np.int64)

        go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
        sample_event_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
        photo_starts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps'][:]
        photo_stops = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:]
        left_lick = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:]
        right_lick = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:]
        tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
        tongue_t = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
        tongue_y_all = tongue[:, 1].astype(np.float32)

        choice = decode_arr(trials['trial_instruction'][:])
        outcome = decode_arr(trials['outcome'][:])
        early = decode_arr(trials['early_lick'][:])
        auto_water = trials['auto_water'][:]
        free_water = trials['free_water'][:]
        trial_start = trials['start_time'][:]
        trial_stop = trials['stop_time'][:]

        n_match = min(n_trials, len(go_times))
        sess_neural, sess_input, sess_output = [], [], []
        sess_tongue_cont = []
        valid_trial_inds = []

        for i in range(n_match):
            go = go_times[i]
            if not np.isfinite(go):
                continue
            if go + T_START < trial_start[i] or go + T_END > trial_stop[i]:
                continue
            if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP:
                continue
            neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
            sample_time = find_last_event_within_trial(sample_event_times, trial_start[i], trial_stop[i], before_time=go)
            if not np.isfinite(sample_time):
                continue
            inp0 = build_time_from_tone(sample_time, go)
            ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
            pe = photo_stops[(photo_stops >= trial_start[i]) & (photo_stops <= trial_stop[i] + 1e-6)]
            m = min(len(ps), len(pe))
            inp1 = build_photostim_series(ps[:m], pe[:m], go)
            ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
            sess_tongue_cont.append(ty)
            sess_neural.append(neural.astype(np.float32))
            sess_input.append(np.stack([inp0, inp1], axis=0).astype(np.float32))
            valid_trial_inds.append(i)

        if len(sess_neural) < 2:
            return None

        all_tongue = np.concatenate([x[np.isfinite(x)] for x in sess_tongue_cont if np.isfinite(x).any()]) if any(np.isfinite(x).any() for x in sess_tongue_cont) else np.array([0,1], dtype=np.float32)
        q40, q60 = np.percentile(all_tongue, [40, 60])
        for ty, i in zip(sess_tongue_cont, valid_trial_inds):
            disc = np.full(N_BINS, 1, dtype=np.int64)
            disc[ty < q40] = 0
            disc[ty > q60] = 2
            out = np.vstack([
                np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64),
                np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64),
                np.full(N_BINS, EARLY_MAP[early[i]], dtype=np.int64),
                disc,
            ])
            sess_output.append(out)

        return {
            'session_id': get_session_id(nwb_path),
            'subject': get_subject_id(nwb_path),
            'neural': sess_neural,
            'input': sess_input,
            'output': sess_output,
            'brain_region_idx': region_idx,
            'n_units_kept': int(len(kept_idx)),
            'n_trials_kept': int(len(sess_neural)),
            'q40': float(q40),
            'q60': float(q60),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true')
    g.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    files = sorted(Path('data').rglob('*.nwb'))
    files = choose_sessions(files, sample=args.sample)

    subjects = []
    subject_map = {}
    brain_regions = []
    data = {
        'neural': [],
        'input': [],
        'output': [],
        'subjects': subjects,
        'subject_idx': [],
        'brain_regions': brain_regions,
        'brain_region_idx': [],
        'input_names': ['time_from_tone_onset', 'photostim_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y'],
        'output_values': [
            ['left', 'right'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['low', 'mid', 'high'],
        ],
        'metadata': {
            'task_description': 'Audio delay task; decode choice, outcome, early lick, and discretized tongue y from neural activity.',
            'time_bin_size': 50.0,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': T_START,
            'off_end': T_END,
            'n_timepoints': N_BINS,
        }
    }

    for fp in files:
        sess = load_session(fp, brain_regions)
        if sess is None:
            print('SKIP', fp)
            continue
        subj = sess['subject']
        if subj not in subject_map:
            subject_map[subj] = len(subjects)
            subjects.append(subj)
        data['neural'].append(sess['neural'])
        data['input'].append(sess['input'])
        data['output'].append(sess['output'])
        data['subject_idx'].append(subject_map[subj])
        data['brain_region_idx'].append(sess['brain_region_idx'])
        print('SESSION', sess['session_id'], 'units', sess['n_units_kept'], 'trials', sess['n_trials_kept'])

    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)

    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)

    print('saved', args.outpicklefile)
    print('sessions', len(data['neural']))
    print('subjects', len(data['subjects']))
    print('brain_regions', len(data['brain_regions']))
    print('elapsed_sec', time.time() - t0)

if __name__ == '__main__':
    main()
