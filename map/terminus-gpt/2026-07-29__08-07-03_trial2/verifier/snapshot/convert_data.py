#!/usr/bin/env python3
import argparse
import json
import pickle
import time
from pathlib import Path

import h5py
import numpy as np

BIN_SIZE = 0.05
WINDOW_START = -2.5
WINDOW_END = 1.5
N_BINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))
BIN_EDGES = np.arange(WINDOW_START, WINDOW_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2

# From methods.txt for the auditory delayed response task:
# sample epoch: 3 x 150 ms tones with 100 ms inter-tone intervals => 0.65 s total
# delay epoch: 1.2 s
# go cue at 1.85 s after trial start
TONE_ONSET_FROM_TRIAL_START = 0.0
GO_CUE_FROM_TRIAL_START = 1.85

CHOICE_MAP = {b'left': 0, b'right': 1, 'left': 0, 'right': 1}
OUTCOME_MAP = {b'ignore': 0, b'miss': 1, b'hit': 2, 'ignore': 0, 'miss': 1, 'hit': 2}
EARLY_MAP = {b'no early': 0, b'early': 1, 'no early': 0, 'early': 1}


def _decode(x):
    if isinstance(x, bytes):
        return x.decode()
    return x


def parse_optional_time(x):
    x = _decode(x)
    if isinstance(x, str) and x == 'N/A':
        return None
    try:
        return float(x)
    except Exception:
        return None


def get_subject_id(f, path):
    try:
        sid = f['general/subject/subject_id'][()]
        return _decode(sid)
    except Exception:
        return path.parent.name


def get_unit_spike_times(units_group):
    spikes = units_group['spike_times'][:]
    index = units_group['spike_times_index'][:]
    out = []
    start = 0
    for stop in index:
        out.append(spikes[start:stop])
        start = stop
    return out


def choose_tongue_group(f):
    bts = f.get('acquisition/BehavioralTimeSeries')
    if bts is None:
        return None
    candidates = [k for k in bts.keys() if 'TongueTracking' in k]
    if not candidates:
        return None
    # Prefer Camera0 if present, otherwise first available
    candidates = sorted(candidates, key=lambda x: (not x.startswith('Camera0'), x))
    return bts[candidates[0]]


def extract_session(session_path, show_processing=False):
    t0 = time.time()
    with h5py.File(session_path, 'r') as f:
        trials = f['intervals/trials']
        n_trials = len(trials['id'])
        trial_start = trials['start_time'][:].astype(float)
        go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
        tone_onset_times = trial_start + TONE_ONSET_FROM_TRIAL_START

        trial_instruction = np.array([_decode(x) for x in trials['trial_instruction'][:]])
        outcome = np.array([_decode(x) for x in trials['outcome'][:]])
        early_lick = np.array([_decode(x) for x in trials['early_lick'][:]])
        photostim_onset = [parse_optional_time(x) for x in trials['photostim_onset'][:]]
        photostim_duration = [parse_optional_time(x) for x in trials['photostim_duration'][:]]

        # Trial validity: required labels present
        valid_trials = np.array([
            ti in ('left', 'right') and oc in ('ignore', 'miss', 'hit') and el in ('no early', 'early')
            for ti, oc, el in zip(trial_instruction, outcome, early_lick)
        ], dtype=bool)

        units = f['units']
        unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
        good_mask = unit_quality == 'good'
        spike_times_list = get_unit_spike_times(units)
        good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]
        if len(good_spike_times):
            spike_t_min = min(float(st[0]) for st in good_spike_times if len(st) > 0) if any(len(st) > 0 for st in good_spike_times) else 0.0
            spike_t_max = max(float(st[-1]) for st in good_spike_times if len(st) > 0) if any(len(st) > 0 for st in good_spike_times) else 0.0
        else:
            spike_t_min, spike_t_max = 0.0, 0.0

        go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
        window_ok = (go_cue_times + WINDOW_START >= spike_t_min) & (go_cue_times + WINDOW_END <= spike_t_max)
        valid_trials = valid_trials & window_ok

        # Derive per-unit brain region from associated electrode location metadata.
        brain_region_labels = []
        try:
            elec = units['electrodes'][:]
            eidx = units['electrodes_index'][:]
            loc = f['general/extracellular_ephys/electrodes/location'][:]
            start_i = 0
            all_unit_regions = []
            for stop_i in eidx:
                inds = elec[start_i:stop_i]
                start_i = stop_i
                if len(inds) == 0:
                    all_unit_regions.append('unknown')
                    continue
                raw = loc[int(inds[0])]
                raw = raw.decode() if isinstance(raw, bytes) else str(raw)
                region = 'unknown'
                try:
                    meta = json.loads(raw)
                    region = meta.get('brain_regions', 'unknown')
                except Exception:
                    region = raw
                all_unit_regions.append(region)
            brain_region_labels = [all_unit_regions[i] for i in np.where(good_mask)[0]]
        except Exception:
            brain_region_labels = ['unknown' for _ in good_spike_times]

        tongue_group = choose_tongue_group(f)
        tongue_t = None
        tongue_y = None
        if tongue_group is not None:
            tongue_data = tongue_group['data'][:]
            tongue_t = tongue_group['timestamps'][:].astype(float)
            # Convention in these tracking arrays is assumed [x, y, likelihood]
            if tongue_data.ndim == 2 and tongue_data.shape[1] >= 2:
                tongue_y = tongue_data[:, 1].astype(float)
                # mask low-confidence frames if likelihood available
                if tongue_data.shape[1] >= 3:
                    lik = tongue_data[:, 2].astype(float)
                    tongue_y = tongue_y.copy()
                    tongue_y[lik < 0.5] = np.nan

        # Session-wise discretization thresholds from all valid tongue samples
        if tongue_y is not None and np.isfinite(tongue_y).any():
            q40, q60 = np.nanpercentile(tongue_y, [40, 60])
        else:
            q40, q60 = np.nan, np.nan

        neural_trials = []
        input_trials = []
        output_trials = []

        for i in range(n_trials):
            if not valid_trials[i]:
                continue
            align = go_cue_times[i]
            edges_abs = align + BIN_EDGES
            centers_rel = BIN_CENTERS.copy()
            centers_abs = align + centers_rel

            # neural: spike counts per bin
            fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
            for n, st in enumerate(good_spike_times):
                mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
                if np.any(mask):
                    counts, _ = np.histogram(st[mask], bins=edges_abs)
                    fr[n] = counts.astype(np.float32) / BIN_SIZE

            # input 0: time from tone onset in seconds, time-varying
            time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)

            # input 1: photostimulation on/off, time-varying
            photo = np.zeros(N_BINS, dtype=np.float32)
            p_on = photostim_onset[i]
            p_dur = photostim_duration[i]
            if p_on is not None and p_dur is not None:
                p_start = trial_start[i] + p_on
                p_stop = p_start + p_dur
                photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
            inp = np.stack([time_from_tone, photo], axis=0)

            # outputs: 3 per-trial categorical rows repeated across time, plus tongue y discrete time-varying
            choice_val = CHOICE_MAP[trial_instruction[i]]
            outcome_val = OUTCOME_MAP[outcome[i]]
            early_val = EARLY_MAP[early_lick[i]]
            choice_ts = np.full(N_BINS, choice_val, dtype=np.int64)
            outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
            early_ts = np.full(N_BINS, early_val, dtype=np.int64)

            if tongue_t is not None and tongue_y is not None and np.isfinite([q40, q60]).all():
                valid_idx = np.flatnonzero(np.isfinite(tongue_y))
                tongue_disc = np.full(N_BINS, 1, dtype=np.int64)
                if len(valid_idx) > 0:
                    vt = tongue_t[valid_idx]
                    vy = tongue_y[valid_idx]
                    idx = np.searchsorted(vt, centers_abs, side='left')
                    idx = np.clip(idx, 0, len(vt) - 1)
                    y = vy[idx]
                    far = np.abs(vt[idx] - centers_abs) > 0.1
                    y = y.copy()
                    y[far] = np.nan
                    finite = np.isfinite(y)
                    tongue_disc[finite & (y < q40)] = 0
                    tongue_disc[finite & (y > q60)] = 2
            else:
                tongue_disc = np.full(N_BINS, 1, dtype=np.int64)

            out = np.stack([choice_ts, outcome_ts, early_ts, tongue_disc], axis=0)

            if np.all(fr == 0):
                continue

            neural_trials.append(fr)
            input_trials.append(inp)
            output_trials.append(out)

        info = {
            'subject': get_subject_id(f, session_path),
            'brain_region_labels': brain_region_labels,
            'n_trials_raw': n_trials,
            'n_trials_kept': len(neural_trials),
            'n_units_raw': len(unit_quality),
            'n_units_good': int(good_mask.sum()),
            'q40': float(q40) if np.isfinite(q40) else None,
            'q60': float(q60) if np.isfinite(q60) else None,
            'session_id': session_path.stem,
            'load_seconds': time.time() - t0,
        }
        return neural_trials, input_trials, output_trials, info


def build_dataset(paths, show_processing=False):
    neural = []
    inputs = []
    outputs = []
    subjects = []
    subject_to_idx = {}
    subject_idx = []
    brain_region_idx = []
    session_info = []
    brain_regions = []
    brain_region_to_idx = {}

    for path in paths:
        ntr, itr, otr, info = extract_session(path, show_processing=show_processing)
        if len(ntr) < 2 or (len(ntr) > 0 and ntr[0].shape[0] == 0):
            print(f'Skipping {path.name}: insufficient kept trials or no good units')
            continue
        neural.append(ntr)
        inputs.append(itr)
        outputs.append(otr)
        sid = info['subject']
        if sid not in subject_to_idx:
            subject_to_idx[sid] = len(subjects)
            subjects.append(sid)
        subject_idx.append(subject_to_idx[sid])
        sess_reg_idx = []
        for r in info['brain_region_labels']:
            if r not in brain_region_to_idx:
                brain_region_to_idx[r] = len(brain_regions)
                brain_regions.append(r)
            sess_reg_idx.append(brain_region_to_idx[r])
        brain_region_idx.append(np.asarray(sess_reg_idx, dtype=np.int64))
        session_info.append(info)
        print(f"Processed {path.name}: kept {info['n_trials_kept']}/{info['n_trials_raw']} trials, good units {info['n_units_good']}/{info['n_units_raw']}, {info['load_seconds']:.2f}s")

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_tone_onset_sec', 'photostim_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_pos_discrete'],
        'output_values': [
            ['left', 'right'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['lt_40pct', '40_to_60pct', 'gt_60pct'],
        ],
        'metadata': {
            'task_description': 'Auditory delayed-response task; decode choice, outcome, early lick, and discretized tongue y from neural activity.',
            'time_bin_size': 50.0,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': WINDOW_START,
            'off_end': WINDOW_END,
            'tone_onset_from_trial_start_sec': TONE_ONSET_FROM_TRIAL_START,
            'go_cue_from_trial_start_sec': GO_CUE_FROM_TRIAL_START,
            'session_info': session_info,
            'notes': 'Go cue and tone onset derived from methods-defined task timing relative to trial start; brain region metadata unavailable in inspected NWB unit tables, so fallback region label `unknown` used.'
        }
    }
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true', help='Process all sessions')
    g.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    ap.add_argument('--show-processing', action='store_true', help='Reserved; plotting omitted in this implementation')
    args = ap.parse_args()

    paths = sorted(Path('data').rglob('*.nwb'))
    if args.sample:
        paths = paths[:2]
    data = build_dataset(paths, show_processing=args.show_processing)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved {args.outpicklefile}')
    print(f"Sessions: {len(data['neural'])}, subjects: {len(data['subjects'])}")


if __name__ == '__main__':
    main()
