import argparse
import math
import os
import pickle
import time
from pathlib import Path

import h5py
import numpy as np


def decode_arr(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode())
        else:
            out.append(str(x))
    return np.array(out, dtype=object)


def get_ragged_row(values, index, i):
    start = 0 if i == 0 else int(index[i - 1])
    end = int(index[i])
    return values[start:end]


def infer_brain_region_names(f, good_mask):
    units = f['units']
    region = None
    if 'anno_name' in units:
        arr = decode_arr(units['anno_name'][()])
        if np.any(arr != ''):
            region = arr
    if region is None and 'electrode_group' in units:
        arr = decode_arr(units['electrode_group'][()])
        region = np.array([x.split('_')[0] if x else 'unknown' for x in arr], dtype=object)
    if region is None:
        region = np.array(['unknown'] * len(good_mask), dtype=object)
    region = region[good_mask]
    region = np.array(['unknown' if (x is None or x == 'nan' or x == '') else x for x in region], dtype=object)
    return region


def load_trial_table(f):
    tr = f['intervals/trials']
    out = {}
    for k in tr.keys():
        arr = tr[k][()]
        if getattr(arr, 'dtype', None) is not None and arr.dtype.kind in ('S', 'O', 'U'):
            out[k] = decode_arr(arr)
        else:
            out[k] = arr
    return out


def load_event_times(f, name):
    grp = f['acquisition/BehavioralEvents'][name]
    return np.asarray(grp['timestamps'][()], dtype=float)


def load_tongue(f):
    grp = f['acquisition/BehavioralTimeSeries']['Camera0_side_TongueTracking']
    data = np.asarray(grp['data'][()], dtype=float)
    ts = np.asarray(grp['timestamps'][()], dtype=float)
    return ts, data


def choose_tongue_y_column(data):
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError('Unexpected tongue tracking shape')
    return 1


def build_edges(pre, post, bin_size):
    n_bins = int(round((pre + post) / bin_size))
    edges = np.linspace(-pre, post, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers


def build_time_from_tone(bin_centers, tone_time_rel):
    return bin_centers - tone_time_rel


def build_photostim_vector(bin_centers, onset_rel, duration):
    x = np.zeros(bin_centers.shape[0], dtype=np.float32)
    if onset_rel is None or duration is None:
        return x
    if not np.isfinite(onset_rel) or not np.isfinite(duration):
        return x
    off = onset_rel + duration
    x[(bin_centers >= onset_rel) & (bin_centers < off)] = 1.0
    return x


def parse_optional_float(x):
    if isinstance(x, str):
        if x in ('N/A', 'nan', ''):
            return np.nan
        return float(x)
    return float(x)


def find_choice_from_licks(left_licks, right_licks, go_time, stop_time):
    l = left_licks[(left_licks >= go_time) & (left_licks <= stop_time)]
    r = right_licks[(right_licks >= go_time) & (right_licks <= stop_time)]
    tl = l[0] if len(l) else np.inf
    tr = r[0] if len(r) else np.inf
    if tl == np.inf and tr == np.inf:
        return None
    return 0 if tl < tr else 1


def process_session(path, edges, bin_centers, show_processing=False):
    t0 = time.time()
    with h5py.File(path, 'r') as f:
        trial = load_trial_table(f)
        go_times = load_event_times(f, 'go_start_times')
        sample_event_times = load_event_times(f, 'sample_start_times')
        left_licks = load_event_times(f, 'left_lick_times')
        right_licks = load_event_times(f, 'right_lick_times')
        tongue_ts, tongue_data = load_tongue(f)
        ycol = choose_tongue_y_column(tongue_data)
        tongue_y = tongue_data[:, ycol]

        cls = decode_arr(f['units']['classification'][()])
        good_mask = cls == 'good'
        good_idx = np.flatnonzero(good_mask)
        if len(good_idx) == 0:
            return None
        spike_times = f['units']['spike_times'][()]
        spike_index = f['units']['spike_times_index'][()]
        region_names = infer_brain_region_names(f, good_mask)
        is_good_trials = np.asarray(f['units']['is_good_trials'][()], dtype=bool) if ('is_good_trials' in f['units'] and f['units']['is_good_trials'].shape[1] == len(trial['start_time'])) else None

        n_trials = len(trial['start_time'])
        assert len(go_times) == n_trials, (len(go_times), n_trials)

        sample_times = np.full(n_trials, np.nan, dtype=float)
        for i in range(n_trials):
            hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) & (sample_event_times <= trial['stop_time'][i])]
            if len(hits):
                sample_times[i] = hits[0]

        # Pre-bin all good-unit spikes once over the session timeline, then slice trial windows.
        bin_size = edges[1] - edges[0]
        n_time = len(bin_centers)
        session_t0 = float(np.min(go_times) + edges[0])
        session_t1 = float(np.max(go_times) + edges[-1])
        global_edges = np.arange(session_t0, session_t1 + bin_size * 1.0001, bin_size)
        global_centers = (global_edges[:-1] + global_edges[1:]) / 2
        global_rates = np.zeros((len(good_idx), len(global_centers)), dtype=np.float32)
        for jj, unit_i in enumerate(good_idx):
            st = get_ragged_row(spike_times, spike_index, int(unit_i))
            counts, _ = np.histogram(st, bins=global_edges)
            global_rates[jj] = counts.astype(np.float32) / bin_size

        go_bin_start = np.rint((go_times + edges[0] - session_t0) / bin_size).astype(int)

        session_trial_neural = []
        session_trial_input = []
        session_trial_output = []
        valid_trial_ids = []
        all_binned_y = []

        for i in range(n_trials):
            go = float(go_times[i])
            start = go + edges[0]
            stop = go + edges[-1]
            if start < 0:
                continue

            if is_good_trials is not None:
                valid_units = is_good_trials[good_idx, i]
                if not np.any(valid_units):
                    continue
            else:
                valid_units = np.ones(len(good_idx), dtype=bool)

            start_idx = int(go_bin_start[i])
            end_idx = start_idx + len(bin_centers)
            if start_idx < 0 or end_idx > global_rates.shape[1]:
                continue
            trial_mat = global_rates[:, start_idx:end_idx].copy()
            if is_good_trials is not None:
                trial_mat[~valid_units, :] = 0.0

            tone_rel = float(sample_times[i] - go) if np.isfinite(sample_times[i]) else np.nan
            inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)

            ps_on = parse_optional_float(trial['photostim_onset'][i]) if 'photostim_onset' in trial else np.nan
            ps_dur = parse_optional_float(trial['photostim_duration'][i]) if 'photostim_duration' in trial else np.nan
            inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
            inp = np.stack([inp0, inp1], axis=0)

            out_outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
            out_early_map = {'no early': 0, 'early': 1}
            outcome = out_outcome_map[str(trial['outcome'][i])]
            early = out_early_map[str(trial['early_lick'][i])]

            choice = find_choice_from_licks(left_licks, right_licks, go, float(trial['stop_time'][i]))
            if choice is None:
                instr = str(trial['trial_instruction'][i])
                choice = 0 if instr == 'left' else 1

            mask = (tongue_ts >= start) & (tongue_ts < stop)
            yt = tongue_y[mask]
            tt = tongue_ts[mask] - go
            binned_y = np.full(len(bin_centers), np.nan, dtype=np.float32)
            if len(tt):
                inds = np.digitize(tt, edges) - 1
                ok = (inds >= 0) & (inds < len(bin_centers)) & np.isfinite(yt)
                if np.any(ok):
                    sums = np.zeros(len(bin_centers), dtype=np.float64)
                    cnts = np.zeros(len(bin_centers), dtype=np.int64)
                    np.add.at(sums, inds[ok], yt[ok])
                    np.add.at(cnts, inds[ok], 1)
                    nz = cnts > 0
                    binned_y[nz] = (sums[nz] / cnts[nz]).astype(np.float32)
            all_binned_y.append(binned_y)

            session_trial_neural.append(trial_mat)
            session_trial_input.append(inp)
            session_trial_output.append([choice, outcome, early, binned_y])
            valid_trial_ids.append(i)

        if len(session_trial_neural) < 2:
            return None

        all_y = np.concatenate([x[np.isfinite(x)] for x in all_binned_y if np.any(np.isfinite(x))]) if any(np.any(np.isfinite(x)) for x in all_binned_y) else np.array([], dtype=float)
        if len(all_y) == 0:
            return None
        q40, q60 = np.percentile(all_y, [40, 60])

        final_outputs = []
        for choice, outcome, early, binned_y in session_trial_output:
            ycat = np.full(len(bin_centers), 1, dtype=np.int64)
            finite = np.isfinite(binned_y)
            ycat[finite & (binned_y < q40)] = 0
            ycat[finite & (binned_y > q60)] = 2
            ycat[finite & (binned_y >= q40) & (binned_y <= q60)] = 1
            out = np.zeros((4, len(bin_centers)), dtype=np.int64)
            out[0, :] = choice
            out[1, :] = outcome
            out[2, :] = early
            out[3, :] = ycat
            final_outputs.append(out)

        subject = f['general']['subject']['subject_id'][()]
        if isinstance(subject, bytes):
            subject = subject.decode()
        session_id = path.stem

    info = {
        'session_id': session_id,
        'subject': subject,
        'neural': session_trial_neural,
        'input': session_trial_input,
        'output': final_outputs,
        'region_names': region_names.tolist(),
        'n_good_units': len(region_names),
        'n_trials': len(final_outputs),
        'elapsed_sec': time.time() - t0,
    }
    print(f'processed {session_id}: trials={info["n_trials"]} good_units={info["n_good_units"]} time={info["elapsed_sec"]:.2f}s')
    return info


def maybe_plot(session_info, outdir='.'):
    import matplotlib.pyplot as plt
    sid = session_info['session_id']
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    trial0 = 0
    neural = session_info['neural'][trial0]
    inp = session_info['input'][trial0]
    out = session_info['output'][trial0]
    axs[0].imshow(neural, aspect='auto', interpolation='nearest')
    axs[0].set_title(f'{sid} neural trial0')
    axs[1].plot(inp[0], label='time_from_tone_onset')
    axs[1].plot(inp[1], label='photostimulation_on')
    axs[1].legend(loc='upper left')
    axs[2].plot(out[3], label='tongue_y_cat')
    axs[2].plot(out[0], label='choice')
    axs[2].plot(out[1], label='outcome')
    axs[2].plot(out[2], label='early')
    axs[2].legend(loc='upper left', ncol=4)
    fig.savefig(Path(outdir) / f'processing_{sid}.png', dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true')
    g.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()

    pre = 2.5
    post = 1.5
    bin_size = 0.05
    edges, bin_centers = build_edges(pre, post, bin_size)

    files = sorted(Path('data').glob('sub-*/*.nwb'))
    if args.sample:
        files = files[:2]
    sessions = []
    for p in files:
        info = process_session(p, edges, bin_centers, show_processing=args.show_processing)
        if info is not None:
            sessions.append(info)
            if args.show_processing and len(sessions) <= 2:
                maybe_plot(info)

    subjects = sorted({s['subject'] for s in sessions})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    all_regions = sorted({r for s in sessions for r in s['region_names']})
    region_to_idx = {r: i for i, r in enumerate(all_regions)}

    data = {
        'neural': [s['neural'] for s in sessions],
        'input': [s['input'] for s in sessions],
        'output': [s['output'] for s in sessions],
        'subjects': subjects,
        'subject_idx': np.array([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
        'brain_regions': all_regions,
        'brain_region_idx': [np.array([region_to_idx[r] for r in s['region_names']], dtype=np.int64) for s in sessions],
        'input_names': ['time_from_tone_onset', 'photostimulation_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['lt_40pct', '40to60pct', 'gt_60pct'],
        ],
        'metadata': {
            'task_description': 'Memory-guided lick task with go-cue alignment, neural activity predicting choice, outcome, early lick, and discretized tongue y-position; includes photostimulation input.',
            'time_bin_size': 50.0,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': -2.5,
            'off_end': 1.5,
            'n_sessions': len(sessions),
            'bin_centers_s': bin_centers.astype(np.float32),
        }
    }

    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {args.outpicklefile} with {len(sessions)} sessions')


if __name__ == '__main__':
    main()
