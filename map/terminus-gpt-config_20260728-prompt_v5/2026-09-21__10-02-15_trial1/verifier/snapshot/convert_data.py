#!/usr/bin/env python3
import argparse
import os
import pickle
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


BIN_SIZE = 0.05
WIN_START = -2.5
WIN_END = 1.5
N_BINS = int(round((WIN_END - WIN_START) / BIN_SIZE))
BIN_EDGES = np.linspace(WIN_START, WIN_END, N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
DATA_DIR = Path('/app/data')


def _decode_arr(arr):
    out = []
    for v in arr:
        if isinstance(v, (bytes, np.bytes_)):
            out.append(v.decode())
        else:
            out.append(str(v))
    return np.array(out, dtype=object)


def load_event_series(group, name):
    if name not in group:
        return None, None
    g = group[name]
    data = np.asarray(g['data'])
    ts = np.asarray(g['timestamps'])
    return data, ts


def infer_trial_event_times(event_ts, go_times, default_offset=-0.6, min_pre=0.05, max_pre=5.0):
    out = np.full(go_times.shape, np.nan, dtype=float)
    if event_ts is None or len(event_ts) == 0:
        return go_times + default_offset
    event_ts = np.asarray(event_ts, dtype=float)
    for i, gt in enumerate(go_times):
        cand = event_ts[(event_ts <= gt - min_pre) & (event_ts >= gt - max_pre)]
        if cand.size:
            out[i] = cand[-1]
        else:
            out[i] = gt + default_offset
    return out


def get_unit_spike_times(units_group, unit_index):
    st_ds = units_group['spike_times']
    try:
        x = st_ds[unit_index]
        arr = np.asarray(x, dtype=float)
        if arr.ndim >= 1 and arr.size > 0:
            return arr.ravel()
    except Exception:
        pass
    st_all = st_ds[()]
    if getattr(st_all, 'dtype', None) == object:
        return np.asarray(st_all[unit_index], dtype=float).ravel()
    idx = np.asarray(units_group['spike_times_index'][()])
    start = 0 if unit_index == 0 else idx[unit_index - 1]
    end = idx[unit_index]
    return np.asarray(st_all[start:end], dtype=float).ravel()


def choose_tongue_columns(tongue_data):
    if tongue_data.shape[1] < 3:
        raise ValueError('Tongue tracking data expected to have at least 3 columns')
    # assume x, y, likelihood
    return 1, 2


def session_from_file(path, make_plot=False):
    t0 = time.time()
    with h5py.File(path, 'r') as h:
        units = h['units']
        trials = h['intervals']['trials']
        acq = h['acquisition']

        unit_class = _decode_arr(units['classification'][()]) if 'classification' in units else None
        good_mask = unit_class == 'good' if unit_class is not None else np.ones(len(units['id']), dtype=bool)
        anno_name = _decode_arr(units['anno_name'][()]) if 'anno_name' in units else np.array(['unknown'] * len(good_mask), dtype=object)
        good_inds = np.flatnonzero(good_mask)
        good_spike_times = [get_unit_spike_times(units, int(ui)) for ui in good_inds]

        beh_events = acq['BehavioralEvents']
        beh_ts = acq['BehavioralTimeSeries']

        go_times = np.asarray(beh_events['go_start_times']['timestamps'][()]).astype(float)
        sample_times = np.asarray(beh_events['sample_start_times']['timestamps'][()]).astype(float) if 'sample_start_times' in beh_events else None
        photostim_start = np.asarray(beh_events['photostim_start_times']['timestamps'][()]).astype(float) if 'photostim_start_times' in beh_events else np.array([])
        photostim_stop = np.asarray(beh_events['photostim_stop_times']['timestamps'][()]).astype(float) if 'photostim_stop_times' in beh_events else np.array([])
        left_lick_times = np.asarray(beh_events['left_lick_times']['timestamps'][()]).astype(float) if 'left_lick_times' in beh_events else np.array([])
        right_lick_times = np.asarray(beh_events['right_lick_times']['timestamps'][()]).astype(float) if 'right_lick_times' in beh_events else np.array([])

        n_trials = len(go_times)
        trial_instruction = _decode_arr(trials['trial_instruction'][()])[:n_trials]
        outcome = _decode_arr(trials['outcome'][()])[:n_trials]
        early_lick = _decode_arr(trials['early_lick'][()])[:n_trials]
        valid_trial_mask = np.ones(n_trials, dtype=bool)
        if 'is_good_trials' in units:
            igt = np.asarray(units['is_good_trials'][()])
            n_valid_cols = min(igt.shape[1], n_trials)
            trial_good_frac = igt[good_inds, :n_valid_cols].mean(axis=0) if len(good_inds) else np.zeros(n_valid_cols)
            valid_trial_mask[:] = False
            valid_trial_mask[:n_valid_cols] = trial_good_frac > 0

        if sample_times is None or len(sample_times) != n_trials:
            sample_times = infer_trial_event_times(sample_times, go_times)
        else:
            sample_times = infer_trial_event_times(sample_times[:], go_times)

        if 'photostim_duration' in trials:
            pdur = _decode_arr(trials['photostim_duration'][()])[:n_trials]
            pon = _decode_arr(trials['photostim_onset'][()])[:n_trials] if 'photostim_onset' in trials else np.array(['N/A'] * n_trials, dtype=object)
        else:
            pdur = np.array(['N/A'] * n_trials, dtype=object)
            pon = np.array(['N/A'] * n_trials, dtype=object)

        tongue = np.asarray(beh_ts['Camera0_side_TongueTracking']['data'][()]).astype(float)
        tongue_t = np.asarray(beh_ts['Camera0_side_TongueTracking']['timestamps'][()]).astype(float)
        y_col, vis_col = choose_tongue_columns(tongue)
        tongue_y = tongue[:, y_col]
        tongue_vis = tongue[:, vis_col]
        visible = np.isfinite(tongue_y) & np.isfinite(tongue_vis) & (tongue_vis > 0.5)
        if np.any(visible):
            q40, q60 = np.quantile(tongue_y[visible], [0.4, 0.6])
        else:
            q40, q60 = 0.0, 0.0

        brain_regions = anno_name[good_inds]
        neural_trials = []
        input_trials = []
        output_trials = []

        for tr in range(n_trials):
            if not valid_trial_mask[tr]:
                continue
            gt = go_times[tr]
            abs_edges = gt + BIN_EDGES
            abs_centers = gt + BIN_CENTERS

            neural = np.zeros((len(good_inds), N_BINS), dtype=np.float32)
            for j, st in enumerate(good_spike_times):
                rel = st - gt
                counts, _ = np.histogram(rel, bins=BIN_EDGES)
                neural[j] = counts.astype(np.float32) / BIN_SIZE

            stime = sample_times[tr] if np.isfinite(sample_times[tr]) else (gt - 0.6)
            time_from_tone = abs_centers - stime
            photostim_on = np.zeros(N_BINS, dtype=np.float32)
            if pon[tr] != 'N/A' and pdur[tr] != 'N/A':
                try:
                    pstart = float(pon[tr])
                    pd = float(pdur[tr])
                    photostim_on = ((abs_centers >= pstart) & (abs_centers < pstart + pd)).astype(np.float32)
                except Exception:
                    pass
            else:
                for ps, pe in zip(photostim_start, photostim_stop):
                    if pe >= abs_edges[0] and ps <= abs_edges[-1]:
                        photostim_on |= ((abs_centers >= ps) & (abs_centers < pe))
                photostim_on = photostim_on.astype(np.float32)

            # choice from first lick after go cue within response window
            lmask = (left_lick_times >= gt) & (left_lick_times < gt + WIN_END)
            rmask = (right_lick_times >= gt) & (right_lick_times < gt + WIN_END)
            lfirst = left_lick_times[lmask][0] if np.any(lmask) else np.inf
            rfirst = right_lick_times[rmask][0] if np.any(rmask) else np.inf
            if np.isfinite(lfirst) and (lfirst < rfirst):
                choice = 0  # left
            elif np.isfinite(rfirst) and (rfirst < lfirst):
                choice = 1  # right
            else:
                choice = 2  # no lick

            outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
            early_map = {'no early': 0, 'early': 1}
            out_val = outcome_map.get(outcome[tr], 0)
            early_val = early_map.get(early_lick[tr], 0)

            idx = np.searchsorted(tongue_t, abs_centers, side='left')
            idx = np.clip(idx, 0, len(tongue_t) - 1)
            ty = tongue_y[idx]
            tv = tongue_vis[idx]
            tongue_disc = np.full(N_BINS, 3, dtype=np.int64)
            vis_now = np.isfinite(ty) & np.isfinite(tv) & (tv > 0.5)
            tongue_disc[vis_now & (ty < q40)] = 0
            tongue_disc[vis_now & (ty >= q40) & (ty <= q60)] = 1
            tongue_disc[vis_now & (ty > q60)] = 2

            inp = np.vstack([time_from_tone.astype(np.float32), photostim_on.astype(np.float32)])
            out = np.vstack([
                np.full(N_BINS, choice, dtype=np.int64),
                np.full(N_BINS, out_val, dtype=np.int64),
                np.full(N_BINS, early_val, dtype=np.int64),
                tongue_disc.astype(np.int64),
            ])

            if np.all(neural == 0):
                continue
            neural_trials.append(neural)
            input_trials.append(inp)
            output_trials.append(out)

        subject = path.parent.name.replace('sub-', '')
        session_info = {
            'file': str(path),
            'subject': subject,
            'n_trials': n_trials,
            'n_good_units': len(good_inds),
            'load_time_sec': time.time() - t0,
        }

        if make_plot:
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            tr = min(0, n_trials - 1)
            axes[0].imshow(neural_trials[tr], aspect='auto', interpolation='nearest', cmap='viridis')
            axes[0].set_ylabel('neurons')
            axes[0].set_title(f'{path.stem} trial0 neural')
            axes[1].plot(BIN_CENTERS, input_trials[tr][0], label='time_from_tone')
            axes[1].plot(BIN_CENTERS, input_trials[tr][1], label='photostim_on')
            axes[1].legend(loc='upper left')
            axes[2].step(BIN_CENTERS, output_trials[tr][3], where='mid')
            axes[2].set_ylabel('tongue_y_bin')
            axes[3].step(BIN_CENTERS, output_trials[tr][0], where='mid', label='choice')
            axes[3].step(BIN_CENTERS, output_trials[tr][1], where='mid', label='outcome')
            axes[3].step(BIN_CENTERS, output_trials[tr][2], where='mid', label='early')
            axes[3].legend(loc='upper left')
            axes[3].set_xlabel('time from go cue (s)')
            fig.tight_layout()
            fig.savefig(f'/app/processing_{path.stem}.png', dpi=150)
            plt.close(fig)

        return neural_trials, input_trials, output_trials, brain_regions.tolist(), subject, session_info


def build_dataset(files, show_processing=False):
    neural_all, input_all, output_all = [], [], []
    subjects = []
    subject_idx = []
    brain_regions_master = []
    brain_region_idx_all = []
    session_info = []

    for i, f in enumerate(files):
        print(f'processing {i+1}/{len(files)}: {f}', flush=True)
        neural, inp, out, sess_regions, subj, info = session_from_file(f, make_plot=show_processing and i < 2)
        if len(neural) < 2 or neural[0].shape[0] == 0:
            print(f'skipping {f} due to insufficient trials or neurons', flush=True)
            continue
        if subj not in subjects:
            subjects.append(subj)
        subject_idx.append(subjects.index(subj))
        sess_region_idx = []
        for r in sess_regions:
            r = r if r != '' else 'unknown'
            if r not in brain_regions_master:
                brain_regions_master.append(r)
            sess_region_idx.append(brain_regions_master.index(r))
        neural_all.append(neural)
        input_all.append(inp)
        output_all.append(out)
        brain_region_idx_all.append(np.asarray(sess_region_idx, dtype=np.int64))
        session_info.append(info)

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': subjects,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': brain_regions_master,
        'brain_region_idx': brain_region_idx_all,
        'input_names': ['time_from_tone_onset_sec', 'photostimulation_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right', 'no_lick'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['lt_40th', '40th_to_60th', 'gt_60th', 'not_visible'],
        ],
        'metadata': {
            'task_description': 'Memory-guided directional licking task with optogenetic perturbation; decode choice, outcome, early lick, and tongue y-position from neural activity.',
            'time_bin_size': BIN_SIZE * 1000.0,
            'temporal_alignment_event': 'Go cue onset (BehavioralEvents/go_start_times)',
            'off_start': WIN_START,
            'off_end': WIN_END,
            'n_timepoints': N_BINS,
            'session_info': session_info,
        },
    }
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='Process all sessions')
    mode.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    ap.add_argument('--show-processing', action='store_true', help='Save processing plots for up to 2 sessions')
    args = ap.parse_args()

    files = sorted(DATA_DIR.rglob('*.nwb'))
    if args.sample:
        files = files[:2]
    t0 = time.time()
    data = build_dataset(files, show_processing=args.show_processing)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {args.outpicklefile}')
    print(f'sessions={len(data["neural"])} subjects={len(data["subjects"])} total_time_sec={time.time()-t0:.2f}')


if __name__ == '__main__':
    main()
