#!/usr/bin/env python3
import argparse
import math
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.behavior_ophys_experiment import BehaviorOphysExperiment


DATASET_ROOT = Path('data/visual-behavior-ophys-1.1.0')
PROJECT_METADATA = DATASET_ROOT / 'project_metadata'


OUTPUT_NAMES = [
    'image_identity',
    'image_change',
    'running_speed_bin',
    'pupil_diameter_bin',
    'trial_outcome',
]

TRIAL_OUTCOME_VALUES = ['hit', 'miss', 'false_alarm', 'correct_reject']
RUN_BIN_VALUES = [f'bin_{i}' for i in range(5)]
PUPIL_BIN_VALUES = [f'bin_{i}' for i in range(5)]
IMAGE_CHANGE_VALUES = ['no_change', 'change']


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='Process all sessions (default)')
    mode.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    ap.add_argument('--show-processing', action='store_true', help='Save processing plots for up to 2 sessions')
    return ap.parse_args()


def list_nwb_files():
    files = sorted(DATASET_ROOT.rglob('*.nwb'))
    if not files:
        raise FileNotFoundError(f'No NWB files found under {DATASET_ROOT}')
    return files


def choose_files(sample=False):
    files = list_nwb_files()
    return files[:2] if sample else files


def get_trial_outcome(row):
    if bool(row.get('hit', False)):
        return 0
    if bool(row.get('miss', False)):
        return 1
    if bool(row.get('false_alarm', False)):
        return 2
    if bool(row.get('correct_reject', False)):
        return 3
    return None


def valid_trials_df(trials):
    trials = trials.copy()
    keep = (trials['go'].fillna(False) | trials['catch'].fillna(False))
    keep &= ~trials['aborted'].fillna(False)
    keep &= ~trials['auto_rewarded'].fillna(False)
    trials = trials.loc[keep].copy()
    trials['trial_outcome_idx'] = trials.apply(get_trial_outcome, axis=1)
    trials = trials.loc[trials['trial_outcome_idx'].notna()].copy()
    return trials


def extract_events_matrix(events_df):
    event_col = 'events'
    if event_col not in events_df.columns:
        raise KeyError(f'Expected {event_col} column in events dataframe')
    arrs = [np.asarray(x, dtype=np.float32) for x in events_df[event_col].values]
    lengths = {a.shape[0] for a in arrs}
    if len(lengths) != 1:
        raise ValueError(f'Inconsistent event trace lengths: {lengths}')
    return np.stack(arrs, axis=0)


def build_image_labels_for_trial(trial_timestamps, stim_df, image_to_idx):
    image_idx = np.full(trial_timestamps.shape, -1, dtype=np.int64)
    image_change = np.zeros(trial_timestamps.shape, dtype=np.int64)
    for _, srow in stim_df.iterrows():
        start = float(srow['start_time'])
        end = float(srow['end_time'])
        mask = (trial_timestamps >= start) & (trial_timestamps < end)
        if not np.any(mask):
            continue
        img_name = srow.get('image_name', None)
        omitted_val = srow.get('omitted', False)
        omitted = False if pd.isna(omitted_val) else bool(omitted_val)
        if (img_name is not None) and (not omitted) and (str(img_name) != 'omitted'):
            image_idx[mask] = image_to_idx.setdefault(str(img_name), len(image_to_idx))
        is_change_val = srow.get('is_change', False)
        image_change[mask] = int(False if pd.isna(is_change_val) else bool(is_change_val))
    return image_idx, image_change


def interp_to_ophys(src_t, src_v, dst_t):
    src_t = np.asarray(src_t, dtype=np.float64)
    src_v = np.asarray(src_v, dtype=np.float64)
    dst_t = np.asarray(dst_t, dtype=np.float64)
    good = np.isfinite(src_t) & np.isfinite(src_v)
    if good.sum() < 2:
        return np.full(dst_t.shape, np.nan, dtype=np.float32)
    src_t = src_t[good]
    src_v = src_v[good]
    order = np.argsort(src_t)
    src_t = src_t[order]
    src_v = src_v[order]
    return np.interp(dst_t, src_t, src_v, left=np.nan, right=np.nan).astype(np.float32)


def pupil_diameter_series(eye_tracking_df):
    et = eye_tracking_df.copy()
    blink = et['likely_blink'].fillna(False).to_numpy(dtype=bool) if 'likely_blink' in et.columns else np.zeros(len(et), dtype=bool)
    if 'pupil_area' in et.columns:
        area = et['pupil_area'].to_numpy(dtype=np.float64)
        diam = 2.0 * np.sqrt(area / math.pi)
    elif 'pupil_width' in et.columns and 'pupil_height' in et.columns:
        diam = np.sqrt(et['pupil_width'].to_numpy(dtype=np.float64) * et['pupil_height'].to_numpy(dtype=np.float64))
    else:
        raise KeyError('No pupil area/width-height columns available')
    diam[blink] = np.nan
    return et['timestamps'].to_numpy(dtype=np.float64), diam.astype(np.float32)


def compute_bin_edges(values, n_bins=5):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges


def digitize_with_edges(values, edges):
    values = np.asarray(values, dtype=np.float64)
    out = np.digitize(values, edges[1:-1], right=False).astype(np.int64)
    bad = ~np.isfinite(values)
    if np.any(bad):
        finite = np.where(np.isfinite(values))[0]
        fill = int(np.median(out[finite])) if finite.size else 0
        out[bad] = fill
    out = np.clip(out, 0, len(edges) - 2)
    return out


def process_experiment(nwb_path):
    t0 = time.time()
    exp = BehaviorOphysExperiment.from_nwb_path(str(nwb_path))
    meta = exp.metadata
    ophys_t = np.asarray(exp.ophys_timestamps, dtype=np.float64)
    neural_full = extract_events_matrix(exp.events)
    trials = valid_trials_df(exp.trials)
    valid_trial_ids = set(trials.index.tolist())
    trial_outcome_map = trials['trial_outcome_idx'].to_dict()
    trial_go_map = trials['go'].fillna(False).astype(bool).to_dict()
    trial_catch_map = trials['catch'].fillna(False).astype(bool).to_dict()

    stim = exp.stimulus_presentations.copy().sort_values('start_time')
    stim = stim[stim['trials_id'].notna()].copy()
    stim = stim[stim['trials_id'].isin(valid_trial_ids)].copy()
    if 'active' in stim.columns:
        stim = stim[stim['active'].fillna(False)].copy()
    if 'omitted' in stim.columns:
        stim = stim[~stim['omitted'].fillna(False)].copy()
    stim = stim[stim['image_name'].notna()].copy()
    stim = stim[stim['start_time'].notna() & stim['end_time'].notna()].copy()

    stim_ids = stim.index.to_numpy()
    stim_trial_ids = stim['trials_id'].to_numpy()
    stim_start = stim['start_time'].to_numpy(dtype=np.float64)
    stim_end = stim['end_time'].to_numpy(dtype=np.float64)
    stim_image_name = stim['image_name'].astype(str).to_numpy()
    if 'is_change' in stim.columns:
        stim_is_change = stim['is_change'].fillna(False).astype(bool).to_numpy()
    else:
        stim_is_change = np.zeros(len(stim), dtype=bool)

    run_t = exp.running_speed['timestamps'].to_numpy(dtype=np.float64)
    run_v = exp.running_speed['speed'].to_numpy(dtype=np.float32)
    pupil_t, pupil_v = pupil_diameter_series(exp.eye_tracking)

    session = {
        'session_id': int(meta['ophys_experiment_id']),
        'subject': str(meta['mouse_id']),
        'region': str(meta['targeted_structure']),
        'meta': dict(meta),
        'trials': [],
        'running_raw_all': [],
        'pupil_raw_all': [],
    }

    for stim_id, parent_id, start, stop, img_name, is_change in zip(
        stim_ids, stim_trial_ids, stim_start, stim_end, stim_image_name, stim_is_change
    ):
        left = np.searchsorted(ophys_t, start, side='left')
        right = np.searchsorted(ophys_t, stop, side='left')
        if right - left < 2:
            continue
        trial_t = ophys_t[left:right]
        neural = neural_full[:, left:right].astype(np.float32, copy=False)
        run_aligned = interp_to_ophys(run_t, run_v, trial_t)
        pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
        outcome = int(trial_outcome_map[parent_id])
        session['running_raw_all'].append(run_aligned)
        session['pupil_raw_all'].append(pupil_aligned)
        session['trials'].append({
            'trial_id': int(stim_id) if isinstance(stim_id, (int, np.integer)) else len(session['trials']),
            'start_time': float(start),
            'stop_time': float(stop),
            'ophys_timestamps': trial_t.astype(np.float32),
            'neural': neural,
            'image_name': img_name,
            'image_change': np.full(right - left, int(is_change), dtype=np.int64),
            'running_raw': run_aligned,
            'pupil_raw': pupil_aligned,
            'trial_outcome': outcome,
            'go': bool(trial_go_map[parent_id]),
            'catch': bool(trial_catch_map[parent_id]),
        })

    dt = time.time() - t0
    print(f'processed experiment {session["session_id"]} with {len(session["trials"])} image-presentation trials in {dt:.2f}s')
    return session

def make_processing_plot(session, out_png):
    if not session['trials']:
        return
    tr = session['trials'][0]
    t = tr['ophys_timestamps'] - tr['ophys_timestamps'][0]
    fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    axs[0].imshow(tr['neural'], aspect='auto', interpolation='nearest')
    axs[0].set_ylabel('neurons')
    axs[0].set_title(f"session {session['session_id']} first kept trial")
    axs[1].plot(t, tr['running_raw'])
    axs[1].set_ylabel('run')
    axs[2].plot(t, tr['pupil_raw'])
    axs[2].set_ylabel('pupil')
    axs[3].step(t, tr['image_change'], where='post', label='image_change')
    img_plot = np.zeros_like(t)
    axs[3].plot(t, img_plot, '.', ms=2, label=f"image={tr.get('image_name', 'NA')}")
    axs[3].legend(loc='upper right', fontsize=8)
    axs[3].set_ylabel('stim')
    axs[3].set_xlabel('time from trial start (s)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def assemble_dataset(processed_sessions, run_edges, pupil_edges):
    subjects = sorted({s['subject'] for s in processed_sessions})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    brain_regions = sorted({s['region'] for s in processed_sessions})
    region_to_idx = {r: i for i, r in enumerate(brain_regions)}
    image_names = sorted({tr['image_name'] for s in processed_sessions for tr in s['trials']})
    image_to_idx = {name: i for i, name in enumerate(image_names)}
    image_values = image_names

    data = {
        'neural': [],
        'input': [],
        'output': [],
        'subjects': subjects,
        'subject_idx': np.array([subject_to_idx[s['subject']] for s in processed_sessions], dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [],
        'input_names': [],
        'output_names': OUTPUT_NAMES,
        'output_values': [
            image_values,
            IMAGE_CHANGE_VALUES,
            RUN_BIN_VALUES,
            PUPIL_BIN_VALUES,
            TRIAL_OUTCOME_VALUES,
        ],
        'metadata': {
            'task_description': 'Allen Visual Behavior Ophys; decode image identity, image change, running speed bin, pupil diameter bin, and trial outcome from calcium event activity.',
            'time_bin_size': None,
            'temporal_alignment_event': 'native ophys timestamps within each trial defined by SDK trial start/stop times',
            'off_start': 0.0,
            'off_end': None,
            'neural_signal': 'detected calcium events',
            'running_speed_bin_edges': run_edges.tolist(),
            'pupil_bin_edges': pupil_edges.tolist(),
        }
    }

    for s in processed_sessions:
        sess_neural = []
        sess_input = []
        sess_output = []
        if not s['trials']:
            continue
        n_neurons = s['trials'][0]['neural'].shape[0]
        data['brain_region_idx'].append(np.full(n_neurons, region_to_idx[s['region']], dtype=np.int64))
        for tr in s['trials']:
            T = tr['neural'].shape[1]
            run_bin = digitize_with_edges(tr['running_raw'], run_edges)
            pupil_bin = digitize_with_edges(tr['pupil_raw'], pupil_edges)
            out = np.vstack([
                np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
                tr['image_change'],
                run_bin,
                pupil_bin,
                np.full(T, tr['trial_outcome'], dtype=np.int64),
            ]).astype(np.int64)
            sess_neural.append(tr['neural'])
            sess_input.append(np.zeros((0, T), dtype=np.float32))
            sess_output.append(out)
        if len(sess_neural) >= 2:
            data['neural'].append(sess_neural)
            data['input'].append(sess_input)
            data['output'].append(sess_output)
        else:
            data['brain_region_idx'].pop()

    data['subject_idx'] = data['subject_idx'][:len(data['neural'])]
    return data


def main():
    args = parse_args()
    sample = args.sample or not args.full
    files = choose_files(sample=sample)
    print(f'found {len(list_nwb_files())} nwb files; processing {len(files)}')

    processed = []
    all_run = []
    all_pupil = []

    use_parallel = (not sample) and len(files) > 4
    if use_parallel:
        n_workers = min(max((os.cpu_count() or 2) // 2, 2), 8)
        print(f'using multiprocessing with {n_workers} workers')
        with Pool(processes=n_workers) as pool:
            for i, sess in enumerate(pool.imap_unordered(process_experiment, files), start=1):
                print(f'[{i}/{len(files)}] finished session {sess["session_id"]} trials={len(sess["trials"])}')
                if len(sess['trials']) >= 2:
                    processed.append(sess)
                    all_run.extend(sess['running_raw_all'])
                    all_pupil.extend(sess['pupil_raw_all'])
                else:
                    print(f'skipping session {sess["session_id"]}: <2 valid trials')
    else:
        for i, f in enumerate(files, start=1):
            print(f'[{i}/{len(files)}] loading {f}')
            sess = process_experiment(f)
            if len(sess['trials']) >= 2:
                processed.append(sess)
                all_run.extend(sess['running_raw_all'])
                all_pupil.extend(sess['pupil_raw_all'])
            else:
                print(f'skipping session {sess["session_id"]}: <2 valid trials')

    if not processed:
        raise RuntimeError('No sessions with at least 2 valid trials')

    run_edges = compute_bin_edges(np.concatenate(all_run))
    pupil_edges = compute_bin_edges(np.concatenate(all_pupil))
    print('running bin edges:', run_edges)
    print('pupil bin edges:', pupil_edges)

    if args.show_processing:
        for sess in processed[:2]:
            out_png = f'processing_{sess["session_id"]}.png'
            make_processing_plot(sess, out_png)
            print('wrote', out_png)

    data = assemble_dataset(processed, run_edges, pupil_edges)

    n_sessions = len(data['neural'])
    n_trials = sum(len(x) for x in data['neural'])
    n_neurons = sum(sess[0].shape[0] for sess in data['neural'] if sess)
    print(f'final dataset: sessions={n_sessions} trials={n_trials} neurons(sum over sessions)={n_neurons}')

    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print('wrote', args.outpicklefile)


if __name__ == '__main__':
    main()
