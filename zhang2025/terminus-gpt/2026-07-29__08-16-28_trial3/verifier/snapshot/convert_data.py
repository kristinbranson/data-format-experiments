import argparse
import pickle
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from one.api import ONE


BINSIZE = 0.02
T_START = -0.5
T_END = 1.5
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
TIME_CENTERS = TIME_BINS + BINSIZE / 2
N_BINS = len(TIME_BINS)


def find_latest_file(session_alf: Path, pattern: str):
    matches = sorted(session_alf.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def load_trials_table(session_alf: Path):
    pqt = find_latest_file(session_alf, '#*/_ibl_trials.table.pqt')
    if pqt is None:
        pqt = session_alf / '_ibl_trials.table.pqt'
    if not pqt.exists():
        raise FileNotFoundError(f'No trials table found in {session_alf}')
    return pd.read_parquet(pqt), pqt


def list_session_dirs(data_root: Path):
    sessions = []
    for p in data_root.glob('*/Subjects/*/*/*'):
        if p.is_dir() and p.name.isdigit() and (p / 'alf').exists():
            sessions.append(p)
    return sorted(sessions)


def get_subject_from_session(session_dir: Path):
    return session_dir.parts[-3]


def get_session_id(session_dir: Path):
    return '/'.join(session_dir.parts[-5:])


def choose_sessions(session_dirs, mode):
    valid = []
    for s in session_dirs:
        alf = s / 'alf'
        if find_latest_file(alf, '#*/_ibl_trials.table.pqt') or (alf / '_ibl_trials.table.pqt').exists():
            valid.append(s)
    if mode == 'sample':
        return valid[:2]
    return valid


def trial_number_in_block(prob_left):
    out = np.zeros(len(prob_left), dtype=np.float32)
    if len(prob_left) == 0:
        return out
    cur = prob_left[0]
    c = 0
    for i, p in enumerate(prob_left):
        if i == 0 or p != cur:
            cur = p
            c = 1
        else:
            c += 1
        out[i] = c
    return out


def map_choice(vals):
    vals = np.asarray(vals)
    out = np.full(vals.shape, -1, dtype=np.int64)
    out[vals == 1] = 0   # left
    out[vals == -1] = 1  # right
    return out


def map_prior(vals):
    vals = np.asarray(vals)
    out = np.full(vals.shape, -1, dtype=np.int64)
    out[np.isclose(vals, 0.2)] = 0
    out[np.isclose(vals, 0.5)] = 1
    out[np.isclose(vals, 0.8)] = 2
    return out


def load_spikes_for_session_dir(session_dir):
    spikes_list = []
    clusters_list = []
    for probe_dir in sorted((session_dir / 'alf').glob('probe*/pykilosort')):
        revs = sorted([d for d in probe_dir.iterdir() if d.is_dir() and d.name.startswith('#')])
        src = revs[-1] if revs else probe_dir
        st_p = src / 'spikes.times.npy'
        sc_p = src / 'spikes.clusters.npy'
        metrics_p = src / 'clusters.metrics.pqt'
        if not (st_p.exists() and sc_p.exists() and metrics_p.exists()):
            continue
        st = np.load(st_p)
        sc = np.load(sc_p).astype(int)
        metrics = pd.read_parquet(metrics_p)
        nclu = len(metrics)
        labels = metrics['label'].to_numpy() if 'label' in metrics.columns else np.ones(nclu)
        cl_chan_p = src / 'clusters.channels.npy'
        ch_brain_p = src / 'channels.brainLocationIds_ccf_2017.npy'
        acr = np.array(['void'] * nclu, dtype=object)
        if cl_chan_p.exists() and ch_brain_p.exists():
            cluster_channels = np.load(cl_chan_p).astype(int)
            channel_region_ids = np.load(ch_brain_p).astype(int)
            valid_ch = (cluster_channels >= 0) & (cluster_channels < len(channel_region_ids))
            region_ids = np.full(nclu, 0, dtype=int)
            region_ids[valid_ch] = channel_region_ids[cluster_channels[valid_ch]]
            acr = np.array([str(x) for x in region_ids], dtype=object)
        good = labels >= 1
        offset = sum(len(c['acronym']) for c in clusters_list)
        remap = -np.ones(nclu, dtype=int)
        remap[good] = np.arange(good.sum()) + offset
        keep_spk = (sc >= 0) & (sc < nclu)
        sc2 = sc[keep_spk]
        st2 = st[keep_spk]
        keep2 = good[sc2]
        spikes_list.append((st2[keep2], remap[sc2[keep2]]))
        clusters_list.append({'acronym': acr[good]})
    if not spikes_list:
        return None, None
    spike_times = np.concatenate([x[0] for x in spikes_list])
    spike_clusters = np.concatenate([x[1] for x in spikes_list])
    order = np.argsort(spike_times)
    spike_times = spike_times[order]
    spike_clusters = spike_clusters[order]
    acronyms = np.concatenate([c['acronym'] for c in clusters_list])
    return {'times': spike_times, 'clusters': spike_clusters}, {'acronym': acronyms}


def bin_spikes(spike_times, spike_clusters, n_neurons, stim_on):
    mats = []
    for t0 in stim_on:
        edges = t0 + TIME_BINS
        out = np.zeros((n_neurons, N_BINS), dtype=np.float32)
        lo = np.searchsorted(spike_times, edges[0], side='left')
        hi = np.searchsorted(spike_times, t0 + T_END, side='left')
        ts = spike_times[lo:hi] - t0
        cl = spike_clusters[lo:hi]
        bins = np.floor((ts - T_START) / BINSIZE).astype(int)
        m = (bins >= 0) & (bins < N_BINS) & (cl >= 0) & (cl < n_neurons)
        np.add.at(out, (cl[m], bins[m]), 1)
        mats.append(out)
    return mats


def load_wheel(session_alf: Path):
    tp = session_alf / '_ibl_wheel.timestamps.npy'
    pp = session_alf / '_ibl_wheel.position.npy'
    if not tp.exists() or not pp.exists():
        return None, None
    return np.load(tp), np.load(pp)


def interp_wheel_speed(timestamps, position, stim_on):
    timestamps = np.asarray(timestamps, dtype=float)
    position = np.asarray(position, dtype=float)
    if len(timestamps) < 2 or len(position) < 2:
        return [np.full(N_BINS, np.nan, dtype=np.float32) for _ in stim_on]
    keep = np.isfinite(timestamps) & np.isfinite(position)
    timestamps = timestamps[keep]
    position = position[keep]
    if len(timestamps) < 2:
        return [np.full(N_BINS, np.nan, dtype=np.float32) for _ in stim_on]
    uniq_t, uniq_idx = np.unique(timestamps, return_index=True)
    timestamps = uniq_t
    position = position[uniq_idx]
    if len(timestamps) < 2:
        return [np.full(N_BINS, np.nan, dtype=np.float32) for _ in stim_on]
    vel = np.gradient(position, timestamps)
    trials = []
    for t0 in stim_on:
        x = t0 + TIME_CENTERS
        y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
        trials.append(y.astype(np.float32))
    return trials


def load_motion_energy(session_alf: Path):
    left_t = find_latest_file(session_alf, '#*/_ibl_leftCamera.times.npy')
    right_t = find_latest_file(session_alf, '#*/_ibl_rightCamera.times.npy')
    left_me = find_latest_file(session_alf, '#*/leftCamera.ROIMotionEnergy.npy')
    right_me = find_latest_file(session_alf, '#*/rightCamera.ROIMotionEnergy.npy')
    streams = []
    for tp, mp in [(left_t, left_me), (right_t, right_me)]:
        if tp is not None and mp is not None and tp.exists() and mp.exists():
            streams.append((np.load(tp), np.load(mp).astype(np.float32)))
    if not streams:
        return None
    return streams


def interp_motion_energy(streams, stim_on):
    trials = []
    for t0 in stim_on:
        x = t0 + TIME_CENTERS
        ys = []
        for ts, me in streams:
            ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
        arr = np.stack(ys, axis=0)
        valid = np.isfinite(arr)
        denom = valid.sum(axis=0)
        summed = np.where(valid, arr, 0.0).sum(axis=0)
        y = np.divide(summed, denom, out=np.full(arr.shape[1], np.nan, dtype=np.float32), where=denom > 0)
        trials.append(y.astype(np.float32))
    return trials


def discretize_tertiles(list_of_arrays):
    allv = np.concatenate([x[np.isfinite(x)] for x in list_of_arrays if np.isfinite(x).any()])
    q1, q2 = np.quantile(allv, [1/3, 2/3]) if len(allv) else (0.0, 1.0)
    out = []
    for x in list_of_arrays:
        y = np.zeros_like(x, dtype=np.int64)
        y[x > q1] = 1
        y[x > q2] = 2
        y[~np.isfinite(x)] = 0
        out.append(y)
    return out, (float(q1), float(q2))


def make_plot(session_id, neural_trials, wheel_trials, whisk_trials, outpath):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].imshow(neural_trials[0], aspect='auto', interpolation='nearest')
    axs[0].set_title(f'{session_id} trial0 spikes')
    axs[1].plot(TIME_CENTERS, wheel_trials[0])
    axs[1].set_title('wheel speed trial0')
    axs[2].plot(TIME_CENTERS, whisk_trials[0])
    axs[2].set_title('whisker motion energy trial0')
    axs[2].set_xlabel('time from stim onset (s)')
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true')
    g.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()

    mode = 'sample' if args.sample else 'full'
    t0 = time.time()
    data_root = Path('data/one_cache')

    session_dirs = choose_sessions(list_session_dirs(data_root), mode)
    print(f'Found {len(session_dirs)} candidate sessions for mode={mode}')

    subjects = []
    subject_to_idx = {}
    brain_regions = []
    region_to_idx = {}

    out = {
        'neural': [], 'input': [], 'output': [],
        'subjects': subjects, 'subject_idx': [],
        'brain_regions': brain_regions, 'brain_region_idx': [],
        'input_names': ['time_since_stim_onset', 'trial_number_in_block'],
        'output_names': ['choice', 'prior_left_prob', 'wheel_speed_bin', 'whisker_motion_energy_bin'],
        'output_values': [['left', 'right'], ['0.2', '0.5', '0.8'], ['low', 'mid', 'high'], ['low', 'mid', 'high']],
        'metadata': {
            'task_description': 'Decode choice, prior block probability, wheel speed bin, and whisker motion energy bin from neural activity in IBL task sessions.',
            'time_bin_size': BINSIZE * 1000.0,
            'temporal_alignment_event': 'stimulus onset',
            'off_start': T_START,
            'off_end': T_END,
            'source': 'IBL brain-wide map local ONE cache',
        }
    }

    kept = 0
    for si, session_dir in enumerate(session_dirs):
        st = time.time()
        session_alf = session_dir / 'alf'
        try:
            trials_df, trial_path = load_trials_table(session_alf)
            if 'stimOn_times' not in trials_df.columns or 'choice' not in trials_df.columns or 'probabilityLeft' not in trials_df.columns:
                print('skip missing required trial columns', session_dir)
                continue
            stim_on = np.asarray(trials_df['stimOn_times'], dtype=float)
            choice = map_choice(trials_df['choice'].to_numpy())
            prior = map_prior(trials_df['probabilityLeft'].to_numpy())
            trial_in_block = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())
            valid = np.isfinite(stim_on) & (choice >= 0) & (prior >= 0)
            if valid.sum() < 2:
                print('skip too few valid trials', session_dir)
                continue

            rel = session_dir.relative_to(data_root)
            parts = rel.parts
            lab, _, subject, date, number = parts[0], parts[1], parts[2], parts[3], parts[4]
            spikes, clusters = load_spikes_for_session_dir(session_dir)
            if spikes is None or len(clusters['acronym']) == 0:
                print('skip no spikes', session_dir)
                continue
            neural_trials = bin_spikes(spikes['times'], spikes['clusters'], len(clusters['acronym']), stim_on[valid])

            wt, wp = load_wheel(session_alf)
            if wt is None:
                print('skip no wheel', session_dir)
                continue
            wheel_trials = interp_wheel_speed(wt, wp, stim_on[valid])

            me_streams = load_motion_energy(session_alf)
            if me_streams is None:
                print('skip no whisker motion energy', session_dir)
                continue
            whisk_trials = interp_motion_energy(me_streams, stim_on[valid])

            wheel_bins, wheel_thr = discretize_tertiles(wheel_trials)
            whisk_bins, whisk_thr = discretize_tertiles(whisk_trials)

            sess_inputs = []
            sess_outputs = []
            for i, tr in enumerate(np.where(valid)[0]):
                inp = np.vstack([
                    TIME_CENTERS.astype(np.float32),
                    np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
                ])
                out_trial = np.vstack([
                    np.full(N_BINS, choice[tr], dtype=np.int64),
                    np.full(N_BINS, prior[tr], dtype=np.int64),
                    wheel_bins[i].astype(np.int64),
                    whisk_bins[i].astype(np.int64),
                ])
                sess_inputs.append(inp)
                sess_outputs.append(out_trial)

            subj = subject
            if subj not in subject_to_idx:
                subject_to_idx[subj] = len(subjects)
                subjects.append(subj)
            out['subject_idx'].append(subject_to_idx[subj])

            reg_idx = []
            for reg in clusters['acronym']:
                if reg not in region_to_idx:
                    region_to_idx[reg] = len(brain_regions)
                    brain_regions.append(str(reg))
                reg_idx.append(region_to_idx[reg])

            out['neural'].append(neural_trials)
            out['input'].append(sess_inputs)
            out['output'].append(sess_outputs)
            out['brain_region_idx'].append(np.asarray(reg_idx, dtype=np.int64))

            if args.show_processing and kept < 2:
                make_plot(get_session_id(session_dir), neural_trials, wheel_trials, whisk_trials, f'processing_{kept}.png')

            kept += 1
            print(f'kept session {kept}: {session_dir} trials={len(neural_trials)} neurons={len(reg_idx)} wheel_thr={wheel_thr} whisk_thr={whisk_thr} dt={time.time()-st:.2f}s')
        except Exception as e:
            print('skip session due to error', session_dir, repr(e))

    out['subject_idx'] = np.asarray(out['subject_idx'], dtype=np.int64)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(out, f)
    print(f'Saved {args.outpicklefile} with {len(out["neural"])} sessions in {time.time()-t0:.2f}s')


if __name__ == '__main__':
    main()
