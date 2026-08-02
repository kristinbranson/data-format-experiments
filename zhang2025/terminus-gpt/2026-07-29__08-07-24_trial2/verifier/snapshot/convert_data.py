#!/usr/bin/env python3
import argparse
import pickle
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpickle')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='Process all sessions')
    mode.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    ap.add_argument('--show-processing', action='store_true', help='Save processing plots for up to 2 sessions')
    return ap.parse_args()


def find_sessions(base=Path('data/one_cache')):
    trial_tables = sorted(base.rglob('_ibl_trials.table.pqt'))
    sessions = []
    for p in trial_tables:
        sess = p.parent
        while sess.name != 'alf' and sess != sess.parent:
            sess = sess.parent
        if sess.name == 'alf':
            sessions.append(sess.parent)
    out = sorted(set(sessions))
    return out


def load_trials(session_path: Path):
    trial_files = sorted((session_path / 'alf').rglob('_ibl_trials.table.pqt'))
    if not trial_files:
        return None
    # prefer latest versioned file if multiple exist
    trial_file = trial_files[-1]
    return pd.read_parquet(trial_file)


def compute_trial_number_in_block(prob_left):
    prob_left = np.asarray(prob_left)
    out = np.zeros(len(prob_left), dtype=np.float32)
    c = 0
    prev = None
    for i, v in enumerate(prob_left):
        if i == 0 or v != prev:
            c = 1
            prev = v
        else:
            c += 1
        out[i] = c
    return out


def map_choice(choice_vals):
    arr = np.asarray(choice_vals)
    # IBL convention often left=1 right=-1, but confirm from data later; current mapping assumes left=1 -> 0, right=-1 -> 1
    out = np.full(arr.shape, -1, dtype=np.int64)
    out[arr == 1] = 0
    out[arr == -1] = 1
    return out


def map_prior(prob_left):
    m = {0.2: 0, 0.5: 1, 0.8: 2}
    return np.array([m.get(float(x), -1) for x in prob_left], dtype=np.int64)


def load_wheel(session_path: Path):
    alf = session_path / 'alf'
    posf = alf / '_ibl_wheel.position.npy'
    tsf = alf / '_ibl_wheel.timestamps.npy'
    if not (posf.exists() and tsf.exists()):
        return None, None
    return np.load(posf), np.load(tsf)


def load_motion_energy(session_path: Path):
    alf = session_path / 'alf'
    left_me = sorted(alf.rglob('leftCamera.ROIMotionEnergy.npy'))
    right_me = sorted(alf.rglob('rightCamera.ROIMotionEnergy.npy'))
    left_t = sorted(alf.rglob('_ibl_leftCamera.times.npy'))
    right_t = sorted(alf.rglob('_ibl_rightCamera.times.npy'))
    streams = []
    if left_me and left_t:
        streams.append((np.load(left_me[-1]), np.load(left_t[-1]), 'left'))
    if right_me and right_t:
        streams.append((np.load(right_me[-1]), np.load(right_t[-1]), 'right'))
    return streams


def session_probe_dirs(session_path: Path):
    return sorted((session_path / 'alf').glob('probe*'))


def load_curated_spikes(session_path: Path):
    all_times = []
    all_clusters = []
    region_names = []
    neuron_region_idx = []
    offset = 0
    for probe_dir in session_probe_dirs(session_path):
        pks = sorted(probe_dir.rglob('pykilosort'))
        if not pks:
            pks = [probe_dir]
        cand = sorted(probe_dir.rglob('clusters.metrics.pqt'))
        if not cand:
            continue
        metrics = pd.read_parquet(cand[-1])
        chan_file = sorted(probe_dir.rglob('clusters.channels.npy'))[-1]
        clu_channels = np.load(chan_file)
        reg_id_files = sorted(probe_dir.rglob('channels.brainLocationIds_ccf_2017.npy'))
        reg_ids = np.load(reg_id_files[-1]) if reg_id_files else None
        spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
        spike_clusters = np.load(sorted(probe_dir.rglob('spikes.clusters.npy'))[-1])
        # provisional curation using available fields; amplitude criterion may require conversion/field interpretation refinement later
        keep = np.ones(len(metrics), dtype=bool)
        if 'noise_cutoff' in metrics.columns:
            keep &= metrics['noise_cutoff'].to_numpy() < 20
        if 'label' in metrics.columns:
            keep &= metrics['label'].to_numpy() >= 1
        kept_ids = np.where(keep)[0]
        if len(kept_ids) == 0:
            continue
        remap = {old: i + offset for i, old in enumerate(kept_ids)}
        mask = np.isin(spike_clusters, kept_ids)
        sc = spike_clusters[mask]
        st = spike_times[mask]
        sc = np.array([remap[c] for c in sc], dtype=np.int64)
        all_times.append(st)
        all_clusters.append(sc)
        # placeholder region labels by channel id until atlas-name mapping is implemented
        for cid in kept_ids:
            ch = int(clu_channels[cid])
            reg = f'ccf_{int(reg_ids[ch])}' if reg_ids is not None and ch < len(reg_ids) else f'channel_{ch}'
            if reg not in region_names:
                region_names.append(reg)
            neuron_region_idx.append(region_names.index(reg))
        offset += len(kept_ids)
    if not all_times:
        return None, None, None
    return np.concatenate(all_times), np.concatenate(all_clusters), (region_names, np.array(neuron_region_idx, dtype=np.int64))


def bin_spikes_for_trials(spike_times, spike_clusters, stim_on, n_neurons, t0=-0.2, t1=1.0, bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
    trial_mats = []
    for s in stim_on:
        rel = spike_times - s
        mask = (rel >= t0) & (rel < t1)
        rel = rel[mask]
        clu = spike_clusters[mask]
        mat = np.zeros((n_neurons, len(edges) - 1), dtype=np.float32)
        if len(rel):
            tb = np.floor((rel - t0) / bin_size).astype(int)
            good = (tb >= 0) & (tb < mat.shape[1]) & (clu >= 0) & (clu < n_neurons)
            np.add.at(mat, (clu[good], tb[good]), 1)
        trial_mats.append(mat)
    return trial_mats, edges


def interp_to_trial_bins(values, timestamps, stim_on, edges, reducer='linear'):
    centers = (edges[:-1] + edges[1:]) / 2
    out = []
    for s in stim_on:
        t = s + centers
        y = np.interp(t, timestamps, values)
        out.append(y.astype(np.float32))
    return out


def tertile_thresholds(arrays):
    x = np.concatenate([np.asarray(a).ravel() for a in arrays if a is not None and len(a) > 0])
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return 0.0, 0.0
    q1, q2 = np.quantile(x[finite], [1/3, 2/3])
    return float(q1), float(q2)

def discretize_with_thresholds(x, q1, q2):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=np.int64)
    y[x > q1] = 1
    y[x > q2] = 2
    return y

def save_processing_plot(session_id, neural_trial, input_trial, output_trial, outpath):
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axs[0].imshow(neural_trial, aspect='auto', interpolation='nearest')
    axs[0].set_ylabel('neurons')
    axs[0].set_title(session_id)
    axs[1].plot(input_trial[0], label='time_since_stim')
    axs[1].plot(input_trial[1], label='trial_in_block')
    axs[1].legend(loc='upper right', fontsize=8)
    axs[2].plot(output_trial[2], label='wheel_bin')
    axs[2].plot(output_trial[3], label='whisker_bin')
    axs[2].legend(loc='upper right', fontsize=8)
    axs[3].plot(output_trial[0], label='choice')
    axs[3].plot(output_trial[1], label='prior')
    axs[3].legend(loc='upper right', fontsize=8)
    axs[3].set_xlabel('time bin')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def process_session(session_path: Path, show_processing=False):
    trials = load_trials(session_path)
    if trials is None or len(trials) < 2:
        return None
    required = ['stimOn_times', 'choice', 'probabilityLeft']
    if any(c not in trials.columns for c in required):
        return None
    valid = trials['stimOn_times'].notna() & trials['choice'].isin([-1, 1]) & trials['probabilityLeft'].notna()
    trials = trials.loc[valid].reset_index(drop=True)
    if len(trials) < 2:
        return None
    spike_data = load_curated_spikes(session_path)
    if spike_data[0] is None:
        return None
    spike_times, spike_clusters, region_pack = spike_data
    region_names, neuron_region_idx = region_pack
    n_neurons = int(neuron_region_idx.shape[0])
    neural, edges = bin_spikes_for_trials(spike_times, spike_clusters, trials['stimOn_times'].to_numpy(), n_neurons)
    keep_trial = np.array([np.any(m != 0) for m in neural], dtype=bool)
    trials = trials.loc[keep_trial].reset_index(drop=True)
    neural = [m for m, k in zip(neural, keep_trial) if k]
    if len(neural) < 2:
        return None
    centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
    block_trial = compute_trial_number_in_block(trials['probabilityLeft'].to_numpy())
    inputs = []
    choice = map_choice(trials['choice'].to_numpy())
    prior = map_prior(trials['probabilityLeft'].to_numpy())
    wheel_pos, wheel_ts = load_wheel(session_path)
    if wheel_pos is not None:
        order = np.argsort(wheel_ts)
        wheel_ts = np.asarray(wheel_ts)[order]
        wheel_pos = np.asarray(wheel_pos)[order]
        uniq_mask = np.concatenate([[True], np.diff(wheel_ts) > 0])
        wheel_ts = wheel_ts[uniq_mask]
        wheel_pos = wheel_pos[uniq_mask]
        if len(wheel_ts) >= 2:
            dt = np.diff(wheel_ts)
            dp = np.diff(wheel_pos)
            speed_mid = np.abs(dp / dt).astype(np.float32)
            ts_mid = ((wheel_ts[:-1] + wheel_ts[1:]) / 2).astype(np.float64)
            wheel_trials = interp_to_trial_bins(speed_mid, ts_mid, trials['stimOn_times'].to_numpy(), edges)
        else:
            wheel_trials = [np.zeros_like(centers) for _ in range(len(trials))]
    else:
        wheel_trials = [np.zeros_like(centers) for _ in range(len(trials))]
    me_streams = load_motion_energy(session_path)
    if me_streams:
        aligned = []
        for vals, ts, side in me_streams:
            aligned.append(interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges))
        if len(aligned) == 1:
            me_trials = aligned[0]
        else:
            me_trials = [np.mean(np.vstack([aligned[0][i], aligned[1][i]]), axis=0) for i in range(len(trials))]
    else:
        me_trials = [np.zeros_like(centers) for _ in range(len(trials))]
    wheel_q1, wheel_q2 = tertile_thresholds(wheel_trials)
    me_q1, me_q2 = tertile_thresholds(me_trials)
    outputs = []
    for i in range(len(trials)):
        inp = np.vstack([
            centers,
            np.full_like(centers, block_trial[i], dtype=np.float32),
        ]).astype(np.float32)
        out = np.vstack([
            np.full_like(centers, choice[i], dtype=np.int64),
            np.full_like(centers, prior[i], dtype=np.int64),
            discretize_with_thresholds(wheel_trials[i], wheel_q1, wheel_q2),
            discretize_with_thresholds(me_trials[i], me_q1, me_q2),
        ])
        inputs.append(inp)
        outputs.append(out)
        neural[i] = neural[i].astype(np.float32)
    subject = session_path.parts[-3]
    if show_processing and len(neural) > 0:
        save_processing_plot('/'.join(session_path.parts[-4:]), neural[0], inputs[0], outputs[0], f'processing_{subject}_{session_path.parts[-2]}_{session_path.parts[-1]}.png')
    return {
        'session_id': '/'.join(session_path.parts[-4:]),
        'subject': subject,
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'brain_regions': region_names,
        'brain_region_idx': neuron_region_idx,
    }


def build_dataset(processed, bin_size=0.02):
    subjects = sorted({p['subject'] for p in processed})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    brain_regions = sorted({r for p in processed for r in p['brain_regions']})
    region_to_idx = {r: i for i, r in enumerate(brain_regions)}
    data = {
        'neural': [p['neural'] for p in processed],
        'input': [p['input'] for p in processed],
        'output': [p['output'] for p in processed],
        'subjects': subjects,
        'subject_idx': np.array([subject_to_idx[p['subject']] for p in processed], dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [np.array([region_to_idx[p['brain_regions'][i]] for i in p['brain_region_idx']], dtype=np.int64) for p in processed],
        'input_names': ['time_since_stimulus_onset', 'trial_number_in_block'],
        'output_names': ['choice', 'prior_probability_left', 'wheel_speed', 'whisker_motion_energy'],
        'output_values': [
            ['left', 'right'],
            ['0.2', '0.5', '0.8'],
            ['low', 'medium', 'high'],
            ['low', 'medium', 'high'],
        ],
        'metadata': {
            'task_description': 'Decode choice, prior probability of left, wheel speed bin, and whisker motion-energy bin from stimulus-aligned neural activity.',
            'time_bin_size': bin_size * 1000.0,
            'temporal_alignment_event': 'stimulus onset',
            'off_start': -0.2,
            'off_end': 1.0,
        }
    }
    return data


def main():
    args = parse_args()
    t0 = time.time()
    sessions = find_sessions()
    if args.sample:
        sessions = sessions[:2]
    processed = []
    for i, sess in enumerate(sessions, 1):
        st = time.time()
        try:
            p = process_session(sess, show_processing=args.show_processing)
        except Exception as e:
            print(f'[WARN] failed session {sess}: {e}')
            p = None
        if p is not None:
            processed.append(p)
        print(f'processed {i}/{len(sessions)} sessions; kept {len(processed)}; dt={time.time()-st:.2f}s')
    data = build_dataset(processed)
    with open(args.outpickle, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {args.outpickle} with {len(processed)} sessions in {time.time()-t0:.2f}s')


if __name__ == '__main__':
    main()
