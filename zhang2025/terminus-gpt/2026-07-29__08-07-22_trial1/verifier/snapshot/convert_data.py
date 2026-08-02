#!/usr/bin/env python3
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

BIN_SIZE_S = 0.02
ALIGN_EVENT = 'stimulus onset'
T_START = -0.2
T_END = 1.0


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true')
    g.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    return ap.parse_args()


def build_time_edges():
    return np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)


def build_time_centers():
    e = build_time_edges()
    return (e[:-1] + e[1:]) / 2


def session_dir_from_row(row):
    return Path('data') / 'one_cache' / str(row['lab']) / 'Subjects' / str(row['subject']) / str(row['date']) / f"{int(row['number']):03d}"


def latest_file(pattern):
    matches = sorted(pattern.parent.glob(pattern.name))
    return matches[-1] if matches else None


def find_trial_table(session_dir):
    alf = session_dir / 'alf'
    cands = sorted(alf.glob('#*/_ibl_trials.table.pqt'))
    if cands:
        return cands[-1]
    p = alf / '_ibl_trials.table.pqt'
    return p if p.exists() else None


def find_motion_energy(session_dir):
    alf = session_dir / 'alf'
    left = sorted(alf.glob('#*/leftCamera.ROIMotionEnergy.npy'))
    right = sorted(alf.glob('#*/rightCamera.ROIMotionEnergy.npy'))
    return (left[-1] if left else None), (right[-1] if right else None)


def find_probe_dirs(session_dir):
    alf = session_dir / 'alf'
    return sorted([p for p in alf.glob('probe*/pykilosort') if p.is_dir()])


def latest_revision_dir(pykilo_dir):
    revs = sorted([p for p in pykilo_dir.glob('#*#') if p.is_dir()])
    return revs[-1] if revs else pykilo_dir


def load_trials(session_dir):
    p = find_trial_table(session_dir)
    if p is None:
        return None
    return pd.read_parquet(p)


def trial_mask(trials):
    need = ['stimOn_times', 'choice', 'probabilityLeft']
    mask = np.ones(len(trials), dtype=bool)
    for c in need:
        if c in trials.columns:
            mask &= np.isfinite(trials[c].to_numpy())
    return mask


def trial_number_in_block(prob_left):
    out = np.zeros(len(prob_left), dtype=np.float32)
    if len(prob_left) == 0:
        return out
    run = 1
    out[0] = 1
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            run += 1
        else:
            run = 1
        out[i] = run
    return out


def map_choice(v):
    # IBL usually: 1=CCW(left), -1=CW(right)
    return 0 if v == 1 else 1 if v == -1 else np.nan


def map_prior(v):
    if np.isclose(v, 0.2):
        return 0
    if np.isclose(v, 0.5):
        return 1
    if np.isclose(v, 0.8):
        return 2
    return np.nan


def discretize_three_bins(x):
    x = np.asarray(x, dtype=np.float32)
    valid = np.isfinite(x)
    out = np.zeros(x.shape, dtype=np.int64)
    if valid.sum() == 0:
        return out
    q1, q2 = np.quantile(x[valid], [1/3, 2/3])
    out[valid] = np.digitize(x[valid], [q1, q2], right=False).astype(np.int64)
    return out


def load_wheel(session_dir):
    alf = session_dir / 'alf'
    pos = alf / '_ibl_wheel.position.npy'
    ts = alf / '_ibl_wheel.timestamps.npy'
    if not (pos.exists() and ts.exists()):
        return None, None
    return np.load(ts), np.load(pos)


def wheel_speed(ts, pos):
    if ts is None or pos is None or len(ts) < 2:
        return None, None
    dt = np.diff(ts)
    dp = np.diff(pos)
    good = dt > 0
    v = np.zeros_like(pos, dtype=np.float32)
    mids = ts[:-1][good] + dt[good] / 2
    vv = np.abs(dp[good] / dt[good]).astype(np.float32)
    return mids, vv


def load_motion_energy(session_dir):
    left_p, right_p = find_motion_energy(session_dir)
    vals = []
    for p in [left_p, right_p]:
        if p is not None and p.exists():
            vals.append(np.load(p).astype(np.float32))
    if not vals:
        return None
    n = min(len(v) for v in vals)
    arr = np.mean(np.vstack([v[:n] for v in vals]), axis=0)
    return arr


def guess_motion_timestamps(session_dir, n):
    # Fallback: use right camera features if present; otherwise spread across session interval.
    alf = session_dir / 'alf'
    feats = sorted(alf.glob('_ibl_*Camera.features.pqt'))
    if feats:
        try:
            df = pd.read_parquet(feats[0])
            for c in df.columns:
                if 'times' in c.lower() or 'timestamp' in c.lower():
                    x = df[c].to_numpy()
                    if len(x) >= n:
                        return x[:n]
        except Exception:
            pass
    trials = load_trials(session_dir)
    if trials is not None and 'stimOn_times' in trials.columns:
        t0 = np.nanmin(trials['stimOn_times'].to_numpy()) + T_START
        t1 = np.nanmax(trials['stimOn_times'].to_numpy()) + T_END
        return np.linspace(t0, t1, n, dtype=np.float32)
    return np.arange(n, dtype=np.float32) * BIN_SIZE_S


def load_spikes_and_regions(session_dir):
    probe_dirs = find_probe_dirs(session_dir)
    all_times = []
    all_clusters = []
    region_names = []
    offset = 0
    for pd_ in probe_dirs:
        base = latest_revision_dir(pd_)
        st = base / 'spikes.times.npy'
        sc = base / 'spikes.clusters.npy'
        cm = base / 'clusters.metrics.pqt'
        ch = base / 'clusters.channels.npy'
        if not (st.exists() and sc.exists()):
            continue
        spikes_t = np.load(st)
        spikes_c = np.load(sc)
        good_ids = None
        if cm.exists():
            m = pd.read_parquet(cm)
            cols = {c.lower(): c for c in m.columns}
            if 'label' in cols:
                good_ids = np.flatnonzero(m[cols['label']].to_numpy() == 1)
            elif 'ks2_label' in cols:
                good_ids = np.flatnonzero(m[cols['ks2_label']].astype(str).str.lower().eq('good').to_numpy())
        if good_ids is None:
            good_ids = np.unique(spikes_c)
        keep = np.isin(spikes_c, good_ids)
        spikes_t = spikes_t[keep]
        spikes_c = spikes_c[keep]
        uniq = np.array(sorted(np.unique(spikes_c)))
        remap = {cid: i + offset for i, cid in enumerate(uniq)}
        spikes_c = np.array([remap[c] for c in spikes_c], dtype=np.int64)
        offset += len(uniq)
        all_times.append(spikes_t)
        all_clusters.append(spikes_c)
        region_names.extend(['unknown'] * len(uniq))
    if not all_times:
        return None, None, None
    return np.concatenate(all_times), np.concatenate(all_clusters), region_names


def bin_signal(ts, values, centers):
    if ts is None or values is None or len(ts) == 0:
        return np.full(len(centers), np.nan, dtype=np.float32)
    edges = np.concatenate([[centers[0] - BIN_SIZE_S / 2], centers + BIN_SIZE_S / 2])
    out = np.full(len(centers), np.nan, dtype=np.float32)
    idx = np.digitize(ts, edges) - 1
    for i in range(len(centers)):
        m = idx == i
        if np.any(m):
            out[i] = np.nanmean(values[m])
    return out


def bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, stim_time):
    edges = stim_time + build_time_edges()
    mat = np.zeros((n_neurons, len(edges) - 1), dtype=np.float32)
    lo = np.searchsorted(spike_times, edges[0], side='left')
    hi = np.searchsorted(spike_times, edges[-1], side='right')
    st = spike_times[lo:hi]
    sc = spike_clusters[lo:hi]
    if len(st) == 0:
        return mat
    tbin = np.digitize(st, edges) - 1
    good = (tbin >= 0) & (tbin < mat.shape[1])
    st = st[good]
    sc = sc[good]
    tbin = tbin[good]
    np.add.at(mat, (sc, tbin), 1)
    return mat


def load_session(row):
    session_dir = session_dir_from_row(row)
    if not session_dir.exists():
        return None
    trials = load_trials(session_dir)
    if trials is None or 'stimOn_times' not in trials.columns:
        return None
    mask = trial_mask(trials)
    trials = trials.loc[mask].reset_index(drop=True)
    if len(trials) < 2:
        return None
    spike_times, spike_clusters, region_names = load_spikes_and_regions(session_dir)
    if spike_times is None:
        return None
    n_neurons = len(region_names)
    centers = build_time_centers().astype(np.float32)
    prob_left = trials['probabilityLeft'].to_numpy(dtype=np.float32)
    trial_in_block = trial_number_in_block(prob_left)
    choice = np.array([map_choice(v) for v in trials['choice'].to_numpy()], dtype=np.float32)
    prior = np.array([map_prior(v) for v in prob_left], dtype=np.float32)

    wheel_ts, wheel_pos = load_wheel(session_dir)
    wheel_v_ts, wheel_v = wheel_speed(wheel_ts, wheel_pos)
    me = load_motion_energy(session_dir)
    me_ts = guess_motion_timestamps(session_dir, len(me)) if me is not None else None

    neural_trials, input_trials, output_trials = [], [], []
    all_wheel_cont, all_me_cont = [], []
    tmp_wheel, tmp_me = [], []
    stim_times = trials['stimOn_times'].to_numpy(dtype=np.float64)
    for i, st in enumerate(stim_times):
        neural_trials.append(bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, st))
        input_trials.append(np.vstack([centers, np.full_like(centers, trial_in_block[i])]).astype(np.float32))
        ws = bin_signal(wheel_v_ts, wheel_v, st + centers) if wheel_v is not None else np.full(len(centers), np.nan, dtype=np.float32)
        ms = bin_signal(me_ts, me, st + centers) if me is not None else np.full(len(centers), np.nan, dtype=np.float32)
        tmp_wheel.append(ws)
        tmp_me.append(ms)
        all_wheel_cont.append(ws[np.isfinite(ws)])
        all_me_cont.append(ms[np.isfinite(ms)])
    all_wheel_cont = np.concatenate(all_wheel_cont) if any(len(x) for x in all_wheel_cont) else np.array([], dtype=np.float32)
    all_me_cont = np.concatenate(all_me_cont) if any(len(x) for x in all_me_cont) else np.array([], dtype=np.float32)
    wheel_bins_all = discretize_three_bins(all_wheel_cont) if len(all_wheel_cont) else np.array([], dtype=np.float32)
    me_bins_all = discretize_three_bins(all_me_cont) if len(all_me_cont) else np.array([], dtype=np.float32)
    # session-wise thresholds for reproducibility
    wq = np.quantile(all_wheel_cont, [1/3, 2/3]) if len(all_wheel_cont) else [0, 0]
    mq = np.quantile(all_me_cont, [1/3, 2/3]) if len(all_me_cont) else [0, 0]
    valid_trial_keep = []
    for i in range(len(trials)):
        if not np.isfinite(choice[i]) or not np.isfinite(prior[i]):
            continue
        wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0] if np.ndim(wq) else 0.0), wq, right=False).astype(np.int64)
        mb = np.digitize(np.nan_to_num(tmp_me[i], nan=mq[0] if np.ndim(mq) else 0.0), mq, right=False).astype(np.int64)
        output_trials.append(np.vstack([
            np.full(centers.shape, int(choice[i]), dtype=np.int64),
            np.full(centers.shape, int(prior[i]), dtype=np.int64),
            wb,
            mb,
        ]).astype(np.int64))
        valid_trial_keep.append(i)
    neural_trials = [neural_trials[i] for i in valid_trial_keep]
    input_trials = [input_trials[i] for i in valid_trial_keep]
    subject = str(row['subject'])
    region_idx = np.zeros(n_neurons, dtype=np.int64)
    return neural_trials, input_trials, output_trials, subject, region_names, region_idx


def build_dataset(sample=False):
    sessions = pd.read_parquet('data/one_cache/Brainwidemap/sessions.pqt')
    all_neural, all_input, all_output = [], [], []
    subjects, subject_map, subject_idx = [], {}, []
    brain_regions = ['unknown']
    brain_region_idx = []
    n_target = 2 if sample else len(sessions)
    for i, (_, row) in enumerate(sessions.iterrows()):
        if sample and len(all_neural) >= 2:
            break
        log(f'processing session {i+1}/{len(sessions)}: {row["lab"]}/{row["subject"]}/{row["date"]}/{int(row["number"]):03d}')
        loaded = load_session(row)
        if loaded is None:
            log('  skipped: missing required data')
            continue
        neural, inp, out, subj, region_names, region_idx = loaded
        all_neural.append(neural)
        all_input.append(inp)
        all_output.append(out)
        if subj not in subject_map:
            subject_map[subj] = len(subjects)
            subjects.append(subj)
        subject_idx.append(subject_map[subj])
        brain_region_idx.append(region_idx)
    return {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subjects,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_since_stimulus_onset', 'trial_number_in_block'],
        'output_names': ['choice', 'prior_probability_left', 'wheel_speed_bin', 'whisker_motion_energy_bin'],
        'output_values': [
            ['left', 'right'],
            ['0.2', '0.5', '0.8'],
            ['low', 'mid', 'high'],
            ['low', 'mid', 'high'],
        ],
        'metadata': {
            'task_description': 'IBL brain-wide map ephys sessions aligned to stimulus onset; decode choice, block prior, wheel speed bin, whisker motion energy bin.',
            'time_bin_size': 20.0,
            'temporal_alignment_event': ALIGN_EVENT,
            'off_start': T_START,
            'off_end': T_END,
        }
    }


def main():
    args = parse_args()
    if not args.full and not args.sample:
        args.full = True
    data = build_dataset(sample=args.sample)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    log(f'wrote {args.outpicklefile}')
    log(f"sessions={len(data['neural'])} subjects={len(data['subjects'])}")


if __name__ == '__main__':
    main()
