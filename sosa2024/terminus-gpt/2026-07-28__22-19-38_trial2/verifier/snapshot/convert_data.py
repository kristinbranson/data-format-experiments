import argparse
import pickle
import time
from pathlib import Path
from collections import Counter

import h5py
import numpy as np
import matplotlib.pyplot as plt


INPUT_NAMES = [
    'time_from_trial_start_s',
    'environment_type',
    'trial_number',
    'previous_trial_outcome',
]

OUTPUT_NAMES = [
    'distance_to_reward_zone_bin',
    'absolute_position_bin',
    'speed_bin',
    'lick',
    'reward_zone_location',
    'reward_outcome',
]

OUTPUT_VALUES = [
    ['lt_-50', '-50_to_-10', '-10_to_lt0', '0', 'gt0_to_10', '10_to_50', 'gt_50'],
    ['bin0', 'bin1', 'bin2', 'bin3', 'bin4'],
    ['lt_2', '2_to_10', '10_to_20', '20_to_40', 'gt_40'],
    ['no', 'yes'],
    ['A', 'B', 'C'],
    ['no', 'yes'],
]


def dec(x):
    if isinstance(x, bytes):
        return x.decode()
    if hasattr(x, 'tolist'):
        x = x.tolist()
    return x




def parse_location_from_identifier(identifier):
    if 'LocationA' in identifier:
        return 0
    if 'LocationB' in identifier:
        return 1
    if 'LocationC' in identifier:
        return 2
    return 0


def parse_env_from_identifier(identifier):
    if 'Env2' in identifier:
        return 1
    return 0

def infer_env_binary(identifier, env_stream):
    vals = np.unique(env_stream[~np.isnan(env_stream)])
    vals = np.sort(vals)
    if len(vals) == 1:
        if 'Env2' in identifier:
            return np.ones_like(env_stream, dtype=np.int64)
        return np.zeros_like(env_stream, dtype=np.int64)
    mapping = {vals[0]: 0, vals[-1]: 1}
    return np.vectorize(lambda z: mapping.get(z, 0))(env_stream).astype(np.int64)


def infer_reward_zone_map(all_values):
    vals = sorted(float(v) for v in np.unique(all_values[~np.isnan(all_values)]))
    return {v: i for i, v in enumerate(vals[:3])}


def discretize_distance(x):
    out = np.zeros_like(x, dtype=np.int64)
    out[x < -50] = 0
    out[(x >= -50) & (x <= -10)] = 1
    out[(x > -10) & (x < 0)] = 2
    out[x == 0] = 3
    out[(x > 0) & (x <= 10)] = 4
    out[(x > 10) & (x <= 50)] = 5
    out[x > 50] = 6
    return out


def discretize_abs_position(pos, pmin, pmax):
    edges = np.linspace(pmin, pmax, 6)
    bins = np.digitize(pos, edges[1:-1], right=False)
    bins = np.clip(bins, 0, 4)
    return bins.astype(np.int64)


def discretize_speed(speed):
    out = np.zeros_like(speed, dtype=np.int64)
    out[(speed >= 2) & (speed < 10)] = 1
    out[(speed >= 10) & (speed < 20)] = 2
    out[(speed >= 20) & (speed <= 40)] = 3
    out[speed > 40] = 4
    return out


def nearest_sample(values, ts, qts):
    idx = np.searchsorted(ts, qts)
    idx = np.clip(idx, 1, len(ts) - 1)
    left = idx - 1
    choose_left = np.abs(qts - ts[left]) <= np.abs(ts[idx] - qts)
    idx = np.where(choose_left, left, idx)
    return values[idx]


def load_session(path, reward_zone_value_map, show_processing=False):
    t0 = time.time()
    with h5py.File(path, 'r') as h:
        identifier = dec(h['identifier'][()])
        subject = dec(h['general/subject/subject_id'][()])
        beh = h['processing/behavior/BehavioralTimeSeries']

        pos = np.array(beh['position/data'], dtype=np.float32)
        pos_t = np.array(beh['position/timestamps'], dtype=np.float64)
        speed = np.array(beh['speed/data'], dtype=np.float32)
        lick = np.array(beh['lick/data'], dtype=np.float32)
        env = np.array(beh['environment/data'], dtype=np.float32)
        rz = np.array(beh['reward_zone/data'], dtype=np.float32)
        trial_num = np.array(beh['trial number/data'], dtype=np.int64)
        trial_start = np.array(beh['trial_start/data'], dtype=np.float32)
        scanning = np.array(beh['scanning/data'], dtype=np.float32)
        reward_event = np.array(beh['Reward/data'], dtype=np.float32)
        reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)

        dt = float(np.median(np.diff(pos_t)))

        deconv_planes = []
        plane_names = sorted(h['processing/ophys/Deconvolved'].keys())
        for plane in plane_names:
            arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
            deconv_planes.append(arr)
        neural_full = np.concatenate(deconv_planes, axis=1)

        env_binary = infer_env_binary(identifier, env)
        env_from_identifier = parse_env_from_identifier(identifier)
        session_reward_location = parse_location_from_identifier(identifier)
        trial_ids = np.unique(trial_num)
        trial_ids = trial_ids[trial_ids >= 0]

        reward_zone_centers = {}
        for zval in np.unique(rz[~np.isnan(rz)]):
            mask = rz == zval
            if np.any(mask):
                reward_zone_centers[zval] = float(np.nanmedian(pos[mask & (speed > -np.inf)]))

        neural_trials = []
        input_trials = []
        output_trials = []

        prev_outcome = 0
        abs_pos_global_min = float(np.nanmin(pos))
        abs_pos_global_max = float(np.nanmax(pos))

        for tr in trial_ids:
            idx = np.where((trial_num == tr) & (scanning > 0))[0]
            if len(idx) < 2:
                continue
            q_idx = idx
            qts = pos_t[q_idx]
            trial_t0 = qts[0]
            rel_t = (qts - trial_t0).astype(np.float32)

            neural_trial = neural_full[q_idx, :].T.astype(np.float16)
            pos_trial = pos[q_idx]
            speed_trial = speed[q_idx]
            lick_trial = (lick[q_idx] > 0).astype(np.int64)
            env_trial = env_binary[q_idx]
            rz_trial = rz[q_idx]

            rz_vals = rz_trial[~np.isnan(rz_trial)]
            if len(rz_vals) == 0:
                continue
            rz_mode = float(Counter(rz_vals.tolist()).most_common(1)[0][0])
            rz_loc = session_reward_location
            rz_center = reward_zone_centers.get(rz_mode, float(np.nanmedian(pos_trial)))
            dist = pos_trial - rz_center
            dist_bin = discretize_distance(dist)

            abs_pos_bin = discretize_abs_position(pos_trial, abs_pos_global_min, abs_pos_global_max)
            speed_bin = discretize_speed(speed_trial)

            t_lo, t_hi = qts[0], qts[-1]
            reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
            outcome = int(np.any(reward_in_trial > 0))

            inp = np.vstack([
                rel_t,
                np.full(len(q_idx), float(env_from_identifier), dtype=np.float32),
                np.full(len(q_idx), float(tr), dtype=np.float32),
                np.full(len(q_idx), float(prev_outcome), dtype=np.float32),
            ]).astype(np.float32)

            inp = inp.astype(np.float32)

            out = np.vstack([
                dist_bin,
                abs_pos_bin,
                speed_bin,
                lick_trial,
                np.full(len(q_idx), rz_loc, dtype=np.int64),
                np.full(len(q_idx), outcome, dtype=np.int64),
            ]).astype(np.int16)

            neural_trials.append(neural_trial)
            input_trials.append(inp)
            output_trials.append(out)
            prev_outcome = outcome

        brain_region_idx = np.zeros(neural_full.shape[1], dtype=np.int16)

        if show_processing:
            fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
            axs[0].plot(pos_t[:2000], pos[:2000])
            axs[0].set_title(f'{path.name} position')
            axs[1].plot(pos_t[:2000], speed[:2000])
            axs[1].set_title('speed')
            axs[2].plot(pos_t[:2000], lick[:2000])
            axs[2].set_title('lick')
            axs[3].imshow(neural_full[:1000, : min(100, neural_full.shape[1])].T, aspect='auto', interpolation='nearest')
            axs[3].set_title('deconvolved neural sample')
            fig.tight_layout()
            fig.savefig(f'processing_{path.stem}.png', dpi=150)
            plt.close(fig)

    info = {
        'subject': subject,
        'identifier': identifier,
        'brain_region_idx': brain_region_idx,
        'dt_s': dt,
        'n_trials': len(neural_trials),
        'load_time_s': time.time() - t0,
    }
    return neural_trials, input_trials, output_trials, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true')
    g.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()

    files = sorted(Path('data').rglob('*.nwb'))
    if args.sample:
        files = files[:2]

    all_rz = []
    for f in files:
        with h5py.File(f, 'r') as h:
            all_rz.append(np.array(h['processing/behavior/BehavioralTimeSeries/reward_zone/data'], dtype=np.float32))
    reward_zone_value_map = infer_reward_zone_map(np.concatenate(all_rz))

    sessions_neural = []
    sessions_input = []
    sessions_output = []
    subject_names = []
    subject_idx = []
    brain_regions = ['CA1']
    brain_region_idx = []
    session_info = []
    dts = []

    for i, f in enumerate(files):
        print(f'[{i+1}/{len(files)}] loading {f}', flush=True)
        neural_trials, input_trials, output_trials, info = load_session(
            f, reward_zone_value_map, show_processing=args.show_processing and i < 2
        )
        if len(neural_trials) < 2:
            print(f'  skipping {f} because <2 valid trials')
            continue
        if info['subject'] not in subject_names:
            subject_names.append(info['subject'])
        subject_idx.append(subject_names.index(info['subject']))
        sessions_neural.append(neural_trials)
        sessions_input.append(input_trials)
        sessions_output.append(output_trials)
        brain_region_idx.append(info['brain_region_idx'])
        session_info.append({'file': str(f), 'identifier': info['identifier'], 'n_trials': info['n_trials']})
        dts.append(info['dt_s'])
        print(f"  kept {len(neural_trials)} trials, dt={info['dt_s']:.6f}s, load_time={info['load_time_s']:.2f}s", flush=True)

    data = {
        'neural': sessions_neural,
        'input': sessions_input,
        'output': sessions_output,
        'subjects': subject_names,
        'subject_idx': np.asarray(subject_idx, dtype=np.int16),
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': 'Trial-aligned hippocampal calcium decoder dataset for environment, reward, position, speed, and licking variables.',
            'time_bin_size': float(np.median(dts) * 1000.0) if dts else None,
            'temporal_alignment_event': 'start of trial',
            'off_start': 0.0,
            'off_end': None,
            'session_info': session_info,
            'source_signal': 'deconvolved calcium activity',
        }
    }

    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {args.outpicklefile}')
    print(f'sessions={len(sessions_neural)} subjects={len(subject_names)}')


if __name__ == '__main__':
    main()
