#!/usr/bin/env python3
"""Convert Sosa et al. NWB calcium/behavior data to decoder format.

Usage: python -u /app/convert_data.py OUT.pkl [--full|--sample] [--show-processing]
NWB access is exclusively through pynwb.NWBHDF5IO.
"""
from __future__ import annotations
import argparse, gc, pickle, re, time
from pathlib import Path
import numpy as np
from pynwb import NWBHDF5IO

DATA_ROOT = Path('/app/data')
TARGET_RATE = 15.5078125
BIN_MS = 1000.0 / TARGET_RATE
ZONE_START = np.array([80.0, 200.0, 320.0], dtype=np.float32)
ZONE_END = ZONE_START + 50.0


def parse_zone_sequence(identifier: str) -> tuple[int, int]:
    """Return source/destination A/B/C indices parsed from the NWB identifier."""
    name = identifier.rstrip('/').split('/')[-1]
    letters = re.findall(r'(?:Location)?([ABC])', name)
    if '_to_' in name and len(letters) >= 2:
        return ord(letters[-2]) - 65, ord(letters[-1]) - 65
    if letters:
        z = ord(letters[-1]) - 65
        return z, z
    raise ValueError(f'Cannot parse reward location from identifier: {identifier}')


def zone_for_trial(src: int, dst: int, trial_number: int) -> int:
    return src if trial_number < 30 else dst


def distance_classes(position: np.ndarray, zone: int) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance to nearest point in the 50-cm reward-zone interval."""
    lo, hi = float(ZONE_START[zone]), float(ZONE_END[zone])
    d = np.where(position < lo, position - lo,
                 np.where(position > hi, position - hi, 0.0)).astype(np.float32)
    # 0:<-50; 1:[-50,-10); 2:[-10,0); 3:0; 4:(0,10]; 5:(10,50]; 6:>50
    c = np.empty(d.shape, dtype=np.int64)
    c[d < -50] = 0
    c[(d >= -50) & (d < -10)] = 1
    c[(d >= -10) & (d < 0)] = 2
    c[d == 0] = 3
    c[(d > 0) & (d <= 10)] = 4
    c[(d > 10) & (d <= 50)] = 5
    c[d > 50] = 6
    return d, c


def position_classes(position: np.ndarray) -> np.ndarray:
    p = np.clip(position, 0.0, 450.0)
    c = np.zeros(p.shape, dtype=np.int64)
    c[p >= 90] = 1
    c[p >= 180] = 2
    c[p >= 270] = 3
    c[p > 360] = 4
    return c


def speed_classes(speed: np.ndarray) -> np.ndarray:
    v = np.maximum(speed, 0.0)
    c = np.zeros(v.shape, dtype=np.int64)
    c[v >= 2] = 1
    c[v >= 10] = 2
    c[v >= 20] = 3
    c[v > 40] = 4
    return c


def pair_bounds(starts: np.ndarray, ends: np.ndarray) -> list[tuple[int, int]]:
    """Pair each start with the next unused teleport; return half-open [start,end)."""
    pairs, j = [], 0
    for s in starts:
        while j < len(ends) and ends[j] < s:
            j += 1
        if j >= len(ends):
            break
        pairs.append((int(s), int(ends[j])))
        j += 1
    return pairs


def aggregate_behavior(x: np.ndarray, factor: int, mode: str) -> np.ndarray:
    n = (len(x) // factor) * factor
    x = np.asarray(x[:n])
    if factor == 1:
        return x
    x = x.reshape(-1, factor)
    if mode == 'mean':
        return np.nanmean(x, axis=1)
    if mode == 'max':
        return np.nanmax(x, axis=1)
    if mode == 'first':
        return x[:, 0]
    raise ValueError(mode)


def load_neural_trial(series_info, start: int, end: int, factor: int) -> np.ndarray:
    """Read accepted cells from all planes and return cells x common-rate time."""
    planes = []
    n = ((end - start) // factor) * factor
    if n <= 0:
        raise ValueError('Trial has no complete output bins')
    for rs, accepted_cols in series_info:
        # pynwb exposes the NWB Data object; no direct h5py loading is used.
        block = np.asarray(rs.data[start:start+n, :], dtype=np.float32)
        block = block[:, accepted_cols]
        if factor > 1:
            block = block.reshape(n // factor, factor, block.shape[1]).sum(axis=1)
        planes.append(block.T)
    return np.concatenate(planes, axis=0).astype(np.float32, copy=False)


def process_session(path: Path):
    t0 = time.time()
    with NWBHDF5IO(str(path), 'r') as io:
        nwb = io.read()
        subject = str(nwb.subject.subject_id)
        bts = nwb.processing['behavior']['BehavioralTimeSeries'].time_series
        deconv = nwb.processing['ophys']['Deconvolved'].roi_response_series

        series_info = []
        rates = []
        for key in sorted(deconv.keys()):
            rs = deconv[key]
            region = np.asarray(rs.rois.data[:], dtype=np.int64)
            labels = np.asarray(rs.rois.table['iscell'][:])
            keep = labels[region, 0] > 0.5
            accepted_local_cols = np.flatnonzero(keep)
            if not len(accepted_local_cols):
                raise ValueError(f'{path}: no accepted cells in {key}')
            series_info.append((rs, accepted_local_cols))
            rates.append(float(rs.rate))
        if not np.allclose(rates, rates[0]):
            raise ValueError(f'{path}: plane rates differ: {rates}')
        # NWB two-plane RoiResponseSeries metadata report 31.015625 Hz, but their
        # rows are already one-to-one with behavior timestamps at 15.5078125 Hz.
        # Use the synchronized behavior clock, not the inconsistent series rate.
        pos_clock = np.asarray(bts['position'].timestamps[:], dtype=np.float64)
        effective_rate = 1.0 / float(np.median(np.diff(pos_clock)))
        if not np.isclose(effective_rate, TARGET_RATE, rtol=2e-3):
            raise ValueError(f'{path}: unsupported behavior-clock rate {effective_rate}')
        factor = 1

        # Behavior arrays are small and synchronized to processed neural samples.
        names = ['trial_start', 'teleport', 'trial number', 'environment',
                 'position', 'speed', 'lick', 'reward_zone']
        beh = {k: np.asarray(bts[k].data[:]) for k in names}
        common_n = min([len(v) for v in beh.values()] +
                       [int(rs.data.shape[0]) for rs, _ in series_info])
        starts = np.flatnonzero(beh['trial_start'][:common_n] > 0)
        ends = np.flatnonzero(beh['teleport'][:common_n] > 0)
        pairs = pair_bounds(starts, ends)
        if len(pairs) != len(starts) or len(starts) != len(ends):
            raise ValueError(f'{path}: unmatched trial markers {len(starts)}/{len(ends)}/{len(pairs)}')

        pos_ts = np.asarray(bts['position'].timestamps[:common_n], dtype=np.float64)
        reward_ts = np.asarray(bts['Reward'].timestamps[:], dtype=np.float64)
        src_zone, dst_zone = parse_zone_sequence(nwb.identifier)
        is_switch = src_zone != dst_zone

        neural_trials, input_trials, output_trials = [], [], []
        outcomes, trial_info = [], []
        lick_artifacts = 0
        for qi, (s, e) in enumerate(pairs):
            trial_num = int(round(float(beh['trial number'][s])))
            if trial_num < 0:
                raise ValueError(f'{path}: negative trial number at start {s}')
            # Reward timestamps are exactly aligned, but timestamp comparison is robust.
            rewarded = int(np.any((reward_ts >= pos_ts[s]) & (reward_ts < pos_ts[e])))
            outcomes.append(rewarded)
            raw_lick = beh['lick'][s:e]
            lick_artifact = bool(np.mean(raw_lick > 2) > 0.30)
            lick_artifacts += int(lick_artifact)

            n_complete = ((e - s) // factor) * factor
            neural = load_neural_trial(series_info, s, e, factor)
            pos = aggregate_behavior(beh['position'][s:e], factor, 'mean').astype(np.float32)
            speed = aggregate_behavior(beh['speed'][s:e], factor, 'mean').astype(np.float32)
            lick = (aggregate_behavior((raw_lick > 0).astype(np.int8), factor, 'max') > 0).astype(np.int64)
            env_native = aggregate_behavior(beh['environment'][s:e], factor, 'first')
            env = int(round(float(env_native[0])))
            if env not in (0, 1) or not np.all(env_native == env_native[0]):
                raise ValueError(f'{path}: invalid/nonconstant environment in trial {qi}')
            T = neural.shape[1]
            if not (T == len(pos) == len(speed) == len(lick) == n_complete // factor):
                raise AssertionError('Resampling length mismatch')

            zone = zone_for_trial(src_zone, dst_zone, trial_num)
            _, dist_cls = distance_classes(pos, zone)
            pos_cls = position_classes(pos)
            speed_cls = speed_classes(speed)
            prev = outcomes[-2] if len(outcomes) > 1 else 0
            elapsed = np.arange(T, dtype=np.float32) / np.float32(TARGET_RATE)
            inp = np.vstack([
                elapsed,
                np.full(T, env, dtype=np.float32),
                np.full(T, trial_num, dtype=np.float32),
                np.full(T, prev, dtype=np.float32),
            ]).astype(np.float32, copy=False)
            out = np.vstack([
                dist_cls, pos_cls, speed_cls, lick,
                np.full(T, zone, dtype=np.int64),
                np.full(T, rewarded, dtype=np.int64),
            ]).astype(np.int64, copy=False)
            if not (neural.shape[1] == inp.shape[1] == out.shape[1]):
                raise AssertionError('Trial stream lengths differ')
            if not np.isfinite(neural).all() or not np.isfinite(inp).all():
                raise ValueError(f'{path}: non-finite converted data')
            neural_trials.append(neural)
            input_trials.append(inp)
            output_trials.append(out)
            trial_info.append({'trial_number': trial_num, 'start': s, 'end': e,
                               'environment': env, 'zone': zone, 'rewarded': rewarded,
                               'lick_artifact': lick_artifact, 'T': T})

        info = {
            'path': str(path), 'subject': subject, 'session_id': str(nwb.session_id),
            'identifier': str(nwb.identifier), 'native_rate': rates[0], 'effective_rate': effective_rate,
            'downsample_factor': factor, 'n_neurons': neural_trials[0].shape[0],
            'n_trials': len(pairs), 'n_rewarded': int(sum(outcomes)),
            'lick_artifact_trials': lick_artifacts, 'is_switch': is_switch,
            'trial_info': trial_info, 'seconds': time.time() - t0,
        }
    return neural_trials, input_trials, output_trials, info


def plot_processing(data, infos, outdir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for si in range(min(2, len(infos))):
        info = infos[si]; q = min(30, len(data['neural'][si]) - 1)
        n = data['neural'][si][q]; x = data['input'][si][q]; y = data['output'][si][q]
        t = x[0]
        fig, ax = plt.subplots(7, 1, figsize=(13, 16), sharex=True,
                               gridspec_kw={'height_ratios':[3,1,1,1,1,1,1]})
        show = n[:min(100, n.shape[0])]
        ax[0].imshow(show, aspect='auto', origin='lower', extent=[t[0],t[-1],0,show.shape[0]], cmap='magma')
        ax[0].set_ylabel('accepted cells'); ax[0].set_title(f"{Path(info['path']).stem}, trial {q}, factor={info['downsample_factor']}")
        labels = ['distance class','position class','speed class','lick','zone','reward']
        for k in range(6):
            ax[k+1].step(t, y[k], where='post'); ax[k+1].set_ylabel(labels[k]); ax[k+1].grid(alpha=.2)
        ax[-1].set_xlabel('seconds from trial start')
        fig.tight_layout()
        safe = f"{info['subject']}_ses-{info['session_id']}"
        fig.savefig(outdir / f'processing_{safe}.png', dpi=140)
        plt.close(fig)


def choose_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.rglob('*.nwb'))
    if not sample:
        return files
    single = files[0]
    multi = next(p for p in files if p.parent.name in ('sub-m17', 'sub-m18'))
    return [single, multi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='process all sessions (default)')
    mode.add_argument('--sample', action='store_true', help='process one single- and one two-plane session')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()
    files = choose_files(args.sample)
    print(f'Processing {len(files)} sessions; target bin {BIN_MS:.6f} ms', flush=True)
    neural, inputs, outputs, infos = [], [], [], []
    t0 = time.time()
    for i, p in enumerate(files, 1):
        ns, xs, ys, info = process_session(p)
        neural.append(ns); inputs.append(xs); outputs.append(ys); infos.append(info)
        print(f"[{i}/{len(files)}] {p.name}: neurons={info['n_neurons']} trials={info['n_trials']} "
              f"rewarded={info['n_rewarded']} factor={info['downsample_factor']} "
              f"lick_bad={info['lick_artifact_trials']} time={info['seconds']:.2f}s", flush=True)
        gc.collect()

    subjects = sorted(set(x['subject'] for x in infos), key=lambda z: int(re.sub(r'\D','',z)))
    subject_idx = np.array([subjects.index(x['subject']) for x in infos], dtype=np.int64)
    data = {
        'neural': neural, 'input': inputs, 'output': outputs,
        'subjects': subjects, 'subject_idx': subject_idx,
        'brain_regions': ['CA1'],
        'brain_region_idx': [np.zeros(x['n_neurons'], dtype=np.int64) for x in infos],
        'input_names': ['time from trial start', 'environment type', 'trial number', 'previous trial outcome'],
        'output_names': ['distance to reward zone', 'absolute position', 'speed', 'lick', 'reward zone location', 'reward outcome'],
        'output_values': [
            ['< -50 cm', '-50 to < -10 cm', '-10 to < 0 cm', '0 cm (inside reward zone)', '> 0 to +10 cm', '> +10 to +50 cm', '> +50 cm'],
            ['< 90 cm', '90 to < 180 cm', '180 to < 270 cm', '270 to 360 cm', '> 360 cm'],
            ['< 2 cm/s', '2 to < 10 cm/s', '10 to < 20 cm/s', '20 to 40 cm/s', '> 40 cm/s'],
            ['no lick', 'lick'], ['A', 'B', 'C'], ['omitted', 'rewarded'],
        ],
        'metadata': {
            'task_description': 'Head-fixed mice navigate a 450 cm virtual corridor for reward at one of three 50 cm zones; neural activity predicts position, movement, licking, reward-zone identity, and outcome.',
            'time_bin_size': float(BIN_MS),
            'temporal_alignment_event': 'trial start (entry to linear track)',
            'off_start': 0.0, 'off_end': None,
            'neural_signal': 'Suite2p deconvolved calcium events, accepted cells only',
            'trial_interval': '[trial_start, teleport); teleport/ITI frame excluded',
            'reward_zone_intervals_cm': {'A':[80.0,130.0], 'B':[200.0,250.0], 'C':[320.0,370.0]},
            'effective_rate_hz': float(TARGET_RATE),
            'roi_response_metadata_rates_hz': sorted(set(x['native_rate'] for x in infos)),
            'session_info': infos,
            'conversion_notes': 'All processed neural rows are one-to-one with 15.5078125 Hz behavior timestamps. Two-plane series metadata report 31.015625 Hz but arrays must not be downsampled. Per-trial variables are repeated over time.',
        },
    }
    # Global structural checks.
    assert len(data['neural']) == len(data['input']) == len(data['output']) == len(infos)
    assert all(len(neural[s]) >= 2 for s in range(len(neural)))
    print(f"Totals: subjects={len(subjects)} sessions={len(infos)} trials={sum(len(x) for x in neural)} "
          f"session-neurons={sum(x['n_neurons'] for x in infos)} lick_bad={sum(x['lick_artifact_trials'] for x in infos)}", flush=True)
    out = Path(args.outpicklefile)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {out} ({out.stat().st_size/1e9:.3f} GB), elapsed {time.time()-t0:.2f}s', flush=True)
    if args.show_processing:
        plot_processing(data, infos, out.parent)
        print('Wrote processing plots', flush=True)

if __name__ == '__main__':
    main()
