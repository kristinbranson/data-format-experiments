#!/usr/bin/env python3
"""Convert MAP NWB sessions to the neural-decoder pickle format.

Usage: python -u /app/convert_data.py OUTFILE [--full|--sample] [--show-processing]
"""
import argparse
import glob
import json
import os
import pickle
import time
from collections import Counter

import h5py
import numpy as np

DATA_ROOT = '/app/data'
BIN_SIZE = 0.05
OFF_START = -2.5
OFF_END = 1.5
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
N_TIME = len(CENTERS_REL)
LIKELIHOOD_CUTOFF = 0.9
MAX_VIDEO_DT = 0.020


def decode_array(x):
    """Decode an HDF5 string vector to a NumPy unicode array."""
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in x])


def classifier_mask(f):
    return decode_array(f['units/classification'][:]) == 'good'


def electrode_regions(f, unit_mask):
    """Map selected units through electrode row indices to target-region JSON."""
    electrode_idx = f['units/electrodes'][:][unit_mask]
    raw = f['general/extracellular_ephys/electrodes/location'][:]
    labels = []
    for value in raw:
        value = value.decode() if isinstance(value, bytes) else str(value)
        try:
            labels.append(json.loads(value).get('brain_regions', value))
        except (json.JSONDecodeError, TypeError):
            labels.append(value)
    return [labels[i] for i in electrode_idx]


def session_inventory(paths):
    """Small metadata prepass: reject sessions without classifier-good units."""
    kept, all_regions = [], set()
    for path in paths:
        with h5py.File(path, 'r') as f:
            class_good = classifier_mask(f)
            n_trials = f['units/is_good_trials'].shape[1]
            always_valid = np.all(f['units/is_good_trials'][:, :], axis=1)
            mask = class_good & always_valid
            if not mask.any():
                print(f"SKIP {os.path.basename(path)}: no classifier-good units", flush=True)
                continue
            represented = f['units/is_good_trials'][mask, :]
            if represented.shape != (int(mask.sum()), n_trials) or not represented.all():
                raise ValueError(f'{path}: curated units contain invalid represented trials')
            regions = electrode_regions(f, mask)
            all_regions.update(regions)
            kept.append((path, int(mask.sum()), n_trials))
    return kept, sorted(all_regions)


def nearest_indices(source_t, target_t):
    """Indices of nearest sorted source timestamp for each target."""
    idx = np.searchsorted(source_t, target_t)
    idx = np.clip(idx, 1, len(source_t) - 1)
    choose_previous = np.abs(source_t[idx - 1] - target_t) < np.abs(source_t[idx] - target_t)
    idx[choose_previous] -= 1
    return idx


def clean_tongue_y(timestamps, data):
    """Apply reference five-sigma velocity cleanup to high-confidence tongue y."""
    y = np.asarray(data[:, 1], dtype=np.float64).copy()
    likelihood = np.asarray(data[:, 2], dtype=np.float64)
    visible = np.isfinite(y) & np.isfinite(likelihood) & (likelihood >= LIKELIHOOD_CUTOFF)
    pair = visible[1:] & visible[:-1]
    dt = np.diff(timestamps)
    velocity = np.full(len(y) - 1, np.nan)
    valid_pair = pair & np.isfinite(dt) & (dt > 0)
    velocity[valid_pair] = np.diff(y)[valid_pair] / dt[valid_pair]
    values = velocity[valid_pair]
    outlier = np.zeros(len(y), dtype=bool)
    if values.size:
        center = np.nanmean(values)
        threshold = 5.0 * np.nanstd(values)
        bad_pair = valid_pair & (np.abs(velocity - center) > threshold)
        # Mark the destination frame of each implausible jump. Iterative artifacts
        # are avoided by interpolation from all remaining high-confidence frames.
        outlier[1:] = bad_pair
    clean_visible = visible & ~outlier
    if clean_visible.sum() < 2:
        raise ValueError('Fewer than two clean visible tongue samples')
    if outlier.any():
        y[outlier] = np.interp(timestamps[outlier], timestamps[clean_visible], y[clean_visible])
    # Outliers were reference-imputed and remain visible; low-confidence samples do not.
    final_visible = visible
    q40, q60 = np.percentile(y[final_visible], [40, 60])
    return y, likelihood, final_visible, outlier, np.array([q40, q60])


def trial_event_mapping(starts, stops, event_times, name, before=None, use_last=False):
    """Map events into trial intervals, optionally requiring times before another event."""
    result = np.empty(len(starts), dtype=np.float64)
    for i, (a, b) in enumerate(zip(starts, stops)):
        hi = b if before is None else min(b, before[i])
        lo_idx = np.searchsorted(event_times, a, side='left')
        hi_idx = np.searchsorted(event_times, hi, side='left' if before is not None else 'right')
        candidates = event_times[lo_idx:hi_idx]
        if len(candidates) == 0:
            raise ValueError(f'trial {i}: no {name} event in [{a}, {hi}]')
        if not use_last and len(candidates) != 1:
            raise ValueError(f'trial {i}: expected one {name}, found {len(candidates)}')
        result[i] = candidates[-1] if use_last else candidates[0]
    return result


def bin_selected_units(f, unit_indices, absolute_edges):
    """Vectorized ragged-spike binning, returning trials x neurons x time in Hz."""
    spike_data = f['units/spike_times']
    endpoints = f['units/spike_times_index'][:]
    starts = np.r_[0, endpoints[:-1]]
    n_trials = absolute_edges.shape[0]
    rates = np.empty((n_trials, len(unit_indices), N_TIME), dtype=np.float32)
    flat_edges = absolute_edges.ravel()
    # Trial edges are globally time-ordered; search all trial edges in one C call/unit.
    for j, unit in enumerate(unit_indices):
        spikes = spike_data[starts[unit]:endpoints[unit]]
        positions = np.searchsorted(spikes, flat_edges, side='left').reshape(n_trials, N_TIME + 1)
        rates[:, j, :] = np.diff(positions, axis=1).astype(np.float32) / BIN_SIZE
    return rates


def build_photostim(centers_abs, starts, stops):
    state = np.zeros(centers_abs.shape, dtype=bool)
    for a, b in zip(starts, stops):
        state |= (centers_abs >= a) & (centers_abs < b)
    return state.astype(np.float32)


def process_session(path, region_lookup, make_plot=False):
    tic = time.time()
    with h5py.File(path, 'r') as f:
        class_good = classifier_mask(f)
        n_trials = f['units/is_good_trials'].shape[1]
        always_valid = np.all(f['units/is_good_trials'][:, :], axis=1)
        mask = class_good & always_valid
        unit_indices = np.flatnonzero(mask)
        if n_trials < 2 or not f['units/is_good_trials'][mask, :].all():
            raise ValueError(f'{path}: invalid represented-trial matrix')

        tr = f['intervals/trials']
        trial_starts = tr['start_time'][:n_trials]
        trial_stops = tr['stop_time'][:n_trials]
        events = f['acquisition/BehavioralEvents']
        go_events = events['go_start_times/timestamps'][:]
        sample_events = events['sample_start_times/timestamps'][:]
        go = trial_event_mapping(trial_starts, trial_stops, go_events, 'go cue')
        tone = trial_event_mapping(trial_starts, trial_stops, sample_events,
                                   'sample/tone onset', before=go, use_last=True)
        edges_abs = go[:, None] + EDGES_REL[None, :]
        centers_abs = go[:, None] + CENTERS_REL[None, :]

        # `is_good_trials` is the source validity mask. NWB obs_intervals rows
        # mirror behavioral trial start/stop boundaries rather than continuous
        # ephys availability; the required go-centered window may legitimately
        # extend into adjacent ITI, so it must not be clipped to those rows.

        rates = bin_selected_units(f, unit_indices, edges_abs)
        # Some source trials contain physiologically impossible population-wide
        # raw-spike gaps despite true is_good_trials flags. Exclude these invalid
        # data periods and apply the identical mask to every aligned stream.
        trial_keep = np.any(rates != 0, axis=(1, 2))
        trial_indices = np.flatnonzero(trial_keep)
        n_zero = int((~trial_keep).sum())
        if n_zero:
            print(f"  excluding {n_zero} population-wide zero-spike trials", flush=True)
        rates = rates[trial_keep]
        trial_starts = trial_starts[trial_keep]
        trial_stops = trial_stops[trial_keep]
        go = go[trial_keep]
        tone = tone[trial_keep]
        edges_abs = edges_abs[trial_keep]
        centers_abs = centers_abs[trial_keep]
        n_trials = len(trial_indices)
        if n_trials < 2:
            raise ValueError(f'{path}: fewer than two valid nonzero neural trials')
        print(f"  neural binned: {rates.shape} in {time.time()-tic:.2f}s", flush=True)

        # Inputs.
        tone_elapsed = (centers_abs - tone[:, None]).astype(np.float32)
        laser_starts = events['photostim_start_times/timestamps'][:]
        laser_stops = events['photostim_stop_times/timestamps'][:]
        photo = build_photostim(centers_abs, laser_starts, laser_stops)
        inputs = np.stack([tone_elapsed, photo], axis=1)  # trials x 2 x time

        # Per-trial categorical outputs.
        outcome_str = decode_array(tr['outcome'][:])[trial_indices]
        instruction = decode_array(tr['trial_instruction'][:])[trial_indices]
        early_str = decode_array(tr['early_lick'][:])[trial_indices]
        outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
        early_map = {'no early': 0, 'early': 1}
        outcome = np.asarray([outcome_map[x] for x in outcome_str], dtype=np.int64)
        early = np.asarray([early_map[x] for x in early_str], dtype=np.int64)
        choice = np.empty(n_trials, dtype=np.int64)
        for i, (o, side) in enumerate(zip(outcome_str, instruction)):
            if o == 'ignore': choice[i] = 2
            elif o == 'hit': choice[i] = 0 if side == 'left' else 1
            else: choice[i] = 1 if side == 'left' else 0

        # Tongue output.
        tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
        video_t = tongue['timestamps'][:]
        video_data = tongue['data'][:]
        clean_y, likelihood, visible, outliers, percentiles = clean_tongue_y(video_t, video_data)
        target = centers_abs.ravel()
        video_idx = nearest_indices(video_t, target)
        close = np.abs(video_t[video_idx] - target) <= MAX_VIDEO_DT
        sampled_visible = visible[video_idx] & close
        sampled_y = clean_y[video_idx]
        tongue_class = np.full(target.shape, 3, dtype=np.int64)
        tongue_class[sampled_visible & (sampled_y < percentiles[0])] = 0
        tongue_class[sampled_visible & (sampled_y >= percentiles[0]) & (sampled_y <= percentiles[1])] = 1
        tongue_class[sampled_visible & (sampled_y > percentiles[1])] = 2
        tongue_class = tongue_class.reshape(n_trials, N_TIME)

        outputs = np.empty((n_trials, 4, N_TIME), dtype=np.int64)
        outputs[:, 0, :] = choice[:, None]
        outputs[:, 1, :] = outcome[:, None]
        outputs[:, 2, :] = early[:, None]
        outputs[:, 3, :] = tongue_class

        labels = electrode_regions(f, mask)
        region_idx = np.asarray([region_lookup[x] for x in labels], dtype=np.int64)
        subject = os.path.basename(os.path.dirname(path)).removeprefix('sub-')
        session_id = os.path.basename(path).split('_behavior')[0]

        if make_plot:
            make_processing_plot(session_id, rates, inputs, outputs, percentiles,
                                 video_t, clean_y, likelihood, visible, outliers, go[0])

    # Each list element is an independent array to meet the requested nested format.
    neural_trials = [np.ascontiguousarray(rates[i]) for i in range(n_trials)]
    input_trials = [np.ascontiguousarray(inputs[i]) for i in range(n_trials)]
    output_trials = [np.ascontiguousarray(outputs[i]) for i in range(n_trials)]
    stats = {
        'session_id': session_id, 'subject': subject, 'n_trials': n_trials,
        'n_neurons': len(unit_indices), 'tongue_q40': float(percentiles[0]),
        'tongue_q60': float(percentiles[1]), 'tongue_outliers': int(outliers.sum()),
        'choice_counts': np.bincount(choice, minlength=3).tolist(),
        'outcome_counts': np.bincount(outcome, minlength=3).tolist(),
        'early_counts': np.bincount(early, minlength=2).tolist(),
        'tongue_counts': np.bincount(tongue_class.ravel(), minlength=4).tolist(),
        'photostim_on_bins': int(photo.sum()), 'excluded_zero_spike_trials': n_zero,
    }
    print(f"  completed {session_id}: {n_trials} trials, {len(unit_indices)} neurons, "
          f"{time.time()-tic:.2f}s", flush=True)
    return neural_trials, input_trials, output_trials, subject, region_idx, stats


def make_processing_plot(session_id, rates, inputs, outputs, q, video_t, y, likelihood,
                         visible, outliers, first_go):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(12, 12), constrained_layout=True)
    im = ax[0].imshow(rates[0, :min(100, rates.shape[1])], aspect='auto', origin='lower',
                      extent=[OFF_START, OFF_END, 0, min(100, rates.shape[1])])
    ax[0].axvline(0, color='w', ls='--'); ax[0].set_title('Trial 0 firing rates (first 100 units)')
    ax[0].set_ylabel('unit'); fig.colorbar(im, ax=ax[0], label='Hz')
    ax[1].plot(CENTERS_REL, inputs[0, 0], label='time from tone (s)')
    ax[1].step(CENTERS_REL, inputs[0, 1], where='mid', label='photostim on')
    ax[1].axvline(0, color='k', ls='--'); ax[1].legend(); ax[1].set_title('Aligned decoder inputs')
    keep = (video_t >= first_go + OFF_START) & (video_t < first_go + OFF_END)
    ax[2].plot(video_t[keep]-first_go, y[keep], lw=.7, label='clean tongue y')
    ax[2].scatter(video_t[keep & ~visible]-first_go, y[keep & ~visible], s=1, alpha=.2, label='not visible')
    ax[2].axhline(q[0], color='C1', ls='--', label='q40'); ax[2].axhline(q[1], color='C2', ls='--', label='q60')
    if np.any(keep & outliers): ax[2].scatter(video_t[keep & outliers]-first_go, y[keep & outliers], c='r', s=8, label='5-sigma outlier')
    ax[2].legend(ncol=4); ax[2].set_title('Tongue processing and session thresholds')
    ax[3].step(CENTERS_REL, outputs[0, 3], where='mid', label='tongue class')
    ax[3].plot(CENTERS_REL, likelihood[nearest_indices(video_t, first_go+CENTERS_REL)], alpha=.6, label='likelihood')
    ax[3].set_yticks([0,1,2,3]); ax[3].legend(); ax[3].set_xlabel('time from go cue (s)')
    ax[3].set_title('Final time-varying tongue output (3 = not visible)')
    fig.suptitle(session_id)
    out = f'/app/processing_{session_id}.png'
    fig.savefig(out, dpi=140); plt.close(fig); print(f'  saved {out}', flush=True)


def validate_result(data):
    ns = len(data['neural'])
    assert ns == len(data['input']) == len(data['output']) == len(data['subject_idx']) == len(data['brain_region_idx'])
    for s in range(ns):
        nt = len(data['neural'][s])
        assert nt >= 2 and nt == len(data['input'][s]) == len(data['output'][s])
        nn = len(data['brain_region_idx'][s])
        for n, x, y in zip(data['neural'][s], data['input'][s], data['output'][s]):
            assert n.shape == (nn, N_TIME) and n.dtype == np.float32 and np.isfinite(n).all() and (n >= 0).all()
            assert x.shape == (2, N_TIME) and np.isfinite(x).all()
            assert y.shape == (4, N_TIME) and np.isfinite(y).all()
            assert set(np.unique(y[0])).issubset({0,1,2})
            assert set(np.unique(y[1])).issubset({0,1,2})
            assert set(np.unique(y[2])).issubset({0,1})
            assert set(np.unique(y[3])).issubset({0,1,2,3})
    print('Internal validation passed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='process all sessions (default)')
    mode.add_argument('--sample', action='store_true', help='process first two valid sessions')
    parser.add_argument('--show-processing', action='store_true', help='save processing plots for up to two sessions')
    parser.add_argument('outpicklefile')
    args = parser.parse_args()
    total_tic = time.time()

    paths = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
    inventory, brain_regions = session_inventory(paths)
    print(f'Inventory: {len(paths)} source files, {len(inventory)} usable sessions, regions={brain_regions}', flush=True)
    if args.sample:
        inventory = inventory[:2]
    region_lookup = {x:i for i,x in enumerate(brain_regions)}
    subjects = sorted({os.path.basename(os.path.dirname(x[0])).removeprefix('sub-') for x in inventory})
    subject_lookup = {x:i for i,x in enumerate(subjects)}

    neural, inputs, outputs, subject_idx, region_indices, session_info = [], [], [], [], [], []
    for i, (path, expected_units, expected_trials) in enumerate(inventory):
        print(f'[{i+1}/{len(inventory)}] {os.path.basename(path)} expected={expected_trials}x{expected_units}', flush=True)
        n, x, y, subject, ridx, stats = process_session(path, region_lookup,
                                                        make_plot=args.show_processing and i < 2)
        neural.append(n); inputs.append(x); outputs.append(y)
        subject_idx.append(subject_lookup[subject]); region_indices.append(ridx); session_info.append(stats)

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': region_indices,
        'input_names': ['time from tone onset (s)', 'photostimulation on'],
        'output_names': ['lick direction choice', 'outcome', 'early lick', 'tongue y-position'],
        'output_values': [
            ['left', 'right', 'no lick'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['below 40th percentile', '40th to 60th percentile', 'above 60th percentile', 'not visible'],
        ],
        'metadata': {
            'task_description': 'Auditory delayed-response task; decode lick choice, outcome, early lick, and time-varying tongue y-position from classifier-curated neural activity.',
            'time_bin_size': 50.0,
            'time_bin_units': 'ms',
            'temporal_alignment_event': 'go cue onset',
            'off_start': OFF_START,
            'off_end': OFF_END,
            'n_timepoints': N_TIME,
            'time_bin_edges_seconds': EDGES_REL.astype(np.float32),
            'time_bin_centers_seconds': CENTERS_REL.astype(np.float32),
            'neural_units': 'spikes/s',
            'neuron_curation': "NWB units/classification == 'good' (paper region-specific classifier QC) and valid on every represented trial according to is_good_trials",
            'trial_curation': 'Trials represented by curated-unit observation intervals/is_good_trials and containing at least one population spike in the requested window; required early/ignore/miss/photostim categories retained.',
            'tongue_visibility_likelihood_cutoff': LIKELIHOOD_CUTOFF,
            'tongue_discretization': 'Per-session 40th/60th percentiles of cleaned visible tongue y; low-confidence or missing samples are class 3.',
            'source': 'Mesoscale Activity Map Dataset, DANDI 000363 v0.230822.0128',
            'session_info': session_info,
        },
    }
    validate_result(data)
    print(f'Writing {args.outpicklefile} ...', flush=True)
    with open(args.outpicklefile, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    size_gb = os.path.getsize(args.outpicklefile) / 1e9
    print(f'Wrote {args.outpicklefile}: {size_gb:.3f} GB; total time {time.time()-total_tic:.2f}s', flush=True)


if __name__ == '__main__':
    main()
