#!/usr/bin/env python3

import gc
import glob
import pickle
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d


DATA_ROOT = Path("/app/data")
OUTPUT_PATH = Path("/app/converted_data.pkl")

TRACK_LENGTH_CM = 450.0
FRAME_RATE_HZ = 15.5078125
BIN_FRAMES = 8
TIME_BIN_SIZE_S = BIN_FRAMES / FRAME_RATE_HZ
LICK_ERROR_THRESHOLD = 0.35
SWITCH_TRIAL = 30
INTERNEURON_SPEED_CORR_THRESHOLD = 0.5
NEUROPIL_COEF = 0.7

REWARD_ZONES = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
ENV_TO_INT = {"Env1": 0, "Env2": 1}


def sort_session_paths(paths):
    def key(path_str):
        name = Path(path_str).name
        match = re.search(r"sub-m(\d+)_ses-(\d+)_", name)
        if match is None:
            raise ValueError(f"Could not parse subject/session from {name}")
        return int(match.group(1)), int(match.group(2))

    return sorted(paths, key=key)


def parse_scene(scene):
    match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
    if match:
        env, zone = match.groups()
        return {
            "env_before": env,
            "env_after": env,
            "zone_before": zone,
            "zone_after": zone,
            "switch_trial": None,
        }

    match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)
    if match:
        env, zone_before, zone_after = match.groups()
        return {
            "env_before": env,
            "env_after": env,
            "zone_before": zone_before,
            "zone_after": zone_after,
            "switch_trial": SWITCH_TRIAL,
        }

    match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
    if match:
        env_before, zone_before, env_after, zone_after = match.groups()
        return {
            "env_before": env_before,
            "env_after": env_after,
            "zone_before": zone_before,
            "zone_after": zone_after,
            "switch_trial": SWITCH_TRIAL,
        }

    raise ValueError(f"Unrecognized scene format: {scene}")


def expected_trial_labels(scene, n_trials):
    info = parse_scene(scene)
    if info["switch_trial"] is None:
        zones = [info["zone_before"]] * n_trials
        envs = [ENV_TO_INT[info["env_before"]]] * n_trials
        return zones, envs

    pre_n = min(info["switch_trial"], n_trials)
    post_n = max(0, n_trials - pre_n)
    zones = [info["zone_before"]] * pre_n + [info["zone_after"]] * post_n
    envs = [ENV_TO_INT[info["env_before"]]] * pre_n + [ENV_TO_INT[info["env_after"]]] * post_n
    return zones, envs


def smooth_ignore_nan(arr, sigma, axis):
    nanmask = np.isnan(arr)
    arr_filled = np.where(nanmask, 0.0, arr)
    weights = np.where(nanmask, 0.001, 1.0).astype(np.float32)
    smoothed = gaussian_filter1d(arr_filled, sigma=sigma, axis=axis, mode="nearest")
    weights = gaussian_filter1d(weights, sigma=sigma, axis=axis, mode="nearest")
    return smoothed / weights


def pair_trial_segments(trial_start_signal, teleport_signal):
    starts = np.flatnonzero(trial_start_signal > 0)
    teleports = np.flatnonzero(teleport_signal > 0)
    segments = []
    teleport_ptr = 0

    for start in starts:
        while teleport_ptr < len(teleports) and teleports[teleport_ptr] <= start:
            teleport_ptr += 1
        if teleport_ptr >= len(teleports):
            break
        stop = teleports[teleport_ptr]
        teleport_ptr += 1
        if stop - start >= 1:
            segments.append((int(start), int(stop)))

    return segments


def load_roi_response_matrix(h5file, base_path, cell_idx, plane_idx_all):
    plane_idx_selected = plane_idx_all[cell_idx]
    plane_ids = np.unique(plane_idx_selected)
    out = None

    for plane_id in plane_ids:
        selected_positions = np.flatnonzero(plane_idx_selected == plane_id)
        selected_global_idx = cell_idx[selected_positions]
        plane_global_idx = np.flatnonzero(plane_idx_all == plane_id)
        local_idx = np.searchsorted(plane_global_idx, selected_global_idx)
        plane_data = np.asarray(
            h5file[f"{base_path}/plane{plane_id}/data"][:, local_idx],
            dtype=np.float32,
        )
        if out is None:
            out = np.empty((plane_data.shape[0], len(cell_idx)), dtype=np.float32)
        out[:, selected_positions] = plane_data

    if out is None:
        raise ValueError(f"No ROI response data found at {base_path}")
    return out


def bin_2d_mean(arr, bin_frames):
    n_bins = arr.shape[1] // bin_frames
    if n_bins == 0:
        return None
    trimmed = arr[:, : n_bins * bin_frames]
    return trimmed.reshape(arr.shape[0], n_bins, bin_frames).mean(axis=2)


def bin_1d_mean(arr, bin_frames):
    n_bins = arr.shape[0] // bin_frames
    if n_bins == 0:
        return None
    trimmed = arr[: n_bins * bin_frames]
    return trimmed.reshape(n_bins, bin_frames).mean(axis=1)


def bin_1d_max(arr, bin_frames):
    n_bins = arr.shape[0] // bin_frames
    if n_bins == 0:
        return None
    trimmed = arr[: n_bins * bin_frames]
    return trimmed.reshape(n_bins, bin_frames).max(axis=1)


def reward_distance_cm(position_cm, zone_bounds):
    start_cm, end_cm = zone_bounds
    distance = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < start_cm
    after = position_cm > end_cm
    distance[before] = position_cm[before] - start_cm
    distance[after] = position_cm[after] - end_cm
    return distance


def discretize_reward_distance(distance_cm):
    out = np.empty(distance_cm.shape, dtype=np.uint8)
    out[distance_cm < -50.0] = 0
    out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
    out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
    out[distance_cm == 0.0] = 3
    out[(distance_cm > 0.0) & (distance_cm <= 10.0)] = 4
    out[(distance_cm > 10.0) & (distance_cm <= 50.0)] = 5
    out[distance_cm > 50.0] = 6
    return out


def discretize_position(position_cm):
    out = np.empty(position_cm.shape, dtype=np.uint8)
    out[position_cm < 90.0] = 0
    out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    out[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    out[(position_cm >= 270.0) & (position_cm < 360.0)] = 3
    out[position_cm >= 360.0] = 4
    return out


def discretize_speed(speed_cm_s):
    out = np.empty(speed_cm_s.shape, dtype=np.uint8)
    out[speed_cm_s < 2.0] = 0
    out[(speed_cm_s >= 2.0) & (speed_cm_s < 10.0)] = 1
    out[(speed_cm_s >= 10.0) & (speed_cm_s < 20.0)] = 2
    out[(speed_cm_s >= 20.0) & (speed_cm_s <= 40.0)] = 3
    out[speed_cm_s > 40.0] = 4
    return out


def find_putative_interneurons(fluorescence, neuropil, speed, trial_segments):
    n_cells = fluorescence.shape[0]
    sum_x = np.zeros(n_cells, dtype=np.float64)
    sum_y = np.zeros(n_cells, dtype=np.float64)
    sum_x2 = np.zeros(n_cells, dtype=np.float64)
    sum_y2 = np.zeros(n_cells, dtype=np.float64)
    sum_xy = np.zeros(n_cells, dtype=np.float64)
    count = np.zeros(n_cells, dtype=np.float64)

    for start, stop in trial_segments:
        f_trial = fluorescence[:, start:stop]
        fneu_trial = neuropil[:, start:stop]
        speed_trial = speed[start:stop].astype(np.float32, copy=False)

        signal = f_trial - NEUROPIL_COEF * fneu_trial
        signal = signal + NEUROPIL_COEF * np.nanmean(fneu_trial, axis=1, keepdims=True)

        baseline = smooth_ignore_nan(signal, sigma=15, axis=1)
        baseline = minimum_filter1d(baseline, size=300, axis=1, mode="nearest")
        baseline = maximum_filter1d(baseline, size=300, axis=1, mode="nearest")

        dff = (signal - baseline) / np.abs(baseline)
        dff[~np.isfinite(dff)] = np.nan
        dff = smooth_ignore_nan(dff, sigma=2, axis=1)

        valid_speed = np.isfinite(speed_trial)
        if not np.any(valid_speed):
            continue

        dff = dff[:, valid_speed]
        speed_valid = speed_trial[valid_speed]
        valid = np.isfinite(dff)

        dff_zeroed = np.where(valid, dff, 0.0)
        speed_row = speed_valid[np.newaxis, :]

        count += valid.sum(axis=1)
        sum_x += dff_zeroed.sum(axis=1)
        sum_x2 += (dff_zeroed * dff_zeroed).sum(axis=1)
        sum_y += (speed_row * valid).sum(axis=1)
        sum_y2 += ((speed_row * speed_row) * valid).sum(axis=1)
        sum_xy += (dff_zeroed * speed_row).sum(axis=1)

    numerator = count * sum_xy - sum_x * sum_y
    denominator = np.sqrt((count * sum_x2 - sum_x * sum_x) * (count * sum_y2 - sum_y * sum_y))
    correlation = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=(denominator > 0) & (count > 1),
    )
    return correlation > INTERNEURON_SPEED_CORR_THRESHOLD, correlation


def convert_session(path):
    with h5py.File(path, "r") as f:
        scene = f["identifier"][()].decode().split("/")[-1]
        subject = f["general/subject/subject_id"][()].decode()
        session_id = f["general/session_id"][()].decode()

        behavior = f["processing/behavior/BehavioralTimeSeries"]
        frame_times = np.asarray(behavior["position/timestamps"][:], dtype=np.float64)
        position = np.asarray(behavior["position/data"][:], dtype=np.float32)
        speed = np.asarray(behavior["speed/data"][:], dtype=np.float32)
        lick_counts = np.asarray(behavior["lick/data"][:], dtype=np.float32)
        environment = np.asarray(behavior["environment/data"][:], dtype=np.float32)
        trial_number_signal = np.asarray(behavior["trial number/data"][:], dtype=np.float32)
        trial_start_signal = np.asarray(behavior["trial_start/data"][:], dtype=np.float32)
        teleport_signal = np.asarray(behavior["teleport/data"][:], dtype=np.float32)
        reward_timestamps = np.asarray(behavior["Reward/timestamps"][:], dtype=np.float64)

        iscell = np.asarray(
            f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][:],
            dtype=np.float32,
        )
        curated_cell_idx = np.flatnonzero(iscell[:, 0] > 0.5)
        plane_idx_all = np.asarray(
            f["processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx"][:],
            dtype=np.int64,
        )

        trial_segments = pair_trial_segments(trial_start_signal, teleport_signal)
        zone_labels, expected_envs = expected_trial_labels(scene, len(trial_segments))

        fluorescence = load_roi_response_matrix(
            f,
            "processing/ophys/Fluorescence",
            curated_cell_idx,
            plane_idx_all,
        ).T
        neuropil = load_roi_response_matrix(
            f,
            "processing/ophys/Neuropil",
            curated_cell_idx,
            plane_idx_all,
        ).T
        is_interneuron, speed_corr = find_putative_interneurons(
            fluorescence,
            neuropil,
            speed,
            trial_segments,
        )
        keep_cells = ~is_interneuron
        del fluorescence
        del neuropil

        deconvolved = load_roi_response_matrix(
            f,
            "processing/ophys/Deconvolved",
            curated_cell_idx,
            plane_idx_all,
        ).T
        deconvolved = deconvolved[keep_cells]
        plane_idx = plane_idx_all[curated_cell_idx][keep_cells]

    reward_outcomes = []
    trial_info = []
    env_mismatch_trials = 0

    for trial_idx, (start, stop) in enumerate(trial_segments):
        reward_outcome = int(np.any((reward_timestamps >= frame_times[start]) & (reward_timestamps < frame_times[stop])))
        reward_outcomes.append(reward_outcome)

        env_slice = environment[start:stop]
        env_slice = env_slice[np.isfinite(env_slice) & (env_slice >= 0)]
        if env_slice.size:
            env_value = int(np.rint(np.nanmedian(env_slice)))
        else:
            env_value = expected_envs[trial_idx]
        if env_value != expected_envs[trial_idx]:
            env_mismatch_trials += 1

        trialnum_slice = trial_number_signal[start:stop]
        trialnum_slice = trialnum_slice[np.isfinite(trialnum_slice) & (trialnum_slice >= 0)]
        if trialnum_slice.size:
            trial_number = int(np.rint(np.nanmedian(trialnum_slice)))
        else:
            trial_number = trial_idx

        lick_error = bool(np.mean(lick_counts[start:stop] > 2.0) > LICK_ERROR_THRESHOLD)
        trial_info.append(
            {
                "start": start,
                "stop": stop,
                "zone_label": zone_labels[trial_idx],
                "environment": env_value,
                "trial_number": trial_number,
                "reward_outcome": reward_outcome,
                "lick_error": lick_error,
            }
        )

    neural_trials = []
    input_trials = []
    output_trials = []
    zone_counter = Counter()
    lick_error_trials = 0

    for trial_idx, info in enumerate(trial_info):
        start = info["start"]
        stop = info["stop"]

        if info["lick_error"]:
            lick_error_trials += 1
            continue

        n_frames = stop - start
        n_bins = n_frames // BIN_FRAMES
        if n_bins < 1:
            continue

        stop_trimmed = start + n_bins * BIN_FRAMES
        neural = bin_2d_mean(deconvolved[:, start:stop_trimmed], BIN_FRAMES)
        pos_binned = bin_1d_mean(position[start:stop_trimmed], BIN_FRAMES)
        speed_binned = bin_1d_mean(speed[start:stop_trimmed], BIN_FRAMES)

        lick_trial = lick_counts[start:stop_trimmed].copy()
        lick_trial[lick_trial > 1.0] = 1.0
        lick_binned = bin_1d_max(lick_trial, BIN_FRAMES)
        lick_binned = (lick_binned > 0).astype(np.uint8)

        if neural is None or pos_binned is None or speed_binned is None or lick_binned is None:
            continue

        neural = np.nan_to_num(neural, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
        pos_binned = np.clip(pos_binned, 0.0, TRACK_LENGTH_CM)
        speed_binned = np.nan_to_num(speed_binned, nan=0.0, posinf=0.0, neginf=0.0)

        time_from_start = (
            frame_times[start:stop_trimmed:BIN_FRAMES] - frame_times[start]
        ).astype(np.float32)
        prev_outcome = reward_outcomes[trial_idx - 1] if trial_idx > 0 else 0

        input_trial = np.vstack(
            [
                time_from_start,
                np.full(n_bins, info["environment"], dtype=np.float32),
                np.full(n_bins, info["trial_number"], dtype=np.float32),
                np.full(n_bins, prev_outcome, dtype=np.float32),
            ]
        ).astype(np.float32)

        zone_bounds = REWARD_ZONES[info["zone_label"]]
        distance_binned = reward_distance_cm(pos_binned.astype(np.float32), zone_bounds)
        output_trial = np.vstack(
            [
                discretize_reward_distance(distance_binned),
                discretize_position(pos_binned),
                discretize_speed(speed_binned),
                lick_binned,
                np.full(n_bins, ord(info["zone_label"]) - ord("A"), dtype=np.uint8),
                np.full(n_bins, info["reward_outcome"], dtype=np.uint8),
            ]
        ).astype(np.uint8)

        neural_trials.append(neural)
        input_trials.append(input_trial)
        output_trials.append(output_trial)
        zone_counter[info["zone_label"]] += 1

    plane_counts_curated = Counter(plane_idx_all[curated_cell_idx].tolist())
    plane_counts_kept = Counter(plane_idx.tolist())
    session_metadata = {
        "source_file": str(path),
        "subject": subject,
        "session_id": session_id,
        "scene": scene,
        "n_trials_total": len(trial_segments),
        "n_trials_kept": len(neural_trials),
        "n_trials_excluded_lick_error": lick_error_trials,
        "n_curated_cells": int(curated_cell_idx.size),
        "n_cells_kept": int(keep_cells.sum()),
        "n_putative_interneurons_excluded": int(is_interneuron.sum()),
        "putative_interneuron_fraction": float(is_interneuron.mean()) if curated_cell_idx.size else 0.0,
        "putative_interneuron_speed_corr_range": [
            float(np.min(speed_corr)) if speed_corr.size else 0.0,
            float(np.max(speed_corr)) if speed_corr.size else 0.0,
        ],
        "environment_mismatch_trials": env_mismatch_trials,
        "reward_zone_trial_counts_kept": dict(zone_counter),
        "plane_counts_curated": dict(plane_counts_curated),
        "plane_counts_kept": dict(plane_counts_kept),
    }

    return {
        "subject": subject,
        "session_id": session_id,
        "scene": scene,
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "brain_region_idx": np.zeros(int(keep_cells.sum()), dtype=np.int64),
        "metadata": session_metadata,
    }


def build_dataset():
    paths = sort_session_paths(glob.glob(str(DATA_ROOT / "sub-*" / "sub-*_behavior+ophys.nwb")))
    session_results = []

    for path_idx, path in enumerate(paths, start=1):
        print(f"[{path_idx:03d}/{len(paths):03d}] Converting {Path(path).name}", flush=True)
        converted = convert_session(path)
        if len(converted["neural"]) < 2:
            print(
                f"  Skipping {Path(path).name}: only {len(converted['neural'])} usable trial(s) after filtering",
                flush=True,
            )
            continue
        session_results.append(converted)
        gc.collect()

    subjects = sorted({session["subject"] for session in session_results}, key=lambda s: int(s[1:]))
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

    data = {
        "neural": [session["neural"] for session in session_results],
        "input": [session["input"] for session in session_results],
        "output": [session["output"] for session in session_results],
        "subjects": subjects,
        "subject_idx": np.asarray([subject_to_idx[session["subject"]] for session in session_results], dtype=np.int64),
        "brain_regions": ["CA1"],
        "brain_region_idx": [session["brain_region_idx"] for session in session_results],
        "input_names": [
            "time_from_trial_start_sec",
            "environment",
            "trial_number",
            "previous_trial_outcome",
        ],
        "output_names": [
            "distance_to_reward_zone",
            "absolute_position",
            "speed",
            "lick",
            "reward_zone_location",
            "reward_outcome",
        ],
        "output_values": [
            ["lt_-50_cm", "-50_to_lt_-10_cm", "-10_to_lt_0_cm", "0_cm", "gt_0_to_10_cm", "gt_10_to_50_cm", "gt_50_cm"],
            ["lt_90_cm", "90_to_lt_180_cm", "180_to_lt_270_cm", "270_to_lt_360_cm", "gte_360_cm"],
            ["lt_2_cm_per_s", "2_to_lt_10_cm_per_s", "10_to_lt_20_cm_per_s", "20_to_40_cm_per_s", "gt_40_cm_per_s"],
            ["no", "yes"],
            ["A", "B", "C"],
            ["no", "yes"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed mouse virtual-navigation task with hidden 50 cm reward zones that switch across days and environments; "
                "decoder predicts discretized reward-relative distance, absolute position, speed, licking, reward-zone identity, and reward outcome."
            ),
            "time_bin_size": float(TIME_BIN_SIZE_S * 1000.0),
            "temporal_alignment_event": "trial start (entry onto the 450 cm virtual linear track)",
            "off_start": 0.0,
            "off_end": None,
            "source_dataset": "Sosa et al. NWB release",
            "source_signal": "Suite2p deconvolved calcium activity (events)",
            "track_length_cm": TRACK_LENGTH_CM,
            "reward_zone_bounds_cm": {label: list(bounds) for label, bounds in REWARD_ZONES.items()},
            "bin_frames": BIN_FRAMES,
            "frame_rate_hz": FRAME_RATE_HZ,
            "lick_error_threshold_fraction": LICK_ERROR_THRESHOLD,
            "interneuron_exclusion": {
                "used": True,
                "criterion": "Pearson correlation between per-trial maximin dF/F and running speed > 0.5",
            },
            "session_info": [session["metadata"] for session in session_results],
        },
    }

    total_trials = sum(len(session) for session in data["neural"])
    total_bins = sum(trial.shape[1] for session in data["neural"] for trial in session)
    total_neurons = sum(session["brain_region_idx"].shape[0] for session in session_results)
    print(
        f"Built dataset with {len(data['neural'])} sessions, {total_trials} trials, {total_bins} time bins, and {total_neurons} neurons across sessions.",
        flush=True,
    )
    return data


def main():
    data = build_dataset()
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(data, f, protocol=4)
    print(f"Saved converted dataset to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
