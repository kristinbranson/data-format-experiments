#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
from suite2p.extraction import dcnv


DATA_ROOT = Path("/app/data")
FULL_OUTPUT = Path("/app/converted_data.pkl")
SAMPLE_OUTPUT = Path("/app/sample_data.pkl")

ZONE_TO_INDEX = {"A": 0, "B": 1, "C": 2}
ZONE_TO_BOUNDS_CM = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
SCENE_LOCATION_RE = re.compile(r"^Env(?P<env>\d+)_Location(?P<zone>[ABC])$")
SCENE_SWITCH_RE = re.compile(
    r"^Env(?P<env>\d+)_Location(?P<zone0>[ABC])_to_(?P<zone1>[ABC])$"
)
SCENE_ENV_SWITCH_RE = re.compile(
    r"^Env(?P<env0>\d+)_(?P<zone0>[ABC])_to_Env(?P<env1>\d+)_(?P<zone1>[ABC])$"
)
SAMPLE_SESSION_KEYS = {
    ("m3", 1),
    ("m3", 3),
    ("m3", 8),
    ("m3", 14),
    ("m11", 3),
    ("m11", 8),
    ("m11", 14),
    ("m17", 1),
    ("m17", 3),
    ("m17", 8),
    ("m17", 14),
}


@dataclass(frozen=True)
class TrialLabel:
    env: int
    zone: str


@dataclass(frozen=True)
class SessionMeta:
    file_path: Path
    subject: str
    session_num: int
    scene: str
    date: str


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def list_nwb_files(data_root: Path) -> list[SessionMeta]:
    session_meta: list[SessionMeta] = []
    for file_path in sorted(data_root.glob("sub-*/sub-*_behavior+ophys.nwb")):
        with h5py.File(file_path, "r") as f:
            subject = decode_scalar(f["general/subject/subject_id"][()])
            identifier = decode_scalar(f["identifier"][()])
        parts = identifier.strip("/").split("/")
        date = parts[-2]
        scene = parts[-1]
        session_num = int(re.search(r"ses-(\d+)", file_path.name).group(1))
        session_meta.append(
            SessionMeta(
                file_path=file_path,
                subject=subject,
                session_num=session_num,
                scene=scene,
                date=date,
            )
        )
    session_meta.sort(key=lambda s: (int(s.subject[1:]), s.session_num))
    return session_meta


def parse_scene(scene: str, n_trials: int, switch_trial_count: int = 30) -> list[TrialLabel]:
    match = SCENE_LOCATION_RE.match(scene)
    if match:
        env = int(match.group("env")) - 1
        zone = match.group("zone")
        return [TrialLabel(env=env, zone=zone) for _ in range(n_trials)]

    match = SCENE_SWITCH_RE.match(scene)
    if match:
        env = int(match.group("env")) - 1
        zone0 = match.group("zone0")
        zone1 = match.group("zone1")
        return [
            TrialLabel(env=env, zone=zone0 if trial_idx < switch_trial_count else zone1)
            for trial_idx in range(n_trials)
        ]

    match = SCENE_ENV_SWITCH_RE.match(scene)
    if match:
        env0 = int(match.group("env0")) - 1
        env1 = int(match.group("env1")) - 1
        zone0 = match.group("zone0")
        zone1 = match.group("zone1")
        return [
            TrialLabel(
                env=env0 if trial_idx < switch_trial_count else env1,
                zone=zone0 if trial_idx < switch_trial_count else zone1,
            )
            for trial_idx in range(n_trials)
        ]

    raise ValueError(f"Unrecognized scene format: {scene}")


def infer_frame_rate(position_timestamps: np.ndarray) -> float:
    diffs = np.diff(position_timestamps.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(1.0 / np.median(diffs))


def nansmooth_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    mask = np.isfinite(arr)
    filled = np.where(mask, arr, 0.0).astype(np.float32, copy=False)
    smooth = gaussian_filter1d(filled, sigma=sigma, axis=1, mode="nearest")
    weights = gaussian_filter1d(mask.astype(np.float32), sigma=sigma, axis=1, mode="nearest")
    return np.divide(
        smooth,
        weights,
        out=np.full_like(smooth, np.nan, dtype=np.float32),
        where=weights > 1e-8,
    )


def compute_dff_and_events(
    fluorescence: np.ndarray,
    neuropil: np.ndarray,
    start_idx: np.ndarray,
    stop_idx: np.ndarray,
    frame_rate_hz: float,
    neu_coef: float = 0.7,
    tau: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells, n_time = fluorescence.shape
    f = np.full((n_cells, n_time), np.nan, dtype=np.float32)
    f_neu = np.full((n_cells, n_time), np.nan, dtype=np.float32)
    for start, stop in zip(start_idx, stop_idx):
        f[:, start:stop] = fluorescence[:, start:stop]
        f_neu[:, start:stop] = neuropil[:, start:stop]

    f -= neu_coef * f_neu
    valid = np.isfinite(f[0])

    baseline = np.full_like(f, np.nan, dtype=np.float32)
    for start, stop in zip(start_idx, stop_idx):
        sl = slice(start, stop)
        f[:, sl] = f[:, sl] + neu_coef * np.nanmean(f_neu[:, sl], axis=1, keepdims=True)
        baseline[:, sl] = nansmooth_2d(f[:, sl], sigma=15)
        baseline[:, sl] = minimum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")
        baseline[:, sl] = maximum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")

    dff = np.full_like(f, np.nan, dtype=np.float32)
    dff[:, valid] = (f[:, valid] - baseline[:, valid]) / np.abs(baseline[:, valid])

    events = np.full_like(f, np.nan, dtype=np.float32)
    for start, stop in zip(start_idx, stop_idx):
        sl = slice(start, stop)
        dff[:, sl] = nansmooth_2d(dff[:, sl], sigma=2)
        events[:, sl] = dcnv.oasis(dff[:, sl], 2000, tau, frame_rate_hz)

    return dff, events


def discretize_distance_to_zone(position_cm: np.ndarray, zone_bounds: tuple[float, float]) -> np.ndarray:
    zone_start, zone_stop = zone_bounds
    signed_distance = np.where(
        position_cm < zone_start,
        position_cm - zone_start,
        np.where(position_cm > zone_stop, position_cm - zone_stop, 0.0),
    )
    bins = np.zeros_like(position_cm, dtype=np.int8)
    bins[signed_distance < -50.0] = 0
    bins[(signed_distance >= -50.0) & (signed_distance <= -10.0)] = 1
    bins[(signed_distance > -10.0) & (signed_distance < 0.0)] = 2
    bins[signed_distance == 0.0] = 3
    bins[(signed_distance > 0.0) & (signed_distance <= 10.0)] = 4
    bins[(signed_distance > 10.0) & (signed_distance <= 50.0)] = 5
    bins[signed_distance > 50.0] = 6
    return bins


def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, 449.999999)
    bins = np.floor(clipped / 90.0).astype(np.int8)
    bins[bins > 4] = 4
    return bins


def discretize_speed(speed_cm_s: np.ndarray) -> np.ndarray:
    bins = np.zeros_like(speed_cm_s, dtype=np.int8)
    bins[(speed_cm_s >= 2.0) & (speed_cm_s < 10.0)] = 1
    bins[(speed_cm_s >= 10.0) & (speed_cm_s < 20.0)] = 2
    bins[(speed_cm_s >= 20.0) & (speed_cm_s <= 40.0)] = 3
    bins[speed_cm_s > 40.0] = 4
    return bins


def bad_lick_trial_mask(lick_counts: np.ndarray, start_idx: np.ndarray, stop_idx: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(start_idx), dtype=bool)
    for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
        if np.mean(lick_counts[start:stop] > 2.0) > 0.3:
            mask[trial_idx] = True
    return mask


def reward_outcome_per_trial(reward_timestamps: np.ndarray, start_times: np.ndarray, stop_times: np.ndarray) -> np.ndarray:
    outcomes = np.zeros(len(start_times), dtype=np.int8)
    reward_idx = 0
    for trial_idx, (start_time, stop_time) in enumerate(zip(start_times, stop_times)):
        while reward_idx < len(reward_timestamps) and reward_timestamps[reward_idx] < start_time:
            reward_idx += 1
        probe_idx = reward_idx
        while probe_idx < len(reward_timestamps) and reward_timestamps[probe_idx] < stop_time:
            outcomes[trial_idx] = 1
            probe_idx += 1
    return outcomes


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def process_session(session_meta: SessionMeta, block_size: int = 256) -> tuple[dict, dict]:
    with h5py.File(session_meta.file_path, "r") as f:
        behavior = f["processing/behavior/BehavioralTimeSeries"]
        ophys = f["processing/ophys"]
        imgseg = ophys["ImageSegmentation"]["PlaneSegmentation"]
        plane_keys = sorted(ophys["Fluorescence"].keys())
        plane_lengths = [ophys["Fluorescence"][plane_key]["data"].shape[0] for plane_key in plane_keys]

        position = behavior["position"]["data"][:].astype(np.float32)
        speed = behavior["speed"]["data"][:].astype(np.float32)
        lick_counts = behavior["lick"]["data"][:].astype(np.float32)
        env_timeseries = behavior["environment"]["data"][:].astype(np.float32)
        trial_number_series = behavior["trial number"]["data"][:].astype(np.float32)
        trial_start_series = behavior["trial_start"]["data"][:]
        teleport_series = behavior["teleport"]["data"][:]
        timestamps = behavior["position"]["timestamps"][:].astype(np.float64)
        reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
        session_len = min(
            len(position),
            len(speed),
            len(lick_counts),
            len(env_timeseries),
            len(trial_number_series),
            len(trial_start_series),
            len(teleport_series),
            len(timestamps),
            *plane_lengths,
        )
        position = position[:session_len]
        speed = speed[:session_len]
        lick_counts = lick_counts[:session_len]
        env_timeseries = env_timeseries[:session_len]
        trial_number_series = trial_number_series[:session_len]
        trial_start_series = trial_start_series[:session_len]
        teleport_series = teleport_series[:session_len]
        timestamps = timestamps[:session_len]

        frame_rate_hz = infer_frame_rate(timestamps)
        trial_start_idx = np.where(trial_start_series > 0)[0]
        trial_stop_idx = np.where(teleport_series > 0)[0]
        trial_labels = parse_scene(session_meta.scene, len(trial_start_idx))
        reward_outcomes = reward_outcome_per_trial(
            reward_timestamps,
            timestamps[trial_start_idx],
            timestamps[trial_stop_idx],
        )
        prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]]).astype(np.float32)
        lick_error_mask = bad_lick_trial_mask(lick_counts, trial_start_idx, trial_stop_idx)
        kept_trial_indices = np.where(~lick_error_mask)[0]

        plane_idx = imgseg["planeIdx"][:].astype(np.int16)
        iscell = imgseg["iscell"][:]

        session_trial_blocks: list[list[np.ndarray]] = [[] for _ in kept_trial_indices]
        interneuron_flags: list[np.ndarray] = []
        curated_counts_by_plane: list[int] = []
        kept_counts_by_plane: list[int] = []

        row_offset = 0
        for plane_key in plane_keys:
            fluorescence_ds = ophys["Fluorescence"][plane_key]["data"]
            neuropil_ds = ophys["Neuropil"][plane_key]["data"]
            plane_cell_count = fluorescence_ds.shape[1]
            plane_slice = slice(row_offset, row_offset + plane_cell_count)
            plane_iscell = iscell[plane_slice, 0] == 1
            curated_counts_by_plane.append(int(plane_iscell.sum()))
            curated_plane_indices = np.where(plane_iscell)[0]
            kept_this_plane = 0

            for block_start in range(0, len(curated_plane_indices), block_size):
                block_local = curated_plane_indices[block_start:block_start + block_size]
                if block_local.size == 0:
                    continue
                fluorescence = fluorescence_ds[:session_len, block_local].T.astype(np.float32)
                neuropil = neuropil_ds[:session_len, block_local].T.astype(np.float32)
                dff, events = compute_dff_and_events(
                    fluorescence=fluorescence,
                    neuropil=neuropil,
                    start_idx=trial_start_idx,
                    stop_idx=trial_stop_idx,
                    frame_rate_hz=frame_rate_hz,
                )
                valid_mask = np.isfinite(dff[0])
                speed_valid = speed[valid_mask]
                dff_valid = dff[:, valid_mask]
                block_corr = np.array(
                    [
                        np.corrcoef(trace, speed_valid)[0, 1]
                        if np.nanstd(trace) > 0 and np.nanstd(speed_valid) > 0
                        else np.nan
                        for trace in dff_valid
                    ],
                    dtype=np.float32,
                )
                block_interneuron = np.isfinite(block_corr) & (block_corr > 0.5)
                interneuron_flags.append(block_interneuron)
                keep_block = ~block_interneuron
                kept_this_plane += int(keep_block.sum())
                if not np.any(keep_block):
                    continue

                kept_events = events[keep_block]
                for out_idx, trial_idx in enumerate(kept_trial_indices):
                    start = trial_start_idx[trial_idx]
                    stop = trial_stop_idx[trial_idx]
                    session_trial_blocks[out_idx].append(
                        kept_events[:, start:stop].astype(np.float16, copy=False)
                    )

            row_offset += plane_cell_count
            kept_counts_by_plane.append(kept_this_plane)

        neural_trials: list[np.ndarray] = []
        input_trials: list[np.ndarray] = []
        output_trials: list[np.ndarray] = []

        for out_idx, trial_idx in enumerate(kept_trial_indices):
            block_list = session_trial_blocks[out_idx]
            if not block_list:
                raise RuntimeError(f"No neurons kept for {session_meta.file_path.name}, trial {trial_idx + 1}")
            neural_trial = np.concatenate(block_list, axis=0).astype(np.float16, copy=False)

            start = trial_start_idx[trial_idx]
            stop = trial_stop_idx[trial_idx]
            trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
            trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
            trial_speed = speed[start:stop].astype(np.float32)
            trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
            label = trial_labels[trial_idx]
            zone_idx = ZONE_TO_INDEX[label.zone]
            zone_bounds = ZONE_TO_BOUNDS_CM[label.zone]

            trial_input = np.vstack(
                [
                    trial_times,
                    np.full(trial_times.shape, float(label.env), dtype=np.float32),
                    np.full(trial_times.shape, trial_number_series[start], dtype=np.float32),
                    np.full(trial_times.shape, prev_reward_outcomes[trial_idx], dtype=np.float32),
                ]
            )

            trial_output = np.vstack(
                [
                    discretize_distance_to_zone(trial_pos, zone_bounds),
                    discretize_absolute_position(trial_pos),
                    discretize_speed(trial_speed),
                    trial_lick,
                    np.full(trial_times.shape, zone_idx, dtype=np.int8),
                    np.full(trial_times.shape, reward_outcomes[trial_idx], dtype=np.int8),
                ]
            )

            env_in_trial = env_timeseries[start:stop]
            env_in_trial = env_in_trial[env_in_trial >= 0]
            if env_in_trial.size:
                env_mode = int(round(float(np.median(env_in_trial))))
                if env_mode != label.env:
                    raise RuntimeError(
                        f"Environment mismatch in {session_meta.file_path.name} trial {trial_idx + 1}: "
                        f"scene-derived {label.env}, behavior-derived {env_mode}"
                    )

            neural_trials.append(neural_trial)
            input_trials.append(trial_input)
            output_trials.append(trial_output)

        n_kept_neurons = int(sum(kept_counts_by_plane))
        all_interneurons = np.concatenate(interneuron_flags) if interneuron_flags else np.zeros(0, dtype=bool)
        stats = {
            "subject": session_meta.subject,
            "session_num": session_meta.session_num,
            "scene": session_meta.scene,
            "file": session_meta.file_path.name,
            "frame_rate_hz": frame_rate_hz,
            "n_trials_total": int(len(trial_start_idx)),
            "n_trials_removed_lick_error": int(lick_error_mask.sum()),
            "n_trials_kept": int(len(kept_trial_indices)),
            "n_rewarded_trials_total": int(reward_outcomes.sum()),
            "n_curated_rois": int(sum(curated_counts_by_plane)),
            "n_interneurons_removed": int(all_interneurons.sum()),
            "n_neurons_kept": int(n_kept_neurons),
            "interneuron_fraction_of_curated": float(all_interneurons.sum() / max(sum(curated_counts_by_plane), 1)),
            "trial_lengths": [int(arr.shape[1]) for arr in neural_trials],
            "plane_keys": plane_keys,
            "plane_curated_counts": curated_counts_by_plane,
            "plane_kept_counts": kept_counts_by_plane,
            "multi_plane": len(plane_keys) > 1,
        }
        session_data = {
            "neural": neural_trials,
            "input": input_trials,
            "output": output_trials,
            "brain_region_idx": np.zeros(n_kept_neurons, dtype=np.int16),
            "stats": stats,
        }
        return session_data, stats


def build_dataset(session_meta: list[SessionMeta], block_size: int) -> tuple[dict, list[dict]]:
    subjects = sorted({meta.subject for meta in session_meta}, key=lambda s: int(s[1:]))
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

    dataset = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": [],
        "brain_regions": ["CA1"],
        "brain_region_idx": [],
        "input_names": [
            "time_from_trial_start_s",
            "environment_type",
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
            ["lt_-50cm", "-50_to_-10cm", "-10_to_0cm", "in_reward_zone", "0_to_10cm", "10_to_50cm", "gt_50cm"],
            ["0_to_90cm", "90_to_180cm", "180_to_270cm", "270_to_360cm", "360_to_450cm"],
            ["lt_2cm_s", "2_to_10cm_s", "10_to_20cm_s", "20_to_40cm_s", "gt_40cm_s"],
            ["no", "yes"],
            ["A", "B", "C"],
            ["omitted", "rewarded"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed virtual-reality navigation on a 450 cm linear corridor with hidden reward zones "
                "that switch location across experience and across two environments."
            ),
            "time_bin_size": None,
            "temporal_alignment_event": "trial start",
            "off_start": 0.0,
            "off_end": None,
            "neural_signal": (
                "Suite2p ROI fluorescence with 0.7 neuropil subtraction, per-trial maximin dF/F baseline, "
                "2-sample Gaussian smoothing, and OASIS deconvolution."
            ),
            "trial_exclusion": (
                "Trials were dropped when >30% of frame samples had lick counts >2, matching the paper's lick-sensor error criterion."
            ),
            "neuron_exclusion": (
                "Kept curated Suite2p iscell ROIs and excluded putative interneurons with dF/F-speed correlation > 0.5."
            ),
        },
    }

    session_stats: list[dict] = []
    for meta in session_meta:
        print(f"Processing {meta.file_path.name} ({meta.subject}, scene={meta.scene})")
        session_data, stats = process_session(meta, block_size=block_size)
        dataset["neural"].append(session_data["neural"])
        dataset["input"].append(session_data["input"])
        dataset["output"].append(session_data["output"])
        dataset["subject_idx"].append(subject_to_idx[meta.subject])
        dataset["brain_region_idx"].append(session_data["brain_region_idx"])
        session_stats.append(stats)

    dataset["subject_idx"] = np.asarray(dataset["subject_idx"], dtype=np.int16)
    mean_frame_rate = float(np.mean([stats["frame_rate_hz"] for stats in session_stats]))
    dataset["metadata"]["time_bin_size"] = 1000.0 / mean_frame_rate
    dataset["metadata"]["session_info"] = session_stats
    dataset["metadata"]["n_sessions"] = len(session_stats)
    dataset["metadata"]["n_subjects"] = len(subjects)
    dataset["metadata"]["switch_trial_count"] = 30
    return dataset, session_stats


def make_sample_dataset(full_dataset: dict, sample_session_keys: set[tuple[str, int]], trials_per_session: int = 10) -> dict:
    session_info = full_dataset["metadata"]["session_info"]
    keep_session_indices = [
        idx
        for idx, info in enumerate(session_info)
        if (info["subject"], info["session_num"]) in sample_session_keys
    ]

    sample = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": full_dataset["subjects"],
        "subject_idx": [],
        "brain_regions": full_dataset["brain_regions"],
        "brain_region_idx": [],
        "input_names": full_dataset["input_names"],
        "output_names": full_dataset["output_names"],
        "output_values": full_dataset["output_values"],
        "metadata": dict(full_dataset["metadata"]),
    }

    new_session_info = []
    for idx in keep_session_indices:
        neural_trials = full_dataset["neural"][idx][:trials_per_session]
        input_trials = full_dataset["input"][idx][:trials_per_session]
        output_trials = full_dataset["output"][idx][:trials_per_session]
        if len(neural_trials) < 2:
            continue
        sample["neural"].append(neural_trials)
        sample["input"].append(input_trials)
        sample["output"].append(output_trials)
        sample["subject_idx"].append(full_dataset["subject_idx"][idx])
        sample["brain_region_idx"].append(full_dataset["brain_region_idx"][idx])
        info = dict(session_info[idx])
        info["n_trials_kept"] = len(neural_trials)
        info["sample_subset"] = True
        new_session_info.append(info)

    sample["subject_idx"] = np.asarray(sample["subject_idx"], dtype=np.int16)
    sample["metadata"]["session_info"] = new_session_info
    sample["metadata"]["n_sessions"] = len(new_session_info)
    sample["metadata"]["sample_description"] = (
        "Representative subset spanning early, switch, environment-switch, and late sessions, "
        "including both single-plane and multi-plane recordings."
    )
    return sample


def print_summary(session_stats: list[dict]) -> None:
    total_trials = sum(stats["n_trials_total"] for stats in session_stats)
    kept_trials = sum(stats["n_trials_kept"] for stats in session_stats)
    removed_trials = sum(stats["n_trials_removed_lick_error"] for stats in session_stats)
    rewarded_trials = sum(stats["n_rewarded_trials_total"] for stats in session_stats)
    trials_mean, trials_std = mean_std(stats["n_trials_total"] for stats in session_stats)
    kept_mean, kept_std = mean_std(stats["n_trials_kept"] for stats in session_stats)
    curated_mean, curated_std = mean_std(stats["n_curated_rois"] for stats in session_stats)
    kept_neurons_mean, kept_neurons_std = mean_std(stats["n_neurons_kept"] for stats in session_stats)
    int_frac_mean, int_frac_std = mean_std(stats["interneuron_fraction_of_curated"] * 100.0 for stats in session_stats)
    frame_rate_mean, frame_rate_std = mean_std(stats["frame_rate_hz"] for stats in session_stats)
    total_timepoints = sum(sum(stats["trial_lengths"]) for stats in session_stats)
    trial_lengths = [length for stats in session_stats for length in stats["trial_lengths"]]
    trial_len_mean, trial_len_std = mean_std(trial_lengths)
    multi_plane_sessions = sum(int(stats["multi_plane"]) for stats in session_stats)

    print(f"Sessions processed: {len(session_stats)}")
    print(f"Total trials before lick-error exclusion: {total_trials}")
    print(f"Total trials kept: {kept_trials}")
    print(f"Trials removed by lick-error criterion (>30% frames with lick count >2): {removed_trials}")
    print(f"Trials/session mean±sd before exclusion: {trials_mean:.2f} ± {trials_std:.2f}")
    print(f"Trials/session mean±sd after exclusion: {kept_mean:.2f} ± {kept_std:.2f}")
    print(f"Rewarded trial fraction before exclusion: {rewarded_trials / total_trials:.4f}")
    print(f"Curated neurons/session mean±sd: {curated_mean:.2f} ± {curated_std:.2f}")
    print(f"Kept neurons/session mean±sd: {kept_neurons_mean:.2f} ± {kept_neurons_std:.2f}")
    print(f"Interneuron exclusion fraction mean±sd: {int_frac_mean:.3f}% ± {int_frac_std:.3f}%")
    print(f"Frame rate mean±sd (Hz): {frame_rate_mean:.6f} ± {frame_rate_std:.6f}")
    print(f"Total kept trial timepoints: {total_timepoints}")
    print(f"Trial length mean±sd (frames): {trial_len_mean:.2f} ± {trial_len_std:.2f}")
    print(f"Multi-plane sessions: {multi_plane_sessions}")
    shortest = min(session_stats, key=lambda x: x["n_trials_total"])
    print(
        "Shortest session by trial count: "
        f"{shortest['file']} ({shortest['subject']}, {shortest['scene']}), "
        f"{shortest['n_trials_total']} trials"
    )


def save_pickle(path: Path, obj: dict) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NWB sessions into decoder-ready pickles.")
    parser.add_argument("--full-output", type=Path, default=FULL_OUTPUT)
    parser.add_argument("--sample-output", type=Path, default=SAMPLE_OUTPUT)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--sample-from-full", type=Path, default=None)
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--sample-only", action="store_true")
    args = parser.parse_args()

    if args.full_only and args.sample_only:
        raise ValueError("Choose at most one of --full-only and --sample-only")

    if args.sample_only and args.sample_from_full is not None:
        with args.sample_from_full.open("rb") as f:
            full_dataset = pickle.load(f)
        session_stats = full_dataset["metadata"]["session_info"]
        print_summary(session_stats)
    else:
        sessions = list_nwb_files(DATA_ROOT)
        if args.max_sessions is not None:
            sessions = sessions[: args.max_sessions]
        full_dataset, session_stats = build_dataset(sessions, block_size=args.block_size)
        print_summary(session_stats)

        if not args.sample_only:
            save_pickle(args.full_output, full_dataset)
            print(f"Saved full dataset to {args.full_output}")

    if not args.full_only:
        sample_dataset = make_sample_dataset(full_dataset, SAMPLE_SESSION_KEYS, trials_per_session=10)
        save_pickle(args.sample_output, sample_dataset)
        print(f"Saved sample dataset to {args.sample_output}")


if __name__ == "__main__":
    main()
