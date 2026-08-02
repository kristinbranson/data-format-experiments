#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

import joblib
import numpy as np


ANIMALS = [
    "QLAK-CA1-08",
    "QLAK-CA1-30",
    "QLAK-CA1-50",
    "QLAK-CA1-51",
    "QLAK-CA1-56",
    "QLAK-CA1-74",
    "QLAK-CA1-75",
]

FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS
TRIAL_SECONDS = 60.0
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)
NOMINAL_SESSION_SECONDS = 40.0 * 60.0
NOMINAL_SESSION_FRAMES = int(FPS * NOMINAL_SESSION_SECONDS)
N_TRIALS_PER_SESSION = 40
N_POSITION_BINS = 3
N_SPATIAL_CLASSES = N_POSITION_BINS * N_POSITION_BINS
DATA_DIR = "/app/data"


def get_env_mat(env: str) -> np.ndarray:
    if env == "square":
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=float)
    if env == "o":
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float)
    if env == "t":
        return np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=float)
    if env == "u":
        return np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=float)
    if env == "rectangle":
        return np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=float)
    if env == "+":
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=float)
    if env == "i":
        return np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]], dtype=float)
    if env == "l":
        return np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
    if env == "bit donut":
        return np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1]], dtype=float)
    if env == "glenn":
        return np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=float)
    raise ValueError(f"Unknown environment {env!r}")


def env_to_blocked_mask(env: str) -> np.ndarray:
    # The repository's blocked indices follow bottom-up row-major order.
    return (np.flipud(get_env_mat(env)).reshape(-1) == 0).astype(np.float32)


def flatten_numeric(value: Any) -> list[float]:
    out: list[float] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
            continue
        arr = np.asarray(item)
        if arr.dtype == object:
            stack.extend(reversed(arr.ravel().tolist()))
            continue
        out.extend(arr.astype(float).ravel().tolist())
    return out


def parse_env_label(value: Any) -> str:
    arr = np.asarray(value, dtype=object).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"Expected a single env label, got {arr}")
    label = arr[0]
    if isinstance(label, bytes):
        return label.decode("utf-8")
    return str(label)


def parse_blocked_mask(value: Any) -> np.ndarray:
    flat = np.array(flatten_numeric(value), dtype=float)
    if flat.size == 0:
        raise ValueError("Blocked entry was empty")
    if flat.size == 1 and np.isclose(flat[0], -1.0):
        return np.zeros(9, dtype=np.float32)
    blocked = np.unique(flat.astype(int))
    if np.any((blocked < 0) | (blocked > 8)):
        raise ValueError(f"Blocked indices out of range: {blocked.tolist()}")
    mask = np.zeros(9, dtype=np.float32)
    mask[blocked] = 1.0
    return mask


def load_animal(animal: str) -> dict[str, Any]:
    return joblib.load(os.path.join(DATA_DIR, animal))[animal]


def calibrate_animal_arena(dat: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    envs = [parse_env_label(v) for v in dat["envs"]]
    square_days = [idx for idx, env in enumerate(envs) if env == "square"]
    if not square_days:
        raise ValueError("No square sessions found for arena calibration")
    square_positions = [np.asarray(dat["position"][day], dtype=np.float64).T for day in square_days]
    all_square = np.concatenate(square_positions, axis=0)
    arena_min = np.nanmin(all_square, axis=0)
    shifted = all_square - arena_min[None, :]
    arena_side = float(np.nanmax(shifted))
    if not np.isfinite(arena_side) or arena_side <= 0.0:
        raise ValueError(f"Invalid arena side length {arena_side}")
    return arena_min.astype(np.float64), arena_side / N_POSITION_BINS, arena_side


def position_to_bins(
    position_xy: np.ndarray,
    arena_min: np.ndarray,
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    shifted = position_xy - arena_min[None, :]
    coords = np.floor(shifted / bin_size).astype(np.int64)
    coords = np.clip(coords, 0, N_POSITION_BINS - 1)
    # Output classes use bottom-up row-major indexing: y * 3 + x.
    bin_ids = coords[:, 1] * N_POSITION_BINS + coords[:, 0]
    return coords, bin_ids


def clean_blocked_bins(
    position_xy: np.ndarray,
    bin_ids: np.ndarray,
    blocked_mask: np.ndarray,
    arena_min: np.ndarray,
    bin_size: float,
) -> tuple[np.ndarray, int]:
    blocked_ids = np.flatnonzero(blocked_mask > 0.5)
    if blocked_ids.size == 0:
        return bin_ids, 0
    bad = np.isin(bin_ids, blocked_ids)
    if not np.any(bad):
        return bin_ids, 0

    cleaned = bin_ids.copy()
    open_ids = np.flatnonzero(blocked_mask < 0.5)
    centers = []
    for open_id in open_ids:
        y_bin = open_id // N_POSITION_BINS
        x_bin = open_id % N_POSITION_BINS
        centers.append(
            np.array(
                [
                    arena_min[0] + (x_bin + 0.5) * bin_size,
                    arena_min[1] + (y_bin + 0.5) * bin_size,
                ],
                dtype=np.float64,
            )
        )
    centers = np.stack(centers, axis=0)
    for idx in np.flatnonzero(bad):
        d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
        cleaned[idx] = int(open_ids[int(np.argmin(d2))])
    return cleaned, int(np.sum(bad))


def build_trial_slices(n_timepoints: int) -> list[slice]:
    if n_timepoints < (N_TRIALS_PER_SESSION - 1) * TRIAL_FRAMES + 2:
        raise ValueError(f"Session too short for 40 nominal trials: {n_timepoints} frames")
    trial_slices: list[slice] = []
    for trial_idx in range(N_TRIALS_PER_SESSION):
        start = trial_idx * TRIAL_FRAMES
        end = min((trial_idx + 1) * TRIAL_FRAMES, n_timepoints)
        if end <= start:
            raise ValueError(f"Invalid trial slice {trial_idx}: {start}:{end}")
        trial_slices.append(slice(start, end))
    return trial_slices


def deep_subset_dataset(data: dict[str, Any], session_indices: list[int]) -> dict[str, Any]:
    subset: dict[str, Any] = {}
    for key, value in data.items():
        if key in {"neural", "input", "output", "brain_region_idx"}:
            subset[key] = [deepcopy(value[idx]) for idx in session_indices]
        elif key == "subject_idx":
            subset[key] = np.asarray(value[session_indices], dtype=np.int64)
        elif key == "metadata":
            subset[key] = deepcopy(value)
            if "session_info" in subset[key]:
                subset[key]["session_info"] = [deepcopy(value["session_info"][idx]) for idx in session_indices]
        else:
            subset[key] = deepcopy(value)
    return subset


def build_dataset() -> tuple[dict[str, Any], dict[str, Any]]:
    subjects = list(ANIMALS)
    subject_lookup = {animal: idx for idx, animal in enumerate(subjects)}

    data: dict[str, Any] = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": [],
        "brain_regions": ["CA1"],
        "brain_region_idx": [],
        "input_names": [f"blocked_bin_{idx}" for idx in range(9)],
        "output_names": ["spatial_bin"],
        "output_values": [[f"bin_{idx}" for idx in range(N_SPATIAL_CLASSES)]],
        "metadata": {
            "task_description": (
                "CA1 calcium-event decoding of the mouse's 3x3 spatial bin from 30 Hz rise-extracted neural activity, "
                "with the 3x3 blocked-partition geometry as the static decoder input."
            ),
            "time_bin_size": TIME_BIN_MS,
            "temporal_alignment_event": "Start of each consecutive 1-minute segment within a recording day",
            "off_start": 0.0,
            "off_end": TRIAL_SECONDS,
            "nominal_recording_duration_s": NOMINAL_SESSION_SECONDS,
            "nominal_trial_duration_s": TRIAL_SECONDS,
            "fps": FPS,
            "source_dataset_files": [f"{animal}" for animal in ANIMALS],
            "source_reference": "Lee, Keinath, Cianfarano and Brandon (2025), georepca1 dataset and code",
            "session_info": [],
        },
    }

    stats: dict[str, Any] = {
        "animals": {},
        "total_unique_neurons": 0,
        "total_session_neuron_observations": 0,
        "total_sessions": 0,
        "total_trials": 0,
        "short_final_trials": 0,
        "max_trial_length": 0,
        "min_trial_length": math.inf,
        "dropped_trailing_frames": 0,
        "snapped_blocked_frames": 0,
        "blocked_env_mismatches": [],
        "session_env_counts": Counter(),
        "arena_side_lengths": {},
    }

    for animal in ANIMALS:
        dat = load_animal(animal)
        envs = [parse_env_label(v) for v in dat["envs"]]
        arena_min, bin_size, arena_side = calibrate_animal_arena(dat)
        stats["arena_side_lengths"][animal] = arena_side
        stats["total_unique_neurons"] += int(dat["trace"][0].shape[0])

        animal_stats = {
            "n_days": len(envs),
            "registered_neurons_per_day": [],
            "day_frame_counts": [],
            "kept_frame_counts": [],
            "dropped_trailing_frames": 0,
            "snapped_blocked_frames": 0,
            "environments": envs,
        }

        for day_idx, env in enumerate(envs):
            blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
            expected_mask = env_to_blocked_mask(env)
            if not np.array_equal(blocked_mask, expected_mask):
                stats["blocked_env_mismatches"].append(
                    {"animal": animal, "day": day_idx, "env": env, "blocked_mask": blocked_mask.tolist()}
                )

            position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
            trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)

            registered = ~np.isnan(trace).any(axis=1)
            if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
                raise ValueError(f"Found partially NaN neurons in {animal} day {day_idx}")
            trace = trace[registered]

            original_frames = int(position.shape[0])
            keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
            dropped_frames = original_frames - keep_frames
            animal_stats["dropped_trailing_frames"] += dropped_frames
            stats["dropped_trailing_frames"] += dropped_frames
            animal_stats["day_frame_counts"].append(original_frames)
            animal_stats["kept_frame_counts"].append(keep_frames)

            position = position[:keep_frames]
            trace = trace[:, :keep_frames]
            if trace.shape[1] != position.shape[0]:
                raise ValueError(f"Trace/position length mismatch for {animal} day {day_idx}")

            _, spatial_bins = position_to_bins(position, arena_min, bin_size)
            spatial_bins, snapped = clean_blocked_bins(position, spatial_bins, blocked_mask, arena_min, bin_size)
            animal_stats["snapped_blocked_frames"] += snapped
            stats["snapped_blocked_frames"] += snapped
            if np.any(blocked_mask[spatial_bins] > 0.5):
                raise ValueError(f"Blocked bins remained after cleaning for {animal} day {day_idx}")

            trial_slices = build_trial_slices(keep_frames)
            session_neural: list[np.ndarray] = []
            session_input: list[np.ndarray] = []
            session_output: list[np.ndarray] = []
            for trial_slice in trial_slices:
                neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
                output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
                input_trial = blocked_mask.astype(np.float32).copy()

                if np.isnan(neural_trial).any() or np.isnan(output_trial).any() or np.isnan(input_trial).any():
                    raise ValueError(f"NaNs found after conversion for {animal} day {day_idx}")
                if neural_trial.shape[1] != output_trial.shape[1]:
                    raise ValueError(f"Trial length mismatch for {animal} day {day_idx}")

                stats["max_trial_length"] = max(stats["max_trial_length"], int(neural_trial.shape[1]))
                stats["min_trial_length"] = min(stats["min_trial_length"], int(neural_trial.shape[1]))
                if neural_trial.shape[1] < TRIAL_FRAMES:
                    stats["short_final_trials"] += 1

                session_neural.append(neural_trial)
                session_input.append(input_trial)
                session_output.append(output_trial)

            session_neurons = int(trace.shape[0])
            animal_stats["registered_neurons_per_day"].append(session_neurons)
            stats["total_session_neuron_observations"] += session_neurons
            stats["total_sessions"] += 1
            stats["total_trials"] += len(session_neural)
            stats["session_env_counts"][env] += 1

            data["neural"].append(session_neural)
            data["input"].append(session_input)
            data["output"].append(session_output)
            data["subject_idx"].append(subject_lookup[animal])
            data["brain_region_idx"].append(np.zeros(session_neurons, dtype=np.int64))
            data["metadata"]["session_info"].append(
                {
                    "animal": animal,
                    "day_index_within_animal": day_idx,
                    "environment": env,
                    "n_trials": len(session_neural),
                    "original_frames": original_frames,
                    "kept_frames": keep_frames,
                    "dropped_trailing_frames": dropped_frames,
                    "registered_neurons": session_neurons,
                    "snapped_blocked_frames": snapped,
                    "arena_min_xy": arena_min.tolist(),
                    "arena_side_length": arena_side,
                    "bin_size": bin_size,
                }
            )

        stats["animals"][animal] = animal_stats

    data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int64)

    stats["session_env_counts"] = dict(stats["session_env_counts"])
    stats["min_trial_length"] = int(stats["min_trial_length"])
    stats["max_trial_length"] = int(stats["max_trial_length"])
    stats["total_subjects"] = len(subjects)
    stats["total_brain_regions"] = len(data["brain_regions"])
    stats["n_trials_per_session_unique"] = sorted({len(session) for session in data["neural"]})
    stats["session_neuron_range"] = [
        int(min(len(br) for br in data["brain_region_idx"])),
        int(max(len(br) for br in data["brain_region_idx"])),
    ]

    return data, stats


def choose_sample_sessions(data: dict[str, Any]) -> list[int]:
    wanted = []
    for target_animal in ("QLAK-CA1-08", "QLAK-CA1-51"):
        for session_idx, info in enumerate(data["metadata"]["session_info"]):
            if info["animal"] == target_animal and info["day_index_within_animal"] < 10:
                wanted.append(session_idx)
    return wanted


def run_decoder_verify(data: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    from decoder import verify_data_format

    valid, errors, warnings = verify_data_format(data)
    return bool(valid), list(errors), list(warnings)


def save_pickle(path: str, obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def print_summary(stats: dict[str, Any]) -> None:
    print("Conversion summary")
    print(f"  animals: {stats['total_subjects']}")
    print(f"  sessions: {stats['total_sessions']}")
    print(f"  trials: {stats['total_trials']}")
    print(f"  unique neurons across animals: {stats['total_unique_neurons']}")
    print(f"  session neuron observations: {stats['total_session_neuron_observations']}")
    print(f"  trial lengths: min={stats['min_trial_length']} max={stats['max_trial_length']}")
    print(f"  short final trials: {stats['short_final_trials']}")
    print(f"  dropped trailing frames: {stats['dropped_trailing_frames']}")
    print(f"  snapped blocked-bin frames: {stats['snapped_blocked_frames']}")
    print(f"  environment counts: {stats['session_env_counts']}")
    print("  arena side lengths from square sessions:")
    for animal in ANIMALS:
        print(f"    {animal}: {stats['arena_side_lengths'][animal]:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert georepca1 CA1 data into decoder format.")
    parser.add_argument("--output", default="/app/converted_data.pkl", help="Output pickle path for the full dataset.")
    parser.add_argument("--sample-output", default="/app/sample_data.pkl", help="Output pickle path for the sample dataset.")
    parser.add_argument("--stats-json", default=None, help="Optional JSON path for conversion statistics.")
    args = parser.parse_args()

    data, stats = build_dataset()

    valid, errors, warnings = run_decoder_verify(data)
    stats["verify_data_format"] = {"valid": valid, "errors": errors, "warnings": warnings}
    if not valid:
        raise RuntimeError(f"Converted dataset failed format verification: {errors}")

    save_pickle(args.output, data)
    sample_session_indices = choose_sample_sessions(data)
    sample_data = deep_subset_dataset(data, sample_session_indices)
    save_pickle(args.sample_output, sample_data)

    stats["sample_session_indices"] = sample_session_indices
    stats["sample_num_sessions"] = len(sample_session_indices)
    stats["sample_subjects"] = sorted({data["metadata"]["session_info"][idx]["animal"] for idx in sample_session_indices})

    print_summary(stats)
    print("Format verification")
    print(f"  valid: {valid}")
    print(f"  warnings: {len(warnings)}")
    if warnings:
        for warning in warnings:
            print(f"    - {warning}")

    if args.stats_json:
        with open(args.stats_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
