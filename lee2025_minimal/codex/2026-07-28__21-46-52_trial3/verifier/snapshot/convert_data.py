#!/usr/bin/env python3
import argparse
import json
import math
import pickle
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d


ANIMALS = [
    "QLAK-CA1-08",
    "QLAK-CA1-30",
    "QLAK-CA1-50",
    "QLAK-CA1-51",
    "QLAK-CA1-56",
    "QLAK-CA1-74",
    "QLAK-CA1-75",
]

EXPECTED_ENV_COUNTS = {
    "square": 27,
    "o": 20,
    "t": 20,
    "u": 20,
    "rectangle": 20,
    "+": 20,
    "i": 20,
    "l": 20,
    "bit donut": 20,
    "glenn": 20,
}

SAMPLE_SELECTION = {
    "QLAK-CA1-08": list(range(11)),
}

FPS = 30
TRIAL_SECONDS = 60
TRIAL_FRAMES = FPS * TRIAL_SECONDS
TEMPORAL_BIN_FRAMES = 3
TEMPORAL_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS
TRACE_SMOOTH_SIGMA_FRAMES = 3
VELOCITY_SMOOTH_SIGMA_FRAMES = 5
VELOCITY_THRESHOLD_CM_PER_S = 5.0
POSITION_BINS_PER_AXIS = 3
POSITION_BUFFER = 1e-15
MIN_POOLED_TIMEPOINTS_PER_TRIAL = 10


def get_env_mat(env_name: str) -> np.ndarray:
    if env_name == "square":
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    if env_name == "o":
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    if env_name == "t":
        return np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=np.float32)
    if env_name == "u":
        return np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=np.float32)
    if env_name == "rectangle":
        return np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=np.float32)
    if env_name == "+":
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float32)
    if env_name == "i":
        return np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]], dtype=np.float32)
    if env_name == "l":
        return np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32)
    if env_name == "bit donut":
        return np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float32)
    if env_name == "glenn":
        return np.array([[0, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=np.float32)
    raise ValueError(f"Unknown environment name: {env_name}")


def flatten_env_names() -> list[str]:
    names = []
    for r in range(POSITION_BINS_PER_AXIS):
        for c in range(POSITION_BINS_PER_AXIS):
            names.append(f"env_open_r{r}c{c}")
    return names


def position_bin_names() -> list[str]:
    names = []
    for r in range(POSITION_BINS_PER_AXIS):
        for c in range(POSITION_BINS_PER_AXIS):
            names.append(f"r{r}c{c}")
    return names


def deref_string(h5: h5py.File, ref) -> str:
    arr = np.array(h5[ref][()]).astype(int).ravel()
    return "".join(chr(x) for x in arr)


def deref_numeric(h5: h5py.File, ref) -> np.ndarray:
    return np.array(h5[ref][()], dtype=np.float64).ravel()


def deref_array(h5: h5py.File, ref, dtype=np.float32) -> np.ndarray:
    return np.array(h5[ref][()], dtype=dtype)


def avg_pool_2d(arr: np.ndarray, kernel: int) -> np.ndarray:
    if arr.shape[0] < kernel:
        return np.empty((0, arr.shape[1]), dtype=arr.dtype)
    n_bins = arr.shape[0] // kernel
    trimmed = arr[: n_bins * kernel]
    return trimmed.reshape(n_bins, kernel, arr.shape[1]).mean(axis=1)


def compute_movement_mask(position: np.ndarray) -> np.ndarray:
    mask = np.zeros(position.shape[0], dtype=bool)
    if position.shape[0] < 2:
        return mask
    speed = np.linalg.norm((position[1:] - position[:-1]) * FPS, axis=1)
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    speed = gaussian_filter1d(speed, sigma=VELOCITY_SMOOTH_SIGMA_FRAMES)
    mask[1:] = speed > VELOCITY_THRESHOLD_CM_PER_S
    return mask


def blocked_indices_from_open_mask(open_mask: np.ndarray) -> list[int]:
    return [idx for idx, value in enumerate(open_mask.reshape(-1)) if value == 0]


def normalize_blocked(raw_blocked: np.ndarray) -> list[int]:
    if raw_blocked.size == 0:
        return []
    if raw_blocked.size == 1 and raw_blocked[0] == -1:
        return []
    return sorted(int(x) for x in raw_blocked.tolist())


def open_mask_from_blocked(blocked_indices: list[int]) -> np.ndarray:
    open_mask = np.ones(POSITION_BINS_PER_AXIS * POSITION_BINS_PER_AXIS, dtype=np.float32)
    for idx in blocked_indices:
        open_mask[idx] = 0.0
    return open_mask.reshape(POSITION_BINS_PER_AXIS, POSITION_BINS_PER_AXIS)


def remap_invalid_bins(bin_xy: np.ndarray, open_mask_flat: np.ndarray) -> tuple[np.ndarray, int]:
    if bin_xy.size == 0:
        return np.empty((0,), dtype=np.int64), 0

    flat = bin_xy[:, 0] * POSITION_BINS_PER_AXIS + bin_xy[:, 1]
    invalid = open_mask_flat[flat] == 0
    remapped = 0
    if not np.any(invalid):
        return flat.astype(np.int64), remapped

    valid_bins = np.argwhere(open_mask_flat.reshape(POSITION_BINS_PER_AXIS, POSITION_BINS_PER_AXIS) > 0)
    for idx in np.where(invalid)[0]:
        diffs = valid_bins - bin_xy[idx]
        nearest = valid_bins[np.argmin(np.linalg.norm(diffs, axis=1))]
        flat[idx] = nearest[0] * POSITION_BINS_PER_AXIS + nearest[1]
        remapped += 1
    return flat.astype(np.int64), remapped


def summarize_trial_lengths(neural_trials: list[np.ndarray]) -> dict[str, float | int]:
    lengths = np.array([trial.shape[1] for trial in neural_trials], dtype=int)
    return {
        "n_trials": int(lengths.size),
        "min_timepoints": int(lengths.min()),
        "median_timepoints": float(np.median(lengths)),
        "max_timepoints": int(lengths.max()),
    }


def convert_session(
    h5: h5py.File,
    animal: str,
    session_index: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, dict]:
    env_name = deref_string(h5, h5["envs"][0, session_index])
    raw_blocked = normalize_blocked(deref_numeric(h5, h5["blocked"][0, session_index]))
    env_open = open_mask_from_blocked(raw_blocked)

    position = deref_array(h5, h5["position"][session_index, 0], dtype=np.float32)
    trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)

    if position.ndim != 2 or position.shape[1] != 2:
        raise ValueError(f"{animal} session {session_index}: unexpected position shape {position.shape}")
    if trace.ndim != 2 or trace.shape[0] != position.shape[0]:
        raise ValueError(
            f"{animal} session {session_index}: trace shape {trace.shape} does not align with position {position.shape}"
        )

    valid_position_rows = ~np.isnan(position).any(axis=1)
    if not np.all(valid_position_rows):
        position = position[valid_position_rows]
        trace = trace[valid_position_rows]

    registered_mask = ~np.isnan(trace[0])
    if not np.any(registered_mask):
        raise ValueError(f"{animal} session {session_index}: no registered cells")

    trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
    movement_mask = compute_movement_mask(position)

    arena_extent = float(np.nanmax(position) + POSITION_BUFFER)
    if not np.isfinite(arena_extent) or arena_extent <= 0:
        raise ValueError(f"{animal} session {session_index}: invalid arena extent {arena_extent}")
    bin_down = arena_extent / POSITION_BINS_PER_AXIS
    env_open_flat = env_open.reshape(-1).astype(np.float32)

    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []
    remapped_samples = 0
    dropped_stationary_trials = 0
    dropped_short_trials = 0
    dropped_zero_neural_trials = 0

    for trial_idx, start in enumerate(range(0, position.shape[0], TRIAL_FRAMES)):
        end = min(start + TRIAL_FRAMES, position.shape[0])
        pos_trial = position[start:end]
        trace_trial = trace[start:end]
        trial_movement = movement_mask[start:end]

        if not np.any(trial_movement):
            dropped_stationary_trials += 1
            continue

        pos_trial = pos_trial[trial_movement]
        trace_trial = trace_trial[trial_movement]
        if pos_trial.shape[0] < TEMPORAL_BIN_FRAMES:
            dropped_short_trials += 1
            continue

        pos_scaled = pos_trial / bin_down
        pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
        trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
        trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)

        if pos_pooled.shape[0] == 0 or trace_pooled.shape[0] == 0:
            dropped_short_trials += 1
            continue
        if pos_pooled.shape[0] != trace_pooled.shape[0]:
            min_len = min(pos_pooled.shape[0], trace_pooled.shape[0])
            pos_pooled = pos_pooled[:min_len]
            trace_pooled = trace_pooled[:min_len]
        if trace_pooled.shape[0] < MIN_POOLED_TIMEPOINTS_PER_TRIAL:
            dropped_short_trials += 1
            continue

        bin_xy = np.floor(pos_pooled).astype(np.int64)
        bin_xy = np.clip(bin_xy, 0, POSITION_BINS_PER_AXIS - 1)
        flat_bins, n_remapped = remap_invalid_bins(bin_xy, env_open_flat)
        remapped_samples += n_remapped

        neural_trial = trace_pooled.T.astype(np.float32)
        if np.all(neural_trial == 0):
            dropped_zero_neural_trials += 1
            continue

        neural_trials.append(neural_trial)
        input_trials.append(env_open_flat.copy())
        output_trials.append(flat_bins[np.newaxis, :].astype(np.int64))

    if len(neural_trials) < 2:
        raise ValueError(
            f"{animal} session {session_index}: only {len(neural_trials)} non-empty trials after preprocessing"
        )

    session_summary = {
        "animal": animal,
        "session_index_within_animal": session_index,
        "environment": env_name,
        "blocked_bins": raw_blocked,
        "n_registered_cells": int(registered_mask.sum()),
        "n_raw_frames": int(position.shape[0]),
        "n_moving_frames": int(movement_mask.sum()),
        "n_trials_raw": int(math.ceil(position.shape[0] / TRIAL_FRAMES)),
        "n_trials_kept": int(len(neural_trials)),
        "n_trials_dropped_stationary": int(dropped_stationary_trials),
        "n_trials_dropped_short": int(dropped_short_trials),
        "n_trials_dropped_zero_neural": int(dropped_zero_neural_trials),
        "n_remapped_invalid_position_bins": int(remapped_samples),
        "trial_timepoints_summary": summarize_trial_lengths(neural_trials),
    }
    return neural_trials, input_trials, output_trials, np.zeros(int(registered_mask.sum()), dtype=np.int64), session_summary


def build_dataset(data_dir: Path, sample_only: bool) -> tuple[dict, dict]:
    dataset = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": ANIMALS.copy(),
        "subject_idx": [],
        "brain_regions": ["CA1"],
        "brain_region_idx": [],
        "input_names": flatten_env_names(),
        "output_names": ["position_bin"],
        "output_values": [position_bin_names()],
        "metadata": {},
    }

    source_stats = {
        "raw_session_count": 0,
        "raw_unique_neurons": 0,
        "raw_registered_cell_session_count": 0,
        "environment_counts": Counter(),
    }
    session_info = []

    for subject_idx, animal in enumerate(ANIMALS):
        if sample_only and animal not in SAMPLE_SELECTION:
            continue

        path = data_dir / f"{animal}.mat"
        with h5py.File(path, "r") as h5:
            n_sessions = h5["envs"].shape[1]
            source_stats["raw_session_count"] += n_sessions
            source_stats["raw_unique_neurons"] += int(h5["SFPs"].shape[1])

            if sample_only:
                session_indices = SAMPLE_SELECTION[animal]
            else:
                session_indices = list(range(n_sessions))

            for session_index in session_indices:
                env_name = deref_string(h5, h5["envs"][0, session_index])
                source_stats["environment_counts"][env_name] += 1

                neural_trials, input_trials, output_trials, region_idx, summary = convert_session(
                    h5=h5,
                    animal=animal,
                    session_index=session_index,
                )

                dataset["neural"].append(neural_trials)
                dataset["input"].append(input_trials)
                dataset["output"].append(output_trials)
                dataset["subject_idx"].append(subject_idx)
                dataset["brain_region_idx"].append(region_idx)
                session_info.append(summary)
                source_stats["raw_registered_cell_session_count"] += summary["n_registered_cells"]

    dataset["subject_idx"] = np.array(dataset["subject_idx"], dtype=np.int64)
    dataset["metadata"] = {
        "task_description": (
            "Decode the mouse's coarse 3x3 spatial location from CA1 calcium-event activity, "
            "given the static 3x3 environment geometry for each trial."
        ),
        "time_bin_size": float(TEMPORAL_BIN_MS),
        "temporal_alignment_event": "No event alignment; contiguous 1-minute windows from session start.",
        "off_start": None,
        "off_end": None,
        "source_dataset": "Lee, Keinath, Cianfarano and Brandon (2025) georepca1 dataset",
        "source_session_duration_s": 40 * 60,
        "source_fps": FPS,
        "trial_duration_s": TRIAL_SECONDS,
        "temporal_bin_frames": TEMPORAL_BIN_FRAMES,
        "trace_smoothing_sigma_frames": TRACE_SMOOTH_SIGMA_FRAMES,
        "movement_selection": "Frames with Gaussian-smoothed speed > 5 cm/s, matching the paper's within-session decoder.",
        "position_binning": "Session-wise common 3x3 binning using max(x,y)/3 scaling before temporal pooling.",
        "neural_processing": (
            "Binary rise-event traces from the source files, restricted to moving frames, "
            "Gaussian-smoothed (sigma=3 frames), then average pooled in non-overlapping 3-frame bins."
        ),
        "output_processing": (
            "2D position samples restricted to moving frames, scaled to a session-level 3x3 grid, "
            "average pooled in non-overlapping 3-frame bins, floored to categorical bins, and "
            "remapped to the nearest valid open bin if temporal pooling crosses blocked regions."
        ),
        "session_info": session_info,
    }

    dataset_stats = {
        "n_sessions": len(dataset["neural"]),
        "n_trials_total": int(sum(len(session) for session in dataset["neural"])),
        "n_neurons_total_across_sessions": int(sum(len(idx) for idx in dataset["brain_region_idx"])),
        "n_subjects_included": int(len(np.unique(dataset["subject_idx"]))),
        "environment_counts": Counter(info["environment"] for info in session_info),
        "total_kept_moving_frames": int(sum(info["n_moving_frames"] for info in session_info)),
        "total_remapped_invalid_position_bins": int(
            sum(info["n_remapped_invalid_position_bins"] for info in session_info)
        ),
    }

    trial_lengths = np.array(
        [trial.shape[1] for session in dataset["neural"] for trial in session],
        dtype=int,
    )
    dataset_stats["trial_timepoints_min"] = int(trial_lengths.min())
    dataset_stats["trial_timepoints_median"] = float(np.median(trial_lengths))
    dataset_stats["trial_timepoints_max"] = int(trial_lengths.max())

    flat_outputs = np.concatenate(
        [trial.reshape(-1) for session in dataset["output"] for trial in session]
    )
    output_counts = Counter(int(x) for x in flat_outputs.tolist())
    dataset_stats["output_bin_counts"] = {
        position_bin_names()[idx]: int(output_counts.get(idx, 0)) for idx in range(9)
    }

    checks = {
        "matches_paper_session_count": source_stats["raw_session_count"] == 207 if not sample_only else True,
        "matches_paper_unique_neurons": source_stats["raw_unique_neurons"] == 5413 if not sample_only else True,
        "matches_paper_rate_maps": source_stats["raw_registered_cell_session_count"] == 69744 if not sample_only else True,
        "matches_environment_histogram": (
            dict(source_stats["environment_counts"]) == EXPECTED_ENV_COUNTS if not sample_only else True
        ),
    }

    summary = {
        "source_stats": {
            "raw_session_count": int(source_stats["raw_session_count"]),
            "raw_unique_neurons": int(source_stats["raw_unique_neurons"]),
            "raw_registered_cell_session_count": int(source_stats["raw_registered_cell_session_count"]),
            "environment_counts": dict(source_stats["environment_counts"]),
        },
        "dataset_stats": dataset_stats,
        "checks": checks,
    }
    return dataset, summary


def print_summary(summary: dict, sample_only: bool) -> None:
    label = "sample" if sample_only else "full"
    print(f"Built {label} dataset")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert georepca1 CA1 data into decoder format.")
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--output", type=Path, default=Path("/app/converted_data.pkl"))
    parser.add_argument("--sample-output", type=Path, default=Path("/app/sample_data.pkl"))
    parser.add_argument("--sample-only", action="store_true", help="Build only the sample subset.")
    parser.add_argument("--stats-json", type=Path, default=None, help="Optional JSON output for summary stats.")
    args = parser.parse_args()

    if args.sample_only:
        sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
        with args.sample_output.open("wb") as f:
            pickle.dump(sample_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print_summary(sample_summary, sample_only=True)
        if args.stats_json is not None:
            args.stats_json.write_text(json.dumps(sample_summary, indent=2, sort_keys=True))
        return

    full_data, full_summary = build_dataset(args.data_dir, sample_only=False)
    with args.output.open("wb") as f:
        pickle.dump(full_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print_summary(full_summary, sample_only=False)

    sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
    with args.sample_output.open("wb") as f:
        pickle.dump(sample_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print_summary(sample_summary, sample_only=True)

    if args.stats_json is not None:
        stats = {"full": full_summary, "sample": sample_summary}
        args.stats_json.write_text(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
