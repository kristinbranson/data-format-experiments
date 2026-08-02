#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
from collections import Counter
from dataclasses import dataclass, asdict

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
EXPECTED_TOTAL_SESSIONS = 207
EXPECTED_TOTAL_UNIQUE_NEURONS = 5413
EXPECTED_TOTAL_RATE_MAPS = 69744


ENV_TO_MASK = {
    "square": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32),
    "o": np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32),
    "t": np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=np.float32),
    "u": np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=np.float32),
    "rectangle": np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=np.float32),
    "+": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float32),
    "i": np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]], dtype=np.float32),
    "l": np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32),
    "bit donut": np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float32),
    "glenn": np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.float32),
}


@dataclass
class ConversionStats:
    total_sessions: int = 0
    total_trials: int = 0
    total_unique_neurons: int = 0
    total_session_neurons: int = 0
    total_frames_used: int = 0
    total_frames_dropped: int = 0
    total_frames_reassigned: int = 0


def normalize_blocked_entry(entry) -> tuple[int, ...]:
    """Convert MATLAB-ish nested blocked entries into a stable tuple of blocked bin ids."""
    while isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    arr = np.asarray(entry, dtype=float).reshape(-1)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return ()
    arr = arr.astype(int)
    if arr.size == 1 and arr[0] == -1:
        return ()
    return tuple(sorted(arr.tolist()))


def mask_from_blocked(blocked_bins: tuple[int, ...]) -> np.ndarray:
    mask = np.ones((3, 3), dtype=np.float32)
    for idx in blocked_bins:
        mask[idx // 3, idx % 3] = 0.0
    return mask


def build_nearest_valid_lookup(env_mask: np.ndarray) -> np.ndarray:
    valid = np.argwhere(env_mask > 0)
    lookup = np.zeros((3, 3, 2), dtype=np.int64)
    for x in range(3):
        for y in range(3):
            if env_mask[x, y] > 0:
                lookup[x, y] = np.array([x, y], dtype=np.int64)
                continue
            dists = np.sum((valid - np.array([x, y])) ** 2, axis=1)
            lookup[x, y] = valid[np.argmin(dists)]
    return lookup


def bin_position_to_grid(position_xy: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """
    Match the paper code's position binning logic:
    - raw positions are kept in the fixed camera frame
    - positions are divided by per-axis maxima
    - no min subtraction is applied
    """
    if position_xy.shape[0] != 2:
        raise ValueError(f"Expected position shape (2, T), got {position_xy.shape}")
    max_per_axis = np.nanmax(position_xy, axis=1) + 1e-12
    bins = np.floor(position_xy / (max_per_axis[:, None] / n_bins)).astype(np.int64)
    return np.clip(bins, 0, n_bins - 1)


def flatten_envs(envs: np.ndarray) -> np.ndarray:
    return np.asarray(envs).reshape(-1)


def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]


def summarize_dataset(data: dict) -> dict:
    nsessions = len(data["neural"])
    ntrials_per_session = [len(x) for x in data["neural"]]
    nneurons_per_session = [x[0].shape[0] for x in data["neural"]]
    trial_lengths = sorted({trial.shape[1] for sess in data["neural"] for trial in sess})
    output_counts = Counter()
    for session in data["output"]:
        for trial in session:
            vals, counts = np.unique(trial[0], return_counts=True)
            output_counts.update({int(v): int(c) for v, c in zip(vals, counts)})
    return {
        "nsessions": nsessions,
        "ntrials_total": int(sum(ntrials_per_session)),
        "ntrials_per_session_unique": sorted(set(ntrials_per_session)),
        "nneurons_min": int(min(nneurons_per_session)),
        "nneurons_mean": float(np.mean(nneurons_per_session)),
        "nneurons_max": int(max(nneurons_per_session)),
        "trial_lengths": trial_lengths,
        "output_class_counts": {str(k): int(v) for k, v in sorted(output_counts.items())},
    }


def subset_sessions(full_data: dict, session_indices: list[int]) -> dict:
    session_indices = list(session_indices)
    used_subjects = sorted({int(full_data["subject_idx"][i]) for i in session_indices})
    subject_remap = {old: new for new, old in enumerate(used_subjects)}
    out = {
        "neural": [full_data["neural"][i] for i in session_indices],
        "input": [full_data["input"][i] for i in session_indices],
        "output": [full_data["output"][i] for i in session_indices],
        "subjects": [full_data["subjects"][i] for i in used_subjects],
        "subject_idx": np.array(
            [subject_remap[int(full_data["subject_idx"][i])] for i in session_indices],
            dtype=np.int64,
        ),
        "brain_regions": list(full_data["brain_regions"]),
        "brain_region_idx": [full_data["brain_region_idx"][i] for i in session_indices],
        "input_names": list(full_data["input_names"]),
        "output_names": list(full_data["output_names"]),
        "output_values": [list(v) for v in full_data["output_values"]],
        "metadata": dict(full_data["metadata"]),
    }
    out["metadata"]["source_session_indices"] = session_indices
    out["metadata"]["source_subjects"] = out["subjects"]
    return out


def convert_dataset(data_dir: str) -> tuple[dict, dict]:
    neural = []
    decoder_input = []
    decoder_output = []
    subject_idx = []
    brain_region_idx = []

    stats = ConversionStats()
    per_animal_unique_neurons = {}
    per_subject_sessions = Counter()
    env_counts = Counter()
    blocked_patterns_by_env: dict[str, set[tuple[int, ...]]] = {}
    frame_lengths = Counter()
    dropped_seconds_by_session = []

    for subject_id, animal in enumerate(ANIMALS):
        dat = load_animal(data_dir, animal)
        envs = flatten_envs(dat["envs"])
        trace = np.asarray(dat["trace"], dtype=np.float32)
        position = np.asarray(dat["position"], dtype=np.float32)

        if trace.ndim != 3:
            raise ValueError(f"{animal}: trace shape should be (n_days, n_cells, T), got {trace.shape}")
        if position.ndim != 3:
            raise ValueError(f"{animal}: position shape should be (n_days, 2, T), got {position.shape}")
        if trace.shape[0] != position.shape[0]:
            raise ValueError(f"{animal}: trace and position disagree on number of days")
        if trace.shape[2] != position.shape[2]:
            raise ValueError(f"{animal}: trace and position disagree on number of frames")

        registered_any_day = np.any(~np.isnan(trace[:, :, 0]), axis=0)
        per_animal_unique_neurons[animal] = int(np.sum(registered_any_day))
        stats.total_unique_neurons += per_animal_unique_neurons[animal]

        for day in range(trace.shape[0]):
            blocked_entry = normalize_blocked_entry(dat["blocked"][day])
            env_name = str(envs[day])
            env_mask = mask_from_blocked(blocked_entry)
            env_counts[env_name] += 1
            blocked_patterns_by_env.setdefault(env_name, set()).add(blocked_entry)

            day_trace = trace[day]
            day_pos = position[day]
            T = int(day_trace.shape[1])
            frame_lengths[T] += 1

            if np.isnan(day_pos).any():
                raise ValueError(f"{animal} day {day}: position contains NaNs")

            registered_today = ~np.isnan(day_trace[:, 0])
            if np.any(np.isnan(day_trace[registered_today])):
                raise ValueError(f"{animal} day {day}: registered neurons contain NaNs")
            if np.any(~np.isnan(day_trace[~registered_today])):
                raise ValueError(f"{animal} day {day}: unregistered neurons are not consistently NaN")

            session_neural = day_trace[registered_today].astype(np.float32, copy=True)
            n_session_neurons = int(session_neural.shape[0])
            stats.total_session_neurons += n_session_neurons

            binned_xy = bin_position_to_grid(day_pos, n_bins=3)
            lookup = build_nearest_valid_lookup(env_mask)
            projected_xy = lookup[binned_xy[0], binned_xy[1]].T
            reassigned = np.any(projected_xy != binned_xy, axis=0)
            stats.total_frames_reassigned += int(np.sum(reassigned))
            output_position = (projected_xy[0] * 3 + projected_xy[1]).astype(np.int64)

            n_trials = T // TRIAL_FRAMES
            if n_trials < 2:
                raise ValueError(f"{animal} day {day}: only {n_trials} full 1-minute trials")
            dropped = T - n_trials * TRIAL_FRAMES
            stats.total_frames_dropped += int(dropped)
            stats.total_frames_used += int(n_trials * TRIAL_FRAMES)
            dropped_seconds_by_session.append(dropped / FPS)

            input_vec = env_mask.reshape(-1).astype(np.float32)
            session_trials_neural = []
            session_trials_input = []
            session_trials_output = []
            for trial_idx in range(n_trials):
                start = trial_idx * TRIAL_FRAMES
                stop = start + TRIAL_FRAMES
                session_trials_neural.append(session_neural[:, start:stop].copy())
                session_trials_input.append(input_vec.copy())
                session_trials_output.append(output_position[np.newaxis, start:stop].copy())

            neural.append(session_trials_neural)
            decoder_input.append(session_trials_input)
            decoder_output.append(session_trials_output)
            subject_idx.append(subject_id)
            brain_region_idx.append(np.zeros(n_session_neurons, dtype=np.int64))

            stats.total_sessions += 1
            stats.total_trials += n_trials
            per_subject_sessions[animal] += 1

        del dat, trace, position
        gc.collect()

    data = {
        "neural": neural,
        "input": decoder_input,
        "output": decoder_output,
        "subjects": list(ANIMALS),
        "subject_idx": np.array(subject_idx, dtype=np.int64),
        "brain_regions": ["CA1"],
        "brain_region_idx": brain_region_idx,
        "input_names": [f"geometry_x{x}_y{y}" for x in range(3) for y in range(3)],
        "output_names": ["position_bin"],
        "output_values": [[f"x{x}_y{y}" for x in range(3) for y in range(3)]],
        "metadata": {
            "task_description": (
                "CA1 binary rise-event activity during free exploration in deforming 3x3-partitioned arenas; "
                "decode 3x3 spatial position from neural activity with static geometry context."
            ),
            "time_bin_size": TIME_BIN_MS,
            "temporal_alignment_event": "trial start of each contiguous 60 second window within a recording session",
            "off_start": 0.0,
            "off_end": TRIAL_SECONDS,
            "recording_type": "miniscope calcium imaging",
            "signal_type": "binary rising-phase events extracted from calcium traces",
            "fps": FPS,
            "trial_duration_sec": TRIAL_SECONDS,
            "trial_frames": TRIAL_FRAMES,
            "original_session_duration_min_nominal": 40.0,
            "trial_split_policy": "keep only complete contiguous 60 second windows; discard trailing incomplete remainder",
            "position_binning": (
                "Raw x/y positions were discretized with the same max-based spatial scaling used by the paper code: "
                "floor(position / ((per-axis max + eps) / 3))."
            ),
            "position_cleanup": (
                "3x3 position bins landing in geometry-blocked partitions were reassigned to the nearest valid open bin, "
                "analogous to the paper code's nearest-valid-bin cleanup during decoding."
            ),
            "input_representation": "flattened 3x3 open-bin mask, 1=open and 0=blocked",
            "output_representation": "single categorical position label 0..8 with label index = x_bin * 3 + y_bin",
            "session_is_day": True,
            "per_subject_sessions": dict(per_subject_sessions),
            "environment_counts": dict(env_counts),
            "blocked_patterns_by_env": {
                env: [list(x) for x in sorted(patterns)]
                for env, patterns in sorted(blocked_patterns_by_env.items())
            },
            "expected_reference_counts": {
                "sessions": EXPECTED_TOTAL_SESSIONS,
                "unique_neurons": EXPECTED_TOTAL_UNIQUE_NEURONS,
                "rate_maps": EXPECTED_TOTAL_RATE_MAPS,
            },
            "observed_reference_counts": {
                "sessions": stats.total_sessions,
                "unique_neurons": stats.total_unique_neurons,
                "rate_maps": stats.total_session_neurons,
            },
            "per_animal_unique_neurons": per_animal_unique_neurons,
            "session_frame_lengths": {str(k): int(v) for k, v in sorted(frame_lengths.items())},
            "mean_dropped_tail_seconds": float(np.mean(dropped_seconds_by_session)),
            "total_frames_reassigned_to_valid_bins": stats.total_frames_reassigned,
        },
    }

    sanity = {
        "conversion_stats": asdict(stats),
        "summary": summarize_dataset(data),
        "per_animal_unique_neurons": per_animal_unique_neurons,
        "environment_counts": dict(env_counts),
        "blocked_patterns_by_env": {
            env: [list(x) for x in sorted(patterns)]
            for env, patterns in sorted(blocked_patterns_by_env.items())
        },
        "frame_lengths": {str(k): int(v) for k, v in sorted(frame_lengths.items())},
        "mean_dropped_tail_seconds": float(np.mean(dropped_seconds_by_session)),
        "reference_count_matches": {
            "sessions": stats.total_sessions == EXPECTED_TOTAL_SESSIONS,
            "unique_neurons": stats.total_unique_neurons == EXPECTED_TOTAL_UNIQUE_NEURONS,
            "rate_maps": stats.total_session_neurons == EXPECTED_TOTAL_RATE_MAPS,
        },
    }
    return data, sanity


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the georepca1 dataset into decoder format.")
    parser.add_argument("--data-dir", default="/app/data", help="Directory containing animal joblib files.")
    parser.add_argument("--full-out", default="/app/converted_data.pkl", help="Path for the full converted dataset.")
    parser.add_argument("--sample-out", default="/app/sample_data.pkl", help="Path for the sample converted dataset.")
    parser.add_argument(
        "--sample-session-count-per-subject",
        type=int,
        default=10,
        help="How many earliest sessions to keep from each of the first sample subjects.",
    )
    parser.add_argument(
        "--sample-subject-count",
        type=int,
        default=2,
        help="How many earliest subjects to keep in sample_data.pkl.",
    )
    parser.add_argument(
        "--sanity-json",
        default=None,
        help="Optional JSON path to save conversion sanity statistics.",
    )
    args = parser.parse_args()

    data, sanity = convert_dataset(args.data_dir)

    with open(args.full_out, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    sample_session_indices = []
    start = 0
    for subject_id in range(min(args.sample_subject_count, len(ANIMALS))):
        sessions_for_subject = []
        for sess_idx, subj in enumerate(data["subject_idx"]):
            if int(subj) == subject_id:
                sessions_for_subject.append(sess_idx)
        sample_session_indices.extend(sessions_for_subject[: args.sample_session_count_per_subject])
        start += len(sessions_for_subject)
    sample_data = subset_sessions(data, sample_session_indices)
    with open(args.sample_out, "wb") as f:
        pickle.dump(sample_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.sanity_json:
        with open(args.sanity_json, "w") as f:
            json.dump(
                {
                    "full": sanity,
                    "sample": summarize_dataset(sample_data),
                    "sample_session_indices": sample_session_indices,
                },
                f,
                indent=2,
            )

    print("Conversion complete.")
    print(json.dumps(sanity, indent=2))
    print("Sample summary:")
    print(json.dumps(summarize_dataset(sample_data), indent=2))
    print(f"Sample session indices: {sample_session_indices}")


if __name__ == "__main__":
    main()
