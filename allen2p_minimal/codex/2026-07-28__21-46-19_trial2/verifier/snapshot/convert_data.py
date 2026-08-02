#!/usr/bin/env python3
"""Convert Allen Visual Behavior ophys data into decoder format."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from allensdk.brain_observatory.behavior.behavior_project_cache import (  # noqa: E402
    VisualBehaviorOphysProjectCache,
)

from decoder import verify_data_format  # noqa: E402


SELECTION_DESCRIPTION = (
    "Active Visual Behavior task experiments present locally on disk "
    "(VisualBehavior and VisualBehaviorMultiscope; VISp/VISl only), using "
    "AllenSDK events traces and keeping only go/catch trials that are not "
    "aborted and not auto-rewarded."
)

TRIAL_OUTCOMES = ["hit", "miss", "false_alarm", "correct_reject"]
RUNNING_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
PUPIL_BIN_NAMES = [f"q{i}" for i in range(1, 6)]


@dataclass
class TrialData:
    neural: np.ndarray
    image_names: np.ndarray
    image_change: np.ndarray
    running_raw: np.ndarray
    pupil_raw: np.ndarray
    trial_outcome: str
    trial_id: int


@dataclass
class SessionData:
    experiment_id: int
    behavior_session_id: int
    mouse_id: str
    targeted_structure: str
    cre_line: str
    imaging_depth: int
    session_type: str
    trials: list[TrialData]
    n_cells: int
    interval_durations: np.ndarray
    trial_count_before_filter: int
    trial_count_after_filter: int
    omission_count: int
    interval_count: int


def filter_sdk_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*UpdatedStimulusPresentationTableWarning.*",
    )
    warnings.filterwarnings(
        "ignore",
        message="Ignoring the following cached namespace",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Downcasting object dtype arrays on \\.fillna.*",
    )


def get_cache(cache_dir: Path) -> VisualBehaviorOphysProjectCache:
    return VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=str(cache_dir))


def get_local_experiment_ids(cache_dir: Path) -> set[int]:
    experiment_dir = cache_dir / "visual-behavior-ophys-1.1.0" / "behavior_ophys_experiments"
    return {
        int(path.stem.split("_")[-1])
        for path in experiment_dir.glob("behavior_ophys_experiment_*.nwb")
    }


def select_experiments(
    cache: VisualBehaviorOphysProjectCache,
    cache_dir: Path,
) -> pd.DataFrame:
    experiment_table = cache.get_ophys_experiment_table().reset_index()
    local_ids = get_local_experiment_ids(cache_dir)
    selected = experiment_table[
        (experiment_table["ophys_experiment_id"].isin(local_ids))
        & (experiment_table["project_code"].isin(["VisualBehavior", "VisualBehaviorMultiscope"]))
        & (experiment_table["behavior_type"] == "active_behavior")
        & (experiment_table["targeted_structure"].isin(["VISp", "VISl"]))
    ].copy()
    selected.sort_values(
        by=[
            "mouse_id",
            "date_of_acquisition",
            "session_type",
            "targeted_structure",
            "imaging_depth",
            "ophys_experiment_id",
        ],
        inplace=True,
    )
    selected.reset_index(drop=True, inplace=True)
    return selected


def choose_sample_experiments(selected: pd.DataFrame, n_sessions: int) -> pd.DataFrame:
    groups: dict[tuple[str, str, str], list[int]] = {}
    for idx, row in selected.iterrows():
        key = (row["cre_line"], row["targeted_structure"], row["session_type"])
        groups.setdefault(key, []).append(idx)

    chosen: list[int] = []
    while len(chosen) < min(n_sessions, len(selected)):
        progress = False
        for key in sorted(groups):
            if groups[key] and len(chosen) < n_sessions:
                chosen.append(groups[key].pop(0))
                progress = True
        if not progress:
            break

    if len(chosen) < min(n_sessions, len(selected)):
        remaining = [idx for idx in selected.index if idx not in set(chosen)]
        chosen.extend(remaining[: n_sessions - len(chosen)])

    sample = selected.loc[sorted(chosen)].copy()
    sample.reset_index(drop=True, inplace=True)
    return sample


def infer_trial_outcome(trial_row: pd.Series) -> str:
    if bool(trial_row["hit"]):
        return "hit"
    if bool(trial_row["miss"]):
        return "miss"
    if bool(trial_row["false_alarm"]):
        return "false_alarm"
    if bool(trial_row["correct_reject"]):
        return "correct_reject"
    raise ValueError(f"Could not infer trial outcome for trial {trial_row.name}")


def fill_nan_by_time(timestamps: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Series has no finite values")
    if finite.all():
        return values
    values[~finite] = np.interp(
        timestamps[~finite],
        timestamps[finite],
        values[finite],
    ).astype(np.float32)
    return values


def interval_reduce_mean(
    timestamps: np.ndarray,
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    start_idx = np.searchsorted(timestamps, starts, side="left")
    end_idx = np.searchsorted(timestamps, ends, side="left")
    csum = np.concatenate(
        [np.array([0.0], dtype=np.float64), np.cumsum(values, dtype=np.float64)]
    )
    counts = end_idx - start_idx
    sums = csum[end_idx] - csum[start_idx]
    out = np.empty(len(starts), dtype=np.float32)
    valid = counts > 0
    out[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    if (~valid).any():
        centers = (starts + ends) / 2.0
        out[~valid] = np.interp(centers[~valid], timestamps, values).astype(np.float32)
    return out


def interval_reduce_sum_matrix(
    timestamps: np.ndarray,
    matrix: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    start_idx = np.searchsorted(timestamps, starts, side="left")
    end_idx = np.searchsorted(timestamps, ends, side="left")
    csum = np.concatenate(
        [
            np.zeros((matrix.shape[0], 1), dtype=np.float64),
            np.cumsum(matrix, axis=1, dtype=np.float64),
        ],
        axis=1,
    )
    reduced = csum[:, end_idx] - csum[:, start_idx]
    return reduced.astype(np.float32)


def make_change_detection_table(ds: Any) -> pd.DataFrame:
    stim = ds.stimulus_presentations.copy()
    block_mask = stim["stimulus_block_name"].astype(str).str.contains("change_detection")
    active_mask = stim["active"].fillna(False).astype(bool)
    stim = stim.loc[block_mask & active_mask].copy()
    stim.sort_values("start_time", inplace=True)
    stim.reset_index(drop=False, inplace=True)
    if stim.empty:
        raise ValueError("No active change_detection stimulus presentations")

    start_times = stim["start_time"].to_numpy(dtype=np.float64)
    dt = np.diff(start_times)
    median_dt = float(np.median(dt))
    interval_end = np.empty_like(start_times)
    interval_end[:-1] = start_times[1:]
    interval_end[-1] = start_times[-1] + median_dt
    stim["interval_end"] = interval_end
    stim["interval_duration"] = stim["interval_end"] - stim["start_time"]
    return stim


def process_experiment(
    cache: VisualBehaviorOphysProjectCache,
    experiment_id: int,
    max_trials_per_session: int | None = None,
) -> SessionData | None:
    ds = cache.get_behavior_ophys_experiment(int(experiment_id))

    if len(ds.events) == 0:
        print(f"Skipping {experiment_id}: no valid ROI events")
        return None
    if ds.eye_tracking.empty:
        print(f"Skipping {experiment_id}: no eye tracking data")
        return None

    pupil_width = ds.eye_tracking["pupil_width"].to_numpy(dtype=np.float64)
    if not np.isfinite(pupil_width).any():
        print(f"Skipping {experiment_id}: pupil width is entirely NaN")
        return None

    trials = ds.trials.copy()
    keep_trials = (~trials["aborted"]) & (~trials["auto_rewarded"]) & (
        trials["go"] | trials["catch"]
    )
    kept_trials = trials.loc[keep_trials].copy()
    if kept_trials.empty:
        print(f"Skipping {experiment_id}: no kept go/catch trials")
        return None

    stim = make_change_detection_table(ds)
    valid_trial_ids = set(int(x) for x in kept_trials.index.to_numpy())
    stim = stim[stim["trials_id"].isin(valid_trial_ids)].copy()
    if stim.empty:
        print(f"Skipping {experiment_id}: no kept stimulus intervals")
        return None
    stim.reset_index(drop=True, inplace=True)

    ophys_timestamps = ds.ophys_timestamps.astype(np.float64)
    event_matrix = np.vstack(ds.events["events"].values).astype(np.float32)
    interval_starts = stim["start_time"].to_numpy(dtype=np.float64)
    interval_ends = stim["interval_end"].to_numpy(dtype=np.float64)
    neural_by_interval = interval_reduce_sum_matrix(
        timestamps=ophys_timestamps,
        matrix=event_matrix,
        starts=interval_starts,
        ends=interval_ends,
    )

    running_df = ds.running_speed.copy()
    running_t = running_df["timestamps"].to_numpy(dtype=np.float64)
    running_v = running_df["speed"].to_numpy(dtype=np.float64).astype(np.float32)
    running_by_interval = interval_reduce_mean(
        timestamps=running_t,
        values=running_v,
        starts=interval_starts,
        ends=interval_ends,
    )

    eye_df = ds.eye_tracking.copy()
    eye_t = eye_df["timestamps"].to_numpy(dtype=np.float64)
    pupil_filled = fill_nan_by_time(
        timestamps=eye_t,
        values=eye_df["pupil_width"].to_numpy(dtype=np.float64),
    )
    pupil_by_interval = interval_reduce_mean(
        timestamps=eye_t,
        values=pupil_filled,
        starts=interval_starts,
        ends=interval_ends,
    )

    trial_data: list[TrialData] = []
    for trial_id, trial_row in kept_trials.iterrows():
        trial_stim = stim[stim["trials_id"] == trial_id].copy()
        if trial_stim.empty:
            continue
        if max_trials_per_session is not None and len(trial_data) >= max_trials_per_session:
            break

        stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
        outcome = infer_trial_outcome(trial_row)
        trial_data.append(
            TrialData(
                neural=neural_by_interval[:, stim_idx].astype(np.float32, copy=False),
                image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
                image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy(),
                running_raw=running_by_interval[stim_idx].astype(np.float32, copy=False),
                pupil_raw=pupil_by_interval[stim_idx].astype(np.float32, copy=False),
                trial_outcome=outcome,
                trial_id=int(trial_id),
            )
        )

    if len(trial_data) < 2:
        print(f"Skipping {experiment_id}: fewer than 2 kept trials with stimulus intervals")
        return None

    return SessionData(
        experiment_id=int(experiment_id),
        behavior_session_id=int(ds.metadata["behavior_session_id"]),
        mouse_id=str(ds.metadata["mouse_id"]),
        targeted_structure=str(ds.metadata["targeted_structure"]),
        cre_line=str(ds.metadata["cre_line"]),
        imaging_depth=int(ds.metadata["imaging_depth"]),
        session_type=str(ds.metadata["session_type"]),
        trials=trial_data,
        n_cells=int(event_matrix.shape[0]),
        interval_durations=stim["interval_duration"].to_numpy(dtype=np.float64),
        trial_count_before_filter=int(len(trials)),
        trial_count_after_filter=int(len(trial_data)),
        omission_count=int((stim["image_name"] == "omitted").sum()),
        interval_count=int(len(stim)),
    )


def rank_quintiles(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="first")
    bins = pd.qcut(ranks, q=5, labels=False)
    return bins.to_numpy(dtype=np.int64)


def summarize_selected(selected: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_experiments": int(len(selected)),
        "n_ophys_sessions": int(selected["ophys_session_id"].nunique()),
        "n_mice": int(selected["mouse_id"].nunique()),
        "cre_line_counts": {
            str(k): int(v) for k, v in selected["cre_line"].value_counts().to_dict().items()
        },
        "session_type_counts": {
            str(k): int(v) for k, v in selected["session_type"].value_counts().to_dict().items()
        },
        "targeted_structure_counts": {
            str(k): int(v)
            for k, v in selected["targeted_structure"].value_counts().to_dict().items()
        },
    }


def build_decoder_dataset(
    session_data: list[SessionData],
    selected_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    all_running = np.concatenate([trial.running_raw for s in session_data for trial in s.trials])
    all_pupil = np.concatenate([trial.pupil_raw for s in session_data for trial in s.trials])
    running_bins = rank_quintiles(all_running)
    pupil_bins = rank_quintiles(all_pupil)

    image_values = sorted(
        {name for s in session_data for t in s.trials for name in t.image_names.tolist()},
        key=lambda x: (x == "omitted", x),
    )
    image_to_idx = {name: idx for idx, name in enumerate(image_values)}
    outcome_to_idx = {name: idx for idx, name in enumerate(TRIAL_OUTCOMES)}

    subjects = sorted({s.mouse_id for s in session_data})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
    brain_regions = sorted({s.targeted_structure for s in session_data})
    brain_region_to_idx = {region: idx for idx, region in enumerate(brain_regions)}

    neural: list[list[np.ndarray]] = []
    decoder_input: list[list[np.ndarray]] = []
    output: list[list[np.ndarray]] = []
    subject_idx: list[int] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict[str, Any]] = []

    running_cursor = 0
    pupil_cursor = 0
    interval_duration_all = []

    for session in session_data:
        session_neural: list[np.ndarray] = []
        session_input: list[np.ndarray] = []
        session_output: list[np.ndarray] = []

        for trial in session.trials:
            t = trial.neural.shape[1]
            image_idx = np.array([image_to_idx[name] for name in trial.image_names], dtype=np.int64)
            change_idx = trial.image_change.astype(np.int64)
            run_idx = running_bins[running_cursor : running_cursor + t]
            pupil_idx = pupil_bins[pupil_cursor : pupil_cursor + t]
            running_cursor += t
            pupil_cursor += t
            outcome_idx = np.full(t, outcome_to_idx[trial.trial_outcome], dtype=np.int64)

            session_neural.append(trial.neural.astype(np.float32, copy=False))
            session_input.append(np.empty((0, t), dtype=np.float32))
            session_output.append(
                np.vstack(
                    [
                        image_idx,
                        change_idx,
                        run_idx,
                        pupil_idx,
                        outcome_idx,
                    ]
                ).astype(np.int64, copy=False)
            )
            interval_duration_all.append(t)

        neural.append(session_neural)
        decoder_input.append(session_input)
        output.append(session_output)
        subject_idx.append(subject_to_idx[session.mouse_id])
        brain_region_idx.append(
            np.full(
                session.n_cells,
                brain_region_to_idx[session.targeted_structure],
                dtype=np.int64,
            )
        )
        session_info.append(
            {
                "ophys_experiment_id": session.experiment_id,
                "behavior_session_id": session.behavior_session_id,
                "mouse_id": session.mouse_id,
                "cre_line": session.cre_line,
                "targeted_structure": session.targeted_structure,
                "imaging_depth": session.imaging_depth,
                "session_type": session.session_type,
                "n_cells": session.n_cells,
                "n_trials": len(session.trials),
                "n_intervals": int(sum(trial.neural.shape[1] for trial in session.trials)),
            }
        )

    interval_durations = np.concatenate([s.interval_durations for s in session_data])
    running_quantiles = np.quantile(all_running, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
    pupil_quantiles = np.quantile(all_pupil, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()

    data = {
        "neural": neural,
        "input": decoder_input,
        "output": output,
        "subjects": subjects,
        "subject_idx": np.array(subject_idx, dtype=np.int64),
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": [],
        "output_names": [
            "image_identity",
            "image_change",
            "running_speed_bin",
            "pupil_diameter_bin",
            "trial_outcome",
        ],
        "output_values": [
            image_values,
            ["no_change", "change"],
            RUNNING_BIN_NAMES,
            PUPIL_BIN_NAMES,
            TRIAL_OUTCOMES,
        ],
        "metadata": {
            "task_description": (
                "Allen Visual Behavior change-detection task. Trials are filtered to "
                "go/catch only, excluding aborted and auto-rewarded trials. Neural "
                "activity is AllenSDK events summed within successive image-presentation "
                "intervals from the active change_detection block."
            ),
            "selection_description": SELECTION_DESCRIPTION,
            "time_bin_size": float(np.median(interval_durations) * 1000.0),
            "temporal_alignment_event": (
                "Successive image-presentation intervals defined by active "
                "change_detection stimulus onsets and assigned to trials via trials_id; "
                "modalities aggregated using ophys/running/eye timestamps within each interval."
            ),
            "off_start": 0.0,
            "off_end": None,
            "neural_signal": "AllenSDK events trace summed within each interval",
            "image_interval_duration_median_s": float(np.median(interval_durations)),
            "image_interval_duration_mean_s": float(np.mean(interval_durations)),
            "running_bin_quantiles": [float(x) for x in running_quantiles],
            "pupil_bin_quantiles": [float(x) for x in pupil_quantiles],
            "session_info": session_info,
            "selected_experiment_summary": selected_summary,
        },
    }

    summary = {
        "n_sessions": len(session_data),
        "n_trials": int(sum(len(s.trials) for s in session_data)),
        "n_intervals": int(sum(sum(trial.neural.shape[1] for trial in s.trials) for s in session_data)),
        "n_cells_total": int(sum(s.n_cells for s in session_data)),
        "median_cells_per_session": float(np.median([s.n_cells for s in session_data])),
        "median_trials_per_session": float(np.median([len(s.trials) for s in session_data])),
        "median_intervals_per_trial": float(
            np.median([trial.neural.shape[1] for s in session_data for trial in s.trials])
        ),
        "omission_fraction": float(
            np.sum([s.omission_count for s in session_data])
            / np.sum([s.interval_count for s in session_data])
        ),
        "mean_interval_duration_s": float(np.mean(interval_durations)),
        "median_interval_duration_s": float(np.median(interval_durations)),
    }
    return data, summary


def save_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_sample_dataset(data: dict[str, Any], session_limit: int, trial_limit: int) -> dict[str, Any]:
    nsessions = min(session_limit, len(data["neural"]))
    sample = {
        "neural": [session_trials[:trial_limit] for session_trials in data["neural"][:nsessions]],
        "input": [session_trials[:trial_limit] for session_trials in data["input"][:nsessions]],
        "output": [session_trials[:trial_limit] for session_trials in data["output"][:nsessions]],
        "subjects": data["subjects"],
        "subject_idx": data["subject_idx"][:nsessions].copy(),
        "brain_regions": data["brain_regions"],
        "brain_region_idx": [x.copy() for x in data["brain_region_idx"][:nsessions]],
        "input_names": data["input_names"],
        "output_names": data["output_names"],
        "output_values": data["output_values"],
        "metadata": dict(data["metadata"]),
    }
    sample["metadata"]["sample_subset"] = {
        "n_sessions": nsessions,
        "trial_limit_per_session": trial_limit,
    }
    return sample


def run_conversion(args: argparse.Namespace) -> None:
    filter_sdk_warnings()
    cache = get_cache(args.cache_dir)

    selected = select_experiments(cache, args.cache_dir)
    if args.mode == "sample":
        selected = choose_sample_experiments(selected, args.sample_sessions)
        max_trials_per_session = args.sample_trials
    else:
        max_trials_per_session = None

    selected_summary = summarize_selected(selected)
    print("Selected experiments:")
    print(json.dumps(selected_summary, indent=2))

    session_data: list[SessionData] = []
    for idx, experiment_id in enumerate(selected["ophys_experiment_id"].tolist(), start=1):
        print(f"[{idx}/{len(selected)}] loading experiment {experiment_id}")
        session = process_experiment(
            cache=cache,
            experiment_id=int(experiment_id),
            max_trials_per_session=max_trials_per_session,
        )
        if session is not None:
            session_data.append(session)

    if not session_data:
        raise RuntimeError("No sessions were converted")

    data, summary = build_decoder_dataset(session_data, selected_summary)
    valid, errors, warnings_list = verify_data_format(data)
    print("\nConversion summary:")
    print(json.dumps(summary, indent=2))
    print("\nFormat validation:")
    print(json.dumps({"valid": valid, "errors": errors, "warnings": warnings_list}, indent=2))
    if not valid:
        raise RuntimeError("Converted dataset failed decoder format validation")

    save_pickle(args.output, data)
    print(f"\nWrote {args.output}")

    if args.sample_output is not None:
        sample = make_sample_dataset(
            data=data,
            session_limit=args.sample_output_sessions,
            trial_limit=args.sample_output_trials,
        )
        valid_sample, errors_sample, warnings_sample = verify_data_format(sample)
        print("\nSample subset validation:")
        print(
            json.dumps(
                {
                    "valid": valid_sample,
                    "errors": errors_sample,
                    "warnings": warnings_sample,
                },
                indent=2,
            )
        )
        if not valid_sample:
            raise RuntimeError("Sample dataset failed decoder format validation")
        save_pickle(args.sample_output, sample)
        print(f"Wrote {args.sample_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Allen Visual Behavior data.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data",
        help="Local Visual Behavior cache directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "converted_data.pkl",
        help="Output pickle path",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=ROOT / "sample_data.pkl",
        help="Optional output path for a smaller sample subset",
    )
    parser.add_argument(
        "--sample-output-sessions",
        type=int,
        default=8,
        help="Number of sessions to keep in sample_data.pkl",
    )
    parser.add_argument(
        "--sample-output-trials",
        type=int,
        default=20,
        help="Trial limit per session in sample_data.pkl",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "sample"],
        default="full",
        help="Convert the full selected cohort or a small stratified subset",
    )
    parser.add_argument(
        "--sample-sessions",
        type=int,
        default=12,
        help="When --mode sample, number of experiments to convert",
    )
    parser.add_argument(
        "--sample-trials",
        type=int,
        default=25,
        help="When --mode sample, maximum number of kept trials per experiment",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_conversion(parse_args())
