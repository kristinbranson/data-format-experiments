import argparse
import json
import math
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/app/code")

from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (  # noqa: E402
    BehaviorOphysExperiment,
)


warnings.filterwarnings(
    "ignore",
    message=r"Ignoring the following cached namespace\(s\) because another version is already loaded",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Downcasting object dtype arrays on \.fillna, \.ffill, \.bfill is deprecated",
    category=FutureWarning,
)


TIME_BIN_MS_DEFAULT = 750.0
IMAGE_INTERVAL_S = 0.75
NO_IMAGE_LABEL = "gray"

TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
TRIAL_OUTCOME_TO_INT = {name: idx for idx, name in enumerate(TRIAL_OUTCOME_VALUES)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Allen Visual Behavior ophys data to decoder format."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/app/data/visual-behavior-ophys-1.1.0"),
        help="Path to the Visual Behavior ophys release directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/converted_data.pkl"),
        help="Output pickle path.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("/app/sample_data.pkl"),
        help="Sample dataset pickle path.",
    )
    parser.add_argument(
        "--sample-session-count",
        type=int,
        default=12,
        help="Number of sessions to include in sample_data.pkl.",
    )
    parser.add_argument(
        "--time-bin-ms",
        type=float,
        default=TIME_BIN_MS_DEFAULT,
        help="Fixed bin size in milliseconds. This converter uses the native 750 ms image-presentation interval.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )
    return parser.parse_args()


def get_available_experiment_ids(data_root: Path):
    experiment_dir = data_root / "behavior_ophys_experiments"
    pattern = re.compile(r"behavior_ophys_experiment_(\d+)\.nwb$")
    experiment_ids = []
    for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb")):
        match = pattern.match(path.name)
        if match is not None:
            experiment_ids.append(int(match.group(1)))
    return experiment_ids


def load_experiment_table(data_root: Path):
    exp_table = pd.read_csv(data_root / "project_metadata" / "ophys_experiment_table.csv")
    exp_table = exp_table.set_index("ophys_experiment_id", drop=False)
    return exp_table


def select_experiments(exp_table: pd.DataFrame, available_ids, max_sessions=None):
    available_ids = set(available_ids)
    selected = exp_table.loc[exp_table.index.intersection(available_ids)].copy()
    selected = selected[selected["behavior_type"] == "active_behavior"].copy()
    selected = selected.reset_index(drop=True)
    selected = selected.sort_values(
        ["date_of_acquisition", "mouse_id", "ophys_experiment_id"]
    )
    if max_sessions is not None:
        selected = selected.head(max_sessions).copy()
    return selected


def fill_nan_by_time(values, timestamps):
    values = np.asarray(values, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(timestamps)
    if valid.sum() == 0:
        return None
    if valid.sum() == 1:
        filled = np.empty_like(values)
        filled[:] = values[valid][0]
        return filled
    return np.interp(timestamps, timestamps[valid], values[valid]).astype(np.float32)


def reduce_to_bins(values, timestamps, bin_edges):
    values = np.asarray(values)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    n_bins = len(bin_edges) - 1
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    start_idx = np.searchsorted(timestamps, bin_edges[:-1], side="left")
    end_idx = np.searchsorted(timestamps, bin_edges[1:], side="left")

    if values.ndim == 1:
        reduced = np.empty(n_bins, dtype=np.float32)
        for i in range(n_bins):
            lo = start_idx[i]
            hi = end_idx[i]
            if hi > lo:
                reduced[i] = np.nanmean(values[lo:hi], dtype=np.float64)
            else:
                nearest = np.searchsorted(timestamps, centers[i], side="left")
                nearest = min(max(nearest, 0), len(timestamps) - 1)
                if nearest > 0 and abs(timestamps[nearest - 1] - centers[i]) < abs(
                    timestamps[nearest] - centers[i]
                ):
                    nearest -= 1
                reduced[i] = values[nearest]
        return reduced

    reduced = np.empty((values.shape[0], n_bins), dtype=np.float32)
    for i in range(n_bins):
        lo = start_idx[i]
        hi = end_idx[i]
        if hi > lo:
            reduced[:, i] = np.nanmean(values[:, lo:hi], axis=1, dtype=np.float64)
        else:
            nearest = np.searchsorted(timestamps, centers[i], side="left")
            nearest = min(max(nearest, 0), len(timestamps) - 1)
            if nearest > 0 and abs(timestamps[nearest - 1] - centers[i]) < abs(
                timestamps[nearest] - centers[i]
            ):
                nearest -= 1
            reduced[:, i] = values[:, nearest]
    return reduced


def compute_quantile_edges(values, nbins):
    percentiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(values, percentiles)
    edges = np.asarray(edges, dtype=np.float64)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges


def digitize_with_edges(values, edges):
    values = np.asarray(values, dtype=np.float32)
    return np.searchsorted(edges, values, side="right").astype(np.int64)


def get_trial_outcome(row):
    if bool(row["hit"]):
        return TRIAL_OUTCOME_TO_INT["hit"]
    if bool(row["miss"]):
        return TRIAL_OUTCOME_TO_INT["miss"]
    if bool(row["false_alarm"]):
        return TRIAL_OUTCOME_TO_INT["false_alarm"]
    if bool(row["correct_reject"]):
        return TRIAL_OUTCOME_TO_INT["correct_reject"]
    raise ValueError("Trial does not have a valid decoder outcome.")


def load_session(
    experiment_id: int,
    nwb_path: Path,
    meta_row: pd.Series,
):
    dataset = BehaviorOphysExperiment.from_nwb_path(
        str(nwb_path), exclude_invalid_rois=True
    )

    if len(dataset.eye_tracking) == 0:
        return None, {"skip_reason": "missing_eye_tracking"}

    if len(dataset.events) == 0:
        return None, {"skip_reason": "no_valid_rois"}

    eye_timestamps = dataset.eye_tracking["timestamps"].to_numpy(dtype=np.float64)
    pupil_width = fill_nan_by_time(
        dataset.eye_tracking["pupil_width"].to_numpy(),
        eye_timestamps,
    )
    if pupil_width is None:
        return None, {"skip_reason": "all_pupil_nan"}

    running_speed = dataset.running_speed["speed"].to_numpy(dtype=np.float32)
    running_timestamps = dataset.running_speed["timestamps"].to_numpy(dtype=np.float64)
    running_valid = np.isfinite(running_speed) & np.isfinite(running_timestamps)
    if running_valid.sum() == 0:
        return None, {"skip_reason": "missing_running_speed"}
    running_speed = running_speed[running_valid]
    running_timestamps = running_timestamps[running_valid]

    events_matrix = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32)
    ophys_timestamps = dataset.ophys_timestamps.astype(np.float64)

    trials = dataset.trials.copy()
    valid_trials = trials[
        (trials["go"] | trials["catch"])
        & (~trials["aborted"])
        & (~trials["auto_rewarded"])
        & np.isfinite(trials["change_time"])
    ].copy()
    if len(valid_trials) < 2:
        return None, {
            "skip_reason": "too_few_valid_trials",
            "n_valid_trials": int(len(valid_trials)),
        }

    stimulus_presentations = dataset.stimulus_presentations.copy()
    stimulus_presentations = stimulus_presentations[
        stimulus_presentations["stimulus_block_name"]
        .fillna("")
        .str.contains("change_detection")
    ].copy()
    stimulus_presentations = stimulus_presentations.sort_values("start_time")
    if len(stimulus_presentations) == 0:
        return None, {"skip_reason": "missing_change_detection_stimulus_table"}

    session = {
        "experiment_id": int(experiment_id),
        "mouse_id": str(meta_row["mouse_id"]),
        "brain_region": str(meta_row["targeted_structure"]),
        "project_code": str(meta_row["project_code"]),
        "session_type": str(meta_row["session_type"]),
        "experience_level": str(meta_row["experience_level"]),
        "cre_line": str(meta_row["cre_line"]),
        "equipment_name": str(meta_row["equipment_name"]),
        "ophys_session_id": int(meta_row["ophys_session_id"]),
        "n_neurons": int(events_matrix.shape[0]),
        "neural_trials": [],
        "running_cont": [],
        "pupil_cont": [],
        "interval_image_names": [],
        "interval_change_flags": [],
        "trial_outcomes": [],
        "trial_ids": [],
        "trial_interval_counts": [],
        "go_flags": [],
        "sanity_total_omitted_intervals": 0,
        "sanity_change_interval_omission_count": 0,
        "sanity_pre_change_omission_count": 0,
        "sanity_change_flag_mismatch_count": 0,
    }

    for trial_id, row in valid_trials.iterrows():
        trial_start = float(row["start_time"])
        trial_stop = float(row["stop_time"])
        trial_stim = stimulus_presentations[
            (stimulus_presentations["start_time"] >= trial_start - 1e-6)
            & (stimulus_presentations["start_time"] < trial_stop + 1e-6)
        ].copy()
        if len(trial_stim) == 0:
            continue

        stim_start_times = trial_stim["start_time"].to_numpy(dtype=np.float64)
        bin_edges = np.concatenate(
            [stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]]
        )

        neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
        running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
        pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
        if (
            np.any(~np.isfinite(neural_trial))
            or np.any(~np.isfinite(running_trial))
            or np.any(~np.isfinite(pupil_trial))
        ):
            continue
        if np.all(neural_trial == 0):
            continue

        omitted_flags = trial_stim["omitted"].fillna(False).to_numpy(dtype=bool)
        change_flags = trial_stim["is_change"].fillna(False).to_numpy(dtype=bool)
        if bool(row["go"]) and int(change_flags.sum()) != 1:
            session["sanity_change_flag_mismatch_count"] += 1
            continue
        if bool(row["catch"]) and int(change_flags.sum()) != 0:
            session["sanity_change_flag_mismatch_count"] += 1
            continue

        change_idx = np.flatnonzero(change_flags)
        session["sanity_total_omitted_intervals"] += int(omitted_flags.sum())
        if len(change_idx) > 0:
            session["sanity_change_interval_omission_count"] += int(
                omitted_flags[change_idx].sum()
            )
            if change_idx[0] > 0:
                session["sanity_pre_change_omission_count"] += int(
                    omitted_flags[change_idx[0] - 1]
                )

        session["neural_trials"].append(neural_trial.astype(np.float32))
        session["running_cont"].append(running_trial.astype(np.float32))
        session["pupil_cont"].append(pupil_trial.astype(np.float32))
        session["interval_image_names"].append(
            [
                NO_IMAGE_LABEL if omitted else str(image_name)
                for image_name, omitted in zip(
                    trial_stim["image_name"].tolist(),
                    omitted_flags.tolist(),
                )
            ]
        )
        session["interval_change_flags"].append(change_flags.astype(bool).tolist())
        session["trial_outcomes"].append(get_trial_outcome(row))
        session["trial_ids"].append(int(trial_id))
        session["trial_interval_counts"].append(int(neural_trial.shape[1]))
        session["go_flags"].append(bool(row["go"]))

    if len(session["neural_trials"]) < 2:
        return None, {
            "skip_reason": "too_few_binned_trials",
            "n_valid_trials": int(len(session["neural_trials"])),
        }

    stats = {
        "n_trials_before_filter": int(len(trials)),
        "n_trials_after_filter": int(len(valid_trials)),
        "n_trials_kept": int(len(session["neural_trials"])),
        "n_intervals_kept": int(sum(session["trial_interval_counts"])),
        "min_intervals_per_trial": int(min(session["trial_interval_counts"])),
        "max_intervals_per_trial": int(max(session["trial_interval_counts"])),
        "n_neurons": int(events_matrix.shape[0]),
        "ophys_rate_hz": float(1.0 / np.mean(np.diff(ophys_timestamps))),
        "eye_tracking_rows": int(len(dataset.eye_tracking)),
        "running_rows": int(len(dataset.running_speed)),
        "omitted_interval_count": int(session["sanity_total_omitted_intervals"]),
        "change_interval_omission_count": int(
            session["sanity_change_interval_omission_count"]
        ),
        "pre_change_omission_count": int(
            session["sanity_pre_change_omission_count"]
        ),
        "change_flag_mismatch_count": int(
            session["sanity_change_flag_mismatch_count"]
        ),
    }
    return session, stats


def convert_sessions_to_dataset(
    sessions,
    time_bin_ms,
    running_edges,
    pupil_edges,
    image_name_to_idx,
):
    brain_regions = sorted({session["brain_region"] for session in sessions})
    brain_region_to_idx = {name: idx for idx, name in enumerate(brain_regions)}

    subjects = sorted({session["mouse_id"] for session in sessions})
    subject_to_idx = {name: idx for idx, name in enumerate(subjects)}

    neural = []
    decoder_input = []
    output = []
    subject_idx = []
    brain_region_idx = []

    total_go_trials = 0
    total_catch_trials = 0
    outcome_counts = {name: 0 for name in TRIAL_OUTCOME_VALUES}

    for session in sessions:
        neural_trials = []
        input_trials = []
        output_trials = []

        for trial_idx, neural_trial in enumerate(session["neural_trials"]):
            T = neural_trial.shape[1]
            running_bins = digitize_with_edges(
                session["running_cont"][trial_idx], running_edges
            )
            pupil_bins = digitize_with_edges(
                session["pupil_cont"][trial_idx], pupil_edges
            )
            image_identity = np.asarray(
                [
                    image_name_to_idx[name]
                    for name in session["interval_image_names"][trial_idx]
                ],
                dtype=np.int64,
            )
            image_change = np.asarray(
                session["interval_change_flags"][trial_idx], dtype=np.int64
            )
            trial_outcome = np.full(
                T, session["trial_outcomes"][trial_idx], dtype=np.int64
            )

            if session["go_flags"][trial_idx]:
                total_go_trials += 1
            else:
                total_catch_trials += 1
            outcome_counts[TRIAL_OUTCOME_VALUES[session["trial_outcomes"][trial_idx]]] += 1

            output_trial = np.vstack(
                [
                    image_identity,
                    image_change,
                    running_bins,
                    pupil_bins,
                    trial_outcome,
                ]
            ).astype(np.int64)

            neural_trials.append(neural_trial.astype(np.float32))
            input_trials.append(np.zeros((0, T), dtype=np.float32))
            output_trials.append(output_trial)

        neural.append(neural_trials)
        decoder_input.append(input_trials)
        output.append(output_trials)
        subject_idx.append(subject_to_idx[session["mouse_id"]])
        brain_region_idx.append(
            np.full(
                session["n_neurons"],
                brain_region_to_idx[session["brain_region"]],
                dtype=np.int64,
            )
        )

    data = {
        "neural": neural,
        "input": decoder_input,
        "output": output,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
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
            [name for name, _ in sorted(image_name_to_idx.items(), key=lambda x: x[1])],
            ["no_change", "change"],
            [f"bin_{i}" for i in range(5)],
            [f"bin_{i}" for i in range(5)],
            TRIAL_OUTCOME_VALUES,
        ],
        "metadata": {
            "task_description": (
                "Allen Visual Behavior change-detection task represented as variable-length "
                "AllenSDK trials. Each timepoint is one native 750 ms image-presentation "
                "interval, and neural events, running, pupil, and stimulus labels are "
                "aggregated over those intervals."
            ),
            "time_bin_size": float(time_bin_ms),
            "temporal_alignment_event": (
                "Successive image-presentation interval onsets within each AllenSDK trial"
            ),
            "off_start": 0.0,
            "off_end": None,
            "neural_signal": "AllenSDK ophys inferred events",
            "selection": {
                "included_behavior_type": "active_behavior",
                "included_trial_types": ["go", "catch"],
                "excluded_trial_types": ["aborted", "auto_rewarded"],
                "included_projects_in_local_subset": sorted(
                    {session["project_code"] for session in sessions}
                ),
                "included_experience_levels_in_local_subset": sorted(
                    {session["experience_level"] for session in sessions}
                ),
            },
            "running_speed_bin_edges": [float(x) for x in running_edges.tolist()],
            "pupil_diameter_bin_edges": [float(x) for x in pupil_edges.tolist()],
            "go_trial_count": int(total_go_trials),
            "catch_trial_count": int(total_catch_trials),
            "trial_outcome_counts": outcome_counts,
        },
    }
    return data


def choose_sample_sessions(num_sessions, sample_session_count):
    if num_sessions <= sample_session_count:
        return np.arange(num_sessions, dtype=int)
    return np.unique(np.linspace(0, num_sessions - 1, sample_session_count, dtype=int))


def summarize_sessions(session_stats, sessions):
    rows = []
    for session in sessions:
        stats = session_stats[session["experiment_id"]]
        rows.append(
            {
                "ophys_experiment_id": session["experiment_id"],
                "ophys_session_id": session["ophys_session_id"],
                "mouse_id": session["mouse_id"],
                "brain_region": session["brain_region"],
                "project_code": session["project_code"],
                "session_type": session["session_type"],
                "experience_level": session["experience_level"],
                "cre_line": session["cre_line"],
                "n_neurons": stats["n_neurons"],
                "n_trials_before_filter": stats["n_trials_before_filter"],
                "n_trials_after_filter": stats["n_trials_after_filter"],
                "n_trials_kept": stats["n_trials_kept"],
                "n_intervals_kept": stats["n_intervals_kept"],
                "min_intervals_per_trial": stats["min_intervals_per_trial"],
                "max_intervals_per_trial": stats["max_intervals_per_trial"],
                "ophys_rate_hz": stats["ophys_rate_hz"],
                "omitted_interval_count": stats["omitted_interval_count"],
                "change_interval_omission_count": stats["change_interval_omission_count"],
                "pre_change_omission_count": stats["pre_change_omission_count"],
                "change_flag_mismatch_count": stats["change_flag_mismatch_count"],
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if not math.isclose(args.time_bin_ms, TIME_BIN_MS_DEFAULT, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "This converter uses native 750 ms image-presentation intervals. "
            "Keep --time-bin-ms=750."
        )

    data_root = args.data_root
    available_ids = get_available_experiment_ids(data_root)
    exp_table = load_experiment_table(data_root)
    selected_df = select_experiments(exp_table, available_ids, args.max_sessions)

    print(f"Available NWB files: {len(available_ids)}", flush=True)
    print(f"Selected active experiments: {len(selected_df)}", flush=True)
    print(f"Selected mice: {selected_df['mouse_id'].nunique()}", flush=True)
    print(
        f"Selected project codes: {selected_df['project_code'].value_counts().to_dict()}",
        flush=True,
    )
    print(
        f"Selected experience levels: {selected_df['experience_level'].value_counts().to_dict()}",
        flush=True,
    )
    print(
        f"Selected session types: {selected_df['session_type'].value_counts().to_dict()}",
        flush=True,
    )
    print(f"Time bin size (ms): {args.time_bin_ms}", flush=True)
    print("Temporal unit: native 750 ms image-presentation intervals", flush=True)

    sessions = []
    session_stats = {}
    skipped = []
    all_running_values = []
    all_pupil_values = []
    image_names = {NO_IMAGE_LABEL}

    for idx, (_, meta_row) in enumerate(selected_df.iterrows(), start=1):
        experiment_id = int(meta_row["ophys_experiment_id"])
        nwb_path = (
            data_root
            / "behavior_ophys_experiments"
            / f"behavior_ophys_experiment_{experiment_id}.nwb"
        )
        session, stats = load_session(
            experiment_id=experiment_id,
            nwb_path=nwb_path,
            meta_row=meta_row,
        )
        if session is None:
            skipped.append({"ophys_experiment_id": int(experiment_id), **stats})
            continue

        sessions.append(session)
        session_stats[int(experiment_id)] = stats
        all_running_values.extend(session["running_cont"])
        all_pupil_values.extend(session["pupil_cont"])
        for trial_image_names in session["interval_image_names"]:
            image_names.update(trial_image_names)

        if idx % 25 == 0 or idx == len(selected_df):
            print(
                f"Loaded {idx}/{len(selected_df)} experiments; kept sessions so far: {len(sessions)}; "
                f"skipped: {len(skipped)}",
                flush=True,
            )

    if not sessions:
        raise RuntimeError("No sessions remained after filtering.")

    all_running_values = np.concatenate(all_running_values).astype(np.float32)
    all_pupil_values = np.concatenate(all_pupil_values).astype(np.float32)
    running_edges = compute_quantile_edges(all_running_values, nbins=5)
    pupil_edges = compute_quantile_edges(all_pupil_values, nbins=5)
    image_name_to_idx = {name: idx for idx, name in enumerate(sorted(image_names))}

    full_data = convert_sessions_to_dataset(
        sessions=sessions,
        time_bin_ms=args.time_bin_ms,
        running_edges=running_edges,
        pupil_edges=pupil_edges,
        image_name_to_idx=image_name_to_idx,
    )

    session_summary = summarize_sessions(session_stats, sessions)
    full_data["metadata"]["session_summary"] = session_summary.to_dict(orient="records")
    full_data["metadata"]["skipped_sessions"] = skipped
    full_data["metadata"]["available_nwb_count"] = int(len(available_ids))
    full_data["metadata"]["selected_active_experiment_count"] = int(len(selected_df))
    full_data["metadata"]["included_session_count"] = int(len(sessions))
    full_data["metadata"]["included_mouse_count"] = int(len(full_data["subjects"]))
    full_data["metadata"]["image_name_count"] = int(len(image_name_to_idx))

    sample_indices = choose_sample_sessions(len(sessions), args.sample_session_count)
    sample_sessions = [sessions[i] for i in sample_indices]
    sample_data = convert_sessions_to_dataset(
        sessions=sample_sessions,
        time_bin_ms=args.time_bin_ms,
        running_edges=running_edges,
        pupil_edges=pupil_edges,
        image_name_to_idx=image_name_to_idx,
    )
    sample_data["metadata"]["parent_session_indices"] = sample_indices.astype(int).tolist()
    sample_data["metadata"]["selected_from_full_experiment_ids"] = [
        int(sessions[i]["experiment_id"]) for i in sample_indices
    ]

    with args.output.open("wb") as f:
        pickle.dump(full_data, f)
    with args.sample_output.open("wb") as f:
        pickle.dump(sample_data, f)

    outcome_counts = full_data["metadata"]["trial_outcome_counts"]
    running_bins = np.concatenate(
        [trial[2] for session_trials in full_data["output"] for trial in session_trials]
    )
    pupil_bins = np.concatenate(
        [trial[3] for session_trials in full_data["output"] for trial in session_trials]
    )
    image_change_bins = np.concatenate(
        [trial[1] for session_trials in full_data["output"] for trial in session_trials]
    )
    total_intervals = sum(
        trial.shape[1] for session_trials in full_data["neural"] for trial in session_trials
    )

    print()
    print("Conversion summary")
    print(f"  Included sessions: {len(full_data['neural'])}")
    print(f"  Included subjects: {len(full_data['subjects'])}")
    print(f"  Total trials: {sum(len(x) for x in full_data['neural'])}")
    print(f"  Total image intervals: {total_intervals}")
    print(
        f"  Trial count range per session: "
        f"{min(len(x) for x in full_data['neural'])} - {max(len(x) for x in full_data['neural'])}"
    )
    print(
        f"  Interval count range per trial: "
        f"{min(trial.shape[1] for session_trials in full_data['neural'] for trial in session_trials)} - "
        f"{max(trial.shape[1] for session_trials in full_data['neural'] for trial in session_trials)}"
    )
    print(
        f"  Neuron count range per session: "
        f"{min(arr.shape[0] for arr in [sess[0] for sess in full_data['neural'] if sess])} - "
        f"{max(arr.shape[0] for arr in [sess[0] for sess in full_data['neural'] if sess])}"
    )
    print(f"  Unique image labels: {len(full_data['output_values'][0])}")
    print(f"  Outcome counts: {json.dumps(outcome_counts, sort_keys=True)}")
    print(
        f"  Running bin counts: {dict(zip(range(5), np.bincount(running_bins, minlength=5).tolist()))}"
    )
    print(
        f"  Pupil bin counts: {dict(zip(range(5), np.bincount(pupil_bins, minlength=5).tolist()))}"
    )
    print(
        f"  Change-interval positives: {int(image_change_bins.sum())} / {int(len(image_change_bins))}"
    )
    print(
        f"  Omitted intervals in kept trials: {int(session_summary['omitted_interval_count'].sum())}"
    )
    print(
        f"  Omission sanity checks: "
        f"change intervals omitted = {int(session_summary['change_interval_omission_count'].sum())}, "
        f"pre-change intervals omitted = {int(session_summary['pre_change_omission_count'].sum())}, "
        f"change-flag mismatches = {int(session_summary['change_flag_mismatch_count'].sum())}"
    )
    print(f"  Full dataset written to: {args.output}")
    print(f"  Sample dataset written to: {args.sample_output}")


if __name__ == "__main__":
    main()
