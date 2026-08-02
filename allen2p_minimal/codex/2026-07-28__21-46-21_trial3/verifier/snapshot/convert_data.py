import argparse
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent
SDK_DIR = REPO_DIR / "code"
if str(SDK_DIR) not in sys.path:
    sys.path.insert(0, str(SDK_DIR))

from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (  # noqa: E402
    BehaviorOphysExperiment,
)
from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_project_cache import (  # noqa: E402
    UpdatedStimulusPresentationTableWarning,
    VisualBehaviorOphysProjectCache,
)


DATA_DIR_DEFAULT = REPO_DIR / "data"
NWB_DIRNAME = "visual-behavior-ophys-1.1.0/behavior_ophys_experiments"
TIME_BIN_MS_DEFAULT = 100.0
GRAY_LABEL = "gray"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Allen Visual Behavior ophys data to the decoder format."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR_DEFAULT,
        help="Root directory containing the Visual Behavior cache.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_DIR / "converted_data.pkl",
        help="Path for the converted pickle output.",
    )
    parser.add_argument(
        "--time-bin-ms",
        type=float,
        default=TIME_BIN_MS_DEFAULT,
        help="Common time bin size in milliseconds for all sessions.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="If set, stop after including this many sessions.",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Optional limit on candidate experiments to inspect, for debugging.",
    )
    parser.add_argument(
        "--experiment-ids",
        type=str,
        default=None,
        help=(
            "Optional comma-separated Allen ophys experiment IDs to include. "
            "Useful for reproducing a fixed subset such as sample_data.pkl."
        ),
    )
    return parser.parse_args()


def available_experiment_table(data_dir: Path) -> pd.DataFrame:
    nwb_root = data_dir / NWB_DIRNAME
    local_paths = {}
    for path in sorted(nwb_root.glob("behavior_ophys_experiment_*.nwb")):
        match = re.search(r"(\d+)\.nwb$", path.name)
        if match:
            local_paths[int(match.group(1))] = path

    warnings.filterwarnings("ignore", category=UpdatedStimulusPresentationTableWarning)
    cache = VisualBehaviorOphysProjectCache.from_local_cache(str(data_dir))
    experiments = cache.get_ophys_experiment_table()
    experiments = experiments.loc[experiments.index.intersection(local_paths.keys())].copy()
    experiments["local_path"] = [str(local_paths[int(idx)]) for idx in experiments.index]
    experiments = experiments.sort_values(
        [
            "date_of_acquisition",
            "mouse_id",
            "ophys_session_id",
            "targeted_structure",
            "imaging_depth",
        ]
    )
    return experiments


def make_target_times(start_time: float, stop_time: float, bin_size_s: float) -> np.ndarray:
    edges = np.arange(start_time, stop_time, bin_size_s, dtype=np.float64)
    centers = edges + (0.5 * bin_size_s)
    return centers[centers < stop_time]


def nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    if source_times.ndim != 1:
        raise ValueError("source_times must be 1D")
    if source_times.size == 0:
        raise ValueError("source_times cannot be empty")
    if source_times.size == 1:
        return np.zeros(target_times.shape[0], dtype=np.int64)

    idx = np.searchsorted(source_times, target_times, side="left")
    idx = np.clip(idx, 1, source_times.size - 1)
    left = idx - 1
    right = idx
    choose_right = np.abs(source_times[right] - target_times) < np.abs(
        target_times - source_times[left]
    )
    return np.where(choose_right, right, left).astype(np.int64)


def interp_signal(
    source_times: np.ndarray, source_values: np.ndarray, target_times: np.ndarray
) -> np.ndarray | None:
    valid = np.isfinite(source_times) & np.isfinite(source_values)
    if valid.sum() == 0:
        return None
    times = np.asarray(source_times[valid], dtype=np.float64)
    values = np.asarray(source_values[valid], dtype=np.float64)
    if times.size == 1:
        return np.full(target_times.shape[0], values[0], dtype=np.float32)

    order = np.argsort(times)
    times = times[order]
    values = values[order]
    unique_times, unique_idx = np.unique(times, return_index=True)
    values = values[unique_idx]
    times = unique_times
    interp = np.interp(target_times, times, values, left=values[0], right=values[-1])
    return interp.astype(np.float32)


def outcome_to_index(trial_row: pd.Series) -> int | None:
    if bool(trial_row["hit"]):
        return 0
    if bool(trial_row["miss"]):
        return 1
    if bool(trial_row["false_alarm"]):
        return 2
    if bool(trial_row["correct_reject"]):
        return 3
    return None


def remap_to_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=False).astype(np.int16)


def build_output_array(
    image_labels: np.ndarray,
    change_labels: np.ndarray,
    running_bins: np.ndarray,
    pupil_bins: np.ndarray,
    outcome_idx: int,
) -> np.ndarray:
    t = image_labels.shape[0]
    outcome = np.full(t, outcome_idx, dtype=np.int16)
    return np.vstack(
        [
            image_labels.astype(np.int16),
            change_labels.astype(np.int16),
            running_bins.astype(np.int16),
            pupil_bins.astype(np.int16),
            outcome,
        ]
    )


def convert_dataset(
    experiments: pd.DataFrame,
    time_bin_ms: float,
    stop_after_sessions: int | None = None,
) -> tuple[dict, dict]:
    bin_size_s = time_bin_ms / 1000.0

    warnings.filterwarnings("ignore", category=UpdatedStimulusPresentationTableWarning)
    warnings.filterwarnings(
        "ignore",
        message=r"Ignoring the following cached namespace\(s\).*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Downcasting object dtype arrays on \.fillna.*",
        category=FutureWarning,
    )

    experiments = experiments.loc[~experiments["passive"]].copy()
    if stop_after_sessions is not None:
        print(f"Will stop after including {stop_after_sessions} sessions.")

    subjects: list[str] = []
    subject_to_idx: dict[str, int] = {}
    brain_regions: list[str] = []
    brain_region_to_idx: dict[str, int] = {}
    image_values = [GRAY_LABEL]
    image_to_idx = {GRAY_LABEL: 0}

    neural_sessions = []
    input_sessions = []
    output_temp_sessions = []
    brain_region_idx_sessions = []
    subject_idx = []
    session_info = []

    all_running = []
    all_pupil = []
    native_dt_ms = []
    excluded_sessions = []

    n_candidates = len(experiments)
    print(f"Inspecting {n_candidates} active candidate experiments.")

    for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
        if stop_after_sessions is not None and len(neural_sessions) >= stop_after_sessions:
            break

        if candidate_number % 10 == 1 or candidate_number == n_candidates:
            print(
                f"[{candidate_number}/{n_candidates}] loading experiment {experiment_id} "
                f"({row['session_type']}, mouse {row['mouse_id']}, {row['targeted_structure']})"
            )

        try:
            dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
        except Exception as exc:  # pragma: no cover - robust logging
            excluded_sessions.append(
                {
                    "experiment_id": int(experiment_id),
                    "reason": f"load_failed:{type(exc).__name__}",
                }
            )
            print(f"  skip {experiment_id}: failed to load ({exc})")
            continue

        ophys_timestamps = np.asarray(dataset.ophys_timestamps, dtype=np.float64)
        if ophys_timestamps.size < 2:
            excluded_sessions.append(
                {"experiment_id": int(experiment_id), "reason": "too_few_ophys_timestamps"}
            )
            print(f"  skip {experiment_id}: not enough ophys timestamps")
            continue
        native_dt_ms.append(float(np.median(np.diff(ophys_timestamps)) * 1000.0))

        events_df = dataset.events
        if len(events_df) == 0:
            excluded_sessions.append({"experiment_id": int(experiment_id), "reason": "no_events"})
            print(f"  skip {experiment_id}: no valid event traces")
            continue
        neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)

        running_df = dataset.running_speed
        running_interp_source = interp_signal(
            running_df["timestamps"].to_numpy(dtype=np.float64),
            running_df["speed"].to_numpy(dtype=np.float64),
            np.array([ophys_timestamps[0]], dtype=np.float64),
        )
        if running_interp_source is None:
            excluded_sessions.append({"experiment_id": int(experiment_id), "reason": "no_running"})
            print(f"  skip {experiment_id}: no running signal")
            continue

        eye_df = dataset.eye_tracking
        pupil_diameter = 2.0 * np.sqrt(
            np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi
        )
        if not np.isfinite(pupil_diameter).any():
            excluded_sessions.append(
                {"experiment_id": int(experiment_id), "reason": "no_valid_pupil"}
            )
            print(f"  skip {experiment_id}: no valid pupil signal")
            continue

        stimulus_presentations = dataset.stimulus_presentations
        change_detection = stimulus_presentations[
            stimulus_presentations["stimulus_block_name"].str.contains(
                "change_detection", na=False
            )
        ].copy()
        change_detection = change_detection.sort_values("start_time")

        trials = dataset.trials.copy()
        trial_mask = (
            (trials["go"] | trials["catch"])
            & (~trials["aborted"])
            & (~trials["auto_rewarded"])
        )
        trials = trials.loc[trial_mask].copy()
        if len(trials) < 2:
            excluded_sessions.append(
                {"experiment_id": int(experiment_id), "reason": "fewer_than_two_valid_trials"}
            )
            print(f"  skip {experiment_id}: fewer than two valid go/catch trials")
            continue

        neural_trials = []
        input_trials = []
        output_temp_trials = []

        for trial_id, trial_row in trials.iterrows():
            start_time = float(trial_row["start_time"])
            stop_time = float(trial_row["stop_time"])
            target_times = make_target_times(start_time, stop_time, bin_size_s)
            if target_times.size < 2:
                continue

            nearest = nearest_indices(ophys_timestamps, target_times)
            neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)

            running_trial = interp_signal(
                running_df["timestamps"].to_numpy(dtype=np.float64),
                running_df["speed"].to_numpy(dtype=np.float64),
                target_times,
            )
            pupil_trial = interp_signal(
                eye_df["timestamps"].to_numpy(dtype=np.float64),
                pupil_diameter,
                target_times,
            )
            if running_trial is None or pupil_trial is None:
                continue

            image_labels = np.full(target_times.shape[0], image_to_idx[GRAY_LABEL], dtype=np.int16)
            change_labels = np.zeros(target_times.shape[0], dtype=np.int16)

            trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
            trial_stim = trial_stim.sort_values("start_time")
            for stim_row in trial_stim.itertuples():
                if bool(stim_row.omitted) or stim_row.image_name == "omitted":
                    continue
                if stim_row.image_name not in image_to_idx:
                    image_to_idx[stim_row.image_name] = len(image_values)
                    image_values.append(stim_row.image_name)
                image_idx = image_to_idx[stim_row.image_name]
                mask = (target_times >= float(stim_row.start_time)) & (
                    target_times < float(stim_row.end_time)
                )
                image_labels[mask] = image_idx
                if bool(stim_row.is_change):
                    change_labels[mask] = 1

            outcome_idx = outcome_to_index(trial_row)
            if outcome_idx is None:
                continue

            neural_trials.append(neural_trial)
            input_trials.append(np.zeros((0, target_times.shape[0]), dtype=np.float32))
            output_temp_trials.append(
                {
                    "image_labels": image_labels,
                    "change_labels": change_labels,
                    "running": running_trial,
                    "pupil": pupil_trial,
                    "outcome_idx": outcome_idx,
                }
            )
            all_running.append(running_trial)
            all_pupil.append(pupil_trial)

        if len(neural_trials) < 2:
            excluded_sessions.append(
                {"experiment_id": int(experiment_id), "reason": "fewer_than_two_kept_trials"}
            )
            print(f"  skip {experiment_id}: fewer than two kept trials after alignment")
            continue

        subject = str(row["mouse_id"])
        if subject not in subject_to_idx:
            subject_to_idx[subject] = len(subjects)
            subjects.append(subject)

        region = str(row["targeted_structure"])
        if region not in brain_region_to_idx:
            brain_region_to_idx[region] = len(brain_regions)
            brain_regions.append(region)

        subject_idx.append(subject_to_idx[subject])
        brain_region_idx_sessions.append(
            np.full(neural_full.shape[0], brain_region_to_idx[region], dtype=np.int16)
        )
        neural_sessions.append(neural_trials)
        input_sessions.append(input_trials)
        output_temp_sessions.append(output_temp_trials)
        session_info.append(
            {
                "experiment_id": int(experiment_id),
                "ophys_session_id": int(row["ophys_session_id"]),
                "behavior_session_id": int(row["behavior_session_id"]),
                "mouse_id": subject,
                "project_code": str(row["project_code"]),
                "session_type": str(row["session_type"]),
                "experience_level": str(row["experience_level"]),
                "targeted_structure": region,
                "imaging_depth": int(row["imaging_depth"]),
                "n_neurons": int(neural_full.shape[0]),
                "n_trials": int(len(neural_trials)),
                "native_frame_interval_ms_median": float(native_dt_ms[-1]),
                "local_path": str(row["local_path"]),
            }
        )

    if not neural_sessions:
        raise RuntimeError("No sessions were converted successfully.")

    running_values = np.concatenate(all_running, axis=0)
    pupil_values = np.concatenate(all_pupil, axis=0)
    running_edges = np.percentile(running_values, [20, 40, 60, 80]).astype(np.float64)
    pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80]).astype(np.float64)

    output_sessions = []
    for output_temp_trials in output_temp_sessions:
        output_trials = []
        for trial in output_temp_trials:
            running_bins = remap_to_bins(trial["running"], running_edges)
            pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)
            output_trials.append(
                build_output_array(
                    image_labels=trial["image_labels"],
                    change_labels=trial["change_labels"],
                    running_bins=running_bins,
                    pupil_bins=pupil_bins,
                    outcome_idx=trial["outcome_idx"],
                )
            )
        output_sessions.append(output_trials)

    metadata = {
        "task_description": (
            "Allen Visual Behavior change-detection ophys data converted into "
            "trial-segmented sessions with event-trace neural activity and "
            "time-varying stimulus and behavior labels."
        ),
        "time_bin_size": float(time_bin_ms),
        "temporal_alignment_event": "trial start time from AllenSDK trials table",
        "off_start": 0.0,
        "off_end": None,
        "trial_inclusion": "keep go and catch trials; exclude aborted and auto-rewarded trials",
        "neural_signal": "AllenSDK discrete calcium events",
        "stimulus_processing": (
            "Use change_detection stimulus block only; label flashed image "
            "identity during non-gray image epochs and gray otherwise."
        ),
        "running_processing": "AllenSDK processed running speed aligned by timestamp and binned into global quintiles",
        "pupil_processing": (
            "Equivalent pupil diameter derived from AllenSDK processed pupil_area, "
            "aligned by timestamp and binned into global quintiles"
        ),
        "source_subset": {
            "on_disk_nwb_experiments": int(len(experiments)),
            "included_active_sessions": int(len(neural_sessions)),
            "excluded_sessions": excluded_sessions,
            "project_codes": sorted({str(x) for x in experiments["project_code"].unique()}),
            "session_types": sorted({str(x) for x in experiments["session_type"].unique()}),
        },
        "running_bin_edges": [float(x) for x in running_edges],
        "pupil_bin_edges": [float(x) for x in pupil_edges],
        "session_info": session_info,
        "native_ophys_frame_interval_ms_summary": {
            "mean": float(np.mean(native_dt_ms)),
            "median": float(np.median(native_dt_ms)),
            "min": float(np.min(native_dt_ms)),
            "max": float(np.max(native_dt_ms)),
        },
    }

    data = {
        "neural": neural_sessions,
        "input": input_sessions,
        "output": output_sessions,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int16),
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx_sessions,
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
            [f"bin_{i}" for i in range(5)],
            [f"bin_{i}" for i in range(5)],
            ["hit", "miss", "false_alarm", "correct_reject"],
        ],
        "metadata": metadata,
    }

    summary = {
        "n_sessions": len(neural_sessions),
        "n_subjects": len(subjects),
        "n_trials_total": int(sum(len(session) for session in neural_sessions)),
        "n_neurons_total": int(
            sum(session_trials[0].shape[0] for session_trials in neural_sessions)
        ),
        "brain_regions": brain_regions,
        "subjects": subjects,
        "running_bin_edges": [float(x) for x in running_edges],
        "pupil_bin_edges": [float(x) for x in pupil_edges],
        "native_dt_ms_summary": metadata["native_ophys_frame_interval_ms_summary"],
        "excluded_sessions_count": len(excluded_sessions),
    }
    return data, summary


def main() -> None:
    args = parse_args()

    experiments = available_experiment_table(args.data_dir)
    if args.experiment_ids:
        requested_ids = {
            int(part.strip()) for part in args.experiment_ids.split(",") if part.strip()
        }
        experiments = experiments.loc[experiments.index.intersection(requested_ids)].copy()
        missing_ids = sorted(requested_ids.difference(set(experiments.index.tolist())))
        print(f"Filtered to {len(experiments)} requested experiment IDs.")
        if missing_ids:
            print(f"Requested experiment IDs not found locally: {missing_ids}")
    if args.max_experiments is not None:
        experiments = experiments.head(args.max_experiments).copy()

    print(f"Found {len(experiments)} locally available ophys experiment files.")
    print("Available project counts:")
    print(experiments["project_code"].value_counts().to_string())
    print("Available active session types:")
    print(experiments.loc[~experiments["passive"], "session_type"].value_counts().to_string())

    data, summary = convert_dataset(
        experiments=experiments,
        time_bin_ms=args.time_bin_ms,
        stop_after_sessions=args.sample_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nSaved converted dataset to:", args.output)
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
