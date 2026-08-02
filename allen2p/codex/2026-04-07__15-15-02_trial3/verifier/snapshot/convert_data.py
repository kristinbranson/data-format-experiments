#!/usr/bin/env python3
"""Convert local Visual Behavior Ophys data into decoder format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

from __future__ import annotations

import argparse
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]
RUNNING_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
PUPIL_BIN_NAMES = [f"q{i}" for i in range(1, 6)]


@dataclass(frozen=True)
class SessionMeta:
    ophys_experiment_id: int
    ophys_session_id: int
    behavior_session_id: int
    mouse_id: str
    targeted_structure: str
    session_type: str
    experience_level: str
    project_code: str
    filepath: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all eligible sessions")
    mode.add_argument("--sample", action="store_true", help="Process only two eligible sessions")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save diagnostic processing plots for up to 2 sessions",
    )
    return parser.parse_args()


def decode_str_array(values: np.ndarray) -> np.ndarray:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        elif isinstance(value, str):
            out.append(value)
        elif hasattr(value, "decode"):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return np.asarray(out, dtype=object)


def read_dataset(ds: h5py.Dataset) -> np.ndarray:
    values = ds[:]
    if values.dtype.kind in {"S", "O", "U"}:
        return decode_str_array(values)
    return np.asarray(values)


def read_interval_table(group: h5py.Group, columns: Iterable[str]) -> pd.DataFrame:
    data = {}
    for column in columns:
        if column not in group:
            continue
        data[column] = read_dataset(group[column])
    return pd.DataFrame(data)


def build_trial_bins(start_time: float, stop_time: float) -> np.ndarray:
    if not np.isfinite(start_time) or not np.isfinite(stop_time) or stop_time <= start_time:
        return np.asarray([], dtype=np.float64)
    n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
    valid = centers < (stop_time + 1e-9)
    return centers[valid]


def linear_resample_vector(
    src_time: np.ndarray,
    src_value: np.ndarray,
    dst_time: np.ndarray,
) -> np.ndarray:
    if dst_time.size == 0:
        return np.asarray([], dtype=np.float32)
    if src_time.size == 0:
        raise ValueError("Cannot resample from an empty source time series")
    return np.interp(dst_time, src_time, src_value).astype(np.float32, copy=False)


def linear_resample_matrix(
    src_time: np.ndarray,
    src_value: np.ndarray,
    dst_time: np.ndarray,
) -> np.ndarray:
    """Resample a time x features matrix onto dst_time."""
    if dst_time.size == 0:
        return np.zeros((src_value.shape[1], 0), dtype=np.float32)

    idx_hi = np.searchsorted(src_time, dst_time, side="left")
    idx_hi = np.clip(idx_hi, 1, len(src_time) - 1)
    idx_lo = idx_hi - 1

    t0 = src_time[idx_lo]
    t1 = src_time[idx_hi]
    denom = np.where(t1 > t0, t1 - t0, 1.0)
    w = ((dst_time - t0) / denom).astype(np.float32)
    interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
    return interp.T.astype(np.float32, copy=False)


def fill_nan_by_time(time_axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=True)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        raise ValueError("All values are NaN")
    if finite.all():
        return values
    values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
    return values


def compute_quantile_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    if values.size == 0:
        raise ValueError("Cannot compute quantile edges from empty values")
    probs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(values, probs)
    if np.unique(edges).size < edges.size:
        eps = np.finfo(np.float64).eps
        edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
    return edges


def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, edges[0], edges[-1])
    bins = np.searchsorted(edges[1:-1], clipped, side="right")
    return bins.astype(np.int64, copy=False)


def get_local_session_metadata(data_root: Path) -> list[SessionMeta]:
    table_path = data_root / "visual-behavior-ophys-1.1.0" / "project_metadata" / "ophys_experiment_table.csv"
    exp_table = pd.read_csv(table_path)

    experiment_dir = data_root / "visual-behavior-ophys-1.1.0" / "behavior_ophys_experiments"
    available_files = {
        int(path.stem.split("_")[-1]): path
        for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb"))
    }

    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(available_files)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()
    exp_table = exp_table.sort_values("ophys_experiment_id")

    sessions = []
    for row in exp_table.itertuples(index=False):
        sessions.append(
            SessionMeta(
                ophys_experiment_id=int(row.ophys_experiment_id),
                ophys_session_id=int(row.ophys_session_id),
                behavior_session_id=int(row.behavior_session_id),
                mouse_id=str(row.mouse_id),
                targeted_structure=str(row.targeted_structure),
                session_type=str(row.session_type),
                experience_level=str(row.experience_level),
                project_code=str(row.project_code),
                filepath=available_files[int(row.ophys_experiment_id)],
            )
        )
    return sessions


def get_task_presentations(f: h5py.File) -> pd.DataFrame:
    rows = []
    for name, group in f["intervals"].items():
        if name == "trials":
            continue
        if "stimulus_block_name" not in group or "start_time" not in group:
            continue
        block_names = decode_str_array(group["stimulus_block_name"][:])
        keep = np.array(["change_detection" in x for x in block_names], dtype=bool)
        if not keep.any():
            continue
        columns = [
            "start_time",
            "stop_time",
            "image_name",
            "omitted",
            "is_change",
            "trials_id",
            "stimulus_block_name",
            "active",
            "duration",
        ]
        df = read_interval_table(group, columns)
        df = df.loc[keep].copy()
        if "image_name" not in df:
            continue
        df["stimulus_block_name"] = block_names[keep]
        rows.append(df)
    if not rows:
        return pd.DataFrame(
            columns=["start_time", "stop_time", "image_name", "omitted", "is_change", "trials_id", "stimulus_block_name"]
        )
    out = pd.concat(rows, ignore_index=True)
    out["image_name"] = out["image_name"].astype(str)
    out["omitted"] = out["omitted"].fillna(0).astype(bool)
    out["is_change"] = out["is_change"].fillna(0).astype(bool)
    out["trials_id"] = out["trials_id"].astype(np.int64)
    return out.sort_values("start_time").reset_index(drop=True)


def get_trial_table(f: h5py.File) -> pd.DataFrame:
    columns = [
        "id",
        "start_time",
        "stop_time",
        "go",
        "catch",
        "aborted",
        "auto_rewarded",
        "hit",
        "miss",
        "false_alarm",
        "correct_reject",
        "change_time",
        "initial_image_name",
        "change_image_name",
    ]
    trials = read_interval_table(f["intervals"]["trials"], columns)
    if "id" not in trials:
        trials["id"] = np.arange(len(trials), dtype=np.int64)
    else:
        trials["id"] = trials["id"].astype(np.int64)
    for column in ["go", "catch", "aborted", "auto_rewarded", "hit", "miss", "false_alarm", "correct_reject"]:
        trials[column] = trials[column].astype(bool)
    for column in ["initial_image_name", "change_image_name"]:
        if column in trials:
            trials[column] = trials[column].astype(str)
    return trials


def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events


def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed


def get_pupil_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    if "EyeTracking" not in f["acquisition"]:
        raise KeyError("Missing EyeTracking acquisition")
    eye_group = f["acquisition"]["EyeTracking"]
    timestamps = np.asarray(eye_group["eye_tracking"]["timestamps"][:], dtype=np.float64)
    width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
    height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
    diameter = 2.0 * np.maximum(width, height)
    diameter = fill_nan_by_time(timestamps, diameter)
    return timestamps, diameter


def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)


def trial_outcome_index(trial_row: pd.Series) -> int:
    if bool(trial_row["hit"]):
        return 0
    if bool(trial_row["miss"]):
        return 1
    if bool(trial_row["false_alarm"]):
        return 2
    if bool(trial_row["correct_reject"]):
        return 3
    raise ValueError("Trial has no valid outcome label")


def collect_global_statistics(sessions: list[SessionMeta]) -> tuple[np.ndarray, np.ndarray, list[SessionMeta], set[str]]:
    running_values = []
    pupil_values = []
    image_names = set(["gray"])
    kept_sessions = []

    for idx, session in enumerate(sessions, start=1):
        t0 = time.time()
        try:
            with h5py.File(session.filepath, "r") as f:
                trials = get_trial_table(f)
                trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
                if len(trials) < 2:
                    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: fewer than 2 kept trials")
                    continue

                presentations = get_task_presentations(f)
                if presentations.empty:
                    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: no task presentations")
                    continue
                image_names.update(x for x in presentations["image_name"].unique() if x != "omitted")

                running_time, running_speed = get_running_data(f)
                pupil_time, pupil_diameter = get_pupil_data(f)

                for trial in trials.itertuples(index=False):
                    centers = build_trial_bins(float(trial.start_time), float(trial.stop_time))
                    if centers.size == 0:
                        continue
                    running_trial = linear_resample_vector(running_time, running_speed, centers)
                    pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
                    running_values.append(running_trial)
                    pupil_values.append(pupil_trial)

                kept_sessions.append(session)
                print(
                    f"[pass1 {idx}/{len(sessions)}] kept {session.ophys_experiment_id} "
                    f"trials={len(trials)} elapsed={time.time() - t0:.2f}s"
                )
        except KeyError as exc:
            print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")

    if not kept_sessions:
        raise RuntimeError("No sessions were retained after pass 1")

    running_all = np.concatenate(running_values).astype(np.float64, copy=False)
    pupil_all = np.concatenate(pupil_values).astype(np.float64, copy=False)
    running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
    pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
    return running_edges, pupil_edges, kept_sessions, image_names


def make_processing_plot(
    session: SessionMeta,
    trial_row: pd.Series,
    neural_raw: np.ndarray,
    neural_resampled: np.ndarray,
    ophys_time: np.ndarray,
    centers: np.ndarray,
    running_time: np.ndarray,
    running_speed: np.ndarray,
    pupil_time: np.ndarray,
    pupil_diameter: np.ndarray,
    running_resampled: np.ndarray,
    pupil_resampled: np.ndarray,
    presentations_trial: pd.DataFrame,
    image_trace: np.ndarray,
    image_change_trace: np.ndarray,
    running_edges: np.ndarray,
    pupil_edges: np.ndarray,
    image_value_names: list[str],
) -> None:
    trial_start = float(trial_row["start_time"])
    trial_stop = float(trial_row["stop_time"])
    pad = 0.5

    raw_neural_time = ophys_time - trial_start
    resampled_time = centers - trial_start

    running_mask = (running_time >= (trial_start - pad)) & (running_time <= (trial_stop + pad))
    pupil_mask = (pupil_time >= (trial_start - pad)) & (pupil_time <= (trial_stop + pad))
    running_time_plot = running_time[running_mask] - trial_start
    pupil_time_plot = pupil_time[pupil_mask] - trial_start
    running_speed_plot = running_speed[running_mask]
    pupil_diameter_plot = pupil_diameter[pupil_mask]

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.0, 1.0, 0.9])

    ax0 = fig.add_subplot(gs[0, :])
    if neural_raw.shape[0] > 0:
        sample_neurons = min(5, neural_raw.shape[0])
        activity_order = np.argsort(np.nanmax(neural_resampled, axis=1))[::-1][:sample_neurons]
        offset = 0.0
        for neuron_idx in activity_order:
            ax0.plot(raw_neural_time, neural_raw[neuron_idx] + offset, alpha=0.5, linewidth=0.8)
            ax0.plot(resampled_time, neural_resampled[neuron_idx] + offset, alpha=0.95, linewidth=1.0)
            step = max(
                0.15,
                float(np.nanpercentile(neural_raw[neuron_idx], 99))
                + float(np.nanpercentile(neural_resampled[neuron_idx], 99))
                + 0.05,
            )
            offset += step
    for row in presentations_trial.itertuples(index=False):
        color = "#f0ad4e" if not bool(row.omitted) else "#d9d9d9"
        ax0.axvspan(float(row.start_time) - trial_start, float(row.stop_time) - trial_start, color=color, alpha=0.15)
    if np.isfinite(trial_row["change_time"]):
        ax0.axvline(float(trial_row["change_time"]) - trial_start, color="red", linestyle="--", linewidth=1.5)
    ax0.set_title(
        f"Session {session.ophys_experiment_id} Trial {int(trial_row['id'])}: "
        "raw vs 30 Hz-resampled events"
    )
    ax0.set_ylabel("Sample neuron event traces")
    ax0.set_xlim(-pad, (trial_stop - trial_start) + pad)

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.plot(running_time_plot, running_speed_plot, color="tab:green", alpha=0.4, label="raw")
    ax1.plot(resampled_time, running_resampled, color="tab:green", linewidth=1.0, label="resampled")
    ax1.set_ylabel("Running speed")
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(gs[1, 1], sharex=ax0)
    ax2.plot(pupil_time_plot, pupil_diameter_plot, color="tab:purple", alpha=0.4, label="raw/interp")
    ax2.plot(resampled_time, pupil_resampled, color="tab:purple", linewidth=1.0, label="resampled")
    ax2.set_ylabel("Pupil diameter")
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(gs[2, :], sharex=ax0)
    ax3.step(resampled_time, image_trace + 0.02, where="mid", label="image_identity_code")
    ax3.step(resampled_time, image_change_trace + len(image_value_names) + 0.5, where="mid", label="image_change")
    ax3.set_ylabel("Stimulus codes")
    ax3.set_yticks(list(range(len(image_value_names))) + [len(image_value_names) + 0.5])
    ax3.set_yticklabels(image_value_names + ["change"])
    ax3.legend(loc="upper right")
    ax3.set_xlabel("Time from trial start (s)")

    ax4 = fig.add_subplot(gs[3, 0])
    ax4.hist(running_resampled, bins=40, color="tab:green", alpha=0.7)
    for edge in running_edges[1:-1]:
        ax4.axvline(edge, color="black", linestyle="--", linewidth=1)
    ax4.set_title("Running bin edges")

    ax5 = fig.add_subplot(gs[3, 1])
    ax5.hist(pupil_resampled, bins=40, color="tab:purple", alpha=0.7)
    for edge in pupil_edges[1:-1]:
        ax5.axvline(edge, color="black", linestyle="--", linewidth=1)
    ax5.set_title("Pupil bin edges")

    fig.tight_layout()
    out_path = Path(f"processing_{session.ophys_experiment_id}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def convert_session(
    session: SessionMeta,
    running_edges: np.ndarray,
    pupil_edges: np.ndarray,
    image_value_to_idx: dict[str, int],
    region_to_idx: dict[str, int],
    show_processing: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    with h5py.File(session.filepath, "r") as f:
        trials = get_trial_table(f)
        trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
        trials = trials.sort_values("id").reset_index(drop=True)

        presentations = get_task_presentations(f)
        ophys_time, events = get_neural_data(f)
        running_time, running_speed = get_running_data(f)
        pupil_time, pupil_diameter = get_pupil_data(f)
        brain_region_idx = get_cell_count_and_region_idx(f, region_to_idx[session.targeted_structure])

        neural_trials: list[np.ndarray] = []
        output_trials: list[np.ndarray] = []
        plotted = False

        for trial_idx, trial in trials.iterrows():
            centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
            if centers.size == 0:
                continue

            neural_trial = linear_resample_matrix(ophys_time, events, centers)
            running_trial = linear_resample_vector(running_time, running_speed, centers)
            pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)

            running_bin = digitize_with_edges(running_trial, running_edges)
            pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)

            image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
            image_change = np.zeros(centers.shape[0], dtype=np.int64)

            trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
            for row in trial_presentations.itertuples(index=False):
                mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
                if not mask.any():
                    continue
                if bool(row.omitted) or str(row.image_name) == "omitted":
                    image_identity[mask] = image_value_to_idx["gray"]
                else:
                    image_identity[mask] = image_value_to_idx[str(row.image_name)]
                if bool(row.is_change):
                    image_change[mask] = 1

            outcome_idx = trial_outcome_index(trial)
            outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)

            output_trial = np.vstack(
                [
                    image_identity,
                    image_change,
                    running_bin,
                    pupil_bin,
                    outcome_trace,
                ]
            )

            neural_trials.append(neural_trial.astype(np.float32, copy=False))
            output_trials.append(output_trial)

            if show_processing and not plotted:
                start = max(0, np.searchsorted(ophys_time, float(trial["start_time"])) - 5)
                stop = min(len(ophys_time), np.searchsorted(ophys_time, float(trial["stop_time"]), side="right") + 5)
                raw_window = events[start:stop].T.astype(np.float32, copy=False)
                make_processing_plot(
                    session=session,
                    trial_row=trial,
                    neural_raw=raw_window,
                    neural_resampled=neural_trial,
                    ophys_time=ophys_time[start:stop],
                    centers=centers,
                    running_time=running_time,
                    running_speed=running_speed,
                    pupil_time=pupil_time,
                    pupil_diameter=pupil_diameter,
                    running_resampled=running_trial,
                    pupil_resampled=pupil_trial,
                    presentations_trial=trial_presentations,
                    image_trace=image_identity,
                    image_change_trace=image_change,
                    running_edges=running_edges,
                    pupil_edges=pupil_edges,
                    image_value_names=[name for name, _ in sorted(image_value_to_idx.items(), key=lambda kv: kv[1])],
                )
                plotted = True

        if len(neural_trials) < 2:
            raise RuntimeError(f"Session {session.ophys_experiment_id} has fewer than 2 usable trials")

        return neural_trials, output_trials, brain_region_idx


def main() -> None:
    args = parse_args()
    if not args.full and not args.sample:
        args.full = True

    repo_root = Path(__file__).resolve().parent
    data_root = repo_root / "data"

    all_sessions = get_local_session_metadata(data_root)
    print(f"Found {len(all_sessions)} local active experiment files")

    if args.sample:
        sessions = all_sessions[:2]
        print("Sample mode: processing first 2 active local sessions before QC filters")
    else:
        sessions = all_sessions

    t_global_start = time.time()
    print("Pass 1: collecting global running/pupil statistics")
    running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
    print(f"Pass 1 retained {len(kept_sessions)} sessions")
    print(f"Running bin edges: {running_edges}")
    print(f"Pupil bin edges: {pupil_edges}")

    image_values = sorted(image_names)
    if "gray" in image_values:
        image_values = ["gray"] + [x for x in image_values if x != "gray"]
    image_value_to_idx = {name: idx for idx, name in enumerate(image_values)}

    subjects = sorted({session.mouse_id for session in kept_sessions})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
    brain_regions = sorted({session.targeted_structure for session in kept_sessions})
    region_to_idx = {region: idx for idx, region in enumerate(brain_regions)}

    if args.sample:
        kept_sessions = kept_sessions[:2]

    neural_all: list[list[np.ndarray]] = []
    input_all: list[list[np.ndarray]] = []
    output_all: list[list[np.ndarray]] = []
    subject_idx = np.zeros(len(kept_sessions), dtype=np.int64)
    brain_region_idx_all: list[np.ndarray] = []

    print("Pass 2: converting sessions")
    for idx, session in enumerate(kept_sessions, start=1):
        t0 = time.time()
        neural_trials, output_trials, brain_region_idx = convert_session(
            session=session,
            running_edges=running_edges,
            pupil_edges=pupil_edges,
            image_value_to_idx=image_value_to_idx,
            region_to_idx=region_to_idx,
            show_processing=args.show_processing and idx <= 2,
        )
        input_trials = [np.zeros((0, trial.shape[1]), dtype=np.float32) for trial in neural_trials]

        neural_all.append(neural_trials)
        input_all.append(input_trials)
        output_all.append(output_trials)
        subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
        brain_region_idx_all.append(brain_region_idx)

        mean_t = float(np.mean([trial.shape[1] for trial in neural_trials]))
        print(
            f"[pass2 {idx}/{len(kept_sessions)}] session={session.ophys_experiment_id} "
            f"trials={len(neural_trials)} neurons={brain_region_idx.shape[0]} "
            f"mean_T={mean_t:.1f} elapsed={time.time() - t0:.2f}s"
        )

    data = {
        "neural": neural_all,
        "input": input_all,
        "output": output_all,
        "subjects": subjects,
        "subject_idx": subject_idx,
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx_all,
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
            OUTCOME_NAMES,
        ],
        "metadata": {
            "task_description": (
                "Active Visual Behavior image-change detection trials from local behavior_ophys_experiment NWBs; "
                "outputs are image identity, image-change indicator, running-speed bin, pupil-diameter bin, "
                "and trial outcome."
            ),
            "time_bin_size": TIME_BIN_MS,
            "temporal_alignment_event": "trial start",
            "off_start": 0.0,
            "off_end": None,
            "sampling_grid_hz": 30.0,
            "neural_signal": "ophys event magnitudes from NWB event_detection",
            "trial_filter": {
                "include_go": True,
                "include_catch": True,
                "exclude_aborted": True,
                "exclude_auto_rewarded": True,
                "exclude_passive_sessions": True,
                "require_eye_tracking": True,
            },
            "source_data_root": str(data_root),
            "included_ophys_experiment_ids": [session.ophys_experiment_id for session in kept_sessions],
            "included_behavior_session_ids": [session.behavior_session_id for session in kept_sessions],
            "included_ophys_session_ids": [session.ophys_session_id for session in kept_sessions],
            "running_bin_edges": running_edges.tolist(),
            "pupil_bin_edges": pupil_edges.tolist(),
        },
    }

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    with args.outpicklefile.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_trials = sum(len(session_trials) for session_trials in neural_all)
    total_neurons = sum(brain_idx.shape[0] for brain_idx in brain_region_idx_all)
    print(
        f"Saved {args.outpicklefile} with {len(neural_all)} sessions, "
        f"{total_trials} trials, {total_neurons} neurons in {time.time() - t_global_start:.2f}s"
    )


if __name__ == "__main__":
    main()
