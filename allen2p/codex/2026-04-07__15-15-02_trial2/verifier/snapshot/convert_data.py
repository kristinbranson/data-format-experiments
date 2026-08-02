#!/usr/bin/env python3
"""Convert Allen Visual Behavior Ophys NWB files into decoder format."""

from __future__ import annotations

import argparse
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_ROOT = Path("/app/data")
RELEASE_ROOT = DATA_ROOT / "visual-behavior-ophys-1.1.0"
EXPERIMENT_DIR = RELEASE_ROOT / "behavior_ophys_experiments"
METADATA_DIR = RELEASE_ROOT / "project_metadata"
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
RNG = np.random.default_rng(0)


@dataclass(frozen=True)
class SessionInfo:
    ophys_experiment_id: int
    path: Path
    mouse_id: str
    targeted_structure: str
    session_type: str
    project_code: str
    passive: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Visual Behavior ophys data to decoder format."
    )
    parser.add_argument("outpicklefile", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="Process all sessions")
    group.add_argument(
        "--sample", action="store_true", help="Process only the first 2 active sessions"
    )
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing plots for up to 2 sessions as processing_<session_id>.png",
    )
    return parser.parse_args()


def has_required_eye_tracking(path: Path) -> bool:
    with h5py.File(path, "r") as h5f:
        if "acquisition" not in h5f or "EyeTracking" not in h5f["acquisition"]:
            return False
        eye = h5f["acquisition"]["EyeTracking"]
        return "pupil_tracking" in eye and "eye_tracking" in eye


def read_metadata_sessions() -> List[SessionInfo]:
    exp_table = pd.read_csv(METADATA_DIR / "ophys_experiment_table.csv")
    file_map = {
        int(path.stem.split("_")[-1]): path
        for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
    }
    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()
    exp_table = exp_table.sort_values("ophys_experiment_id")

    sessions = [
        SessionInfo(
            ophys_experiment_id=int(row.ophys_experiment_id),
            path=file_map[int(row.ophys_experiment_id)],
            mouse_id=str(int(row.mouse_id)),
            targeted_structure=str(row.targeted_structure),
            session_type=str(row.session_type),
            project_code=str(row.project_code),
            passive=bool(row.passive),
        )
        for row in exp_table.itertuples(index=False)
    ]
    filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
    excluded = len(sessions) - len(filtered_sessions)
    if excluded:
        print(f"[setup] excluded {excluded} active sessions missing eye-tracking pupil data")
    sessions = filtered_sessions
    return sessions


def h5_array(obj: h5py.Dataset | h5py.Group) -> np.ndarray:
    if isinstance(obj, h5py.Dataset):
        return obj[()]
    if isinstance(obj, h5py.Group) and "data" in obj:
        return obj["data"][()]
    raise TypeError(f"Unsupported HDF5 object for array read: {type(obj)}")


def decode_strings(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S" or values.dtype.kind == "O":
        return np.array(
            [v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v) for v in values],
            dtype=object,
        )
    return values


def read_interval_table(group: h5py.Group, columns: Sequence[str]) -> Dict[str, np.ndarray]:
    out = {}
    for col in columns:
        if col not in group:
            raise KeyError(f"Missing interval column '{col}' in {group.name}")
        arr = h5_array(group[col])
        out[col] = decode_strings(np.asarray(arr))
    return out


def choose_task_presentation_group(h5f: h5py.File) -> h5py.Group:
    best_name = None
    best_score = (-1, -1)
    intervals = h5f["intervals"]
    for name in intervals.keys():
        if not name.endswith("_presentations") or name == "trials":
            continue
        grp = intervals[name]
        if "active" not in grp or "image_name" not in grp:
            continue
        active = np.asarray(h5_array(grp["active"])).astype(bool)
        n_active = int(active.sum())
        if n_active == 0:
            continue
        has_change_detection = 0
        if "stimulus_block_name" in grp:
            block_names = decode_strings(np.asarray(h5_array(grp["stimulus_block_name"])))
            has_change_detection = int(
                any("change_detection" in str(x) for x in block_names.tolist())
            )
        score = (has_change_detection, n_active)
        if score > best_score:
            best_score = score
            best_name = name
    if best_name is None:
        raise RuntimeError(f"Could not find active task stimulus table in {h5f.filename}")
    return intervals[best_name]


def session_native_dt(ophys_timestamps: np.ndarray) -> float:
    return float(np.mean(np.diff(ophys_timestamps)))


def session_grid(start: float, stop: float, dt: float = TIME_BIN_SIZE_S) -> np.ndarray:
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt


def interpolate_vector(
    source_t: np.ndarray, source_values: np.ndarray, query_t: np.ndarray
) -> np.ndarray:
    if source_t.ndim != 1 or source_values.ndim != 1:
        raise ValueError("interpolate_vector expects 1D inputs")
    return np.interp(query_t, source_t, source_values).astype(np.float32)


def interpolate_matrix(
    source_t: np.ndarray, source_values: np.ndarray, query_t: np.ndarray
) -> np.ndarray:
    """Linear interpolation for 2D source arrays of shape (T, N)."""
    if source_values.ndim != 2:
        raise ValueError(f"Expected 2D source_values, got {source_values.shape}")

    n_src = source_t.shape[0]
    if n_src == 0:
        raise ValueError("No source timestamps for interpolation")
    if n_src == 1:
        return np.repeat(source_values, len(query_t), axis=0).astype(np.float32)

    right = np.searchsorted(source_t, query_t, side="left")
    right = np.clip(right, 0, n_src - 1)
    left = np.clip(right - 1, 0, n_src - 1)

    same = right == left
    t0 = source_t[left]
    t1 = source_t[right]
    denom = np.where(np.abs(t1 - t0) < 1e-12, 1.0, t1 - t0)
    w = np.where(same, 0.0, (query_t - t0) / denom)

    left_vals = source_values[left]
    right_vals = source_values[right]
    out = left_vals * (1.0 - w[:, None]) + right_vals * w[:, None]
    return out.astype(np.float32)


def interpolate_pupil(
    pupil_t: np.ndarray, pupil_diameter: np.ndarray, query_t: np.ndarray
) -> np.ndarray:
    valid = np.isfinite(pupil_diameter)
    if valid.sum() == 0:
        raise ValueError("No valid pupil samples available")
    if valid.sum() == 1:
        return np.full(query_t.shape, float(pupil_diameter[valid][0]), dtype=np.float32)
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)


def robust_quintile_edges(values: np.ndarray) -> np.ndarray:
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    for i in range(1, len(percentiles)):
        if percentiles[i] <= percentiles[i - 1]:
            percentiles[i] = percentiles[i - 1] + 1e-6
    return percentiles


def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=False).astype(np.int64)


def trial_outcome_code(trials: Dict[str, np.ndarray], idx: int, mapping: Dict[str, int]) -> int:
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
    raise ValueError(f"Trial {idx} has no valid outcome label")


def read_session_raw(
    session: SessionInfo,
    load_events: bool = True,
) -> Dict[str, object]:
    with h5py.File(session.path, "r") as h5f:
        trial_group = h5f["intervals"]["trials"]
        trials = read_interval_table(
            trial_group,
            [
                "go",
                "catch",
                "aborted",
                "auto_rewarded",
                "hit",
                "miss",
                "false_alarm",
                "correct_reject",
                "change_time",
                "start_time",
                "stop_time",
            ],
        )
        for key in ("go", "catch", "aborted", "auto_rewarded", "hit", "miss", "false_alarm", "correct_reject"):
            trials[key] = trials[key].astype(bool)
        trials["change_time"] = trials["change_time"].astype(np.float64)
        trials["start_time"] = trials["start_time"].astype(np.float64)
        trials["stop_time"] = trials["stop_time"].astype(np.float64)

        keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])

        stim_group = choose_task_presentation_group(h5f)
        stim = read_interval_table(
            stim_group,
            [
                "start_time",
                "stop_time",
                "image_name",
                "is_change",
                "omitted",
                "trials_id",
                "active",
                "flashes_since_change",
            ],
        )
        stim["start_time"] = stim["start_time"].astype(np.float64)
        stim["stop_time"] = stim["stop_time"].astype(np.float64)
        stim["image_name"] = np.asarray(stim["image_name"], dtype=object)
        stim["is_change"] = stim["is_change"].astype(bool)
        stim["omitted"] = stim["omitted"].astype(bool)
        stim["trials_id"] = stim["trials_id"].astype(np.int64)
        stim["active"] = stim["active"].astype(bool)
        stim["flashes_since_change"] = stim["flashes_since_change"].astype(np.int64)

        ophys_timestamps = np.asarray(
            h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64
        )
        events: Optional[np.ndarray] = None
        n_neurons: Optional[int] = None
        if load_events:
            events = np.asarray(
                h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
            )

        running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
        running_timestamps = np.asarray(
            h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
        )

        pupil_width = np.asarray(
            h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32
        )
        pupil_height = np.asarray(
            h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32
        )
        pupil_timestamps = np.asarray(
            h5f["acquisition"]["EyeTracking"]["eye_tracking"]["timestamps"], dtype=np.float64
        )
        pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)

        if load_events:
            cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
            n_cell_table = len(cell_table["id"])
            if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
                valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
                if valid_roi.sum() != events.shape[1]:
                    events = events[:, valid_roi]
            n_neurons = events.shape[1]

    return {
        "trials": trials,
        "keep_mask": keep,
        "stimulus": stim,
        "events": events,
        "ophys_timestamps": ophys_timestamps,
        "running_speed": running_speed,
        "running_timestamps": running_timestamps,
        "pupil_diameter": pupil_diameter,
        "pupil_timestamps": pupil_timestamps,
        "n_neurons": n_neurons,
    }


def collect_global_statistics(
    sessions: Sequence[SessionInfo],
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[int, int], Dict[int, float]]:
    running_values: List[np.ndarray] = []
    pupil_values: List[np.ndarray] = []
    image_names: set[str] = set()
    valid_trial_counts: Dict[int, int] = {}
    native_dt_by_session: Dict[int, float] = {}

    t_global_start = time.perf_counter()
    for idx, session in enumerate(sessions, start=1):
        t0 = time.perf_counter()
        raw = read_session_raw(session, load_events=False)
        native_dt_by_session[session.ophys_experiment_id] = session_native_dt(raw["ophys_timestamps"])
        valid_trial_counts[session.ophys_experiment_id] = int(raw["keep_mask"].sum())

        stim = raw["stimulus"]
        image_names.update(
            str(x)
            for x, omitted in zip(stim["image_name"], stim["omitted"])
            if (not omitted) and str(x) not in ("", "None", "nan")
        )

        for trial_idx in np.flatnonzero(raw["keep_mask"]):
            start = float(raw["trials"]["start_time"][trial_idx])
            stop = float(raw["trials"]["stop_time"][trial_idx])
            grid = session_grid(start, stop)
            running_values.append(
                interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
            )
            pupil_values.append(
                interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
            )

        elapsed = time.perf_counter() - t0
        print(
            f"[pass1] {idx:03d}/{len(sessions)} session {session.ophys_experiment_id}: "
            f"{valid_trial_counts[session.ophys_experiment_id]} valid trials in {elapsed:.2f}s"
        )

    running_all = np.concatenate(running_values).astype(np.float32)
    pupil_all = np.concatenate(pupil_values).astype(np.float32)
    running_edges = robust_quintile_edges(running_all)
    pupil_edges = robust_quintile_edges(pupil_all)
    image_values = ["gray"] + sorted(image_names)

    print(
        f"[pass1] collected global stats from {len(sessions)} sessions in "
        f"{time.perf_counter() - t_global_start:.2f}s"
    )
    print(f"[pass1] running edges: {running_edges}")
    print(f"[pass1] pupil edges: {pupil_edges}")
    print(f"[pass1] image categories: {len(image_values)}")
    return running_edges, pupil_edges, image_values, valid_trial_counts, native_dt_by_session


def stimulus_identity_codes(
    stimulus: Dict[str, np.ndarray], query_t: np.ndarray, image_to_code: Dict[str, int]
) -> np.ndarray:
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    image_names = stimulus["image_name"]
    omitted = stimulus["omitted"]

    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
    valid = idx >= 0
    valid &= idx < len(starts)
    idx_valid = idx[valid]
    in_interval = query_t[valid] < stops[idx_valid]
    if np.any(in_interval):
        sub_idx = idx_valid[in_interval]
        names = np.asarray(image_names[sub_idx], dtype=object)
        omit = omitted[sub_idx]
        target = np.full(sub_idx.shape, image_to_code["gray"], dtype=np.int64)
        for i, (name, is_omitted) in enumerate(zip(names, omit)):
            if (not is_omitted) and str(name) in image_to_code:
                target[i] = image_to_code[str(name)]
        assign = np.flatnonzero(valid)[in_interval]
        codes[assign] = target
    return codes


def stimulus_change_codes(stimulus: Dict[str, np.ndarray], query_t: np.ndarray) -> np.ndarray:
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    is_change = stimulus["is_change"]
    omitted = stimulus["omitted"]

    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.zeros(query_t.shape, dtype=np.int64)
    valid = idx >= 0
    valid &= idx < len(starts)
    idx_valid = idx[valid]
    in_interval = query_t[valid] < stops[idx_valid]
    if np.any(in_interval):
        sub_idx = idx_valid[in_interval]
        changed = is_change[sub_idx] & (~omitted[sub_idx])
        assign = np.flatnonzero(valid)[in_interval]
        codes[assign] = changed.astype(np.int64)
    return codes


def make_processing_plot(
    session: SessionInfo,
    raw: Dict[str, object],
    example_trial: Dict[str, np.ndarray],
    running_edges: np.ndarray,
    pupil_edges: np.ndarray,
    image_values: Sequence[str],
    outcome_values: Sequence[str],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)

    keep_mask = raw["keep_mask"]
    trials = raw["trials"]
    counts = {
        "go": int(trials["go"].sum()),
        "catch": int(trials["catch"].sum()),
        "aborted": int(trials["aborted"].sum()),
        "auto_rewarded": int(trials["auto_rewarded"].sum()),
        "kept": int(keep_mask.sum()),
    }
    axes[0, 0].bar(list(counts.keys()), list(counts.values()), color=["C0", "C1", "C3", "C4", "C2"])
    axes[0, 0].set_title("Trial Filtering")
    axes[0, 0].set_ylabel("Count")

    t = example_trial["time_relative"]
    axes[0, 1].step(t, example_trial["image_identity"], where="post")
    axes[0, 1].step(t, example_trial["image_change"] * (len(image_values) - 1), where="post", alpha=0.7)
    axes[0, 1].set_title("Stimulus Outputs")
    axes[0, 1].set_xlabel("Time from trial start (s)")
    axes[0, 1].set_ylabel("Image code / change")

    axes[1, 0].plot(t, example_trial["running_cont"], color="black", lw=1)
    axes[1, 0].step(t, example_trial["running_bin"], where="post", color="C2", lw=1)
    for edge in running_edges:
        axes[1, 0].axhline(edge, color="gray", ls=":", lw=0.7)
    axes[1, 0].set_title("Running Speed -> Quintile Bin")
    axes[1, 0].set_xlabel("Time from trial start (s)")

    axes[1, 1].plot(t, example_trial["pupil_cont"], color="black", lw=1)
    axes[1, 1].step(t, example_trial["pupil_bin"], where="post", color="C1", lw=1)
    for edge in pupil_edges:
        axes[1, 1].axhline(edge, color="gray", ls=":", lw=0.7)
    axes[1, 1].set_title("Pupil Diameter -> Quintile Bin")
    axes[1, 1].set_xlabel("Time from trial start (s)")

    neural = example_trial["neural"]
    nneurons_show = min(30, neural.shape[0])
    axes[2, 0].imshow(
        neural[:nneurons_show],
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="magma",
    )
    axes[2, 0].set_title(f"Neural Events ({nneurons_show} neurons)")
    axes[2, 0].set_xlabel("30 Hz time bins")
    axes[2, 0].set_ylabel("Neuron")

    axes[2, 1].text(
        0.01,
        0.98,
        "\n".join(
            [
                f"session_type: {session.session_type}",
                f"project_code: {session.project_code}",
                f"brain_region: {session.targeted_structure}",
                f"native_dt: {session_native_dt(raw['ophys_timestamps']):.4f}s",
                f"trial_outcome: {outcome_values[int(example_trial['trial_outcome'][0])]}",
                f"n_valid_trials: {int(keep_mask.sum())}",
                f"n_neurons: {raw['n_neurons']}",
            ]
        ),
        va="top",
        ha="left",
        family="monospace",
    )
    axes[2, 1].axis("off")
    axes[2, 1].set_title("Session Summary")

    outpath = Path(f"processing_{session.ophys_experiment_id}.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {outpath}")


def convert_sessions(
    sessions: Sequence[SessionInfo],
    running_edges: np.ndarray,
    pupil_edges: np.ndarray,
    image_values: Sequence[str],
    show_processing: bool,
) -> Dict[str, object]:
    image_to_code = {name: idx for idx, name in enumerate(image_values)}
    outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
    outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": [],
        "subject_idx": [],
        "brain_regions": [],
        "brain_region_idx": [],
        "input_names": [],
        "output_names": [
            "image_identity",
            "image_change",
            "running_speed_bin",
            "pupil_diameter_bin",
            "trial_outcome",
        ],
        "output_values": [
            list(image_values),
            ["no_change", "change"],
            [f"bin_{i}" for i in range(5)],
            [f"bin_{i}" for i in range(5)],
            outcome_values,
        ],
        "metadata": {
            "task_description": (
                "Allen Visual Behavior change-detection task. Neural input is calcium event activity; "
                "outputs are image identity, image change, running-speed bin, pupil-diameter bin, "
                "and trial outcome for active go/catch trials only."
            ),
            "time_bin_size": float(TIME_BIN_SIZE_MS),
            "temporal_alignment_event": "trial start",
            "off_start": 0.0,
            "off_end": None,
            "neural_signal": "precomputed calcium events",
            "included_session_types": sorted({s.session_type for s in sessions}),
            "excluded_session_types": ["passive viewing sessions"],
            "excluded_sessions_missing_eye_tracking": True,
            "session_ophys_experiment_ids": [int(s.ophys_experiment_id) for s in sessions],
            "running_speed_bin_edges": running_edges.astype(float).tolist(),
            "pupil_diameter_bin_edges": pupil_edges.astype(float).tolist(),
            "resampling_reference": "common 30 Hz grid derived from source ophys timestamps",
            "source_release": "visual-behavior-ophys-1.1.0 local subset",
        },
    }

    subject_to_idx: Dict[str, int] = {}
    brain_region_to_idx: Dict[str, int] = {}

    for sess_num, session in enumerate(sessions, start=1):
        t0 = time.perf_counter()
        raw = read_session_raw(session)

        if int(raw["keep_mask"].sum()) < 2:
            print(
                f"[pass2] skipping session {session.ophys_experiment_id} because it has "
                f"{int(raw['keep_mask'].sum())} valid trials"
            )
            continue

        subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
        if subject_idx == len(data["subjects"]):
            data["subjects"].append(session.mouse_id)

        region_idx = brain_region_to_idx.setdefault(
            session.targeted_structure, len(brain_region_to_idx)
        )
        if region_idx == len(data["brain_regions"]):
            data["brain_regions"].append(session.targeted_structure)

        session_neural: List[np.ndarray] = []
        session_input: List[np.ndarray] = []
        session_output: List[np.ndarray] = []

        plotted = False
        for trial_idx in np.flatnonzero(raw["keep_mask"]):
            start = float(raw["trials"]["start_time"][trial_idx])
            stop = float(raw["trials"]["stop_time"][trial_idx])
            grid = session_grid(start, stop)

            neural_trial = interpolate_matrix(
                raw["ophys_timestamps"], raw["events"], grid
            ).T.astype(np.float32)
            running_cont = interpolate_vector(
                raw["running_timestamps"], raw["running_speed"], grid
            )
            pupil_cont = interpolate_pupil(
                raw["pupil_timestamps"], raw["pupil_diameter"], grid
            )
            image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
            image_change = stimulus_change_codes(raw["stimulus"], grid)

            running_bins = digitize_with_edges(running_cont, running_edges)
            pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
            outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
            trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)

            output_trial = np.vstack(
                [
                    image_codes.astype(np.int64),
                    image_change,
                    running_bins,
                    pupil_bins,
                    trial_outcome,
                ]
            )
            input_trial = np.zeros((0, len(grid)), dtype=np.float32)

            session_neural.append(neural_trial)
            session_input.append(input_trial)
            session_output.append(output_trial)

            if show_processing and (sess_num <= 2) and not plotted:
                make_processing_plot(
                    session=session,
                    raw=raw,
                    example_trial={
                        "time_relative": grid - start,
                        "image_identity": image_codes,
                        "image_change": image_change,
                        "running_cont": running_cont,
                        "running_bin": running_bins,
                        "pupil_cont": pupil_cont,
                        "pupil_bin": pupil_bins,
                        "trial_outcome": trial_outcome,
                        "neural": neural_trial,
                    },
                    running_edges=running_edges,
                    pupil_edges=pupil_edges,
                    image_values=image_values,
                    outcome_values=outcome_values,
                )
                plotted = True

        if len(session_neural) < 2:
            print(
                f"[pass2] skipping session {session.ophys_experiment_id} after conversion "
                f"because it has {len(session_neural)} trials"
            )
            continue

        data["neural"].append(session_neural)
        data["input"].append(session_input)
        data["output"].append(session_output)
        data["subject_idx"].append(subject_idx)
        data["brain_region_idx"].append(
            np.full(raw["n_neurons"], region_idx, dtype=np.int64)
        )

        elapsed = time.perf_counter() - t0
        total_bins = sum(trial.shape[1] for trial in session_neural)
        print(
            f"[pass2] {sess_num:03d}/{len(sessions)} session {session.ophys_experiment_id}: "
            f"{len(session_neural)} trials, {raw['n_neurons']} neurons, {total_bins} bins in {elapsed:.2f}s"
        )

    data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int64)
    return data


def summarize_data(data: Dict[str, object]) -> None:
    n_sessions = len(data["neural"])
    n_trials = sum(len(s) for s in data["neural"])
    n_neurons = int(sum(br.shape[0] for br in data["brain_region_idx"]))
    mean_trials = float(np.mean([len(s) for s in data["neural"]])) if n_sessions else 0.0
    mean_neurons = (
        float(np.mean([br.shape[0] for br in data["brain_region_idx"]])) if n_sessions else 0.0
    )
    print(f"[summary] sessions={n_sessions}")
    print(f"[summary] trials={n_trials}, mean/session={mean_trials:.2f}")
    print(f"[summary] neurons={n_neurons}, mean/session={mean_neurons:.2f}")
    print(f"[summary] subjects={len(data['subjects'])}, brain_regions={data['brain_regions']}")


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    sessions = read_metadata_sessions()
    if args.sample:
        sessions = sessions[:2]
    print(f"[setup] processing {len(sessions)} active sessions")

    running_edges, pupil_edges, image_values, valid_counts, native_dt = collect_global_statistics(
        sessions
    )
    print(
        "[setup] native ophys dt range:",
        min(native_dt.values()),
        max(native_dt.values()),
    )
    print(
        "[setup] valid trial count range:",
        min(valid_counts.values()),
        max(valid_counts.values()),
    )

    data = convert_sessions(
        sessions=sessions,
        running_edges=running_edges,
        pupil_edges=pupil_edges,
        image_values=image_values,
        show_processing=args.show_processing,
    )
    summarize_data(data)

    with args.outpicklefile.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.perf_counter() - start_time
    print(f"[done] wrote {args.outpicklefile} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
