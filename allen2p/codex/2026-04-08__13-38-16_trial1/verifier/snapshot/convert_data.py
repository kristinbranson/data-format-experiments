#!/usr/bin/env python3

import argparse
import math
import pickle
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATA_ROOT = Path("/app/data/visual-behavior-ophys-1.1.0")
EXPERIMENT_DIR = DATA_ROOT / "behavior_ophys_experiments"
MANIFEST_PATH = Path("/app/data/visual-behavior-ophys_project_manifest_v1.1.0.json")

BIN_SIZE_SEC = 0.1
SHOW_PROCESSING_MAX = 2

TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
RUN_BIN_VALUES = [f"q{i}" for i in range(1, 6)]
PUPIL_BIN_VALUES = [f"q{i}" for i in range(1, 6)]


@dataclass
class TrialSpec:
    trial_idx: int
    start_time: float
    stop_time: float
    change_time: float
    outcome_idx: int
    is_go: bool
    is_catch: bool


@dataclass
class SessionPreview:
    path: Path
    experiment_id: int
    ophys_session_id: int
    subject_id: str
    brain_region: str
    session_type: str
    has_eye_tracking: bool
    trial_specs: List[TrialSpec]
    running_values: np.ndarray
    pupil_values: np.ndarray
    unique_images: List[str]
    raw_trial_count: int
    valid_trial_count: int
    excluded_reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Allen Visual Behavior ophys NWB files into decoder format."
    )
    parser.add_argument("outpicklefile", type=str, help="Output pickle file path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all eligible sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 eligible sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Plot processing diagnostics for up to 2 sessions.",
    )
    return parser.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))


def sort_files_for_mode(files: List[Path], sample_mode: bool) -> List[Path]:
    if not sample_mode:
        return files
    cells_path = DATA_ROOT / "project_metadata" / "ophys_cells_table.csv"
    cells = pd.read_csv(cells_path)
    cell_counts = cells.groupby("ophys_experiment_id").size().to_dict()

    def sort_key(path: Path) -> Tuple[int, int]:
        experiment_id = int(path.stem.split("_")[-1])
        return (-int(cell_counts.get(experiment_id, 0)), experiment_id)

    return sorted(files, key=sort_key)


def decode_scalar(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def decode_string_array(dataset) -> np.ndarray:
    values = dataset[()]
    if isinstance(values, (bytes, str)):
        return np.array([decode_scalar(values)], dtype=object)
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O", "U"}:
        return np.array([decode_scalar(v) for v in values], dtype=object)
    return values.astype(object)


def read_interval_group(group: h5py.Group, names: Iterable[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in names:
        if name in group:
            data = group[name]
            if data.dtype.kind in {"S", "O", "U"}:
                out[name] = decode_string_array(data)
            else:
                out[name] = np.asarray(data)
    return out


def read_trials(f: h5py.File) -> Dict[str, np.ndarray]:
    group = f["/intervals/trials"]
    names = [
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
        "change_frame",
        "initial_image_name",
        "change_image_name",
    ]
    return read_interval_group(group, names)


def read_task_presentations(f: h5py.File) -> Dict[str, np.ndarray]:
    rows: List[Dict[str, np.ndarray]] = []
    interval_root = f["/intervals"]
    keep_names = [
        "start_time",
        "stop_time",
        "image_name",
        "omitted",
        "is_change",
        "is_sham_change",
        "trials_id",
        "active",
    ]
    for key in interval_root.keys():
        if key == "trials":
            continue
        group = interval_root[key]
        if "image_name" not in group or "start_time" not in group or "stop_time" not in group:
            continue
        rows.append(read_interval_group(group, keep_names))

    if not rows:
        return {
            "start_time": np.array([], dtype=np.float64),
            "stop_time": np.array([], dtype=np.float64),
            "image_name": np.array([], dtype=object),
            "omitted": np.array([], dtype=np.float64),
            "is_change": np.array([], dtype=np.float64),
            "is_sham_change": np.array([], dtype=np.float64),
            "trials_id": np.array([], dtype=np.float64),
            "active": np.array([], dtype=bool),
        }

    merged: Dict[str, List[np.ndarray]] = {name: [] for name in keep_names}
    for row in rows:
        for name in keep_names:
            if name in row:
                merged[name].append(row[name])

    out = {
        name: np.concatenate(parts) if parts else np.array([], dtype=np.float64)
        for name, parts in merged.items()
    }
    order = np.argsort(out["start_time"])
    for name in list(out.keys()):
        out[name] = out[name][order]
    return out


def build_trial_specs(trials: Dict[str, np.ndarray]) -> List[TrialSpec]:
    raw_count = len(trials["id"])
    aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)
    auto_rewarded = np.nan_to_num(trials["auto_rewarded"], nan=0.0).astype(bool)
    go = np.nan_to_num(trials["go"], nan=0.0).astype(bool)
    catch = np.nan_to_num(trials["catch"], nan=0.0).astype(bool)
    hit = np.nan_to_num(trials["hit"], nan=0.0).astype(bool)
    miss = np.nan_to_num(trials["miss"], nan=0.0).astype(bool)
    false_alarm = np.nan_to_num(trials["false_alarm"], nan=0.0).astype(bool)
    correct_reject = np.nan_to_num(trials["correct_reject"], nan=0.0).astype(bool)
    change_time = np.asarray(trials["change_time"], dtype=np.float64)

    specs: List[TrialSpec] = []
    for idx in range(raw_count):
        if aborted[idx] or auto_rewarded[idx]:
            continue
        outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
        if sum(int(x) for x in outcome_flags) != 1:
            raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
        outcome_idx = outcome_flags.index(True)
        specs.append(
            TrialSpec(
                trial_idx=idx,
                start_time=float(trials["start_time"][idx]),
                stop_time=float(trials["stop_time"][idx]),
                change_time=float(change_time[idx]) if np.isfinite(change_time[idx]) else math.nan,
                outcome_idx=outcome_idx,
                is_go=bool(go[idx]),
                is_catch=bool(catch[idx]),
            )
        )
    return specs


def build_bin_centers(start_time: float, stop_time: float, bin_size_sec: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
    if starts.size == 0:
        starts = np.array([start_time], dtype=np.float64)
    ends = np.minimum(starts + bin_size_sec, stop_time)
    widths = np.maximum(ends - starts, 1e-6)
    centers = starts + 0.5 * widths
    return starts, ends, centers


def interpolate_series(times: np.ndarray, values: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    finite = np.isfinite(times) & np.isfinite(values)
    if query_times.size == 0:
        return np.array([], dtype=np.float32)
    if finite.sum() == 0:
        return np.full(query_times.shape, np.nan, dtype=np.float32)
    if finite.sum() == 1:
        v = float(values[finite][0])
        return np.full(query_times.shape, v, dtype=np.float32)
    t = np.asarray(times[finite], dtype=np.float64)
    v = np.asarray(values[finite], dtype=np.float64)
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    out = np.interp(query_times, t, v, left=v[0], right=v[-1])
    return out.astype(np.float32)


def pupil_area_to_diameter(area: np.ndarray) -> np.ndarray:
    area = np.asarray(area, dtype=np.float64)
    out = np.full(area.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(area) & (area >= 0)
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)


def collect_time_window_values(
    times: np.ndarray, values: np.ndarray, trial_specs: List[TrialSpec]
) -> np.ndarray:
    if times.size == 0 or values.size == 0 or not trial_specs:
        return np.array([], dtype=np.float32)
    segments: List[np.ndarray] = []
    for spec in trial_specs:
        lo = np.searchsorted(times, spec.start_time, side="left")
        hi = np.searchsorted(times, spec.stop_time, side="left")
        if hi > lo:
            segments.append(values[lo:hi])
    if not segments:
        return np.array([], dtype=np.float32)
    return np.concatenate(segments).astype(np.float32, copy=False)


def unique_nonempty_images(presentations: Dict[str, np.ndarray]) -> List[str]:
    if presentations["image_name"].size == 0:
        return []
    # Omitted flashes are represented as gray in the converted output and should
    # not become a separate image-identity class.
    names = [
        str(x)
        for x in presentations["image_name"]
        if str(x) not in {"", "nan", "None", "omitted"}
    ]
    return sorted(set(names))


def find_presentation_rows(
    presentations: Dict[str, np.ndarray], start_time: float, stop_time: float
) -> np.ndarray:
    if presentations["start_time"].size == 0:
        return np.array([], dtype=np.int64)
    starts = presentations["start_time"]
    ends = presentations["stop_time"]
    mask = (starts < stop_time) & (ends > start_time)
    return np.flatnonzero(mask)


def make_image_series(
    presentations: Dict[str, np.ndarray],
    row_idx: np.ndarray,
    centers: np.ndarray,
    image_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
    change_series = np.zeros(centers.shape, dtype=np.int16)
    for idx in row_idx:
        start = float(presentations["start_time"][idx])
        stop = float(presentations["stop_time"][idx])
        if stop <= start:
            continue
        in_window = (centers >= start) & (centers < stop)
        if not np.any(in_window):
            continue
        omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
        if not omitted:
            image_name = str(presentations["image_name"][idx])
            image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
        is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
        if is_change and not omitted:
            change_series[in_window] = 1
    return image_series, change_series


def compute_bin_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.asarray(edges, dtype=np.float64)


def discretize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.digitize(values, edges, right=False)
    bins = np.clip(bins, 0, len(edges))
    return bins.astype(np.int16)


def session_preview(path: Path, bin_size_sec: float) -> SessionPreview:
    with h5py.File(path, "r") as f:
        experiment_id = int(decode_scalar(f["/identifier"][()]))
        ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
        subject_id = decode_scalar(f["/general/subject/subject_id"][()])
        session_type = decode_scalar(f["/general/metadata"].attrs["session_type"])
        brain_region = decode_scalar(f["/general/optophysiology/imaging_plane_1/location"][()])
        has_eye_tracking = "/acquisition/EyeTracking/pupil_tracking/area_raw" in f

        trials = read_trials(f)
        specs = build_trial_specs(trials)

        presentations = read_task_presentations(f)
        unique_images = unique_nonempty_images(presentations)

        running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
        running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)

        pupil_values_all = np.array([], dtype=np.float32)
        if has_eye_tracking:
            pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
            pupil_area = np.asarray(
                f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64
            )
            blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
            pupil_area = pupil_area.copy()
            pupil_area[blink] = np.nan
            pupil_diameter = pupil_area_to_diameter(pupil_area)
        else:
            pupil_times = np.array([], dtype=np.float64)
            pupil_diameter = np.array([], dtype=np.float32)

        running_concat = collect_time_window_values(running_times, running_values, specs)
        if has_eye_tracking:
            pupil_values_all = collect_time_window_values(pupil_times, pupil_diameter, specs)

        excluded_reason = None
        if not has_eye_tracking:
            excluded_reason = "missing_eye_tracking"
        elif len(specs) < 2:
            excluded_reason = "fewer_than_2_valid_trials"
        elif np.isfinite(pupil_values_all).sum() < 2:
            excluded_reason = "insufficient_valid_pupil_samples"

        return SessionPreview(
            path=path,
            experiment_id=experiment_id,
            ophys_session_id=ophys_session_id,
            subject_id=subject_id,
            brain_region=brain_region,
            session_type=session_type,
            has_eye_tracking=has_eye_tracking,
            trial_specs=specs,
            running_values=running_concat,
            pupil_values=pupil_values_all,
            unique_images=unique_images,
            raw_trial_count=len(trials["id"]),
            valid_trial_count=len(specs),
            excluded_reason=excluded_reason,
        )


def load_neural_events(f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
    event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
    valid_roi = np.asarray(
        f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
    )
    valid_mask = valid_roi[event_rois]
    event_data = event_data[:, valid_mask]
    return event_data, event_rois[valid_mask]


def plot_processing_summary(
    preview: SessionPreview,
    trials: Dict[str, np.ndarray],
    presentations: Dict[str, np.ndarray],
    event_data: np.ndarray,
    ophys_timestamps: np.ndarray,
    running_times: np.ndarray,
    running_values: np.ndarray,
    pupil_times: np.ndarray,
    pupil_diameter: np.ndarray,
    run_edges: np.ndarray,
    pupil_edges: np.ndarray,
    image_to_idx: Dict[str, int],
    bin_size_sec: float,
) -> None:
    if not preview.trial_specs:
        return

    preferred = next((t for t in preview.trial_specs if t.is_go), preview.trial_specs[0])
    trial_rows = find_presentation_rows(presentations, preferred.start_time, preferred.stop_time)
    starts, ends, centers = build_bin_centers(preferred.start_time, preferred.stop_time, bin_size_sec)
    image_series, change_series = make_image_series(presentations, trial_rows, centers, image_to_idx)
    running_interp = interpolate_series(running_times, running_values, centers)
    pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
    running_bins = discretize_with_edges(running_interp, run_edges)
    pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)

    frame_mask = (ophys_timestamps >= preferred.start_time) & (ophys_timestamps < preferred.stop_time)
    trial_event_data = event_data[frame_mask]
    trial_event_times = ophys_timestamps[frame_mask]
    nneurons_plot = min(8, trial_event_data.shape[1])

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), constrained_layout=True)
    fig.suptitle(
        f"Processing summary for experiment {preview.experiment_id} ({preview.session_type})",
        fontsize=14,
    )

    counts = {
        "raw_trials": preview.raw_trial_count,
        "valid_trials": preview.valid_trial_count,
        "go": int(np.nan_to_num(trials["go"], nan=0.0).sum()),
        "catch": int(np.nan_to_num(trials["catch"], nan=0.0).sum()),
        "aborted": int(np.nan_to_num(trials["aborted"], nan=0.0).sum()),
        "auto": int(np.nan_to_num(trials["auto_rewarded"], nan=0.0).sum()),
    }
    axes[0].bar(range(len(counts)), list(counts.values()), color="#4C78A8")
    axes[0].set_xticks(range(len(counts)), list(counts.keys()))
    axes[0].set_ylabel("Count")
    axes[0].set_title("Trial filtering counts")

    for idx in trial_rows:
        start = float(presentations["start_time"][idx])
        stop = float(presentations["stop_time"][idx])
        omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
        color = "#BDBDBD" if omitted else "#72B7B2"
        axes[1].axvspan(start, stop, color=color, alpha=0.4)
        if bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0)):
            axes[1].axvline(start, color="#E45756", linewidth=2)
    axes[1].step(centers, image_series, where="mid", color="#4C78A8", label="image_identity")
    axes[1].step(centers, change_series * image_series.max(), where="mid", color="#E45756", label="image_change")
    axes[1].set_xlim(preferred.start_time, preferred.stop_time)
    axes[1].set_title("Stimulus presentations projected onto rebinned trial")
    axes[1].legend(loc="upper right")

    axes[2].plot(running_times, running_values, color="#59A14F", alpha=0.5, label="raw running")
    axes[2].plot(centers, running_interp, color="#1B9E77", linewidth=2, label="rebinned running")
    axes[2].step(centers, running_bins, where="mid", color="#2F4B7C", linestyle="--", label="running bin")
    axes[2].set_xlim(preferred.start_time, preferred.stop_time)
    axes[2].set_title("Running speed interpolation and discretization")
    axes[2].legend(loc="upper right")

    axes[3].plot(pupil_times, pupil_diameter, color="#F28E2B", alpha=0.5, label="raw pupil diameter")
    axes[3].plot(centers, pupil_interp, color="#D37295", linewidth=2, label="rebinned pupil")
    axes[3].step(centers, pupil_bins, where="mid", color="#7A4D8F", linestyle="--", label="pupil bin")
    axes[3].set_xlim(preferred.start_time, preferred.stop_time)
    axes[3].set_title("Pupil interpolation and discretization")
    axes[3].legend(loc="upper right")

    if trial_event_times.size and nneurons_plot > 0:
        offset = np.arange(nneurons_plot, dtype=np.float32) * 2.0
        sample = trial_event_data[:, :nneurons_plot]
        for i in range(nneurons_plot):
            axes[4].plot(trial_event_times, sample[:, i] + offset[i], linewidth=0.8)
        axes[4].set_title("Raw event traces for sample neurons in representative trial")
    axes[4].set_xlim(preferred.start_time, preferred.stop_time)
    axes[4].set_xlabel("Time (s)")

    out_path = Path(f"processing_{preview.experiment_id}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def convert_session(
    preview: SessionPreview,
    image_to_idx: Dict[str, int],
    running_edges: np.ndarray,
    pupil_edges: np.ndarray,
    brain_region_to_idx: Dict[str, int],
    show_processing: bool,
    bin_size_sec: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, Dict[str, int]]:
    session_start = time.perf_counter()
    with h5py.File(preview.path, "r") as f:
        ophys_timestamps = np.asarray(
            f["/processing/ophys/dff/traces/timestamps"], dtype=np.float64
        )
        event_data, _ = load_neural_events(f)
        running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
        running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
        pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
        pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
        likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
        pupil_area = pupil_area.copy()
        pupil_area[likely_blink] = np.nan
        pupil_diameter = pupil_area_to_diameter(pupil_area)

        trials = read_trials(f)
        presentations = read_task_presentations(f)

        neural_trials: List[np.ndarray] = []
        input_trials: List[np.ndarray] = []
        output_trials: List[np.ndarray] = []

        for spec in preview.trial_specs:
            starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
            frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
            frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
            T = centers.size
            n_neurons = event_data.shape[1]

            neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
            for b in range(T):
                lo = int(frame_starts[b])
                hi = int(frame_ends[b])
                if hi > lo:
                    neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)

            row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
            image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)

            running_interp = interpolate_series(running_times, running_values, centers)
            pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
            if np.any(~np.isfinite(running_interp)):
                raise ValueError(f"Non-finite running interpolation in experiment {preview.experiment_id}")
            if np.any(~np.isfinite(pupil_interp)):
                raise ValueError(f"Non-finite pupil interpolation in experiment {preview.experiment_id}")

            running_bins = discretize_with_edges(running_interp, running_edges)
            pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
            outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)

            output_trial = np.vstack(
                [
                    image_series.astype(np.int16),
                    change_series.astype(np.int16),
                    running_bins,
                    pupil_bins,
                    outcome_series,
                ]
            )
            input_trial = np.empty((0, T), dtype=np.float32)

            neural_trials.append(neural_trial)
            input_trials.append(input_trial)
            output_trials.append(output_trial)

        if show_processing:
            plot_processing_summary(
                preview=preview,
                trials=trials,
                presentations=presentations,
                event_data=event_data,
                ophys_timestamps=ophys_timestamps,
                running_times=running_times,
                running_values=running_values,
                pupil_times=pupil_times,
                pupil_diameter=pupil_diameter,
                run_edges=running_edges,
                pupil_edges=pupil_edges,
                image_to_idx=image_to_idx,
                bin_size_sec=bin_size_sec,
            )

    brain_region_idx = np.full(
        event_data.shape[1], brain_region_to_idx[preview.brain_region], dtype=np.int64
    )
    stats = {
        "experiment_id": preview.experiment_id,
        "trial_count": len(neural_trials),
        "neuron_count": int(event_data.shape[1]),
        "elapsed_sec": int(round(time.perf_counter() - session_start)),
    }
    return neural_trials, input_trials, output_trials, brain_region_idx, stats


def summarize_previews(previews: List[SessionPreview], excluded: List[SessionPreview]) -> None:
    log(
        "Preview summary: "
        f"{len(previews)} eligible sessions, {len(excluded)} excluded sessions"
    )
    if excluded:
        reason_counts = Counter(x.excluded_reason for x in excluded)
        log(f"Excluded reasons: {dict(reason_counts)}")
    total_trials = sum(p.valid_trial_count for p in previews)
    log(f"Eligible trial count after filtering: {total_trials}")


def collect_previews(
    files: List[Path],
    bin_size_sec: float,
    sample_mode: bool,
    required_eligible: int,
) -> Tuple[List[SessionPreview], List[SessionPreview]]:
    eligible: List[SessionPreview] = []
    excluded: List[SessionPreview] = []
    for idx, path in enumerate(files, start=1):
        preview = session_preview(path, bin_size_sec)
        if preview.excluded_reason is None:
            eligible.append(preview)
        else:
            excluded.append(preview)

        if idx % 25 == 0 or idx == len(files):
            log(
                f"Preview progress: {idx}/{len(files)} files, "
                f"{len(eligible)} eligible, {len(excluded)} excluded"
            )

        if sample_mode and len(eligible) >= required_eligible:
            log(f"Sample preview satisfied after {idx} files")
            break

    return eligible, excluded


def main() -> None:
    args = parse_args()
    mode = "sample" if args.sample else "full"
    out_path = Path(args.outpicklefile)

    overall_start = time.perf_counter()
    files = sort_files_for_mode(list_nwb_files(), sample_mode=args.sample)
    log(f"Found {len(files)} NWB experiment files under {EXPERIMENT_DIR}")

    preview_start = time.perf_counter()
    eligible, excluded = collect_previews(
        files=files,
        bin_size_sec=BIN_SIZE_SEC,
        sample_mode=args.sample,
        required_eligible=2,
    )
    summarize_previews(eligible, excluded)
    log(f"Preview pass completed in {time.perf_counter() - preview_start:.1f}s")

    if mode == "sample":
        eligible = eligible[:2]
        log(f"Sample mode selected: processing {len(eligible)} sessions")
    else:
        log(f"Full mode selected: processing {len(eligible)} sessions")

    if len(eligible) < 2:
        raise RuntimeError("Need at least 2 eligible sessions after filtering")

    image_names = sorted({img for p in eligible for img in p.unique_images})
    image_values = ["gray"] + image_names
    image_to_idx = {name: idx for idx, name in enumerate(image_values)}

    running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
    pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
    running_pool = running_pool[np.isfinite(running_pool)]
    pupil_pool = pupil_pool[np.isfinite(pupil_pool)]
    if running_pool.size < 5 or pupil_pool.size < 5:
        raise RuntimeError("Not enough running or pupil samples to compute 5-bin discretization")

    running_edges = compute_bin_edges(running_pool, 5)
    pupil_edges = compute_bin_edges(pupil_pool, 5)
    log(f"Computed global running bin edges: {running_edges.tolist()}")
    log(f"Computed global pupil bin edges: {pupil_edges.tolist()}")

    subjects: List[str] = []
    subject_to_idx: Dict[str, int] = {}
    brain_regions: List[str] = []
    brain_region_to_idx: Dict[str, int] = {}

    neural_sessions: List[List[np.ndarray]] = []
    input_sessions: List[List[np.ndarray]] = []
    output_sessions: List[List[np.ndarray]] = []
    subject_idx: List[int] = []
    brain_region_idx: List[np.ndarray] = []
    session_stats: List[Dict[str, int]] = []

    show_processing_ids = {p.experiment_id for p in eligible[:SHOW_PROCESSING_MAX]} if args.show_processing else set()
    convert_start = time.perf_counter()
    for i, preview in enumerate(eligible, start=1):
        if preview.subject_id not in subject_to_idx:
            subject_to_idx[preview.subject_id] = len(subjects)
            subjects.append(preview.subject_id)
        if preview.brain_region not in brain_region_to_idx:
            brain_region_to_idx[preview.brain_region] = len(brain_regions)
            brain_regions.append(preview.brain_region)

        neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(
            preview=preview,
            image_to_idx=image_to_idx,
            running_edges=running_edges,
            pupil_edges=pupil_edges,
            brain_region_to_idx=brain_region_to_idx,
            show_processing=preview.experiment_id in show_processing_ids,
            bin_size_sec=BIN_SIZE_SEC,
        )

        neural_sessions.append(neural_trials)
        input_sessions.append(input_trials)
        output_sessions.append(output_trials)
        subject_idx.append(subject_to_idx[preview.subject_id])
        brain_region_idx.append(region_idx_arr)
        session_stats.append(stats)
        log(
            f"[{i}/{len(eligible)}] experiment {preview.experiment_id}: "
            f"{stats['trial_count']} trials, {stats['neuron_count']} neurons, "
            f"{stats['elapsed_sec']}s"
        )

    log(f"Conversion pass completed in {time.perf_counter() - convert_start:.1f}s")

    data = {
        "neural": neural_sessions,
        "input": input_sessions,
        "output": output_sessions,
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
            image_values,
            ["no_change", "change"],
            RUN_BIN_VALUES,
            PUPIL_BIN_VALUES,
            TRIAL_OUTCOME_VALUES,
        ],
        "metadata": {
            "task_description": (
                "Allen Visual Behavior ophys trials with neural calcium-event activity "
                "rebinned to 100 ms bins; outputs are image identity, image change, "
                "running-speed bin, pupil-diameter bin, and trial outcome."
            ),
            "time_bin_size": BIN_SIZE_SEC * 1000.0,
            "temporal_alignment_event": "trial start",
            "off_start": 0.0,
            "off_end": None,
            "neural_representation": "ophys event-detection magnitudes",
            "binning_rule": "sum event magnitudes within each 100 ms trial bin",
            "running_bin_edges": running_edges.tolist(),
            "pupil_bin_edges": pupil_edges.tolist(),
            "excluded_sessions": [
                {
                    "experiment_id": p.experiment_id,
                    "session_type": p.session_type,
                    "reason": p.excluded_reason,
                }
                for p in excluded
            ],
            "included_experiment_ids": [p.experiment_id for p in eligible],
            "source_manifest": str(MANIFEST_PATH.name),
            "mode": mode,
        },
    }

    with out_path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.perf_counter() - overall_start
    total_trials = sum(len(x) for x in neural_sessions)
    total_neurons = int(sum(arr.shape[0] for arr in brain_region_idx))
    log(f"Saved converted dataset to {out_path}")
    log(
        f"Summary: {len(neural_sessions)} sessions, {total_trials} trials, "
        f"{total_neurons} neurons, elapsed {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
