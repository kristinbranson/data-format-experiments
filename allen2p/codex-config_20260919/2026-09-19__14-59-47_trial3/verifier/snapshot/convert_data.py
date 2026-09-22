#!/usr/bin/env python3
"""Convert Allen Visual Behavior 2P NWBs to decoder-compatible trials.

Usage
-----
python -u /app/convert_data.py OUTPUT.pkl [--full | --sample]
                                      [--show-processing]

The converter reads only released, valid ROIs from active behavioral
experiments with eye tracking. Detected calcium events and behavioral streams
are synchronized on a common 30 Hz grid derived from the ophys clock.
"""

from __future__ import annotations

import argparse
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


APP_ROOT = Path("/app")
DATA_ROOT = APP_ROOT / "data" / "visual-behavior-ophys-1.1.0"
NWB_ROOT = DATA_ROOT / "behavior_ophys_experiments"
METADATA_PATH = DATA_ROOT / "project_metadata" / "ophys_experiment_table.csv"

TARGET_HZ = 30.0
BIN_SIZE_S = 1.0 / TARGET_HZ
MAX_PUPIL_GAP_FRAMES = 30  # one second at the eye-camera rate
SAMPLE_EXPERIMENT_IDS = (792813858, 809501118)  # image sets A and B

IMAGE_VALUES = [
    "gray",
    "im000",
    "im031",
    "im035",
    "im045",
    "im054",
    "im061",
    "im062",
    "im063",
    "im065",
    "im066",
    "im069",
    "im073",
    "im075",
    "im077",
    "im085",
    "im106",
]
IMAGE_TO_CODE = {name: i for i, name in enumerate(IMAGE_VALUES)}
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
QUINTILE_VALUES = [
    "0-20th percentile",
    "20-40th percentile",
    "40-60th percentile",
    "60-80th percentile",
    "80-100th percentile",
]


@dataclass
class SessionResult:
    neural: list[np.ndarray]
    inputs: list[np.ndarray]
    outputs: list[np.ndarray]
    subject: str
    region: str
    nneurons: int
    info: dict
    diagnostic: dict | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output pickle path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="Process two representative sessions")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing_<experiment_id>.png for up to two sessions",
    )
    return parser.parse_args()


def experiment_id_from_path(path: Path) -> int:
    match = re.search(r"(\d+)\.nwb$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse experiment ID from {path}")
    return int(match.group(1))


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in values],
        dtype=object,
    )


def choose_experiments(sample: bool) -> tuple[pd.DataFrame, list[tuple[int, Path]], list[dict]]:
    """Select active experiments with complete required streams."""
    metadata = pd.read_csv(METADATA_PATH).set_index("ophys_experiment_id", drop=False)
    paths = sorted(NWB_ROOT.glob("behavior_ophys_experiment_*.nwb"))
    available = {experiment_id_from_path(path): path for path in paths}

    if sample:
        candidate_ids = list(SAMPLE_EXPERIMENT_IDS)
        missing = [eid for eid in candidate_ids if eid not in available]
        if missing:
            raise FileNotFoundError(f"Sample NWBs missing: {missing}")
    else:
        candidate_ids = sorted(available)

    selected: list[tuple[int, Path]] = []
    excluded: list[dict] = []
    for eid in candidate_ids:
        if eid not in metadata.index:
            excluded.append({"ophys_experiment_id": eid, "reason": "missing metadata"})
            continue
        row = metadata.loc[eid]
        if row["behavior_type"] != "active_behavior":
            excluded.append({"ophys_experiment_id": eid, "reason": "passive viewing"})
            continue
        path = available[eid]
        with h5py.File(path, "r") as nwb:
            if "EyeTracking" not in nwb["acquisition"]:
                excluded.append({"ophys_experiment_id": eid, "reason": "missing eye tracking"})
                continue
            required = [
                "processing/ophys/event_detection/data",
                "processing/ophys/event_detection/timestamps",
                "processing/running/speed/data",
                "processing/running/speed/timestamps",
                "intervals/trials",
            ]
            absent = [key for key in required if key not in nwb]
            if absent:
                excluded.append(
                    {"ophys_experiment_id": eid, "reason": f"missing required datasets: {absent}"}
                )
                continue
        selected.append((eid, path))

    if sample and len(selected) != 2:
        raise RuntimeError(f"Sample mode requires two usable sessions; got {len(selected)}")
    if not selected:
        raise RuntimeError("No eligible experiments found")
    return metadata, selected, excluded


def fill_short_internal_gaps(
    values: np.ndarray, timestamps: np.ndarray, max_gap_frames: int
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    """Interpolate bounded short NaN runs; return filled and residual runs.

    Runs are half-open index intervals. Edge gaps are deliberately not filled.
    """
    filled = np.asarray(values, dtype=np.float64).copy()
    invalid = ~np.isfinite(filled)
    changes = np.diff(np.r_[False, invalid, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    filled_runs: list[tuple[int, int]] = []
    residual_runs: list[tuple[int, int]] = []
    n = len(filled)
    for start, end in zip(starts, ends):
        bounded = start > 0 and end < n and np.isfinite(filled[start - 1]) and np.isfinite(filled[end])
        if bounded and end - start <= max_gap_frames:
            filled[start:end] = np.interp(
                timestamps[start:end],
                [timestamps[start - 1], timestamps[end]],
                [filled[start - 1], filled[end]],
            )
            filled_runs.append((int(start), int(end)))
        else:
            residual_runs.append((int(start), int(end)))
    return filled, filled_runs, residual_runs


def trials_overlapping_invalid_runs(
    starts: np.ndarray,
    stops: np.ndarray,
    sample_times: np.ndarray,
    residual_runs: list[tuple[int, int]],
) -> np.ndarray:
    """Return a mask for trials touching an unresolved pupil-invalid interval."""
    bad = np.zeros(len(starts), dtype=bool)
    if len(sample_times) < 2:
        return np.ones(len(starts), dtype=bool)
    half_step = 0.5 * float(np.median(np.diff(sample_times)))
    for first, last_exclusive in residual_runs:
        invalid_start = sample_times[first] - half_step
        invalid_stop = sample_times[last_exclusive - 1] + half_step
        bad |= (starts < invalid_stop) & (stops > invalid_start)
    return bad


def trial_centers(start: float, stop: float) -> np.ndarray:
    n_bins = int(np.floor((stop - start) * TARGET_HZ + 1.0e-9))
    if n_bins < 1:
        return np.empty(0, dtype=np.float64)
    return start + (np.arange(n_bins, dtype=np.float64) + 0.5) * BIN_SIZE_S


def interpolate_matrix(
    source_t: np.ndarray, source_values: np.ndarray, target_t: np.ndarray
) -> np.ndarray:
    """Linear interpolation along axis 0, returning float32 time x feature."""
    if len(source_t) < 2:
        raise ValueError("Need at least two source timestamps")
    if target_t[0] < source_t[0] or target_t[-1] > source_t[-1]:
        raise ValueError("Target timestamps outside source range")
    right = np.searchsorted(source_t, target_t, side="left")
    right = np.clip(right, 1, len(source_t) - 1)
    left = right - 1
    denom = source_t[right] - source_t[left]
    if np.any(denom <= 0):
        raise ValueError("Source timestamps must be strictly increasing")
    weight = ((target_t - source_t[left]) / denom).astype(np.float32)
    result = source_values[left] * (1.0 - weight[:, None]) + source_values[right] * weight[:, None]
    return np.asarray(result, dtype=np.float32)


def interpolate_vector(source_t: np.ndarray, source_values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    finite = np.isfinite(source_t) & np.isfinite(source_values)
    if finite.sum() < 2:
        raise ValueError("Insufficient finite points for interpolation")
    valid_t = source_t[finite]
    valid_v = source_values[finite]
    if target_t[0] < valid_t[0] or target_t[-1] > valid_t[-1]:
        raise ValueError("Target timestamps outside finite behavioral stream")
    return np.interp(target_t, valid_t, valid_v).astype(np.float32)


def quantile_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite value passed to quantile binning")
    edges = np.quantile(values.astype(np.float64), [0.2, 0.4, 0.6, 0.8])
    codes = np.searchsorted(edges, values, side="right").astype(np.int8)
    if codes.min() < 0 or codes.max() > 4:
        raise AssertionError("Quintile code outside 0..4")
    return codes, edges


def presentation_group(nwb: h5py.File) -> h5py.Group:
    names = [
        key
        for key in nwb["intervals"].keys()
        if key not in {"trials", "natural_movie_one_presentations", "spontaneous_presentations"}
    ]
    with_images = [key for key in names if "image_name" in nwb[f"intervals/{key}"]]
    if len(with_images) != 1:
        raise RuntimeError(f"Expected one natural-image presentation table, got {with_images}")
    return nwb[f"intervals/{with_images[0]}"]


def read_float32_dataset(dataset: h5py.Dataset) -> np.ndarray:
    destination = np.empty(dataset.shape, dtype=np.float32)
    dataset.read_direct(destination)
    return destination


def process_session(
    eid: int,
    path: Path,
    meta: pd.Series,
    make_diagnostic: bool,
) -> SessionResult:
    start_clock = time.perf_counter()
    with h5py.File(path, "r") as nwb:
        trials = nwb["intervals/trials"]
        go = trials["go"][:].astype(bool)
        catch = trials["catch"][:].astype(bool)
        aborted = trials["aborted"][:].astype(bool)
        auto_rewarded = trials["auto_rewarded"][:].astype(bool)
        eligible = go | catch
        if np.any(eligible & (aborted | auto_rewarded)):
            raise AssertionError(f"{eid}: eligible trial is aborted/auto-rewarded")

        outcome_flags = np.vstack([trials[name][:].astype(bool) for name in OUTCOME_COLUMNS])
        if np.any(outcome_flags[:, eligible].sum(axis=0) != 1):
            raise AssertionError(f"{eid}: retained outcomes are not mutually exclusive/exhaustive")

        starts_all = trials["start_time"][:].astype(np.float64)
        stops_all = trials["stop_time"][:].astype(np.float64)
        candidate_indices = np.flatnonzero(eligible)
        candidate_starts = starts_all[candidate_indices]
        candidate_stops = stops_all[candidate_indices]

        eye_t = nwb["acquisition/EyeTracking/eye_tracking/timestamps"][:].astype(np.float64)
        pupil_width = nwb["acquisition/EyeTracking/pupil_tracking/width"][:].astype(np.float64)
        pupil_height = nwb["acquisition/EyeTracking/pupil_tracking/height"][:].astype(np.float64)
        if not (len(eye_t) == len(pupil_width) == len(pupil_height)):
            raise AssertionError(f"{eid}: pupil stream length mismatch")
        pupil_raw = 2.0 * np.maximum(pupil_width, pupil_height)
        pupil_filled, short_runs, residual_runs = fill_short_internal_gaps(
            pupil_raw, eye_t, MAX_PUPIL_GAP_FRAMES
        )
        invalid_trials = trials_overlapping_invalid_runs(
            candidate_starts, candidate_stops, eye_t, residual_runs
        )
        kept_indices = candidate_indices[~invalid_trials]
        removed_indices = candidate_indices[invalid_trials]
        if len(kept_indices) < 2:
            raise RuntimeError(f"{eid}: only {len(kept_indices)} valid trials after pupil filtering")

        per_trial_t = [trial_centers(starts_all[i], stops_all[i]) for i in kept_indices]
        if any(len(x) < 2 for x in per_trial_t):
            raise AssertionError(f"{eid}: retained trial has fewer than two bins")
        offsets = np.cumsum([0] + [len(x) for x in per_trial_t])
        target_t = np.concatenate(per_trial_t)

        ophys_t = nwb["processing/ophys/event_detection/timestamps"][:].astype(np.float64)
        event_dataset = nwb["processing/ophys/event_detection/data"]
        if event_dataset.shape[0] != len(ophys_t):
            raise AssertionError(f"{eid}: event/timestamp mismatch")
        nneurons = int(event_dataset.shape[1])
        if target_t[0] < ophys_t[0] or target_t[-1] > ophys_t[-1]:
            raise AssertionError(f"{eid}: retained trials outside ophys bounds")
        events = read_float32_dataset(event_dataset)
        if not np.all(np.isfinite(events)):
            raise ValueError(f"{eid}: event data contain NaN/Inf")
        neural_aligned = interpolate_matrix(ophys_t, events, target_t)

        running_t = nwb["processing/running/speed/timestamps"][:].astype(np.float64)
        running = nwb["processing/running/speed/data"][:].astype(np.float64)
        running_aligned = interpolate_vector(running_t, running, target_t)
        pupil_aligned = interpolate_vector(eye_t, pupil_filled, target_t)
        if not np.all(np.isfinite(pupil_aligned)):
            raise ValueError(f"{eid}: nonfinite pupil after valid-trial selection")

        running_codes, running_edges = quantile_codes(running_aligned)
        pupil_codes, pupil_edges = quantile_codes(pupil_aligned)

        presentations = presentation_group(nwb)
        presentation_starts = presentations["start_time"][:].astype(np.float64)
        presentation_stops = presentations["stop_time"][:].astype(np.float64)
        active = presentations["active"][:].astype(bool)
        omitted = np.nan_to_num(presentations["omitted"][:], nan=0.0).astype(bool)
        image_names = decode_strings(presentations["image_name"][:])
        unknown = sorted(set(image_names[active & ~omitted]) - set(IMAGE_TO_CODE))
        if unknown:
            raise ValueError(f"{eid}: unknown image identities {unknown}")

        image_codes = np.zeros(len(target_t), dtype=np.int8)
        pidx = np.searchsorted(presentation_starts, target_t, side="right") - 1
        valid_pidx = pidx >= 0
        shown = np.zeros(len(target_t), dtype=bool)
        shown[valid_pidx] = (
            active[pidx[valid_pidx]]
            & ~omitted[pidx[valid_pidx]]
            & (target_t[valid_pidx] < presentation_stops[pidx[valid_pidx]])
        )
        shown_indices = np.flatnonzero(shown)
        image_codes[shown_indices] = np.asarray(
            [IMAGE_TO_CODE[name] for name in image_names[pidx[shown_indices]]], dtype=np.int8
        )

        change_codes = np.zeros(len(target_t), dtype=np.int8)
        change_times_all = trials["change_time"][:].astype(np.float64)
        for trial_position, raw_index in enumerate(kept_indices):
            if not go[raw_index]:
                continue
            lo, hi = offsets[trial_position], offsets[trial_position + 1]
            local_index = int(np.searchsorted(target_t[lo:hi], change_times_all[raw_index], side="left"))
            if local_index >= hi - lo:
                raise AssertionError(f"{eid}: go change falls after its trial grid")
            change_codes[lo + local_index] = 1

        # Cross-check the true presentation annotation at every retained go change.
        presentation_change = np.nan_to_num(presentations["is_change"][:], nan=0.0).astype(bool)
        true_change_times = presentation_starts[active & presentation_change]
        go_change_times = change_times_all[kept_indices[go[kept_indices]]]
        if len(go_change_times):
            nearest = np.min(np.abs(go_change_times[:, None] - true_change_times[None, :]), axis=1)
            if not np.allclose(nearest, 0.0, rtol=0.0, atol=1.0e-9):
                raise AssertionError(f"{eid}: trial and presentation change times disagree")

        neural_trials: list[np.ndarray] = []
        input_trials: list[np.ndarray] = []
        output_trials: list[np.ndarray] = []
        for trial_position, raw_index in enumerate(kept_indices):
            lo, hi = offsets[trial_position], offsets[trial_position + 1]
            neural_trial = np.ascontiguousarray(neural_aligned[lo:hi].T, dtype=np.float32)
            empty_input = np.empty((0, hi - lo), dtype=np.float32)
            outcome = int(np.flatnonzero(outcome_flags[:, raw_index])[0])
            output_trial = np.empty((5, hi - lo), dtype=np.int16)
            output_trial[0] = image_codes[lo:hi]
            output_trial[1] = change_codes[lo:hi]
            output_trial[2] = running_codes[lo:hi]
            output_trial[3] = pupil_codes[lo:hi]
            output_trial[4].fill(outcome)
            if not np.all(np.isfinite(neural_trial)):
                raise ValueError(f"{eid}: nonfinite neural trial")
            neural_trials.append(neural_trial)
            input_trials.append(empty_input)
            output_trials.append(output_trial)

        cell_ids = nwb[
            "processing/ophys/image_segmentation/cell_specimen_table/cell_specimen_id"
        ][:].astype(np.int64)
        valid_rois = nwb[
            "processing/ophys/image_segmentation/cell_specimen_table/valid_roi"
        ][:].astype(bool)
        if len(cell_ids) != nneurons or not np.all(valid_rois):
            raise AssertionError(f"{eid}: event/cell table mismatch or invalid ROI present")

        native_rate = float(1.0 / np.median(np.diff(ophys_t)))
        outcome_counts = {
            name: int(outcome_flags[j, kept_indices].sum()) for j, name in enumerate(OUTCOME_COLUMNS)
        }
        info = {
            "ophys_experiment_id": int(eid),
            "ophys_session_id": int(meta["ophys_session_id"]),
            "behavior_session_id": int(meta["behavior_session_id"]),
            "mouse_id": str(int(meta["mouse_id"])),
            "session_type": str(meta["session_type"]),
            "experience_level": str(meta["experience_level"]),
            "image_set": str(meta["image_set"]),
            "project_code": str(meta["project_code"]),
            "equipment_name": str(meta["equipment_name"]),
            "cre_line": str(meta["cre_line"]),
            "targeted_structure": str(meta["targeted_structure"]),
            "imaging_depth_um": int(meta["imaging_depth"]),
            "native_ophys_rate_hz": native_rate,
            "n_neurons": nneurons,
            "cell_specimen_ids": cell_ids.tolist(),
            "candidate_go_catch_trials": int(len(candidate_indices)),
            "retained_trials": int(len(kept_indices)),
            "removed_trials_long_pupil_gap": int(len(removed_indices)),
            "removed_raw_trial_ids": trials["id"][:][removed_indices].astype(int).tolist(),
            "retained_raw_trial_ids": trials["id"][:][kept_indices].astype(int).tolist(),
            "short_pupil_gaps_interpolated": int(len(short_runs)),
            "long_or_edge_pupil_gaps": int(len(residual_runs)),
            "pupil_invalid_fraction_raw": float(np.mean(~np.isfinite(pupil_raw))),
            "running_quintile_edges_cm_per_s": running_edges.tolist(),
            "pupil_diameter_quintile_edges_pixels": pupil_edges.tolist(),
            "outcome_counts": outcome_counts,
            "trial_timepoints_min": int(min(map(len, per_trial_t))),
            "trial_timepoints_max": int(max(map(len, per_trial_t))),
        }

        diagnostic = None
        if make_diagnostic:
            first_trial = 0
            lo, hi = offsets[first_trial], offsets[first_trial + 1]
            raw_trial_index = int(kept_indices[first_trial])
            diagnostic = {
                "eid": eid,
                "ophys_t": ophys_t,
                "events_sample": events[:, : min(5, nneurons)],
                "target_trial_t": target_t[lo:hi],
                "neural_trial": neural_aligned[lo:hi, : min(5, nneurons)],
                "image_codes": image_codes[lo:hi],
                "change_codes": change_codes[lo:hi],
                "running": running_aligned[lo:hi],
                "running_codes": running_codes[lo:hi],
                "running_edges": running_edges,
                "eye_t": eye_t,
                "pupil_raw": pupil_raw,
                "pupil_filled": pupil_filled,
                "pupil": pupil_aligned[lo:hi],
                "pupil_codes": pupil_codes[lo:hi],
                "pupil_edges": pupil_edges,
                "trial_start": starts_all[raw_trial_index],
                "trial_stop": stops_all[raw_trial_index],
                "candidate_trials": len(candidate_indices),
                "retained_trials": len(kept_indices),
                "removed_trials": len(removed_indices),
                "outcome_counts": outcome_counts,
            }

    elapsed = time.perf_counter() - start_clock
    print(
        f"[{eid}] {nneurons} neurons, {len(candidate_indices)} candidate -> "
        f"{len(neural_trials)} trials, {sum(x.shape[1] for x in neural_trials):,} bins "
        f"in {elapsed:.2f}s",
        flush=True,
    )
    return SessionResult(
        neural=neural_trials,
        inputs=input_trials,
        outputs=output_trials,
        subject=str(int(meta["mouse_id"])),
        region=str(meta["targeted_structure"]),
        nneurons=nneurons,
        info=info,
        diagnostic=diagnostic,
    )


def plot_processing(diagnostic: dict, output_path: Path) -> None:
    """Plot each material processing stage for one representative trial."""
    fig, axes = plt.subplots(4, 2, figsize=(17, 16), constrained_layout=True)
    axes = axes.ravel()
    eid = diagnostic["eid"]
    target_t = diagnostic["target_trial_t"]
    relative = target_t - diagnostic["trial_start"]

    dt_native = np.diff(diagnostic["ophys_t"])
    axes[0].hist(dt_native * 1000.0, bins=40, alpha=0.8, label="native ophys")
    axes[0].axvline(BIN_SIZE_S * 1000.0, color="k", ls="--", label="target 30 Hz")
    axes[0].set(xlabel="frame interval (ms)", ylabel="count", title="1. Timestamp/binning check")
    axes[0].legend()

    native_t = diagnostic["ophys_t"]
    window = (native_t >= diagnostic["trial_start"]) & (native_t < diagnostic["trial_stop"])
    for j in range(diagnostic["events_sample"].shape[1]):
        axes[1].plot(
            native_t[window] - diagnostic["trial_start"],
            diagnostic["events_sample"][window, j] + j,
            ".",
            ms=2,
            alpha=0.6,
        )
        axes[1].plot(relative, diagnostic["neural_trial"][:, j] + j, lw=0.8)
    axes[1].set(title="2. Raw ophys events (dots) and 30-Hz interpolation", xlabel="trial time (s)")

    axes[2].step(relative, diagnostic["image_codes"], where="mid")
    axes[2].set(title="3. Image identity (gray=0; named images >0)", xlabel="trial time (s)", ylabel="class")

    axes[3].step(relative, diagnostic["change_codes"], where="mid")
    axes[3].set(title="4. True image-change pulse", xlabel="trial time (s)", ylabel="binary", ylim=(-0.1, 1.1))

    axes[4].plot(relative, diagnostic["running"], label="processed cm/s")
    axes[4].scatter(relative, diagnostic["running_codes"], c=diagnostic["running_codes"], s=8, label="quintile code")
    axes[4].set(title=f"5. Running alignment; edges={np.round(diagnostic['running_edges'], 2)}", xlabel="trial time (s)")
    axes[4].legend(fontsize=8)

    eye_t = diagnostic["eye_t"]
    pupil_window = (eye_t >= diagnostic["trial_start"] - 1) & (eye_t <= diagnostic["trial_stop"] + 1)
    axes[5].plot(
        eye_t[pupil_window] - diagnostic["trial_start"],
        diagnostic["pupil_raw"][pupil_window],
        ".",
        ms=2,
        alpha=0.5,
        label="blink-filtered raw",
    )
    axes[5].plot(relative, diagnostic["pupil"], lw=1, label="short-gap fill + aligned")
    axes[5].set(title="6. Pupil validity and alignment", xlabel="trial time (s)", ylabel="diameter (px)")
    axes[5].legend(fontsize=8)

    axes[6].plot(relative, diagnostic["pupil"], color="tab:blue")
    for edge in diagnostic["pupil_edges"]:
        axes[6].axhline(edge, color="gray", lw=0.7, ls="--")
    axes[6].scatter(relative, diagnostic["pupil_codes"], c=diagnostic["pupil_codes"], s=8)
    axes[6].set(title=f"7. Pupil quintiles; edges={np.round(diagnostic['pupil_edges'], 2)}", xlabel="trial time (s)")

    names = list(diagnostic["outcome_counts"])
    counts = [diagnostic["outcome_counts"][name] for name in names]
    axes[7].bar(names, counts)
    axes[7].tick_params(axis="x", rotation=25)
    axes[7].set(
        title=(
            "8. Trial curation/outcomes: "
            f"{diagnostic['candidate_trials']} candidate, {diagnostic['retained_trials']} retained, "
            f"{diagnostic['removed_trials']} pupil-invalid"
        ),
        ylabel="trials",
    )

    fig.suptitle(f"Visual Behavior conversion processing — experiment {eid}", fontsize=16)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved {output_path}", flush=True)


def validate_converted(data: dict) -> None:
    nsessions = len(data["neural"])
    if not (len(data["input"]) == len(data["output"]) == nsessions):
        raise AssertionError("Session nesting mismatch")
    for s in range(nsessions):
        if len(data["neural"][s]) < 2:
            raise AssertionError(f"Session {s} has fewer than two trials")
        if not (len(data["neural"][s]) == len(data["input"][s]) == len(data["output"][s])):
            raise AssertionError(f"Session {s} trial nesting mismatch")
        nneurons = len(data["brain_region_idx"][s])
        for neural, inputs, outputs in zip(data["neural"][s], data["input"][s], data["output"][s]):
            if neural.ndim != 2 or neural.shape[0] != nneurons:
                raise AssertionError("Neural shape mismatch")
            if inputs.shape != (0, neural.shape[1]):
                raise AssertionError("Empty input shape mismatch")
            if outputs.shape != (5, neural.shape[1]):
                raise AssertionError("Output shape mismatch")
            if not np.all(np.isfinite(neural)):
                raise AssertionError("Nonfinite neural value")
            limits = (16, 1, 4, 4, 3)
            if any(outputs[i].min() < 0 or outputs[i].max() > limits[i] for i in range(5)):
                raise AssertionError("Categorical output outside codebook")
            if int(outputs[1].sum()) > 1:
                raise AssertionError("A trial contains more than one change pulse")


def main() -> None:
    args = parse_args()
    all_start = time.perf_counter()
    metadata, selected, excluded = choose_experiments(sample=args.sample)
    print(
        f"Mode: {'sample' if args.sample else 'full'}; selected {len(selected)} experiments; "
        f"excluded {len(excluded)}",
        flush=True,
    )

    results: list[SessionResult] = []
    for position, (eid, path) in enumerate(selected):
        result = process_session(
            eid=eid,
            path=path,
            meta=metadata.loc[eid],
            make_diagnostic=args.show_processing and position < 2,
        )
        results.append(result)
        if result.diagnostic is not None:
            plot_processing(result.diagnostic, APP_ROOT / f"processing_{eid}.png")

    subjects = sorted({result.subject for result in results}, key=int)
    subject_lookup = {name: i for i, name in enumerate(subjects)}
    regions = sorted({result.region for result in results})
    region_lookup = {name: i for i, name in enumerate(regions)}

    data = {
        "neural": [result.neural for result in results],
        "input": [result.inputs for result in results],
        "output": [result.outputs for result in results],
        "subjects": subjects,
        "subject_idx": np.asarray([subject_lookup[result.subject] for result in results], dtype=np.int32),
        "brain_regions": regions,
        "brain_region_idx": [
            np.full(result.nneurons, region_lookup[result.region], dtype=np.int16) for result in results
        ],
        "input_names": [],
        "output_names": [
            "image identity",
            "image change",
            "running speed quintile",
            "pupil diameter quintile",
            "trial outcome",
        ],
        "output_values": [
            IMAGE_VALUES,
            ["no change", "change"],
            QUINTILE_VALUES,
            QUINTILE_VALUES,
            OUTCOME_VALUES,
        ],
        "metadata": {
            "task_description": (
                "Allen Visual Behavior go/no-go natural-image change-detection task; neural activity "
                "predicts image identity/change, running and pupil quintiles, and trial outcome."
            ),
            "time_bin_size": 1000.0 / TARGET_HZ,
            "time_bin_size_units": "ms",
            "temporal_alignment_event": "native trial start on the synchronized ophys clock",
            "off_start": 0.0,
            "off_end": None,
            "neural_signal": "released FastLZero detected calcium event magnitude",
            "target_sampling_rate_hz": TARGET_HZ,
            "trial_definition": "NWB native trial start/stop; go and catch only",
            "session_definition": "one released ophys experiment/imaging plane",
            "session_selection": "active_behavior with complete eye tracking",
            "pupil_definition": "2 * max(blink-filtered pupil ellipse width, height), pixels",
            "pupil_missing_rule": (
                "linearly interpolate bounded gaps <=30 eye frames; exclude trials overlapping longer/edge gaps"
            ),
            "percentile_scope": "within session across retained trial timepoints",
            "image_gray_rule": "gray class for inter-stimulus gaps and omitted presentations",
            "source_release": "visual-behavior-ophys-1.1.0",
            "excluded_sessions": excluded,
            "session_info": [result.info for result in results],
        },
    }
    validate_converted(data)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    write_start = time.perf_counter()
    with temporary.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)
    write_elapsed = time.perf_counter() - write_start

    total_trials = sum(len(x) for x in data["neural"])
    total_cells = sum(x.nneurons for x in results)
    size_gib = output.stat().st_size / 1024**3
    print(
        f"Wrote {output} ({size_gib:.3f} GiB): {len(results)} sessions, "
        f"{len(subjects)} subjects, {total_cells} session-neurons, {total_trials} trials",
        flush=True,
    )
    print(
        f"Timing: processing+plots {write_start - all_start:.2f}s; pickle write {write_elapsed:.2f}s; "
        f"total {time.perf_counter() - all_start:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
