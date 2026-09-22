#!/usr/bin/env python3
"""Convert the supplied Allen Visual Behavior ophys NWBs for neural decoding.

Decisions implemented here
--------------------------
* One released ``behavior_ophys_experiment`` (one imaging plane) is one decoder
  session.  This is the AllenSDK unit that owns one simultaneously sampled
  neural population, and it is also the unit used for plane-wise decoding in
  the reference paper.
* Only active-behavior experiments are used.  Passive replay sessions do not
  contain an animal performing Go/Catch trials.  Released experiments/ROIs
  have passed Allen QC; ``valid_roi`` is nevertheless applied explicitly.
* Trials are the SDK/NWB trial intervals.  We retain (Go OR Catch) AND NOT
  aborted AND NOT auto-rewarded trials, exactly as requested.  No engagement
  or performance threshold is imposed.
* Neural activity is the detected calcium-event magnitude used in the paper,
  not dF/F or the SDK's display-oriented filtered events.
* All streams are placed on 100 ms bins anchored at each trial start.  Calcium
  event magnitudes whose *ophys timestamps* fall in a bin are summed.  Running
  speed and processed (blink-filtered) pupil area are interpolated at bin
  centers on the common synchronized NWB clock.  Thus the ophys timestamps,
  rather than frame number or nominal acquisition rate, define alignment.
* Pupil diameter is the equivalent circular diameter 2*sqrt(area/pi).  Since
  this is monotonic, its percentile labels equal pupil-area percentile labels.
  Short blink gaps (NaNs in the SDK-processed trace) are linearly interpolated.
* Running and pupil values are quintiled separately within each experiment,
  using all retained trial bins.  This removes between-animal/camera scale
  differences and gives percentile classes for the data actually decoded.
* Image identity is an explicit ``gray`` class outside a displayed image
  interval (including omissions).  Image-change is one only in the first bin
  centered at or after a true image-change onset.  Trial outcome is repeated
  over time solely so static and time-varying targets fit one rectangular
  output matrix.

The converter reads NWB arrays directly with h5py.  Paths and semantics are
those exposed by AllenSDK 2.16 for this dataset, but direct reading avoids
materializing image templates and ROI masks that are irrelevant to decoding.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path("/app/data/visual-behavior-ophys-1.1.0")
DEFAULT_OUTPUT = Path("/app/converted_data.pkl")
BIN_SECONDS = 0.100
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
OUTCOME_NAMES = ("hit", "miss", "false alarm", "correct rejection")


def experiment_id(path: Path) -> int:
    match = re.search(r"behavior_ophys_experiment_(\d+)\.nwb$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse experiment id from {path}")
    return int(match.group(1))


def read_array(group: h5py.Group, name: str, dtype=None) -> np.ndarray:
    """Read an HDF5 dataset, optionally casting during the HDF5 read."""
    dataset = group[name]
    if dtype is None or np.dtype(dtype) == dataset.dtype:
        return dataset[:]
    result = np.empty(dataset.shape, dtype=dtype)
    dataset.read_direct(result)
    return result


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
         for x in values],
        dtype=object,
    )


def stimulus_group(nwb: h5py.File) -> h5py.Group:
    """Return the natural-image task presentation table (not movie/spontaneous)."""
    candidates = []
    for name, group in nwb["intervals"].items():
        if isinstance(group, h5py.Group) and "image_name" in group:
            candidates.append((name, group))
    if len(candidates) != 1:
        names = [name for name, _ in candidates]
        raise ValueError(f"Expected one task image table, found {names}")
    return candidates[0][1]


def interpolate_finite(timestamps: np.ndarray, values: np.ndarray,
                       targets: np.ndarray, label: str) -> np.ndarray:
    valid = np.isfinite(timestamps) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        raise ValueError(f"Fewer than two finite {label} samples")
    # np.interp also supplies the nearest valid endpoint outside the sampled
    # range. Retained trials normally lie strictly inside both behavior traces.
    return np.interp(targets, timestamps[valid], values[valid]).astype(np.float32)


def quintile(values: np.ndarray) -> np.ndarray:
    """Return labels 0..4 using empirical 20/40/60/80 percentile edges."""
    edges = np.quantile(values, (0.2, 0.4, 0.6, 0.8))
    return np.searchsorted(edges, values, side="right").astype(np.int16)


def cumulative_bin_sums(cumulative_events: np.ndarray,
                        ophys_timestamps: np.ndarray,
                        edges: np.ndarray) -> np.ndarray:
    """Sum native-frame event magnitudes into bins; return neuron x time."""
    indices = np.searchsorted(ophys_timestamps, edges, side="left")
    starts, stops = indices[:-1], indices[1:]
    n_bins = len(starts)
    n_neurons = cumulative_events.shape[1]
    right = np.zeros((n_bins, n_neurons), dtype=np.float32)
    left = np.zeros_like(right)
    has_right = stops > 0
    has_left = starts > 0
    right[has_right] = cumulative_events[stops[has_right] - 1]
    left[has_left] = cumulative_events[starts[has_left] - 1]
    return (right - left).T


def load_trial_table(group: h5py.Group) -> dict[str, np.ndarray]:
    names = (
        "start_time", "stop_time", "go", "catch", "aborted",
        "auto_rewarded", *OUTCOME_COLUMNS,
    )
    return {name: read_array(group, name) for name in names}


def make_trial_grids(trials: dict[str, np.ndarray], keep: np.ndarray):
    grids = []
    for row in np.flatnonzero(keep):
        start = float(trials["start_time"][row])
        stop = float(trials["stop_time"][row])
        n_bins = int(np.floor((stop - start) / BIN_SECONDS + 1e-9))
        if n_bins < 1:
            continue
        edges = start + np.arange(n_bins + 1, dtype=np.float64) * BIN_SECONDS
        centers = edges[:-1] + BIN_SECONDS / 2.0
        grids.append((int(row), edges, centers))
    return grids


def convert_experiment(path: Path, row: pd.Series,
                       image_to_index: dict[str, int],
                       region_to_index: dict[str, int]):
    with h5py.File(path, "r") as nwb:
        trials = load_trial_table(nwb["intervals/trials"])
        keep = (
            (trials["go"].astype(bool) | trials["catch"].astype(bool))
            & ~trials["aborted"].astype(bool)
            & ~trials["auto_rewarded"].astype(bool)
        )
        grids = make_trial_grids(trials, keep)
        if len(grids) < 2:
            raise ValueError(f"{path.name}: fewer than two retained trials")

        outcomes = np.column_stack(
            [trials[name].astype(bool) for name in OUTCOME_COLUMNS]
        )
        if not np.all(outcomes[keep].sum(axis=1) == 1):
            raise ValueError(f"{path.name}: retained trial lacks a unique outcome")

        ophys = nwb["processing/ophys"]
        event_group = ophys["event_detection"]
        ophys_timestamps = read_array(event_group, "timestamps", np.float64)
        events = read_array(event_group, "data", np.float32)
        valid_roi = read_array(
            ophys["image_segmentation/cell_specimen_table"], "valid_roi"
        ).astype(bool)
        events = np.ascontiguousarray(events[:, valid_roi], dtype=np.float32)
        if events.shape[1] == 0:
            raise ValueError(f"{path.name}: no valid neurons")
        # In-place cumulative sums let arbitrary trial-relative bins be obtained
        # without rereading the large event dataset or allocating a second copy.
        np.cumsum(events, axis=0, dtype=np.float32, out=events)

        running_group = nwb["processing/running/speed"]
        running_t = read_array(running_group, "timestamps", np.float64)
        running_v = read_array(running_group, "data", np.float64)

        pupil_group = nwb["acquisition/EyeTracking/pupil_tracking"]
        eye_t = read_array(
            nwb["acquisition/EyeTracking/eye_tracking"], "timestamps", np.float64
        )
        pupil_area = read_array(pupil_group, "area", np.float64)
        # The NWB ``area`` field is SDK-processed: outliers and dilated likely
        # blink frames are NaN. Equivalent circular diameter is in camera pixels.
        pupil_diameter = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)

        stim = stimulus_group(nwb)
        stim_start = read_array(stim, "start_time", np.float64)
        stim_stop = read_array(stim, "stop_time", np.float64)
        stim_names = decode_strings(read_array(stim, "image_name"))
        omitted = read_array(stim, "omitted").astype(bool)
        is_change_raw = read_array(stim, "is_change", np.float64)
        is_change = np.isfinite(is_change_raw) & is_change_raw.astype(bool)

        all_centers = np.concatenate([centers for _, _, centers in grids])
        running_aligned = interpolate_finite(
            running_t, running_v, all_centers, "running-speed"
        )
        pupil_aligned = interpolate_finite(
            eye_t, pupil_diameter, all_centers, "pupil-diameter"
        )
        running_bins = quintile(running_aligned)
        pupil_bins = quintile(pupil_aligned)

        neural_trials = []
        input_trials = []
        output_trials = []
        offset = 0
        gray_index = image_to_index["gray"]
        for trial_row, edges, centers in grids:
            n_time = len(centers)
            neural = cumulative_bin_sums(events, ophys_timestamps, edges)

            image_identity = np.full(n_time, gray_index, dtype=np.int16)
            image_change = np.zeros(n_time, dtype=np.int16)
            overlaps = np.flatnonzero(
                (stim_stop > edges[0]) & (stim_start < edges[-1])
            )
            for presentation in overlaps:
                if not omitted[presentation] and stim_names[presentation] != "omitted":
                    shown = ((centers >= stim_start[presentation])
                             & (centers < stim_stop[presentation]))
                    image_identity[shown] = image_to_index[stim_names[presentation]]
                if is_change[presentation]:
                    change_bin = int(np.searchsorted(
                        centers, stim_start[presentation], side="left"
                    ))
                    if change_bin < n_time:
                        image_change[change_bin] = 1

            outcome = int(np.flatnonzero(outcomes[trial_row])[0])
            output = np.empty((5, n_time), dtype=np.int16)
            output[0] = image_identity
            output[1] = image_change
            output[2] = running_bins[offset:offset + n_time]
            output[3] = pupil_bins[offset:offset + n_time]
            output[4].fill(outcome)
            offset += n_time

            neural_trials.append(neural.astype(np.float32, copy=False))
            input_trials.append(np.empty((0, n_time), dtype=np.float32))
            output_trials.append(output)

    region_idx = np.full(
        neural_trials[0].shape[0],
        region_to_index[str(row["targeted_structure"])],
        dtype=np.int16,
    )
    info = {
        "ophys_experiment_id": int(row.name),
        "ophys_session_id": int(row["ophys_session_id"]),
        "behavior_session_id": int(row["behavior_session_id"]),
        "mouse_id": str(int(row["mouse_id"])),
        "project_code": str(row["project_code"]),
        "session_type": str(row["session_type"]),
        "experience_level": str(row["experience_level"]),
        "targeted_structure": str(row["targeted_structure"]),
        "imaging_depth_um": int(row["imaging_depth"]),
        "n_neurons": int(neural_trials[0].shape[0]),
        "n_trials": len(neural_trials),
    }
    return neural_trials, input_trials, output_trials, region_idx, info


def discover(data_root: Path):
    nwb_dir = data_root / "behavior_ophys_experiments"
    table_path = data_root / "project_metadata/ophys_experiment_table.csv"
    table = pd.read_csv(table_path).set_index("ophys_experiment_id")
    paths = {experiment_id(path): path for path in nwb_dir.glob("*.nwb")}
    missing_metadata = sorted(set(paths) - set(table.index))
    if missing_metadata:
        raise ValueError(f"No metadata for experiment ids {missing_metadata}")

    supplied = table.loc[sorted(paths)].copy()
    active = supplied[~supplied["passive"].astype(bool)].copy()

    # Pupil diameter is a required output. Excluding sessions in which the NWB
    # lacks eye tracking is preferable to inventing a constant/imputed signal.
    usable_ids = []
    excluded_no_eye = []
    for exp_id in active.index:
        with h5py.File(paths[int(exp_id)], "r") as nwb:
            if "acquisition/EyeTracking/pupil_tracking/area" in nwb:
                usable_ids.append(int(exp_id))
            else:
                excluded_no_eye.append(int(exp_id))
    return table.loc[usable_ids].copy(), paths, excluded_no_eye, len(supplied)


def collect_image_names(selected: pd.DataFrame, paths: dict[int, Path]):
    names = set()
    for exp_id in selected.index:
        with h5py.File(paths[int(exp_id)], "r") as nwb:
            stim = stimulus_group(nwb)
            image_names = decode_strings(read_array(stim, "image_name"))
            omitted = read_array(stim, "omitted").astype(bool)
            names.update(image_names[~omitted])
    names.discard("omitted")
    return ["gray", *sorted(names)]


def convert(data_root: Path, output_path: Path) -> dict:
    selected, paths, excluded_no_eye, supplied_count = discover(data_root)
    image_values = collect_image_names(selected, paths)
    image_to_index = {name: i for i, name in enumerate(image_values)}
    subjects = sorted({str(int(x)) for x in selected["mouse_id"]})
    subject_to_index = {name: i for i, name in enumerate(subjects)}
    brain_regions = sorted(selected["targeted_structure"].astype(str).unique())
    region_to_index = {name: i for i, name in enumerate(brain_regions)}

    neural_sessions = []
    input_sessions = []
    output_sessions = []
    brain_region_idx = []
    subject_idx = []
    session_info = []

    total = len(selected)
    for number, (exp_id, row) in enumerate(selected.iterrows(), start=1):
        print(f"[{number:3d}/{total}] experiment {exp_id}", flush=True)
        neural, inputs, outputs, regions, info = convert_experiment(
            paths[int(exp_id)], row, image_to_index, region_to_index
        )
        neural_sessions.append(neural)
        input_sessions.append(inputs)
        output_sessions.append(outputs)
        brain_region_idx.append(regions)
        subject_idx.append(subject_to_index[str(int(row["mouse_id"]))])
        session_info.append(info)

    data = {
        "neural": neural_sessions,
        "input": input_sessions,
        "output": output_sessions,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int16),
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": [],
        "output_names": [
            "image identity", "image change", "running speed quintile",
            "pupil diameter quintile", "trial outcome",
        ],
        "output_values": [
            image_values,
            ["no change", "change"],
            ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
            ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
            list(OUTCOME_NAMES),
        ],
        "metadata": {
            "task_description": (
                "Allen Visual Behavior flashed-image change-detection task; "
                "neural calcium events predict image identity/change, running "
                "and pupil quintiles, and Go/Catch trial outcome."
            ),
            "time_bin_size": BIN_SECONDS * 1000.0,
            "temporal_alignment_event": (
                "Trial start; synchronized streams binned/aligned using ophys "
                "timestamps on the NWB clock"
            ),
            "off_start": 0.0,
            "off_end": None,
            "trial_window": "Full NWB trial interval; duration varies by trial",
            "neural_measure": (
                "Allen event_detection calcium-event magnitudes, summed in "
                "100 ms bins"
            ),
            "trial_filter": (
                "(go OR catch) AND NOT aborted AND NOT auto_rewarded"
            ),
            "session_unit": "One behavior_ophys_experiment / imaging plane",
            "session_filter": (
                "Active behavior, released QC-passed experiments, valid ROIs, "
                "and eye tracking available"
            ),
            "percentile_scope": "Within experiment over retained trial bins",
            "pupil_measure": (
                "Equivalent circular diameter from SDK blink-filtered pupil area; "
                "finite samples linearly interpolated"
            ),
            "gray_image_value": 0,
            "supplied_experiment_count": supplied_count,
            "converted_experiment_count": len(selected),
            "excluded_passive_experiment_count": int(
                supplied_count - len(selected) - len(excluded_no_eye)
            ),
            "excluded_missing_eye_tracking_ids": excluded_no_eye,
            "session_info": session_info,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output_path)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 2**30:.2f} GiB)")
    return data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.data_root, args.output)
