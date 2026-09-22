#!/usr/bin/env python3
"""Convert the supplied Allen Visual Behavior 2P NWBs for neural decoding.

The conversion follows the task/paper's image-presentation analysis unit: one
sample is the 750 ms interval beginning at each 250 ms image flash (and the
corresponding 750 ms interval for omissions).  Simultaneously acquired imaging
planes are combined into one recording session.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


BIN_SECONDS = 0.750
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")


def _experiment_id(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def _decode_strings(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def _presentation_group(nwb: h5py.File) -> h5py.Group:
    """Return the natural-image presentation table, independent of image set."""
    matches = [
        group for group in nwb["intervals"].values()
        if isinstance(group, h5py.Group)
        and "image_name" in group
        and "omitted" in group
        and "trials_id" in group
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one natural-image presentation table, found {len(matches)}"
        )
    return matches[0]


def _eligible_trial_ids(nwb: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """Get requested Go/Catch trial IDs and their four-way outcomes."""
    trials = nwb["intervals/trials"]
    keep = (
        (trials["go"][:] | trials["catch"][:])
        & ~trials["aborted"][:]
        & ~trials["auto_rewarded"][:]
    )
    trial_ids = trials["id"][:][keep].astype(np.int64)
    outcomes = np.column_stack([trials[name][:][keep] for name in OUTCOME_COLUMNS])
    if not np.all(outcomes.sum(axis=1) == 1):
        raise RuntimeError("Every retained trial must have exactly one trial outcome")
    return trial_ids, outcomes.argmax(axis=1).astype(np.int16)


def _session_presentations(
    nwb: h5py.File, trial_ids: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """Read active presentation times/labels in retained-trial order."""
    presentations = _presentation_group(nwb)
    source_trial_ids = presentations["trials_id"][:]
    active = presentations["active"][:].astype(bool)
    starts_source = presentations["start_time"][:]
    names_source = np.asarray(_decode_strings(presentations["image_name"][:]), dtype=object)
    changes_source = presentations["is_change"][:].astype(bool)

    row_groups: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    names: list[np.ndarray] = []
    changes: list[np.ndarray] = []
    cursor = 0
    for trial_id in trial_ids:
        idx = np.flatnonzero((source_trial_ids == trial_id) & active)
        if len(idx) == 0:
            raise RuntimeError(f"Retained trial {trial_id} has no active presentations")
        # The NWB table is chronological, but make the assumption explicit.
        idx = idx[np.argsort(starts_source[idx])]
        row_groups.append(np.arange(cursor, cursor + len(idx), dtype=np.int64))
        cursor += len(idx)
        starts.append(starts_source[idx])
        names.append(names_source[idx])
        changes.append(changes_source[idx])

    return (
        np.concatenate(starts).astype(np.float64),
        row_groups,
        np.concatenate(names),
        np.concatenate(changes).astype(np.int16),
    )


def _interval_sums(
    timestamps: np.ndarray, values: h5py.Dataset, starts: np.ndarray
) -> np.ndarray:
    """Sum time x cell event magnitudes in fixed presentation intervals."""
    first = int(np.searchsorted(timestamps, starts[0], side="left"))
    last = int(np.searchsorted(timestamps, starts[-1] + BIN_SECONDS, side="left"))
    event_data = np.empty((last - first, values.shape[1]), dtype=np.float32)
    values.read_direct(event_data, source_sel=np.s_[first:last, :])

    # Prefix sums make every half-open [start, start+750 ms) aggregation exact
    # with respect to the plane's own ophys timestamps.
    prefix = np.empty((len(event_data) + 1, event_data.shape[1]), dtype=np.float32)
    prefix[0] = 0.0
    np.cumsum(event_data, axis=0, dtype=np.float32, out=prefix[1:])
    left = np.searchsorted(timestamps, starts, side="left") - first
    right = np.searchsorted(timestamps, starts + BIN_SECONDS, side="left") - first
    return prefix[right] - prefix[left]


def _interval_means(
    reference_timestamps: np.ndarray, values_at_reference: np.ndarray, starts: np.ndarray
) -> np.ndarray:
    """Average a finite behavioral trace over presentation intervals."""
    prefix = np.empty(len(values_at_reference) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values_at_reference, out=prefix[1:])
    left = np.searchsorted(reference_timestamps, starts, side="left")
    right = np.searchsorted(
        reference_timestamps, starts + BIN_SECONDS, side="left"
    )
    counts = right - left
    if np.any(counts == 0):
        raise RuntimeError("A presentation interval contains no ophys timestamps")
    return (prefix[right] - prefix[left]) / counts


def _interpolate_finite(
    source_timestamps: np.ndarray,
    source_values: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    """Linearly align a behavioral stream to ophys timestamps.

    Pupil samples marked as blinks are NaN in the AllenSDK/NWB.  Interpolation
    over finite samples supplies categorical labels without treating blink
    frames as real pupil measurements.
    """
    finite = np.isfinite(source_timestamps) & np.isfinite(source_values)
    if finite.sum() < 2:
        raise RuntimeError("Behavioral stream has fewer than two finite samples")
    return np.interp(
        target_timestamps,
        source_timestamps[finite],
        source_values[finite],
    )


def _quintile_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discretize values using within-session 20/40/60/80th percentiles."""
    edges = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    codes = np.searchsorted(edges, values, side="right").astype(np.int16)
    return codes, edges


def convert(data_root: Path, output_path: Path) -> dict:
    release_root = data_root / "visual-behavior-ophys-1.1.0"
    experiment_dir = release_root / "behavior_ophys_experiments"
    metadata_path = release_root / "project_metadata" / "ophys_experiment_table.csv"

    files = {
        _experiment_id(path): path
        for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb"))
    }
    if not files:
        raise FileNotFoundError(f"No experiment NWBs found under {experiment_dir}")

    table = pd.read_csv(metadata_path)
    table = table[table["ophys_experiment_id"].isin(files)].copy()
    # Passive sessions do not contain the requested Go/Catch trial population.
    table = table[~table["passive"].astype(bool)].copy()
    table.sort_values(["ophys_session_id", "ophys_experiment_id"], inplace=True)

    grouped: list[tuple[int, pd.DataFrame]] = []
    excluded: list[dict] = []
    image_names: set[str] = set()
    for session_id, experiments in table.groupby("ophys_session_id", sort=True):
        representative = files[int(experiments.iloc[0]["ophys_experiment_id"])]
        with h5py.File(representative, "r") as nwb:
            pupil_path = "acquisition/EyeTracking/pupil_tracking/width"
            if pupil_path not in nwb:
                excluded.append({
                    "ophys_session_id": int(session_id),
                    "reason": "missing pupil tracking required by decoder output",
                })
                continue
            if np.isfinite(nwb[pupil_path][:]).sum() < 2:
                excluded.append({
                    "ophys_session_id": int(session_id),
                    "reason": "insufficient finite pupil diameter samples",
                })
                continue
            trial_ids, _ = _eligible_trial_ids(nwb)
            if len(trial_ids) < 2:
                excluded.append({
                    "ophys_session_id": int(session_id),
                    "reason": "fewer than two eligible Go/Catch trials",
                })
                continue
            _, _, names, _ = _session_presentations(nwb, trial_ids)
            image_names.update(names.tolist())
        grouped.append((int(session_id), experiments.copy()))

    # Natural image identifiers are stable strings (imXXX); omissions are a
    # genuine task interval and get their own category after the image classes.
    real_images = sorted(name for name in image_names if name != "omitted")
    image_values = real_images + (["omitted"] if "omitted" in image_names else [])
    image_to_code = {name: idx for idx, name in enumerate(image_values)}

    subjects = sorted({str(int(g.iloc[0]["mouse_id"])) for _, g in grouped})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
    brain_regions = sorted({str(region) for _, g in grouped for region in g["targeted_structure"]})
    region_to_idx = {region: idx for idx, region in enumerate(brain_regions)}

    neural: list[list[np.ndarray]] = []
    decoder_input: list[list[np.ndarray]] = []
    output: list[list[np.ndarray]] = []
    subject_idx: list[int] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict] = []

    for number, (session_id, experiments) in enumerate(grouped, start=1):
        experiment_ids = experiments["ophys_experiment_id"].astype(int).tolist()
        representative = files[experiment_ids[0]]
        with h5py.File(representative, "r") as nwb:
            trial_ids, trial_outcomes = _eligible_trial_ids(nwb)
            starts, row_groups, names, changes = _session_presentations(nwb, trial_ids)
            reference_ts = nwb["processing/ophys/event_detection/timestamps"][:]

            running_ts = nwb["processing/running/speed/timestamps"][:]
            running_raw = nwb["processing/running/speed/data"][:]
            running_at_ophys = _interpolate_finite(running_ts, running_raw, reference_ts)
            running_values = _interval_means(reference_ts, running_at_ophys, starts)

            pupil = nwb["acquisition/EyeTracking/pupil_tracking"]
            pupil_at_ophys = _interpolate_finite(
                pupil["timestamps"][:], pupil["width"][:], reference_ts
            )
            pupil_values = _interval_means(reference_ts, pupil_at_ophys, starts)

        running_codes, running_edges = _quintile_codes(running_values)
        pupil_codes, pupil_edges = _quintile_codes(pupil_values)

        plane_activity: list[np.ndarray] = []
        region_codes: list[np.ndarray] = []
        neuron_counts: list[int] = []
        for experiment_id, region in zip(
            experiment_ids, experiments["targeted_structure"].astype(str)
        ):
            with h5py.File(files[experiment_id], "r") as nwb:
                event_detection = nwb["processing/ophys/event_detection"]
                timestamps = event_detection["timestamps"][:]
                aggregated = _interval_sums(timestamps, event_detection["data"], starts)
            plane_activity.append(aggregated)
            neuron_counts.append(aggregated.shape[1])
            region_codes.append(
                np.full(aggregated.shape[1], region_to_idx[region], dtype=np.int64)
            )

        activity = np.concatenate(plane_activity, axis=1)
        image_codes = np.asarray([image_to_code[name] for name in names], dtype=np.int16)

        session_neural: list[np.ndarray] = []
        session_input: list[np.ndarray] = []
        session_output: list[np.ndarray] = []
        for rows, outcome in zip(row_groups, trial_outcomes):
            timepoints = len(rows)
            session_neural.append(np.ascontiguousarray(activity[rows].T, dtype=np.float32))
            session_input.append(np.empty((0, timepoints), dtype=np.float32))
            session_output.append(np.vstack([
                image_codes[rows],
                changes[rows],
                running_codes[rows],
                pupil_codes[rows],
                np.full(timepoints, outcome, dtype=np.int16),
            ]).astype(np.int16, copy=False))

        neural.append(session_neural)
        decoder_input.append(session_input)
        output.append(session_output)
        mouse_id = str(int(experiments.iloc[0]["mouse_id"]))
        subject_idx.append(subject_to_idx[mouse_id])
        brain_region_idx.append(np.concatenate(region_codes))
        session_info.append({
            "ophys_session_id": session_id,
            "ophys_experiment_ids": experiment_ids,
            "behavior_session_id": int(experiments.iloc[0]["behavior_session_id"]),
            "mouse_id": mouse_id,
            "project_code": str(experiments.iloc[0]["project_code"]),
            "session_type": str(experiments.iloc[0]["session_type"]),
            "experience_level": str(experiments.iloc[0]["experience_level"]),
            "n_trials": len(session_neural),
            "n_neurons": int(activity.shape[1]),
            "neurons_per_experiment": neuron_counts,
            "running_quintile_edges_cm_per_s": running_edges.tolist(),
            "pupil_diameter_quintile_edges_pixels": pupil_edges.tolist(),
        })
        print(
            f"[{number:3d}/{len(grouped)}] session {session_id}: "
            f"{len(session_neural)} trials, {activity.shape[1]} neurons, "
            f"{len(experiment_ids)} plane(s)",
            flush=True,
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
            "running_speed_quintile",
            "pupil_diameter_quintile",
            "trial_outcome",
        ],
        "output_values": [
            image_values,
            ["no_change", "change"],
            ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
            ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
            list(OUTCOME_COLUMNS),
        ],
        "metadata": {
            "task_description": (
                "Head-fixed visual change-detection task. Neural activity predicts "
                "image identity, true image changes, running-speed and pupil-diameter "
                "quintiles, and four-way Go/Catch trial outcome."
            ),
            "time_bin_size": BIN_SECONDS * 1000.0,
            "temporal_alignment_event": (
                "Start of each 750 ms image-presentation interval; all streams are "
                "aligned through ophys timestamps."
            ),
            "off_start": None,
            "off_end": None,
            "neural_signal": "AllenSDK inferred calcium-event magnitude, summed per interval",
            "trial_filter": (
                "Go and Catch trials only; aborted and auto-rewarded trials excluded"
            ),
            "trial_segmentation": (
                "Active stimulus presentations carrying each retained NWB trials_id"
            ),
            "behavior_alignment": (
                "Allen processed running speed and blink-filtered pupil ellipse width "
                "linearly interpolated to ophys timestamps, then averaged per interval"
            ),
            "discretization": (
                "Running speed and pupil diameter use within-session 20th, 40th, "
                "60th, and 80th percentile boundaries over retained intervals"
            ),
            "image_omissions": (
                "Omissions are labeled as their own image-identity category and occupy "
                "the same 750 ms interval used in the reference analysis"
            ),
            "session_grouping": (
                "Experiments sharing an ophys_session_id are simultaneous planes and "
                "are concatenated as neurons in one recording session"
            ),
            "excluded_sessions": excluded,
            "session_info": session_info,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, output_path)
    print(f"Saved {output_path} ({output_path.stat().st_size / 1e9:.3f} GB)")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/app/data"))
    parser.add_argument("--output", type=Path, default=Path("/app/converted_data.pkl"))
    args = parser.parse_args()
    convert(args.data_root, args.output)


if __name__ == "__main__":
    main()
