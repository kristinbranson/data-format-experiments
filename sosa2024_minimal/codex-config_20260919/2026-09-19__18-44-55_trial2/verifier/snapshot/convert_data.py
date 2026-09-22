#!/usr/bin/env python3
"""Convert the Sosa et al. (2025) NWB release for neural decoding.

The conversion follows the paper's calcium-processing methods: manually curated
Suite2p ROIs are neuropil corrected, converted to trial-wise maximin dF/F,
smoothed, deconvolved with OASIS, and screened for putative interneurons.  Only
the on-track interval from each trial-start event up to (but not including) its
teleport event is retained.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
from suite2p.extraction import dcnv


BEHAVIOR = "processing/behavior/BehavioralTimeSeries"
OPHYS = "processing/ophys"
ZONE_STARTS = np.array([80.0, 200.0, 320.0], dtype=np.float32)
ZONE_ENDS = ZONE_STARTS + 50.0
ZONE_CENTERS = (ZONE_STARTS + ZONE_ENDS) / 2.0
SWITCH_DAYS = {3, 5, 7, 8, 10, 12, 14}
NOMINAL_RATE_HZ = 15.5078125


def _decode_scalar(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _natural_key(path: Path):
    subject = re.search(r"sub-m(\d+)", str(path))
    session = re.search(r"ses-(\d+)", path.name)
    return (int(subject.group(1)), int(session.group(1)))


def _read_behavior(group, name: str) -> np.ndarray:
    return np.asarray(group[f"{name}/data"])


def _trial_bounds(behavior) -> tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(_read_behavior(behavior, "trial_start") > 0)
    stops = np.flatnonzero(_read_behavior(behavior, "teleport") > 0)
    if len(starts) != len(stops):
        raise ValueError(f"Unmatched trial starts ({len(starts)}) and teleports ({len(stops)})")
    if np.any(stops <= starts):
        raise ValueError("A teleport did not follow its corresponding trial start")
    return starts, stops


def _sampling_rate(behavior) -> float:
    """Use aligned timestamps, which correctly account for two-plane sessions."""
    timestamps = np.asarray(behavior["position/timestamps"])
    rate = float(1.0 / np.median(np.diff(timestamps)))
    if not np.isclose(rate, NOMINAL_RATE_HZ, rtol=0, atol=1e-4):
        raise ValueError(f"Unexpected aligned sampling rate: {rate} Hz")
    return rate


def _trial_outcomes(reward_timestamps: np.ndarray, timestamps: np.ndarray,
                    starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        left = np.searchsorted(reward_timestamps, timestamps[start], side="left")
        right = np.searchsorted(reward_timestamps, timestamps[stop], side="left")
        outcomes[trial] = right > left
    return outcomes


def _reward_zone_labels(position: np.ndarray, rzone: np.ndarray,
                        starts: np.ndarray, stops: np.ndarray,
                        experiment_day: int) -> np.ndarray:
    """Recover the active zone robustly from the aligned rzone event stream.

    The aligned rzone channel is an integrated event/count channel, not a zone
    label.  Its nonzero samples occur inside the active zone.  The experimental
    protocol has one condition for the session, or two conditions split after
    trial 30 on switch days.  Taking the modal observed zone in each condition
    also fills reward-omission trials where the event channel can be all zero.
    """
    observed = np.full(len(starts), -1, dtype=np.int8)
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        mask = rzone[start:stop] > 0
        if np.any(mask):
            event_position = float(np.median(position[start:stop][mask]))
            observed[trial] = int(np.argmin(np.abs(ZONE_CENTERS - event_position)))

    boundaries = [0, min(30, len(starts)), len(starts)] if experiment_day in SWITCH_DAYS else [0, len(starts)]
    labels = np.empty(len(starts), dtype=np.int8)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        valid = observed[left:right]
        valid = valid[valid >= 0]
        if len(valid) == 0:
            raise ValueError(f"Cannot infer reward zone for trials {left}:{right}")
        labels[left:right] = np.bincount(valid, minlength=3).argmax()
    return labels


def _bad_lick_trials(lick: np.ndarray, starts: np.ndarray,
                     stops: np.ndarray) -> np.ndarray:
    """Paper criterion: >30% of frame samples have cumulative lick count >2."""
    return np.asarray([
        np.mean(lick[start:stop] > 2) > 0.30
        for start, stop in zip(starts, stops)
    ], dtype=bool)


def _dff_and_events(fluorescence: np.ndarray, neuropil: np.ndarray,
                    rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Paper's single-trial maximin dF/F and OASIS processing."""
    # Undo neuropil contamination while adding its within-trial mean back, as
    # in reward_relative.preprocessing.dff (neu_coef=0.7).
    corrected = fluorescence - 0.7 * neuropil
    corrected += 0.7 * neuropil.mean(axis=1, keepdims=True)

    baseline = gaussian_filter1d(corrected, 15, axis=1)
    baseline = minimum_filter1d(baseline, 300, axis=1)
    baseline = maximum_filter1d(baseline, 300, axis=1)
    dff = (corrected - baseline) / np.abs(baseline)
    dff = gaussian_filter1d(dff, 2, axis=1).astype(np.float32, copy=False)
    events = dcnv.oasis(dff, 2000, 0.7, rate_hz)
    return dff, events


def _distance_class(position: np.ndarray, zone: int) -> np.ndarray:
    start, end = ZONE_STARTS[zone], ZONE_ENDS[zone]
    distance = np.where(position < start, position - start,
                        np.where(position > end, position - end, 0.0))
    return np.select(
        [distance < -50, distance < -10, distance < 0, distance == 0,
         distance <= 10, distance <= 50],
        [0, 1, 2, 3, 4, 5], default=6,
    ).astype(np.int8)


def _output_matrix(position: np.ndarray, speed: np.ndarray, lick: np.ndarray,
                   zone: int, rewarded: int) -> np.ndarray:
    absolute_position = np.digitize(position, [90, 180, 270, 360]).astype(np.int8)
    speed_class = np.digitize(speed, [2, 10, 20, 40]).astype(np.int8)
    lick_binary = (lick > 0).astype(np.int8)
    timepoints = len(position)
    return np.vstack([
        _distance_class(position, zone),
        absolute_position,
        speed_class,
        lick_binary,
        np.full(timepoints, zone, dtype=np.int8),
        np.full(timepoints, rewarded, dtype=np.int8),
    ])


def convert_session(path: Path) -> tuple[dict, dict]:
    with h5py.File(path, "r") as nwb:
        behavior = nwb[BEHAVIOR]
        starts, stops = _trial_bounds(behavior)
        rate_hz = _sampling_rate(behavior)
        timestamps = np.asarray(behavior["position/timestamps"])
        position = _read_behavior(behavior, "position")
        speed = _read_behavior(behavior, "speed")
        lick = _read_behavior(behavior, "lick")
        environment = _read_behavior(behavior, "environment")
        trial_number = _read_behavior(behavior, "trial number")
        rzone = _read_behavior(behavior, "reward_zone")

        subject = _decode_scalar(nwb["general/subject/subject_id"][()])
        experiment_day = int(_decode_scalar(nwb["general/session_id"][()]))
        n_planes = len(np.unique(nwb[f"{OPHYS}/ImageSegmentation/PlaneSegmentation/planeIdx"][:]))
        is_cell = nwb[f"{OPHYS}/ImageSegmentation/PlaneSegmentation/iscell"][:, 0].astype(bool)
        roi_indices = np.flatnonzero(is_cell)
        if len(roi_indices) == 0:
            raise ValueError("Session has no manually curated cells")

        outcomes = _trial_outcomes(
            np.asarray(behavior["Reward/timestamps"]), timestamps, starts, stops
        )
        zones = _reward_zone_labels(position, rzone, starts, stops, experiment_day)
        bad_lick = _bad_lick_trials(lick, starts, stops)

        # Single-plane sessions have one RoiResponseSeries; m17/m18 have one
        # series per plane and a shared, concatenated ROI table.  Resolve each
        # series' DynamicTableRegion rather than assuming table indices are
        # response-matrix column indices.
        plane_specs = []
        fluorescence_group = nwb[f"{OPHYS}/Fluorescence"]
        neuropil_group = nwb[f"{OPHYS}/Neuropil"]
        for plane_name in sorted(fluorescence_group.keys()):
            table_rows = np.asarray(fluorescence_group[f"{plane_name}/rois"], dtype=np.int64)
            local_keep = np.flatnonzero(is_cell[table_rows])
            if len(local_keep):
                plane_specs.append((
                    fluorescence_group[f"{plane_name}/data"],
                    neuropil_group[f"{plane_name}/data"],
                    local_keep,
                ))

        events_by_trial: list[np.ndarray] = []
        sum_x = np.zeros(len(roi_indices), dtype=np.float64)
        sum_x2 = np.zeros(len(roi_indices), dtype=np.float64)
        sum_xy = np.zeros(len(roi_indices), dtype=np.float64)
        sum_y = 0.0
        sum_y2 = 0.0
        sample_count = 0

        # Process all trials for the paper's session-wide speed-correlation
        # interneuron screen, including trials whose lick sensor was faulty.
        for start, stop in zip(starts, stops):
            fluorescence = np.concatenate([
                np.asarray(f_data[start:stop, :], dtype=np.float32)[:, local_keep].T
                for f_data, _, local_keep in plane_specs
            ], axis=0)
            neuropil = np.concatenate([
                np.asarray(fneu_data[start:stop, :], dtype=np.float32)[:, local_keep].T
                for _, fneu_data, local_keep in plane_specs
            ], axis=0)
            dff, events = _dff_and_events(fluorescence, neuropil, rate_hz)
            if not np.all(np.isfinite(dff)) or not np.all(np.isfinite(events)):
                raise ValueError("Non-finite neural activity after calcium processing")
            events_by_trial.append(events)

            trial_speed = speed[start:stop].astype(np.float64, copy=False)
            dff64 = dff.astype(np.float64, copy=False)
            sum_x += dff64.sum(axis=1)
            sum_x2 += np.square(dff64).sum(axis=1)
            sum_xy += dff64 @ trial_speed
            sum_y += trial_speed.sum()
            sum_y2 += np.square(trial_speed).sum()
            sample_count += len(trial_speed)

    numerator = sample_count * sum_xy - sum_x * sum_y
    denominator = np.sqrt(
        (sample_count * sum_x2 - np.square(sum_x))
        * (sample_count * sum_y2 - sum_y * sum_y)
    )
    speed_correlation = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    putative_interneuron = speed_correlation > 0.5
    keep_neuron = ~putative_interneuron
    if not np.any(keep_neuron):
        raise ValueError("Interneuron screen removed every cell")

    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []
    retained_source_trials: list[int] = []
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        if bad_lick[trial]:
            continue
        timepoints = stop - start
        trial_env = int(np.rint(np.median(environment[start:stop])))
        source_trial_number = float(np.median(trial_number[start:stop]))
        previous_outcome = int(outcomes[trial - 1]) if trial > 0 else 0
        input_trial = np.vstack([
            np.arange(timepoints, dtype=np.float32) / np.float32(rate_hz),
            np.full(timepoints, trial_env, dtype=np.float32),
            np.full(timepoints, source_trial_number, dtype=np.float32),
            np.full(timepoints, previous_outcome, dtype=np.float32),
        ])
        output_trial = _output_matrix(
            position[start:stop], speed[start:stop], lick[start:stop],
            int(zones[trial]), int(outcomes[trial]),
        )
        neural_trials.append(np.ascontiguousarray(events_by_trial[trial][keep_neuron]))
        input_trials.append(input_trial)
        output_trials.append(output_trial)
        retained_source_trials.append(trial)

    session_info = {
        "source_file": str(path.resolve()),
        "subject": subject,
        "experiment_day": experiment_day,
        "sampling_rate_hz": rate_hz,
        "n_imaging_planes": n_planes,
        "source_trial_count": len(starts),
        "retained_trial_count": len(neural_trials),
        "excluded_faulty_lick_trials": np.flatnonzero(bad_lick).astype(int).tolist(),
        "retained_source_trials": retained_source_trials,
        "manually_curated_roi_count": int(len(roi_indices)),
        "excluded_putative_interneuron_count": int(putative_interneuron.sum()),
        "retained_neuron_count": int(keep_neuron.sum()),
        "reward_zone_by_source_trial": zones.astype(int).tolist(),
        "reward_outcome_by_source_trial": outcomes.astype(int).tolist(),
    }
    arrays = {
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "brain_region_idx": np.zeros(int(keep_neuron.sum()), dtype=np.int64),
    }
    return arrays, session_info


def build_dataset(data_root: Path, max_sessions: int | None = None) -> dict:
    files = sorted(data_root.glob("sub-*/*_behavior+ophys.nwb"), key=_natural_key)
    if max_sessions is not None:
        files = files[:max_sessions]
    if not files:
        raise FileNotFoundError(f"No NWB files found below {data_root}")

    neural, inputs, outputs, regions, session_info = [], [], [], [], []
    session_subjects: list[str] = []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path}", flush=True)
        arrays, info = convert_session(path)
        neural.append(arrays["neural"])
        inputs.append(arrays["input"])
        outputs.append(arrays["output"])
        regions.append(arrays["brain_region_idx"])
        session_info.append(info)
        session_subjects.append(info["subject"])

    subjects = sorted(set(session_subjects), key=lambda value: int(re.search(r"\d+", value).group()))
    subject_lookup = {subject: index for index, subject in enumerate(subjects)}
    subject_idx = np.asarray([subject_lookup[s] for s in session_subjects], dtype=np.int64)

    return {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": subject_idx,
        "brain_regions": ["CA1"],
        "brain_region_idx": regions,
        "input_names": [
            "time from trial start (s)",
            "environment (ENV1=0, ENV2=1)",
            "trial number",
            "previous trial rewarded",
        ],
        "output_names": [
            "signed distance to reward zone",
            "absolute corridor position",
            "speed",
            "lick",
            "reward zone location",
            "reward outcome",
        ],
        "output_values": [
            ["< -50 cm", "-50 to -10 cm", "-10 to <0 cm", "in reward zone (0 cm)",
             ">0 to +10 cm", "+10 to +50 cm", "> +50 cm"],
            ["<90 cm", "90-180 cm", "180-270 cm", "270-360 cm", ">360 cm"],
            ["<2 cm/s", "2-10 cm/s", "10-20 cm/s", "20-40 cm/s", ">40 cm/s"],
            ["no lick", "lick"],
            ["A (80-130 cm)", "B (200-250 cm)", "C (320-370 cm)"],
            ["not rewarded", "rewarded"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed mice traversed a 450 cm virtual corridor in ENV1 or ENV2 and "
                "licked in one of three hidden reward zones; reward location changed after "
                "trial 30 on switch days and reward was omitted on approximately 15% of trials."
            ),
            "time_bin_size": 1000.0 / NOMINAL_RATE_HZ,
            "temporal_alignment_event": "start of trial (entry into the 450 cm corridor)",
            "off_start": 0.0,
            "off_end": None,
            "trial_end_event": "teleport onset; trial end is variable relative to trial start",
            "neural_signal": "OASIS-deconvolved, trial-wise maximin dF/F",
            "neural_processing": (
                "Manual Suite2p cell curation; 0.7 neuropil subtraction with trial mean "
                "restoration; 20 s (300-sample) maximin baseline; two-sample Gaussian "
                "smoothing; OASIS tau=0.7 s; exclude dF/F-speed Pearson r > 0.5."
            ),
            "trial_filtering": (
                "Complete start-to-teleport trials retained except paper-identified faulty lick "
                "trials (>30% of samples with cumulative lick count >2)."
            ),
            "reward_zone_intervals_cm": {"A": [80, 130], "B": [200, 250], "C": [320, 370]},
            "session_info": session_info,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/app/data"))
    parser.add_argument("--output", type=Path, default=Path("/app/converted_data.pkl"))
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="Process only the first N sessions (for smoke tests).")
    args = parser.parse_args()

    data = build_dataset(args.data_root, args.max_sessions)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, args.output)
    print(f"Saved {len(data['neural'])} sessions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
