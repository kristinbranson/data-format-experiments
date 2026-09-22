#!/usr/bin/env python3
"""Convert the Sosa, Plitt & Giocomo NWB release for neural decoding.

The conversion deliberately follows the paper's fluorescence processing rather
than using the NWB ``Deconvolved`` series.  The latter contains Suite2P's
deconvolution of the unnormalised fluorescence, whereas the paper first computes
a trial-local maximin dF/F trace.  See ``methods.txt`` and
``code/src/reward_relative/preprocessing.py::dff``.

Decisions which are specific to this decoder are documented in ``metadata`` in
the resulting pickle.  Run from /app with::

    python convert_data.py
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d


FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
TRACK_END_CM = 450.0
ZONE_BOUNDS = np.asarray([[80.0, 130.0], [200.0, 250.0], [320.0, 370.0]])
ZONE_CENTERS = ZONE_BOUNDS.mean(axis=1)
SWITCH_TRIAL = 30  # zero-based: trials 0--29 are the initial condition
NEUROPIL_COEFFICIENT = 0.7
INTERNEURON_R_THRESHOLD = 0.5
LICK_ERROR_FRACTION = 0.30


def natural_key(path: Path) -> tuple[int, int]:
    """Sort files by numeric mouse and session identifiers."""
    match = re.search(r"sub-m(\d+).*ses-(\d+)", str(path))
    if match is None:
        raise ValueError(f"Cannot parse subject/session from {path}")
    return int(match.group(1)), int(match.group(2))


def read_behavior(group: h5py.Group, name: str) -> np.ndarray:
    return np.asarray(group[name]["data"])


def paper_dff_trial(
    fluorescence: h5py.Group,
    neuropil: h5py.Group,
    cell_indices_by_plane: list[np.ndarray],
    start: int,
    stop: int,
) -> np.ndarray:
    """Reproduce the paper's dF/F processing for one [start, stop) lap.

    NWB arrays are time x ROI; the returned array is cell x time.  Processing
    matches reward_relative.preprocessing.dff: subtract 0.7 neuropil, restore
    each trial's neuropil mean, smooth with sigma=15 samples, min then max filter
    with a 300-sample (~20 s) window, normalise, and smooth dF/F with sigma=2.
    """
    # m17 and m18 were acquired in two interleaved planes and store one response
    # series per plane. Other mice have only plane0. Reading only manually
    # curated cells substantially reduces both I/O and memory.
    f_parts = []
    fneu_parts = []
    for plane, local_indices in enumerate(cell_indices_by_plane):
        if local_indices.size == 0:
            continue
        f_parts.append(
            np.asarray(
                fluorescence[f"plane{plane}"]["data"][start:stop, local_indices],
                dtype=np.float32,
            ).T
        )
        fneu_parts.append(
            np.asarray(
                neuropil[f"plane{plane}"]["data"][start:stop, local_indices],
                dtype=np.float32,
            ).T
        )
    f = np.concatenate(f_parts, axis=0)
    fneu = np.concatenate(fneu_parts, axis=0)

    corrected = f - NEUROPIL_COEFFICIENT * fneu
    corrected += NEUROPIL_COEFFICIENT * np.mean(fneu, axis=1, keepdims=True)
    baseline = gaussian_filter1d(corrected, 15, axis=1)
    baseline = minimum_filter1d(baseline, 300, axis=1)
    baseline = maximum_filter1d(baseline, 300, axis=1)

    # A zero baseline is not expected for fluorescence.  The guard prevents an
    # invalid decoder file if a malformed ROI nevertheless reaches this point.
    denominator = np.abs(baseline)
    denominator[denominator == 0] = np.finfo(np.float32).eps
    dff = (corrected - baseline) / denominator
    dff = gaussian_filter1d(dff, 2, axis=1)
    return np.asarray(dff, dtype=np.float32)


def correlations_from_sums(
    sum_x: np.ndarray,
    sum_x2: np.ndarray,
    sum_xy: np.ndarray,
    sum_y: float,
    sum_y2: float,
    n: int,
) -> np.ndarray:
    numerator = n * sum_xy - sum_x * sum_y
    denom_x = n * sum_x2 - sum_x * sum_x
    denom_y = n * sum_y2 - sum_y * sum_y
    denominator = np.sqrt(np.maximum(denom_x * denom_y, 0.0))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def infer_zone_by_trial(
    timestamps: np.ndarray,
    position: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    reward_timestamps: np.ndarray,
) -> tuple[np.ndarray, list[int], list[float]]:
    """Infer the active A/B/C zone for each trial from reward locations.

    The task uses one condition for trials 0--29 and another condition from
    trial 30 onward on switch days.  Inferring one label per block makes omitted
    trials well-defined and is more robust than assigning a zone independently
    from the sparse delivery events.  On stay days both blocks infer the same
    label.  Reward positions are linearly interpolated at exact delivery times.
    """
    n_trials = len(starts)
    labels = np.empty(n_trials, dtype=np.int8)
    block_labels: list[int] = []
    block_medians: list[float] = []
    boundaries = [(0, min(SWITCH_TRIAL, n_trials)), (min(SWITCH_TRIAL, n_trials), n_trials)]

    for first, last in boundaries:
        if first == last:
            continue
        lo = timestamps[starts[first]]
        hi = timestamps[stops[last - 1]]
        block_rewards = reward_timestamps[(reward_timestamps >= lo) & (reward_timestamps <= hi)]
        if block_rewards.size == 0:
            raise ValueError(f"No reward deliveries in trial block [{first}, {last})")
        reward_positions = np.interp(block_rewards, timestamps, position)
        median_position = float(np.median(reward_positions))
        label = int(np.argmin(np.abs(ZONE_CENTERS - median_position)))
        labels[first:last] = label
        block_labels.append(label)
        block_medians.append(median_position)

    return labels, block_labels, block_medians


def reward_outcomes(
    timestamps: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    reward_timestamps: np.ndarray,
) -> np.ndarray:
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        outcomes[trial] = np.any(
            (reward_timestamps >= timestamps[start])
            & (reward_timestamps <= timestamps[stop])
        )
    return outcomes


def distance_classes(position: np.ndarray, zone: int) -> np.ndarray:
    """Bin signed distance to the closest point in the active reward zone."""
    zone_start, zone_stop = ZONE_BOUNDS[zone]
    distance = np.where(
        position < zone_start,
        position - zone_start,
        np.where(position > zone_stop, position - zone_stop, 0.0),
    )
    result = np.full(position.shape, 3, dtype=np.int8)  # exactly/in the zone
    result[distance < -50.0] = 0
    result[(distance >= -50.0) & (distance < -10.0)] = 1
    result[(distance >= -10.0) & (distance < 0.0)] = 2
    result[(distance > 0.0) & (distance <= 10.0)] = 4
    result[(distance > 10.0) & (distance <= 50.0)] = 5
    result[distance > 50.0] = 6
    return result


def convert_session(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict]:
    mouse_number, session_number = natural_key(path)
    print(f"Converting m{mouse_number} session {session_number:02d}: {path.name}", flush=True)

    with h5py.File(path, "r") as nwb:
        behavior = nwb["processing/behavior/BehavioralTimeSeries"]
        position = read_behavior(behavior, "position").astype(np.float64, copy=False)
        speed = read_behavior(behavior, "speed").astype(np.float64, copy=False)
        lick = read_behavior(behavior, "lick")
        environment = read_behavior(behavior, "environment")
        trial_number_stream = read_behavior(behavior, "trial number")
        timestamps = np.asarray(behavior["position"]["timestamps"], dtype=np.float64)
        reward_timestamps = np.asarray(behavior["Reward"]["timestamps"], dtype=np.float64)
        starts = np.flatnonzero(read_behavior(behavior, "trial_start") > 0)
        stops = np.flatnonzero(read_behavior(behavior, "teleport") > 0)

        if len(starts) != len(stops) or np.any(stops <= starts):
            raise ValueError(f"Invalid start/teleport pairing in {path}")
        frame_intervals = np.diff(timestamps)
        if not np.isclose(np.median(frame_intervals), 1.0 / FRAME_RATE_HZ, rtol=0, atol=1e-6):
            raise ValueError(f"Unexpected behavior sampling interval in {path}")

        zones, block_zone_labels, block_reward_medians = infer_zone_by_trial(
            timestamps, position, starts, stops, reward_timestamps
        )
        outcomes = reward_outcomes(timestamps, starts, stops, reward_timestamps)
        previous_outcomes = np.r_[0, outcomes[:-1]].astype(np.float32)

        segmentation = nwb["processing/ophys/ImageSegmentation/PlaneSegmentation"]
        iscell = np.asarray(segmentation["iscell"][:, 0]) > 0
        plane_index = np.asarray(segmentation["planeIdx"], dtype=np.int64)
        plane_values = np.unique(plane_index)
        if not np.array_equal(plane_values, np.arange(len(plane_values))):
            raise ValueError(f"Non-contiguous imaging plane labels in {path}: {plane_values}")
        cell_indices_by_plane = [
            np.flatnonzero(iscell[plane_index == plane]) for plane in plane_values
        ]
        cell_indices = np.flatnonzero(iscell)
        fluorescence = nwb["processing/ophys/Fluorescence"]
        neuropil = nwb["processing/ophys/Neuropil"]

        neural_trials: list[np.ndarray] = []
        input_trials: list[np.ndarray] = []
        output_trials: list[np.ndarray] = []
        kept_source_trials: list[int] = []
        dropped_lick_trials: list[int] = []

        # Streaming sufficient statistics reproduce np.corrcoef(dff, speed)
        # without materialising a second session-wide dF/F matrix.
        n_cells = len(cell_indices)
        sum_x = np.zeros(n_cells, dtype=np.float64)
        sum_x2 = np.zeros(n_cells, dtype=np.float64)
        sum_xy = np.zeros(n_cells, dtype=np.float64)
        sum_y = 0.0
        sum_y2 = 0.0
        n_samples = 0

        for trial, (start, stop) in enumerate(zip(starts, stops)):
            # The alignment marker is included as t=0; the teleport marker is
            # excluded, leaving only the 450 cm virtual corridor.
            trial_slice = slice(int(start), int(stop))
            trial_lick = lick[trial_slice]
            if np.mean(trial_lick > 2) > LICK_ERROR_FRACTION:
                dropped_lick_trials.append(trial)
                continue

            dff = paper_dff_trial(
                fluorescence, neuropil, cell_indices_by_plane, int(start), int(stop)
            )
            trial_speed = speed[trial_slice]
            sum_x += np.sum(dff, axis=1, dtype=np.float64)
            sum_x2 += np.sum(dff * dff, axis=1, dtype=np.float64)
            sum_xy += np.asarray(dff, dtype=np.float64) @ trial_speed
            sum_y += float(np.sum(trial_speed, dtype=np.float64))
            sum_y2 += float(np.sum(trial_speed * trial_speed, dtype=np.float64))
            n_samples += dff.shape[1]

            trial_position = position[trial_slice]
            time_from_start = timestamps[trial_slice] - timestamps[start]
            env_values = environment[trial_slice]
            valid_env = env_values[env_values >= 0]
            if valid_env.size == 0:
                raise ValueError(f"No environment label in {path}, trial {trial}")
            env = float(np.rint(np.median(valid_env)))
            source_trial_number = float(np.rint(np.median(trial_number_stream[trial_slice])))
            if source_trial_number != trial:
                raise ValueError(
                    f"Unexpected trial number {source_trial_number} for marker {trial} in {path}"
                )

            inputs = np.vstack(
                [
                    time_from_start,
                    np.full(dff.shape[1], env),
                    np.full(dff.shape[1], source_trial_number),
                    np.full(dff.shape[1], previous_outcomes[trial]),
                ]
            ).astype(np.float32)

            outputs = np.vstack(
                [
                    distance_classes(trial_position, int(zones[trial])),
                    np.digitize(trial_position, [90.0, 180.0, 270.0, 360.0]).astype(np.int8),
                    np.digitize(trial_speed, [2.0, 10.0, 20.0, 40.0]).astype(np.int8),
                    (trial_lick > 0).astype(np.int8),
                    np.full(dff.shape[1], zones[trial], dtype=np.int8),
                    np.full(dff.shape[1], outcomes[trial], dtype=np.int8),
                ]
            )

            neural_trials.append(dff)
            input_trials.append(inputs)
            output_trials.append(outputs)
            kept_source_trials.append(trial)

    speed_correlations = correlations_from_sums(
        sum_x, sum_x2, sum_xy, sum_y, sum_y2, n_samples
    )
    keep_neurons = speed_correlations <= INTERNEURON_R_THRESHOLD
    neural_trials = [trial[keep_neurons] for trial in neural_trials]

    info = {
        "source_file": str(path.relative_to(path.parents[2])),
        "subject": f"m{mouse_number}",
        "session": session_number,
        "source_trial_count": int(len(starts)),
        "kept_source_trials": kept_source_trials,
        "dropped_corrupt_lick_trials": dropped_lick_trials,
        "suite2p_roi_count": int(len(iscell)),
        "manually_curated_cell_count": int(len(cell_indices)),
        "putative_interneurons_excluded": int(np.count_nonzero(~keep_neurons)),
        "final_neuron_count": int(np.count_nonzero(keep_neurons)),
        "reward_zone_by_block": block_zone_labels,
        "median_reward_position_cm_by_block": block_reward_medians,
    }
    return neural_trials, input_trials, output_trials, info


def convert(data_dir: Path) -> dict:
    paths = sorted(data_dir.glob("sub-*/*_behavior+ophys.nwb"), key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No NWB files found below {data_dir}")

    subject_names = [f"m{mouse}" for mouse in sorted({natural_key(path)[0] for path in paths})]
    subject_lookup = {subject: index for index, subject in enumerate(subject_names)}
    neural: list[list[np.ndarray]] = []
    inputs: list[list[np.ndarray]] = []
    outputs: list[list[np.ndarray]] = []
    subject_idx: list[int] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict] = []

    for path in paths:
        neural_session, input_session, output_session, info = convert_session(path)
        if len(neural_session) < 2:
            raise ValueError(f"Fewer than two usable trials in {path}")
        neural.append(neural_session)
        inputs.append(input_session)
        outputs.append(output_session)
        subject_idx.append(subject_lookup[info["subject"]])
        brain_region_idx.append(np.zeros(info["final_neuron_count"], dtype=np.int8))
        session_info.append(info)

    return {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subject_names,
        "subject_idx": np.asarray(subject_idx, dtype=np.int16),
        "brain_regions": ["CA1"],
        "brain_region_idx": brain_region_idx,
        "input_names": [
            "time from trial start (s)",
            "environment type (ENV1=0, ENV2=1)",
            "trial number (zero-based)",
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
            ["< -50 cm", "-50 to < -10 cm", "-10 to < 0 cm", "in reward zone (0 cm)", "> 0 to 10 cm", "> 10 to 50 cm", "> 50 cm"],
            ["< 90 cm", "90 to < 180 cm", "180 to < 270 cm", "270 to < 360 cm", ">= 360 cm"],
            ["< 2 cm/s", "2 to < 10 cm/s", "10 to < 20 cm/s", "20 to < 40 cm/s", ">= 40 cm/s"],
            ["no lick", "lick"],
            ["A (80-130 cm)", "B (200-250 cm)", "C (320-370 cm)"],
            ["not rewarded", "rewarded"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed mice navigated a 450 cm virtual corridor and licked in one "
                "of three hidden 50 cm reward zones. The active zone changed across days "
                "and after trial 29 in switch sessions; rewards were randomly omitted on "
                "approximately 15% of trials."
            ),
            "time_bin_size": TIME_BIN_MS,
            "temporal_alignment_event": "entry into the virtual corridor (trial-start marker)",
            "off_start": 0.0,
            "off_end": None,
            "trial_end_event": "teleport onset (excluded from each trial)",
            "track_length_cm": TRACK_END_CM,
            "sampling_rate_hz": FRAME_RATE_HZ,
            "neural_signal": "trial-wise neuropil-corrected, maximin-baselined, Gaussian-smoothed dF/F",
            "neural_processing": {
                "source_rois": "Suite2P ROIs retained by the manually curated iscell mask",
                "neuropil_coefficient": NEUROPIL_COEFFICIENT,
                "baseline": "per-trial maximin; sigma 15 sample smoothing then 300-sample minimum and maximum filters",
                "dff_smoothing_sigma_samples": 2,
                "putative_interneuron_exclusion": "Pearson r(dF/F, speed) > 0.5 over retained corridor samples",
            },
            "trial_filtering": (
                "Trials with lick count >2 in more than 30% of corridor frames were "
                "dropped because the paper identifies these as lick-circuit errors. No "
                "running-speed filter was applied because stopped/slow frames are a "
                "requested speed class."
            ),
            "distance_definition": (
                "Signed linear distance to the nearest point in the active reward zone: "
                "negative before the zone, zero throughout the zone, positive after it."
            ),
            "first_trial_previous_outcome": (
                "Set to 0 because the outcome of the pre-imaging warm-up trial is unavailable."
            ),
            "reward_zone_inference": (
                "Nearest of the known A/B/C zone centers to the median actual reward "
                "delivery position, independently for trials 0-29 and trials >=30. "
                "Omission trials inherit the active block label."
            ),
            "session_info": session_info,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--output", type=Path, default=Path("/app/converted_data.pkl"))
    args = parser.parse_args()

    converted = convert(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(converted, stream, protocol=pickle.HIGHEST_PROTOCOL)
    size_gb = args.output.stat().st_size / 1e9
    print(f"Saved {args.output} ({size_gb:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
