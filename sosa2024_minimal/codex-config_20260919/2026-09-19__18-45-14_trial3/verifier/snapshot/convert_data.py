#!/usr/bin/env python3
"""Convert the Sosa, Plitt & Giocomo NWB release for neural decoding.

Decisions tied to the paper/repository
--------------------------------------
* A session is one NWB recording (one mouse/day); every available session is used.
* Only manually curated Suite2p ROIs (``iscell[:, 0] == 1``) are retained.
* Neural activity is recomputed from ROI and neuropil fluorescence with the paper's
  pipeline: 0.7 neuropil subtraction, trial-local 20 s maximin baseline, dF/F,
  Gaussian smoothing (2 imaging samples), and Suite2p OASIS deconvolution with a
  0.7 s calcium time constant. Putative interneurons whose trial-period dF/F has
  Pearson r > 0.5 with speed are then removed, as described in Methods.
* Trials run from each ``trial_start`` sample up to (not including) its matching
  ``teleport`` sample. Teleport periods are not included, matching the main paper
  analyses. Native aligned samples are retained (~15.5 Hz, 64.48 ms); trials may
  consequently have different durations.
* The Methods report 81 lick-detector failures as trials on which >30% of samples
  had cumulative lick count >2. Those trials are excluded because lick is a
  required categorical decoder target and NaN labels cannot be trained on.
* Reward outcome requires both a reward delivery timestamp and an active reward-
  zone sample, matching behavior.get_trial_types. Previous outcome refers to the
  immediately preceding original trial (not merely the preceding retained trial).
* Signed distance to the reward zone is negative before the zone, zero anywhere
  inside it, and positive after it (distance to the nearest zone boundary).

The dual-plane mice have scanner metadata at 62.03 Hz and per-plane series metadata
at 31.02 Hz, but their arrays and aligned behavior have one sample per plane frame.
The paper states the effective sampling rate is ~15.5 Hz per plane, which also
matches the behavioral timestamp spacing. Thus all sessions share the same native
time-bin duration.
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


FRAME_RATE_HZ = 15.5078125
NEUROPIL_COEF = 0.7
CALCIUM_TAU_S = 0.7
BASELINE_WINDOW_SAMPLES = 300  # paper/code: 20 s at approximately 15 Hz
LICK_ERROR_FRACTION = 0.30
ZONE_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}


def natural_key(path: Path) -> tuple[int, int]:
    """Sort files by numeric mouse and session identifiers."""
    match = re.search(r"sub-m(\d+).*ses-(\d+)", str(path))
    if match is None:
        raise ValueError(f"Cannot parse mouse/session from {path}")
    return int(match.group(1)), int(match.group(2))


def scene_conditions(scene: str, n_trials: int) -> tuple[np.ndarray, np.ndarray]:
    """Return environment (0/1) and reward-zone (A/B/C) for every trial."""
    static = re.fullmatch(r"Env([12])_Location([ABC])", scene)
    same_env_switch = re.fullmatch(r"Env([12])_Location([ABC])_to_([ABC])", scene)
    env_switch = re.fullmatch(r"Env([12])_([ABC])_to_Env([12])_([ABC])", scene)

    if static:
        environments = np.full(n_trials, int(static.group(1)) - 1, dtype=np.int8)
        zones = np.full(n_trials, static.group(2), dtype="U1")
    elif same_env_switch:
        environments = np.full(n_trials, int(same_env_switch.group(1)) - 1, dtype=np.int8)
        zones = np.where(np.arange(n_trials) < 30,
                         same_env_switch.group(2), same_env_switch.group(3))
    elif env_switch:
        environments = np.where(np.arange(n_trials) < 30,
                                int(env_switch.group(1)) - 1,
                                int(env_switch.group(3)) - 1).astype(np.int8)
        zones = np.where(np.arange(n_trials) < 30,
                         env_switch.group(2), env_switch.group(4))
    else:
        raise ValueError(f"Unrecognized scene name: {scene}")
    return environments, zones


def discretize_distance(position: np.ndarray, start: float, stop: float) -> np.ndarray:
    """Categorize signed distance to the nearest point in an interval."""
    distance = np.where(position < start, position - start,
                        np.where(position > stop, position - stop, 0.0))
    result = np.empty(distance.shape, dtype=np.int8)
    result[distance < -50] = 0
    result[(distance >= -50) & (distance < -10)] = 1
    result[(distance >= -10) & (distance < 0)] = 2
    result[distance == 0] = 3
    result[(distance > 0) & (distance <= 10)] = 4
    result[(distance > 10) & (distance <= 50)] = 5
    result[distance > 50] = 6
    return result


def compute_dff_and_events(f: np.ndarray, f_neu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the paper repository's single-trial dF/F and deconvolution steps."""
    # Inputs are cells x time. Work in float32 to keep the full conversion tractable.
    corrected = f - NEUROPIL_COEF * f_neu
    corrected += NEUROPIL_COEF * np.mean(f_neu, axis=1, keepdims=True)

    baseline = gaussian_filter1d(corrected, 15, axis=1)
    baseline = minimum_filter1d(baseline, BASELINE_WINDOW_SAMPLES, axis=1)
    baseline = maximum_filter1d(baseline, BASELINE_WINDOW_SAMPLES, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dff = (corrected - baseline) / np.abs(baseline)
    dff = gaussian_filter1d(dff, 2, axis=1)
    if not np.isfinite(dff).all():
        # Degenerate zero baselines are not expected in curated ROIs. Preserve the
        # usable samples without allowing NaNs/Infs into the decoder.
        dff = np.nan_to_num(dff, copy=False)
    events = dcnv.oasis(dff, 2000, CALCIUM_TAU_S, FRAME_RATE_HZ)
    return dff, np.asarray(events, dtype=np.float32)


def read_curated_trial(
    h5: h5py.File,
    series_name: str,
    start: int,
    stop: int,
    iscell: np.ndarray,
    plane_idx: np.ndarray,
) -> np.ndarray:
    """Read a cells x time trial, preserving the segmentation table's ROI order."""
    pieces: list[np.ndarray] = []
    global_indices: list[np.ndarray] = []
    series = h5[f"processing/ophys/{series_name}"]
    for plane_name in sorted(series.keys(), key=lambda value: int(value.removeprefix("plane"))):
        plane = int(plane_name.removeprefix("plane"))
        plane_global = np.flatnonzero(plane_idx == plane)
        local_keep = np.flatnonzero(iscell[plane_global])
        # Reading a contiguous time slab before selecting columns is substantially
        # faster in h5py than a large two-dimensional fancy selection.
        slab = np.asarray(series[plane_name]["data"][start:stop, :], dtype=np.float32)
        pieces.append(slab[:, local_keep])
        global_indices.append(plane_global[local_keep])

    values = np.concatenate(pieces, axis=1)
    order = np.argsort(np.concatenate(global_indices))
    return np.ascontiguousarray(values[:, order].T)


def correlation_from_sums(
    n: int,
    sx: np.ndarray,
    sx2: np.ndarray,
    sy: float,
    sy2: float,
    sxy: np.ndarray,
) -> np.ndarray:
    numerator = n * sxy - sx * sy
    denominator = np.sqrt(np.maximum(0.0, n * sx2 - sx * sx) *
                          max(0.0, n * sy2 - sy * sy))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def convert_session(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict]:
    """Convert one recording and return its retained trial arrays and metadata."""
    with h5py.File(path, "r") as h5:
        behavior = h5["processing/behavior/BehavioralTimeSeries"]
        starts = np.flatnonzero(behavior["trial_start/data"][:] > 0)
        stops = np.flatnonzero(behavior["teleport/data"][:] > 0)
        if len(starts) != len(stops) or np.any(stops <= starts):
            raise ValueError(f"Invalid trial boundaries in {path}")

        identifier = h5["identifier"][()].decode()
        scene = identifier.rsplit("/", 1)[-1]
        subject = h5["general/subject/subject_id"][()].decode()
        session_id = h5["general/session_id"][()].decode()
        environments, zones = scene_conditions(scene, len(starts))

        timestamps = behavior["position/timestamps"][:]
        position = behavior["position/data"][:]
        speed = behavior["speed/data"][:]
        lick = behavior["lick/data"][:]
        env_stream = behavior["environment/data"][:]
        trial_number_stream = behavior["trial number/data"][:]
        rzone_stream = behavior["reward_zone/data"][:]
        reward_times = behavior["Reward/timestamps"][:]

        iscell_table = h5["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][:, 0]
        iscell = np.asarray(iscell_table == 1)
        plane_idx = np.asarray(
            h5["processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx"][:], dtype=int
        )
        n_curated = int(np.count_nonzero(iscell))

        # Determine reward outcomes on all original trials before applying the lick
        # quality filter, so "previous trial" retains its literal meaning.
        outcomes = np.zeros(len(starts), dtype=np.int8)
        for trial, (start, stop) in enumerate(zip(starts, stops)):
            left = np.searchsorted(reward_times, timestamps[start], side="left")
            right = np.searchsorted(reward_times, timestamps[stop], side="left")
            outcomes[trial] = int(right > left and np.any(rzone_stream[start:stop] > 0))

        trial_events: list[np.ndarray] = []
        retained_trial_indices: list[int] = []
        # Streaming sufficient statistics for dF/F-speed correlations.
        sx = np.zeros(n_curated, dtype=np.float64)
        sx2 = np.zeros(n_curated, dtype=np.float64)
        sxy = np.zeros(n_curated, dtype=np.float64)
        sy = sy2 = 0.0
        n_samples = 0
        excluded_lick_trials = 0

        for trial, (start, stop) in enumerate(zip(starts, stops)):
            trial_lick = lick[start:stop]
            bad_lick_sensor = np.mean(trial_lick > 2) > LICK_ERROR_FRACTION

            # The environment stream is a direct source check on scene parsing.
            stream_environment = int(round(float(np.median(env_stream[start:stop]))))
            if stream_environment != int(environments[trial]):
                raise ValueError(f"Scene/environment mismatch in {path}, trial {trial}")

            f = read_curated_trial(h5, "Fluorescence", start, stop, iscell, plane_idx)
            f_neu = read_curated_trial(h5, "Neuropil", start, stop, iscell, plane_idx)
            dff, events = compute_dff_and_events(f, f_neu)

            # Interneuron curation is a neural-quality step in the paper and uses
            # every valid neural trial, including trials whose lick channel failed.
            trial_speed = np.asarray(speed[start:stop], dtype=np.float64)
            dff64 = np.asarray(dff, dtype=np.float64)
            sx += np.sum(dff64, axis=1)
            sx2 += np.sum(dff64 * dff64, axis=1)
            sxy += dff64 @ trial_speed
            sy += float(np.sum(trial_speed))
            sy2 += float(trial_speed @ trial_speed)
            n_samples += len(trial_speed)

            if bad_lick_sensor:
                excluded_lick_trials += 1
                continue
            trial_events.append(events)
            retained_trial_indices.append(trial)

        speed_correlation = correlation_from_sums(n_samples, sx, sx2, sy, sy2, sxy)
        keep_neuron = speed_correlation <= 0.5
        neural_trials = [np.ascontiguousarray(events[keep_neuron], dtype=np.float32)
                         for events in trial_events]

        input_trials: list[np.ndarray] = []
        output_trials: list[np.ndarray] = []
        zone_to_class = {"A": 0, "B": 1, "C": 2}
        for trial in retained_trial_indices:
            start, stop = starts[trial], stops[trial]
            pos = np.asarray(position[start:stop], dtype=np.float32)
            spd = np.asarray(speed[start:stop], dtype=np.float32)
            time_from_start = np.asarray(timestamps[start:stop] - timestamps[start], dtype=np.float32)
            trial_number = float(np.median(trial_number_stream[start:stop]))
            previous_outcome = float(outcomes[trial - 1]) if trial > 0 else 0.0
            environment = float(environments[trial])
            inputs = np.vstack((
                time_from_start,
                np.full(len(pos), environment, dtype=np.float32),
                np.full(len(pos), trial_number, dtype=np.float32),
                np.full(len(pos), previous_outcome, dtype=np.float32),
            )).astype(np.float32, copy=False)

            zone = str(zones[trial])
            zone_start, zone_stop = ZONE_BOUNDS[zone]
            outputs = np.vstack((
                discretize_distance(pos, zone_start, zone_stop),
                np.digitize(pos, [90.0, 180.0, 270.0, 360.0]).astype(np.int8),
                np.digitize(spd, [2.0, 10.0, 20.0, 40.0]).astype(np.int8),
                (lick[start:stop] > 0).astype(np.int8),
                np.full(len(pos), zone_to_class[zone], dtype=np.int8),
                np.full(len(pos), outcomes[trial], dtype=np.int8),
            ))
            input_trials.append(np.ascontiguousarray(inputs))
            output_trials.append(np.ascontiguousarray(outputs, dtype=np.int8))

        info = {
            "source_file": str(path.relative_to(path.parents[2])),
            "subject": subject,
            "session_id": session_id,
            "scene": scene,
            "n_trials_original": int(len(starts)),
            "n_trials_retained": int(len(neural_trials)),
            "n_trials_excluded_lick_sensor": int(excluded_lick_trials),
            "n_rois_total": int(len(iscell)),
            "n_neurons_manually_curated": n_curated,
            "n_putative_interneurons_excluded": int(np.count_nonzero(~keep_neuron)),
            "n_neurons_retained": int(np.count_nonzero(keep_neuron)),
        }
        if len(neural_trials) < 2:
            raise ValueError(f"Fewer than two retained trials in {path}")
        return neural_trials, input_trials, output_trials, info


def build_dataset(data_root: Path) -> dict:
    files = sorted(data_root.glob("sub-*/*.nwb"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No NWB files found under {data_root}")
    subjects = sorted({f"m{natural_key(path)[0]}" for path in files},
                      key=lambda value: int(value[1:]))
    subject_lookup = {subject: index for index, subject in enumerate(subjects)}

    neural: list[list[np.ndarray]] = []
    inputs: list[list[np.ndarray]] = []
    outputs: list[list[np.ndarray]] = []
    subject_idx: list[int] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict] = []

    for number, path in enumerate(files, start=1):
        print(f"[{number:3d}/{len(files)}] {path.relative_to(data_root)}", flush=True)
        session_neural, session_input, session_output, info = convert_session(path)
        neural.append(session_neural)
        inputs.append(session_input)
        outputs.append(session_output)
        subject_idx.append(subject_lookup[info["subject"]])
        brain_region_idx.append(np.zeros(info["n_neurons_retained"], dtype=np.int8))
        session_info.append(info)
        print(
            f"    {info['n_trials_retained']} trials; {info['n_neurons_retained']} neurons "
            f"({info['n_trials_excluded_lick_sensor']} bad-lick trials and "
            f"{info['n_putative_interneurons_excluded']} putative interneurons excluded)",
            flush=True,
        )

    return {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int16),
        "brain_regions": ["CA1"],
        "brain_region_idx": brain_region_idx,
        "input_names": [
            "time from trial start (s)",
            "environment type",
            "trial number",
            "previous trial outcome",
        ],
        "output_names": [
            "distance to reward zone",
            "absolute corridor position",
            "speed",
            "lick",
            "reward zone location",
            "reward outcome",
        ],
        "output_values": [
            ["< -50 cm", "-50 to < -10 cm", "-10 to < 0 cm", "inside reward zone",
             "> 0 to +10 cm", "> +10 to +50 cm", "> +50 cm"],
            ["< 90 cm", "90 to < 180 cm", "180 to < 270 cm",
             "270 to < 360 cm", ">= 360 cm"],
            ["< 2 cm/s", "2 to < 10 cm/s", "10 to < 20 cm/s",
             "20 to < 40 cm/s", ">= 40 cm/s"],
            ["no lick", "lick"],
            ["A (80-130 cm)", "B (200-250 cm)", "C (320-370 cm)"],
            ["no reward", "rewarded"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed mice traversed a 450 cm virtual corridor with a hidden 50 cm "
                "reward zone that changed among locations A, B, and C and between ENV1/ENV2. "
                "Targets are reward-relative distance, corridor position, speed, licking, "
                "reward-zone identity, and trial reward outcome."
            ),
            "time_bin_size": 1000.0 / FRAME_RATE_HZ,
            "temporal_alignment_event": "start of trial at entry to the 0 cm end of the virtual corridor",
            "off_start": 0.0,
            "off_end": None,
            "neural_signal": "trial-local maximin dF/F, smoothed and OASIS-deconvolved calcium events",
            "trial_interval": "trial_start sample inclusive to teleport sample exclusive",
            "track_length_cm": 450.0,
            "reward_zone_bounds_cm": {key: list(value) for key, value in ZONE_BOUNDS.items()},
            "lick_sensor_exclusion_rule": (
                "exclude a trial when >30% of samples have cumulative lick count >2"
            ),
            "session_info": session_info,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "converted_data.pkl")
    args = parser.parse_args()

    dataset = build_dataset(args.data_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    print(f"Writing {args.output} ...", flush=True)
    with temporary.open("wb") as stream:
        pickle.dump(dataset, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, args.output)
    print("Conversion complete.", flush=True)


if __name__ == "__main__":
    main()
