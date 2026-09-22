#!/usr/bin/env python3
"""Convert the Mesoscale Activity Map NWB files for neural decoding.

Decisions tied to the supplied papers/repository
-------------------------------------------------
* Units are restricted to ``units/classification == "good"``.  This is the
  output of the paper's region-specific QC classifiers and is the population
  used by the supplied preprocessing code.
* Every trial with electrophysiology is retained, except acquisition gaps with
  no spikes from any unit.  The papers excluded early-lick, ignore, free-water,
  and photostimulation trials for particular analyses, but those exclusions
  would remove labels or inputs explicitly requested by this decoder task.
* Spike times, video, tones, licks, and laser epochs are all aligned in their
  common NWB session clock, then expressed relative to each trial's go onset.
* Firing rate is a non-overlapping 50-ms spike histogram divided by 0.05 s.
  The half-open bins exactly tile [-2.5, 1.5), giving 80 samples.
* Tongue position uses the side camera, as in the method paper.  Low-confidence
  DeepLabCut samples are the requested "not visible" category.  Visible
  five-sigma velocity outliers are interpolated from nearby samples, following
  the method paper, before session-wide visible-position percentiles are taken.

The script intentionally uses h5py rather than pynwb: the archived files use an
older NWB schema, while all fields needed here are standard HDF5 datasets.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


BIN_SIZE_S = 0.050
OFF_START_S = -2.5
OFF_END_S = 1.5
N_TIME = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
BIN_EDGES = OFF_START_S + np.arange(N_TIME + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0

# DeepLabCut probabilities in these files are strongly bimodal.  0.9 is a
# conservative standard p-cutoff and cleanly separates tracked from occluded
# tongue samples in this dataset.
TONGUE_LIKELIHOOD_CUTOFF = 0.9


def decode_strings(values: Iterable[object]) -> np.ndarray:
    """Decode an HDF5 string vector and strip padding."""
    return np.asarray(
        [v.decode("utf-8").strip() if isinstance(v, bytes) else str(v).strip()
         for v in values],
        dtype=str,
    )


def parse_optional_floats(values: Iterable[object]) -> np.ndarray:
    """Convert NWB string numbers, mapping 'N/A' to NaN."""
    text = decode_strings(values)
    result = np.full(text.shape, np.nan, dtype=np.float64)
    valid = text != "N/A"
    result[valid] = text[valid].astype(np.float64)
    return result


def nearest_indices(sorted_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Indices of samples nearest each query in a sorted timestamp vector."""
    right = np.searchsorted(sorted_times, query_times, side="left")
    right = np.clip(right, 0, len(sorted_times) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(query_times - sorted_times[left]) <= np.abs(
        sorted_times[right] - query_times
    )
    return np.where(choose_left, left, right)


def clean_tongue_y(
    y: np.ndarray, likelihood: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Interpolate visible five-sigma velocity outliers.

    Occluded samples are deliberately not imputed: the decoder specification
    asks for a distinct not-visible class.  Velocity is assessed only between
    consecutive visible frames so entry into or out of occlusion is not called
    an outlier.
    """
    clean = np.asarray(y, dtype=np.float64).copy()
    visible = (
        np.isfinite(clean)
        & np.isfinite(likelihood)
        & (likelihood >= TONGUE_LIKELIHOOD_CUTOFF)
    )
    consecutive = visible[:-1] & visible[1:]
    speeds = np.abs(np.diff(clean))
    speed_sample = speeds[consecutive & np.isfinite(speeds)]
    outlier = np.zeros(clean.shape, dtype=bool)
    if speed_sample.size >= 2:
        cutoff = speed_sample.mean() + 5.0 * speed_sample.std()
        # The destination of an implausibly fast transition is the sample to
        # impute.  This matches the framewise velocity convention.
        outlier[1:] = consecutive & (speeds > cutoff)

    good = visible & ~outlier
    if np.any(outlier) and np.count_nonzero(good) >= 2:
        x = np.arange(len(clean), dtype=np.float64)
        clean[outlier] = np.interp(x[outlier], x[good], clean[good])
    elif np.any(outlier):
        visible[outlier] = False
    return clean, visible, int(np.count_nonzero(outlier))


def event_choice(
    go_times: np.ndarray,
    left_licks: np.ndarray,
    right_licks: np.ndarray,
) -> np.ndarray:
    """Return first response-epoch lick: left=0, right=1, no lick=2."""
    choice = np.full(len(go_times), 2, dtype=np.int8)
    for trial, go in enumerate(go_times):
        li = np.searchsorted(left_licks, go, side="left")
        ri = np.searchsorted(right_licks, go, side="left")
        left_time = left_licks[li] if li < len(left_licks) else np.inf
        right_time = right_licks[ri] if ri < len(right_licks) else np.inf
        response_end = go + 1.5
        if left_time < response_end and left_time <= right_time:
            choice[trial] = 0
        elif right_time < response_end:
            choice[trial] = 1
    return choice


def final_tone_onsets(
    trial_starts: np.ndarray,
    go_times: np.ndarray,
    sample_starts: np.ndarray,
) -> np.ndarray:
    """Select the final tone onset before go, after any early-lick replay."""
    result = np.empty(len(go_times), dtype=np.float64)
    for trial, (start, go) in enumerate(zip(trial_starts, go_times)):
        stop_idx = np.searchsorted(sample_starts, go, side="right")
        start_idx = np.searchsorted(sample_starts, start, side="left")
        if stop_idx <= start_idx:
            raise ValueError(f"Trial {trial} has no sample/tone onset before go")
        result[trial] = sample_starts[stop_idx - 1]
    return result


def bin_good_units(
    units: h5py.Group,
    good_indices: np.ndarray,
    go_times: np.ndarray,
) -> np.ndarray:
    """Bin absolute spike times into a (unit, trial, time) rate cube."""
    n_units = len(good_indices)
    n_trials = len(go_times)
    # Float32 from the start avoids a second full-size rate cube.
    rates = np.zeros((n_units, n_trials, N_TIME), dtype=np.float32)
    all_spikes = units["spike_times"]
    ends = units["spike_times_index"][:].astype(np.int64, copy=False)
    starts = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
    window_starts = go_times + OFF_START_S
    window_ends = go_times + OFF_END_S

    for output_unit, source_unit in enumerate(good_indices):
        spikes = all_spikes[starts[source_unit]:ends[source_unit]]
        if len(spikes) == 0:
            continue
        # Trial windows do not overlap in this dataset (the shortest go-to-go
        # interval exceeds four seconds), so each in-window spike has one trial.
        trial_idx = np.searchsorted(window_starts, spikes, side="right") - 1
        valid_trial = trial_idx >= 0
        candidate_spikes = spikes[valid_trial]
        candidate_trials = trial_idx[valid_trial]
        in_window = candidate_spikes < window_ends[candidate_trials]
        candidate_spikes = candidate_spikes[in_window]
        candidate_trials = candidate_trials[in_window]
        relative = candidate_spikes - go_times[candidate_trials]
        bin_idx = np.searchsorted(BIN_EDGES, relative, side="right") - 1
        valid_bin = (bin_idx >= 0) & (bin_idx < N_TIME)
        np.add.at(
            rates[output_unit],
            (candidate_trials[valid_bin], bin_idx[valid_bin]),
            np.float32(1.0 / BIN_SIZE_S),
        )
    return rates


def scan_vocab(paths: list[Path]) -> tuple[list[str], list[str], list[Path]]:
    """Find stable subject/region vocabularies and discard empty sessions."""
    subjects: set[str] = set()
    regions: set[str] = set()
    usable: list[Path] = []
    for path in paths:
        with h5py.File(path, "r") as nwb:
            classification = decode_strings(nwb["units/classification"][:])
            good = classification == "good"
            if np.count_nonzero(good) == 0:
                print(f"Skipping {path.name}: no classifier-approved units", flush=True)
                continue
            subjects.add(
                nwb["general/subject/subject_id"][()].decode("utf-8").strip()
            )
            annotations = decode_strings(nwb["units/anno_name"][:])[good]
            if np.any(annotations == ""):
                raise ValueError(f"Good unit without anatomical annotation in {path}")
            regions.update(annotations.tolist())
            usable.append(path)
    return sorted(subjects), sorted(regions), usable


def convert_session(
    path: Path,
    subject_to_idx: dict[str, int],
    region_to_idx: dict[str, int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], int, np.ndarray, dict]:
    """Convert one NWB session."""
    with h5py.File(path, "r") as nwb:
        trials = nwb["intervals/trials"]
        events = nwb["acquisition/BehavioralEvents"]
        units = nwb["units"]

        n_behavior_trials = len(trials["id"])
        all_go_times = events["go_start_times/timestamps"][:]
        if len(all_go_times) != n_behavior_trials:
            raise ValueError(
                f"{path}: {n_behavior_trials} trials but {len(all_go_times)} go cues"
            )

        # A few NWB files contain behavior continuing after the ephys recording.
        # is_good_trials has one column per trial actually covered by ephys (as
        # do each unit's ragged obs_intervals); later behavioral-only trials must
        # not become artificial all-zero neural trials.
        n_ephys_trials = units["is_good_trials"].shape[1]
        if n_ephys_trials > n_behavior_trials:
            raise ValueError(f"{path}: more ephys trials than behavioral trials")
        source_trial_idx = np.arange(n_ephys_trials, dtype=np.int64)
        go_times = all_go_times[source_trial_idx]

        classification = decode_strings(units["classification"][:])
        good_indices = np.flatnonzero(classification == "good")
        annotations = decode_strings(units["anno_name"][:])[good_indices]
        region_idx = np.asarray(
            [region_to_idx[name] for name in annotations], dtype=np.int32
        )
        rates = bin_good_units(units, good_indices, go_times)

        # Some archived trials have an observation interval but no spikes from
        # any unit at all (including unclassified units).  These acquisition
        # gaps cannot supply a neural decoder input and are excluded explicitly.
        has_neural_data = np.any(rates != 0, axis=(0, 2))
        n_zero_neural = int(np.count_nonzero(~has_neural_data))
        source_trial_idx = source_trial_idx[has_neural_data]
        go_times = go_times[has_neural_data]
        rates = rates[:, has_neural_data, :]
        n_trials = len(source_trial_idx)

        # Decoder inputs.
        trial_starts = trials["start_time"][:][source_trial_idx]
        sample_starts = events["sample_start_times/timestamps"][:]
        tone_onsets = final_tone_onsets(trial_starts, go_times, sample_starts)
        time_from_tone = (
            BIN_CENTERS[None, :] + (go_times - tone_onsets)[:, None]
        ).astype(np.float32)

        photo_onset = parse_optional_floats(
            trials["photostim_onset"][:][source_trial_idx]
        )
        photo_duration = parse_optional_floats(
            trials["photostim_duration"][:][source_trial_idx]
        )
        absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
        absolute_photo_start = trial_starts + photo_onset
        absolute_photo_end = absolute_photo_start + photo_duration
        photo_on = (
            (absolute_centers >= absolute_photo_start[:, None])
            & (absolute_centers < absolute_photo_end[:, None])
            & np.isfinite(absolute_photo_start[:, None])
        ).astype(np.float32)

        # Per-trial decoder outputs.
        left_licks = events["left_lick_times/timestamps"][:]
        right_licks = events["right_lick_times/timestamps"][:]
        choice = event_choice(go_times, left_licks, right_licks)
        outcome_text = decode_strings(trials["outcome"][:][source_trial_idx])
        outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
        unknown_outcome = sorted(set(outcome_text) - set(outcome_map))
        if unknown_outcome:
            raise ValueError(f"{path}: unknown outcomes {unknown_outcome}")
        outcome = np.asarray([outcome_map[x] for x in outcome_text], dtype=np.int8)
        early_text = decode_strings(trials["early_lick"][:][source_trial_idx])
        unknown_early = sorted(set(early_text) - {"no early", "early"})
        if unknown_early:
            raise ValueError(f"{path}: unknown early-lick labels {unknown_early}")
        early = (early_text == "early").astype(np.int8)

        # Time-varying tongue output.  Thresholds use all visible session frames,
        # while labels are sampled nearest each firing-rate bin center.
        tongue = nwb[
            "acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"
        ]
        tongue_data = tongue["data"][:]
        tongue_times = tongue["timestamps"][:]
        tongue_y, visible, n_outliers = clean_tongue_y(
            tongue_data[:, 1], tongue_data[:, 2]
        )
        if np.count_nonzero(visible) < 2:
            raise ValueError(f"{path}: fewer than two visible tongue samples")
        q40, q60 = np.percentile(tongue_y[visible], [40.0, 60.0])
        video_idx = nearest_indices(tongue_times, absolute_centers.ravel()).reshape(
            n_trials, N_TIME
        )
        sampled_y = tongue_y[video_idx]
        sampled_visible = visible[video_idx]
        tongue_class = np.full((n_trials, N_TIME), 3, dtype=np.int8)
        tongue_class[sampled_visible & (sampled_y < q40)] = 0
        tongue_class[
            sampled_visible & (sampled_y >= q40) & (sampled_y <= q60)
        ] = 1
        tongue_class[sampled_visible & (sampled_y > q60)] = 2

        neural_trials = [rates[:, trial, :].copy() for trial in range(n_trials)]
        input_trials = [
            np.stack((time_from_tone[trial], photo_on[trial])).astype(
                np.float32, copy=False
            )
            for trial in range(n_trials)
        ]
        output_trials = []
        for trial in range(n_trials):
            output_trial = np.empty((4, N_TIME), dtype=np.int8)
            output_trial[0, :] = choice[trial]
            output_trial[1, :] = outcome[trial]
            output_trial[2, :] = early[trial]
            output_trial[3, :] = tongue_class[trial]
            output_trials.append(output_trial)

        subject = nwb["general/subject/subject_id"][()].decode("utf-8").strip()
        identifier = nwb["identifier"][()].decode("utf-8").strip()
        session_info = {
            "identifier": identifier,
            "source_file": str(path),
            "subject": subject,
            "n_trials": int(n_trials),
            "n_behavior_trials": int(n_behavior_trials),
            "n_ephys_trials": int(n_ephys_trials),
            "n_zero_neural_trials_excluded": n_zero_neural,
            "n_good_units": int(len(good_indices)),
            "tongue_visible_y_percentile_40": float(q40),
            "tongue_visible_y_percentile_60": float(q60),
            "tongue_velocity_outliers_imputed": n_outliers,
            "choice_counts": np.bincount(choice, minlength=3).astype(int).tolist(),
            "outcome_counts": np.bincount(outcome, minlength=3).astype(int).tolist(),
            "early_lick_counts": np.bincount(early, minlength=2).astype(int).tolist(),
        }
        return (
            neural_trials,
            input_trials,
            output_trials,
            subject_to_idx[subject],
            region_idx,
            session_info,
        )


def convert(data_dir: Path, output_path: Path) -> dict:
    paths = sorted(data_dir.glob("sub-*/*.nwb"))
    if not paths:
        raise FileNotFoundError(f"No NWB files found below {data_dir}")
    subjects, brain_regions, paths = scan_vocab(paths)
    subject_to_idx = {name: idx for idx, name in enumerate(subjects)}
    region_to_idx = {name: idx for idx, name in enumerate(brain_regions)}

    neural: list[list[np.ndarray]] = []
    inputs: list[list[np.ndarray]] = []
    outputs: list[list[np.ndarray]] = []
    subject_idx: list[int] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict] = []

    for session, path in enumerate(paths, start=1):
        converted = convert_session(path, subject_to_idx, region_to_idx)
        n, i, o, sidx, ridx, info = converted
        if len(n) < 2:
            print(f"Skipping {path.name}: fewer than two trials", flush=True)
            continue
        neural.append(n)
        inputs.append(i)
        outputs.append(o)
        subject_idx.append(sidx)
        brain_region_idx.append(ridx)
        session_info.append(info)
        print(
            f"[{session:3d}/{len(paths)}] {info['identifier']}: "
            f"{info['n_trials']} trials, {info['n_good_units']} units",
            flush=True,
        )

    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int32),
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": ["time from tone onset (s)", "photostimulation on"],
        "output_names": ["lick direction choice", "outcome", "early lick", "tongue y-position"],
        "output_values": [
            ["left", "right", "no lick"],
            ["ignore", "miss", "hit"],
            ["no", "yes"],
            ["below 40th percentile", "40th to 60th percentile", "above 60th percentile", "not visible"],
        ],
        "metadata": {
            "task_description": (
                "Auditory delayed-response task: mice report a low/high tone "
                "instruction by licking left/right after a go cue; some trials "
                "include late-delay ALM photoinhibition."
            ),
            "time_bin_size": 50.0,
            "temporal_alignment_event": "go cue onset",
            "off_start": OFF_START_S,
            "off_end": OFF_END_S,
            "neural_measure": "firing rate (Hz; half-open non-overlapping bins)",
            "time_bin_centers_s": BIN_CENTERS.tolist(),
            "unit_filter": "NWB units/classification == 'good' (paper classifier QC)",
            "trial_filter": (
                "all ephys-covered trials retained because early lick, ignore, "
                "and photostimulation are requested decoder variables; "
                "behavior-only tails and acquisition gaps with no neural data "
                "are excluded"
            ),
            "excluded_sessions": (
                "sessions with no classifier-approved units or fewer than two trials"
            ),
            "brain_region_definition": "exact units/anno_name Allen CCF annotation",
            "choice_definition": "first left/right lick in [go onset, go onset + 1.5 s); otherwise no lick",
            "tone_onset_definition": "final sample/tone onset preceding go, after any early-lick replay",
            "tongue_tracking_view": "Camera0 side view",
            "tongue_visibility_likelihood_cutoff": TONGUE_LIKELIHOOD_CUTOFF,
            "tongue_discretization": (
                "session-wide 40th/60th percentiles over visible samples after "
                "five-sigma velocity-outlier interpolation; low-likelihood "
                "samples are category 3 (not visible)"
            ),
            "session_info": session_info,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    print(f"Writing {output_path} ...", flush=True)
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, output_path)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 2**30:.2f} GiB)", flush=True)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--output", type=Path, default=Path("/app/converted_data.pkl"))
    args = parser.parse_args()
    convert(args.data_dir, args.output)


if __name__ == "__main__":
    main()
