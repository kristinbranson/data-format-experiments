#!/usr/bin/env python3
"""Convert the MAP NWB dataset to the decoder-compatible pickle format.

Usage
-----
python -u /app/convert_data.py OUTPUT.pkl [--full | --sample] [--show-processing]

The source NWBs use a continuous absolute clock.  This converter maps the
authors' classifier-approved units and explicitly observed trials onto 80
non-overlapping 50-ms bins centered from -2.475 to +1.475 s around go cue.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


DATA_ROOT = Path("/app/data")
TIME_EDGES = np.linspace(-2.5, 1.5, 81, dtype=np.float64)
TIME_CENTERS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
BIN_WIDTH_S = 0.05
TONGUE_LIKELIHOOD_THRESHOLD = 0.9

CHOICE_VALUES = ["left", "right", "no lick"]
OUTCOME_VALUES = ["ignore", "miss", "hit"]
EARLY_VALUES = ["no", "yes"]
TONGUE_VALUES = [
    "< 40th percentile",
    "40th to 60th percentile",
    "> 60th percentile",
    "not visible",
]


def decode_scalar(value) -> str:
    """Decode an HDF5 byte/string scalar to a normal Python string."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def decode_array(dataset) -> np.ndarray:
    return np.asarray([decode_scalar(x) for x in dataset[:]], dtype=object)


@dataclass
class TrialSelection:
    indices: np.ndarray
    observed_indices: np.ndarray
    n_original: int
    n_observed: int
    n_after_unit_trial_qc: int
    n_after_window_qc: int
    recording_start: float
    recording_stop: float


def get_good_units(nwb: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    classifications = decode_array(nwb["units/classification"])
    good_indices = np.flatnonzero(classifications == "good")
    annotations = decode_array(nwb["units/anno_name"])[good_indices]
    if np.any(annotations == ""):
        raise ValueError("A classifier-good unit has an empty CCF annotation")
    return good_indices.astype(np.int64), annotations


def map_observed_trials(nwb: h5py.File, good_units: np.ndarray) -> TrialSelection:
    """Map ragged unit observation intervals to trial-table rows.

    In nine source files, ``is_good_trials`` covers only a contiguous recorded
    subset of the behavioral trial table.  NWB ``obs_intervals`` gives the
    exact start/stop pair for each recorded trial.  All classifier-good units
    in a session share the same observation sequence (checked here on several
    representative units), so a fixed session neuron set is well defined.
    """
    trial_group = nwb["intervals/trials"]
    starts = trial_group["start_time"][:]
    stops = trial_group["stop_time"][:]
    n_original = len(starts)

    obs_data = nwb["units/obs_intervals"]
    obs_index = nwb["units/obs_intervals_index"][:].astype(np.int64)

    first_unit = int(good_units[0])
    a = 0 if first_unit == 0 else int(obs_index[first_unit - 1])
    b = int(obs_index[first_unit])
    common_obs = obs_data[a:b]
    if len(common_obs) == 0:
        raise ValueError("Classifier-good units have no observation intervals")

    # All good units must have the same number and endpoints. Check every
    # endpoint/count cheaply, then compare full sequences for first/middle/last.
    previous = np.r_[0, obs_index[:-1]]
    lengths = obs_index[good_units] - previous[good_units]
    if not np.all(lengths == len(common_obs)):
        raise ValueError("Good units have different observation interval counts")
    for unit in np.unique(good_units[[0, len(good_units) // 2, -1]]):
        ua, ub = int(previous[unit]), int(obs_index[unit])
        if not np.allclose(obs_data[ua:ub], common_obs, atol=1e-9, rtol=0):
            raise ValueError("Good units have different observation intervals")

    observed_trial_indices = np.empty(len(common_obs), dtype=np.int64)
    for j, (obs_start, obs_stop) in enumerate(common_obs):
        matches = np.flatnonzero(
            np.isclose(starts, obs_start, atol=1e-8, rtol=0)
            & np.isclose(stops, obs_stop, atol=1e-8, rtol=0)
        )
        if len(matches) != 1:
            raise ValueError(
                f"Observation interval {obs_start, obs_stop} maps to "
                f"{len(matches)} trial rows"
            )
        observed_trial_indices[j] = matches[0]
    if np.any(np.diff(observed_trial_indices) <= 0):
        raise ValueError("Observed trials are not strictly ordered")

    unit_trial_validity = nwb["units/is_good_trials"]
    if unit_trial_validity.shape[1] != len(common_obs):
        raise ValueError(
            "is_good_trials columns do not match the observed interval count: "
            f"{unit_trial_validity.shape[1]} vs {len(common_obs)}"
        )
    all_units_valid = np.all(unit_trial_validity[good_units, :], axis=0)
    after_unit_qc = observed_trial_indices[all_units_valid]

    go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
    if len(go_times) != n_original:
        raise ValueError("Go-cue count does not match trial-table length")
    recording_start, recording_stop = map(float, (common_obs[0, 0], common_obs[-1, 1]))
    full_window = (
        (go_times[after_unit_qc] + TIME_EDGES[0] >= recording_start - 1e-9)
        & (go_times[after_unit_qc] + TIME_EDGES[-1] <= recording_stop + 1e-9)
    )
    selected = after_unit_qc[full_window]
    if len(selected) < 2:
        raise ValueError(f"Only {len(selected)} fully valid trials remain")

    return TrialSelection(
        indices=selected,
        observed_indices=observed_trial_indices,
        n_original=n_original,
        n_observed=len(observed_trial_indices),
        n_after_unit_trial_qc=len(after_unit_qc),
        n_after_window_qc=len(selected),
        recording_start=recording_start,
        recording_stop=recording_stop,
    )


def final_tone_onsets(nwb: h5py.File, trial_indices: np.ndarray) -> np.ndarray:
    """Return the last sample/tone epoch onset before each selected go cue."""
    sample_starts = nwb[
        "acquisition/BehavioralEvents/sample_start_times/timestamps"
    ][:]
    go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
    trial_starts = nwb["intervals/trials/start_time"][:]
    selected_go = go_times[trial_indices]
    positions = np.searchsorted(sample_starts, selected_go, side="left") - 1
    if np.any(positions < 0):
        raise ValueError("A selected trial has no sample/tone onset before go")
    tones = sample_starts[positions]
    if np.any(tones < trial_starts[trial_indices] - 1e-9):
        raise ValueError("A selected trial has no in-trial sample/tone onset")
    return tones


def construct_inputs(
    nwb: h5py.File, trial_indices: np.ndarray, tone_onsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
    absolute_centers = go_times[trial_indices, None] + TIME_CENTERS[None, :]

    time_from_tone = absolute_centers - tone_onsets[:, None]

    event_root = nwb["acquisition/BehavioralEvents"]
    photo_starts = event_root["photostim_start_times/timestamps"][:]
    photo_stops = event_root["photostim_stop_times/timestamps"][:]
    if len(photo_starts) != len(photo_stops):
        raise ValueError("Photostimulation start/stop event counts differ")
    if np.any(photo_stops <= photo_starts):
        raise ValueError("Non-positive photostimulation interval")
    started = np.searchsorted(photo_starts, absolute_centers, side="right")
    stopped = np.searchsorted(photo_stops, absolute_centers, side="right")
    photostim_on = (started > stopped).astype(np.float32)

    inputs = np.stack(
        [time_from_tone.astype(np.float32), photostim_on], axis=1
    )
    return inputs, absolute_centers


def bin_spikes(
    nwb: h5py.File, good_units: np.ndarray, trial_indices: np.ndarray
) -> np.ndarray:
    """Bin spikes to Hz; return trial x neuron x time float32."""
    go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
    absolute_edges = go_times[trial_indices, None] + TIME_EDGES[None, :]
    if np.any(np.diff(absolute_edges.ravel()) < 0):
        raise ValueError("Trial windows overlap or are not time ordered")

    spike_index = nwb["units/spike_times_index"][:].astype(np.int64)
    previous = np.r_[0, spike_index[:-1]]
    # One sequential read is substantially faster than thousands of HDF5 reads.
    all_spikes = nwb["units/spike_times"][:]
    rates = np.empty(
        (len(trial_indices), len(good_units), len(TIME_CENTERS)), dtype=np.float32
    )
    for out_unit, source_unit in enumerate(good_units):
        spikes = all_spikes[previous[source_unit] : spike_index[source_unit]]
        edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
        rates[:, out_unit, :] = np.diff(edge_positions, axis=1) / BIN_WIDTH_S
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("Invalid firing rates")
    # Rates from integer counts must be integer after multiplying by bin width.
    if not np.allclose(rates * BIN_WIDTH_S, np.rint(rates * BIN_WIDTH_S)):
        raise ValueError("Firing rates are inconsistent with 50-ms spike counts")
    return rates


@dataclass
class TongueProcessing:
    classes: np.ndarray
    p40: float
    p60: float
    visible_fraction_all_frames: float
    visible_fraction_sampled: float
    n_velocity_outliers: int
    timestamps: np.ndarray
    corrected_y: np.ndarray
    likelihood: np.ndarray


def process_tongue(nwb: h5py.File, absolute_centers: np.ndarray) -> TongueProcessing:
    tracking = nwb[
        "acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"
    ]
    timestamps = tracking["timestamps"][:]
    marker = tracking["data"][:]
    if marker.ndim != 2 or marker.shape[1] < 3 or len(marker) != len(timestamps):
        raise ValueError("Unexpected tongue-tracking array")
    x = marker[:, 0].astype(np.float64, copy=False)
    y = marker[:, 1].astype(np.float64, copy=True)
    likelihood = marker[:, 2].astype(np.float64, copy=False)
    visible = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(likelihood)
        & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    )
    if np.sum(visible) < 10:
        raise ValueError("Too few visible tongue frames for session percentiles")

    # Match the method paper's five-SD velocity outlier correction, restricting
    # the reference distribution to adjacent high-confidence frames.
    dt = np.diff(timestamps)
    displacement = np.hypot(np.diff(x), np.diff(y))
    valid_pairs = visible[:-1] & visible[1:] & np.isfinite(dt) & (dt > 0)
    speeds = np.full(len(dt), np.nan, dtype=np.float64)
    speeds[valid_pairs] = displacement[valid_pairs] / dt[valid_pairs]
    reference_speeds = speeds[np.isfinite(speeds)]
    outlier_frames = np.zeros(len(y), dtype=bool)
    if len(reference_speeds) > 1:
        cutoff = np.mean(reference_speeds) + 5 * np.std(reference_speeds)
        outlier_frames[1:] = np.isfinite(speeds) & (speeds > cutoff)
    interpolation_basis = visible & ~outlier_frames
    if np.any(outlier_frames & visible) and np.sum(interpolation_basis) >= 2:
        target = np.flatnonzero(outlier_frames & visible)
        y[target] = np.interp(
            timestamps[target], timestamps[interpolation_basis], y[interpolation_basis]
        )

    p40, p60 = np.percentile(y[visible], [40, 60])
    if not np.isfinite(p40 + p60) or p40 > p60:
        raise ValueError("Invalid tongue percentile thresholds")

    flat_centers = absolute_centers.ravel()
    right = np.searchsorted(timestamps, flat_centers, side="left")
    right_clipped = np.clip(right, 0, len(timestamps) - 1)
    left_clipped = np.clip(right - 1, 0, len(timestamps) - 1)
    choose_left = np.abs(flat_centers - timestamps[left_clipped]) <= np.abs(
        timestamps[right_clipped] - flat_centers
    )
    nearest = np.where(choose_left, left_clipped, right_clipped)
    median_dt = float(np.median(np.diff(timestamps)))
    covered = (
        (flat_centers >= timestamps[0])
        & (flat_centers <= timestamps[-1])
        & (np.abs(timestamps[nearest] - flat_centers) <= 1.5 * median_dt)
    )
    sampled_visible = covered & visible[nearest]
    sampled_y = y[nearest]
    classes = np.full(len(flat_centers), 3, dtype=np.int64)
    classes[sampled_visible & (sampled_y < p40)] = 0
    classes[
        sampled_visible & (sampled_y >= p40) & (sampled_y <= p60)
    ] = 1
    classes[sampled_visible & (sampled_y > p60)] = 2
    classes = classes.reshape(absolute_centers.shape)

    return TongueProcessing(
        classes=classes,
        p40=float(p40),
        p60=float(p60),
        visible_fraction_all_frames=float(np.mean(visible)),
        visible_fraction_sampled=float(np.mean(classes != 3)),
        n_velocity_outliers=int(np.sum(outlier_frames & visible)),
        timestamps=timestamps,
        corrected_y=y,
        likelihood=likelihood,
    )


def construct_outputs(
    nwb: h5py.File, trial_indices: np.ndarray, tongue_classes: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    trials = nwb["intervals/trials"]
    instruction = decode_array(trials["trial_instruction"])[trial_indices]
    outcome_text = decode_array(trials["outcome"])[trial_indices]
    early_text = decode_array(trials["early_lick"])[trial_indices]

    if not set(np.unique(instruction)).issubset({"left", "right"}):
        raise ValueError("Unexpected trial instruction")
    if not set(np.unique(outcome_text)).issubset(set(OUTCOME_VALUES)):
        raise ValueError("Unexpected outcome label")
    if not set(np.unique(early_text)).issubset({"no early", "early"}):
        raise ValueError("Unexpected early-lick label")

    actual_choice = np.full(len(trial_indices), 2, dtype=np.int64)
    hit = outcome_text == "hit"
    miss = outcome_text == "miss"
    actual_choice[hit & (instruction == "left")] = 0
    actual_choice[hit & (instruction == "right")] = 1
    actual_choice[miss & (instruction == "left")] = 1
    actual_choice[miss & (instruction == "right")] = 0

    outcome_map = {name: idx for idx, name in enumerate(OUTCOME_VALUES)}
    outcomes = np.asarray([outcome_map[x] for x in outcome_text], dtype=np.int64)
    early = (early_text == "early").astype(np.int64)

    n_trials, n_time = tongue_classes.shape
    outputs = np.empty((n_trials, 4, n_time), dtype=np.int64)
    outputs[:, 0, :] = actual_choice[:, None]
    outputs[:, 1, :] = outcomes[:, None]
    outputs[:, 2, :] = early[:, None]
    outputs[:, 3, :] = tongue_classes
    labels = {"choice": actual_choice, "outcome": outcomes, "early": early}
    return outputs, labels


def plot_processing(
    session_id: str,
    nwb: h5py.File,
    trial_indices: np.ndarray,
    good_units: np.ndarray,
    rates: np.ndarray,
    inputs: np.ndarray,
    outputs: np.ndarray,
    absolute_centers: np.ndarray,
    tongue: TongueProcessing,
) -> None:
    """Plot source-to-output alignment for every main conversion stage."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trial_pos = 0
    source_trial = int(trial_indices[trial_pos])
    source_unit = int(good_units[0])
    go = float(
        nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][source_trial]
    )
    spike_index = nwb["units/spike_times_index"][:].astype(np.int64)
    a = 0 if source_unit == 0 else int(spike_index[source_unit - 1])
    b = int(spike_index[source_unit])
    spikes = nwb["units/spike_times"][a:b] - go
    spikes = spikes[(spikes >= TIME_EDGES[0]) & (spikes < TIME_EDGES[-1])]

    fig, axes = plt.subplots(5, 1, figsize=(12, 16), constrained_layout=True)
    ax = axes[0]
    ax.vlines(spikes, 0, 1, color="black", lw=0.7, label="raw spikes")
    ax2 = ax.twinx()
    ax2.step(TIME_CENTERS, rates[trial_pos, 0], where="mid", color="tab:blue")
    ax.axvline(0, color="red", ls="--", label="go")
    ax.set_xlim(-2.5, 1.5)
    ax.set_ylabel("raw spike raster")
    ax2.set_ylabel("50-ms rate (Hz)", color="tab:blue")
    ax.set_title("Neural: raw absolute spikes → go-aligned 50-ms firing rate")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(TIME_CENTERS, inputs[trial_pos, 0], label="time from tone (s)")
    ax.axhline(0, color="gray", ls=":")
    ax.axvline(0, color="red", ls="--", label="go")
    ax.set_xlim(-2.5, 1.5)
    ax.set_title("Input 1: continuous time from final tone onset")
    ax.legend()

    ax = axes[2]
    show_trials = np.flatnonzero(np.any(inputs[:, 1] > 0, axis=1))[:8]
    if len(show_trials) == 0:
        show_trials = np.arange(min(8, len(inputs)))
    ax.imshow(
        inputs[show_trials, 1],
        aspect="auto",
        interpolation="nearest",
        extent=[-2.5, 1.5, len(show_trials), 0],
        vmin=0,
        vmax=1,
        cmap="binary",
    )
    ax.axvline(0, color="red", ls="--")
    ax.set_ylabel("selected trials")
    ax.set_title("Input 2: photostimulation state at bin centers")

    ax = axes[3]
    center_times = absolute_centers[trial_pos]
    window = (tongue.timestamps >= go - 2.5) & (tongue.timestamps <= go + 1.5)
    ax.plot(
        tongue.timestamps[window] - go,
        tongue.corrected_y[window],
        color="0.6",
        lw=0.7,
        label="corrected source y",
    )
    sampled_class = outputs[trial_pos, 3]
    visible_bins = sampled_class != 3
    nearest_y = np.interp(center_times, tongue.timestamps, tongue.corrected_y)
    ax.scatter(
        TIME_CENTERS[visible_bins],
        nearest_y[visible_bins],
        c=sampled_class[visible_bins],
        cmap="viridis",
        vmin=0,
        vmax=2,
        s=20,
        label="visible bin class",
    )
    ax.scatter(
        TIME_CENTERS[~visible_bins],
        np.full(np.sum(~visible_bins), tongue.p40),
        marker="x",
        color="red",
        s=18,
        label="not visible",
    )
    ax.axhline(tongue.p40, color="tab:orange", ls="--", label="p40/p60")
    ax.axhline(tongue.p60, color="tab:orange", ls="--")
    ax.axvline(0, color="red", ls=":")
    ax.set_xlim(-2.5, 1.5)
    ax.set_ylabel("tongue y (pixels)")
    ax.set_title("Output 4: tracking confidence, percentiles, and classes")
    ax.legend(ncol=2, fontsize=8)

    ax = axes[4]
    names = ["choice", "outcome", "early", "tongue"]
    for row, name in enumerate(names):
        ax.step(TIME_CENTERS, outputs[trial_pos, row] + row * 4, where="mid", label=name)
    ax.axvline(0, color="red", ls="--")
    ax.set_xlim(-2.5, 1.5)
    ax.set_xlabel("time from go cue (s)")
    ax.set_yticks([])
    ax.set_title("Final categorical outputs (trial labels broadcast; tongue time-varying)")
    ax.legend(ncol=4)

    fig.suptitle(f"MAP conversion processing: {session_id}; source trial {source_trial}")
    output_path = Path("/app") / f"processing_{session_id}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  saved processing plot: {output_path}", flush=True)


def convert_session(path: str, make_plot: bool) -> dict | None:
    started = time.perf_counter()
    session_id = Path(path).stem.replace("_behavior+ecephys+ogen", "").replace(
        "_behavior+ecephys", ""
    )
    with h5py.File(path, "r") as nwb:
        good_units, annotations = get_good_units(nwb)
        if len(good_units) == 0:
            print(f"SKIP {session_id}: no classifier-good units", flush=True)
            return None
        selection = map_observed_trials(nwb, good_units)
        trial_indices = selection.indices
        tone_onsets = final_tone_onsets(nwb, trial_indices)
        inputs, absolute_centers = construct_inputs(nwb, trial_indices, tone_onsets)
        rates = bin_spikes(nwb, good_units, trial_indices)

        # A few source observation rows extend beyond the actual end of every
        # retained unit's spike stream despite being flagged good in NWB.  A
        # four-second population-silent window across hundreds of units is a
        # missing-recording signature, not physiology; exclude it explicitly.
        neural_data_present = np.any(rates > 0, axis=(1, 2))
        n_all_zero_neural_trials = int(np.sum(~neural_data_present))
        if n_all_zero_neural_trials:
            trial_indices = trial_indices[neural_data_present]
            tone_onsets = tone_onsets[neural_data_present]
            inputs = inputs[neural_data_present]
            absolute_centers = absolute_centers[neural_data_present]
            rates = rates[neural_data_present]
        if len(trial_indices) < 2:
            raise ValueError("Fewer than two trials remain after neural-data QC")
        tongue = process_tongue(nwb, absolute_centers)
        outputs, labels = construct_outputs(nwb, trial_indices, tongue.classes)

        # Core dimensional/range checks before any data leave the source file.
        expected_neural = (len(trial_indices), len(good_units), 80)
        if rates.shape != expected_neural:
            raise ValueError(f"Neural shape {rates.shape} != {expected_neural}")
        if inputs.shape != (len(trial_indices), 2, 80):
            raise ValueError(f"Input shape is {inputs.shape}")
        if outputs.shape != (len(trial_indices), 4, 80):
            raise ValueError(f"Output shape is {outputs.shape}")
        for row, n_values in enumerate([3, 3, 2, 4]):
            if outputs[:, row].min() < 0 or outputs[:, row].max() >= n_values:
                raise ValueError(f"Output row {row} is outside its class range")

        if make_plot:
            plot_processing(
                session_id,
                nwb,
                trial_indices,
                good_units,
                rates,
                inputs,
                outputs,
                absolute_centers,
                tongue,
            )

        subject_id = Path(path).parent.name
        source_subject = decode_scalar(nwb["general/subject/subject_id"][()])
        if source_subject not in {subject_id, subject_id.removeprefix("sub-")}:
            raise ValueError(
                f"Path subject {subject_id} disagrees with NWB subject {source_subject}"
            )

        photo_rows = decode_array(nwb["intervals/trials/photostim_onset"])
        n_photo_rows = int(np.sum(photo_rows != "N/A"))
        n_photo_events = len(
            nwb["acquisition/BehavioralEvents/photostim_start_times/timestamps"]
        )
        if n_photo_rows != n_photo_events:
            raise ValueError("Photostimulation trial rows and event counts differ")

        info = {
            "session_id": session_id,
            "source_file": os.path.relpath(path, "/app"),
            "subject": subject_id,
            "n_units": int(len(good_units)),
            "n_trials_original": selection.n_original,
            "n_trials_observed": selection.n_observed,
            "n_trials_after_unit_trial_qc": selection.n_after_unit_trial_qc,
            "n_trials_after_window_qc": selection.n_after_window_qc,
            "n_all_zero_neural_trials_excluded": n_all_zero_neural_trials,
            "n_trials_retained": int(len(trial_indices)),
            "source_trial_indices": trial_indices.astype(np.int32),
            "recording_start_s": selection.recording_start,
            "recording_stop_s": selection.recording_stop,
            "tone_minus_go_s_range": [
                float(np.min(tone_onsets - nwb[
                    "acquisition/BehavioralEvents/go_start_times/timestamps"
                ][trial_indices])),
                float(np.max(tone_onsets - nwb[
                    "acquisition/BehavioralEvents/go_start_times/timestamps"
                ][trial_indices])),
            ],
            "tongue_y_p40": tongue.p40,
            "tongue_y_p60": tongue.p60,
            "tongue_likelihood_threshold": TONGUE_LIKELIHOOD_THRESHOLD,
            "tongue_visible_fraction_all_frames": tongue.visible_fraction_all_frames,
            "tongue_visible_fraction_sampled_bins": tongue.visible_fraction_sampled,
            "tongue_velocity_outliers_imputed": tongue.n_velocity_outliers,
            "photostimulation_event_count": int(n_photo_events),
            "retained_choice_counts": np.bincount(labels["choice"], minlength=3).tolist(),
            "retained_outcome_counts": np.bincount(labels["outcome"], minlength=3).tolist(),
            "retained_early_counts": np.bincount(labels["early"], minlength=2).tolist(),
            "retained_tongue_bin_counts": np.bincount(
                tongue.classes.ravel(), minlength=4
            ).tolist(),
        }

    elapsed = time.perf_counter() - started
    print(
        f"DONE {session_id}: {len(good_units)} units, "
        f"{selection.n_original}->{len(trial_indices)} trials, {elapsed:.2f} s",
        flush=True,
    )
    return {
        "subject": subject_id,
        "annotations": annotations,
        "neural": [rates[i] for i in range(len(rates))],
        "input": [inputs[i] for i in range(len(inputs))],
        "output": [outputs[i] for i in range(len(outputs))],
        "info": info,
        "elapsed": elapsed,
    }


def summarize(data: dict) -> None:
    session_trials = np.asarray([len(x) for x in data["neural"]])
    session_units = np.asarray([x[0].shape[0] for x in data["neural"]])
    output_counts = [np.zeros(n, dtype=np.int64) for n in [3, 3, 2, 4]]
    input_min = np.full(2, np.inf)
    input_max = np.full(2, -np.inf)
    for session_inputs, session_outputs in zip(data["input"], data["output"]):
        for x, y in zip(session_inputs, session_outputs):
            input_min = np.minimum(input_min, np.min(x, axis=1))
            input_max = np.maximum(input_max, np.max(x, axis=1))
            for row in range(4):
                output_counts[row] += np.bincount(
                    y[row].ravel(), minlength=len(output_counts[row])
                )
    print("\nCONVERTED DATA SUMMARY", flush=True)
    print(f"  sessions: {len(data['neural'])}", flush=True)
    print(f"  subjects: {len(data['subjects'])} {data['subjects']}", flush=True)
    print(
        f"  trials: {session_trials.sum()} "
        f"(session min/mean/max {session_trials.min()}/"
        f"{session_trials.mean():.2f}/{session_trials.max()})",
        flush=True,
    )
    print(
        f"  neurons: {session_units.sum()} "
        f"(session min/mean/max {session_units.min()}/"
        f"{session_units.mean():.2f}/{session_units.max()})",
        flush=True,
    )
    print(f"  brain regions: {len(data['brain_regions'])}", flush=True)
    print(f"  time bins: 80 x 50 ms; input ranges: {list(zip(input_min, input_max))}")
    for name, counts in zip(data["output_names"], output_counts):
        fractions = counts / counts.sum()
        print(
            f"  {name}: counts={counts.tolist()} fractions={fractions.round(6).tolist()}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", help="Output pickle path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--full", action="store_true", help="Process all sessions (default)"
    )
    mode.add_argument("--sample", action="store_true", help="Process two sessions")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing_<session_id>.png for up to two sessions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()
    paths = sorted(glob.glob(str(DATA_ROOT / "sub-*" / "*.nwb")))
    if not paths:
        raise FileNotFoundError(f"No NWB files under {DATA_ROOT}")
    target_sessions = 2 if args.sample else None
    print(
        f"Found {len(paths)} source NWBs; mode={'sample' if args.sample else 'full'}",
        flush=True,
    )

    converted_sessions = []
    for path in paths:
        make_plot = args.show_processing and len(converted_sessions) < 2
        result = convert_session(path, make_plot=make_plot)
        if result is None:
            continue
        converted_sessions.append(result)
        if target_sessions is not None and len(converted_sessions) >= target_sessions:
            break

    if not converted_sessions:
        raise RuntimeError("No usable sessions were converted")

    subjects = sorted({x["subject"] for x in converted_sessions})
    subject_to_idx = {name: idx for idx, name in enumerate(subjects)}
    brain_regions = sorted(
        {str(region) for x in converted_sessions for region in x["annotations"]}
    )
    region_to_idx = {name: idx for idx, name in enumerate(brain_regions)}

    data = {
        "neural": [x["neural"] for x in converted_sessions],
        "input": [x["input"] for x in converted_sessions],
        "output": [x["output"] for x in converted_sessions],
        "subjects": subjects,
        "subject_idx": np.asarray(
            [subject_to_idx[x["subject"]] for x in converted_sessions],
            dtype=np.int64,
        ),
        "brain_regions": brain_regions,
        "brain_region_idx": [
            np.asarray([region_to_idx[str(r)] for r in x["annotations"]], dtype=np.int64)
            for x in converted_sessions
        ],
        "input_names": ["time from tone onset (s)", "photostimulation on"],
        "output_names": ["lick direction choice", "outcome", "early lick", "tongue y-position"],
        "output_values": [CHOICE_VALUES, OUTCOME_VALUES, EARLY_VALUES, TONGUE_VALUES],
        "metadata": {
            "task_description": (
                "Auditory delayed-response directional licking task. Neural firing "
                "rates predict actual lick choice, trial outcome, early licking, "
                "and time-varying discretized tongue y-position; contextual inputs "
                "are signed time from the final instruction-tone onset and ALM "
                "photostimulation state."
            ),
            "time_bin_size": 50.0,
            "temporal_alignment_event": "go cue onset (NWB go_start_times)",
            "off_start": -2.5,
            "off_end": 1.5,
            "time_bin_edges_seconds": TIME_EDGES.astype(np.float32),
            "time_bin_centers_seconds": TIME_CENTERS.astype(np.float32),
            "neural_units": "spikes/s (Hz)",
            "neural_bin_definition": (
                "Adjacent half-open 50-ms bins [edge_i, edge_i+1); no smoothing"
            ),
            "unit_filter": "NWB units/classification == 'good'",
            "trial_filter": (
                "Trial listed in common good-unit obs_intervals; every retained "
                "unit is_good_trials; complete requested window lies in overall "
                "recorded interval; exclude population-all-zero windows caused "
                "by source stream truncation. Behavioral categories otherwise retained."
            ),
            "tongue_visibility_rule": (
                "side-camera likelihood >= 0.9 and timestamp within 1.5 camera "
                "frame intervals; otherwise class 3"
            ),
            "tongue_discretization": (
                "Per session visible-frame y percentiles: class 0 < p40; class 1 "
                "p40 through p60 inclusive; class 2 > p60; class 3 not visible"
            ),
            "source_dataset": "DANDI:000363/0.230822.0128 Mesoscale Activity Map Dataset",
            "source_format": "NWB 2.x (HDF5)",
            "session_info": [x["info"] for x in converted_sessions],
        },
    }

    # Final cross-field consistency checks.
    n_sessions = len(data["neural"])
    if not (
        len(data["input"])
        == len(data["output"])
        == len(data["subject_idx"])
        == len(data["brain_region_idx"])
        == n_sessions
    ):
        raise ValueError("Top-level session lists have different lengths")
    for i in range(n_sessions):
        if not (
            len(data["neural"][i])
            == len(data["input"][i])
            == len(data["output"][i])
        ):
            raise ValueError(f"Trial count mismatch in session {i}")
        if len(data["neural"][i]) < 2:
            raise ValueError(f"Session {i} has fewer than two trials")
        if len(data["brain_region_idx"][i]) != data["neural"][i][0].shape[0]:
            raise ValueError(f"Region/unit count mismatch in session {i}")

    summarize(data)
    output_path = Path(args.outpicklefile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_started = time.perf_counter()
    print(f"\nSaving {output_path} ...", flush=True)
    with output_path.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    save_elapsed = time.perf_counter() - save_started
    total_elapsed = time.perf_counter() - run_started
    print(
        f"Saved {output_path} ({output_path.stat().st_size / 2**30:.3f} GiB) "
        f"in {save_elapsed:.2f} s; total elapsed {total_elapsed:.2f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
