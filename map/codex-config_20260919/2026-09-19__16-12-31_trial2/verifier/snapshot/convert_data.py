#!/usr/bin/env python3
"""Convert the MAP NWB release to the neural-decoder pickle format.

Usage:
    python -u /app/convert_data.py OUTPUT.pkl [--full | --sample]
                                           [--show-processing]

The conversion uses classifier-curated units, trials with valid neural/video
coverage, go-cue alignment, and non-overlapping 50-ms firing-rate bins.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import time
from collections import Counter
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path("/app/data")
OFF_START = -2.5
OFF_END = 1.5
BIN_WIDTH_S = 0.050
BIN_EDGES = np.arange(OFF_START, OFF_END + BIN_WIDTH_S / 2, BIN_WIDTH_S)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
N_TIME = len(BIN_CENTERS)
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
    """Decode an HDF5 byte/string scalar."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def decode_array(values) -> np.ndarray:
    return np.asarray([decode_scalar(x) for x in values])


def numeric_or_nan(values) -> np.ndarray:
    """Convert NWB string-valued numeric columns (`N/A` included) to float."""
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i, value in enumerate(values):
        try:
            out[i] = float(decode_scalar(value))
        except (TypeError, ValueError):
            pass
    return out


def session_id_from_path(path: str) -> str:
    name = Path(path).name
    return name.split("_behavior")[0]


def discover_inventory(paths: list[str]) -> tuple[list[str], list[str]]:
    """Return globally sorted subject IDs and fine CCF region annotations."""
    subjects: set[str] = set()
    regions: set[str] = set()
    for path in paths:
        with h5py.File(path, "r") as nwb:
            subject = decode_scalar(nwb["general/subject/subject_id"][()])
            folder_subject = Path(path).parent.name.removeprefix("sub-")
            if subject != folder_subject:
                raise ValueError(f"Subject mismatch in {path}: {subject} vs {folder_subject}")
            subjects.add(subject)
            classification = decode_array(nwb["units/classification"][:])
            good = classification == "good"
            if not np.any(good):
                continue
            annotations = decode_array(nwb["units/anno_name"][:])[good]
            if np.any((annotations == "") | (annotations == "nan")):
                raise ValueError(f"Curated unit without anatomy in {path}")
            regions.update(annotations.tolist())
    return sorted(subjects), sorted(regions)


def first_tone_onsets(
    trial_starts: np.ndarray, go_times: np.ndarray, sample_starts: np.ndarray
) -> np.ndarray:
    """Find the first auditory sample event in each trial without positional pairing."""
    indices = np.searchsorted(sample_starts, trial_starts, side="left")
    if np.any(indices >= len(sample_starts)):
        raise ValueError("A recorded trial has no sample/tone event")
    tone = sample_starts[indices]
    valid = (tone >= trial_starts - 1e-9) & (tone <= go_times + 1e-9)
    if not np.all(valid):
        bad = np.flatnonzero(~valid)[:5]
        raise ValueError(f"Tone event outside trial-to-go interval at trials {bad.tolist()}")
    return tone


def choice_codes(outcomes: np.ndarray, instructions: np.ndarray) -> np.ndarray:
    """Recover the reported lick direction from outcome and instructed direction."""
    if not set(np.unique(outcomes)).issubset(set(OUTCOME_VALUES)):
        raise ValueError(f"Unknown outcomes: {np.unique(outcomes)}")
    if not set(np.unique(instructions)).issubset({"left", "right"}):
        raise ValueError(f"Unknown trial instructions: {np.unique(instructions)}")
    code = np.full(len(outcomes), 2, dtype=np.int8)  # ignore -> no lick
    hit = outcomes == "hit"
    miss = outcomes == "miss"
    code[hit & (instructions == "left")] = 0
    code[hit & (instructions == "right")] = 1
    code[miss & (instructions == "right")] = 0
    code[miss & (instructions == "left")] = 1
    return code


def make_processing_plot(
    session_id: str,
    stats: dict,
    rates: np.ndarray,
    inputs: np.ndarray,
    outputs: np.ndarray,
    sampled_tongue_y: np.ndarray,
    sampled_visible: np.ndarray,
    q40: float,
    q60: float,
    annotations: np.ndarray,
) -> None:
    """Plot every transformation stage for one representative converted trial."""
    trial = 0
    x = BIN_CENTERS
    fig, axes = plt.subplots(4, 2, figsize=(16, 16), constrained_layout=True)

    ax = axes[0, 0]
    labels = ["table", "recorded", "stable", "coverage"]
    vals = [stats[f"n_trials_{k}"] for k in labels]
    ax.bar(labels, vals)
    ax.set_title("Trial filtering")
    ax.set_ylabel("Trials")

    ax = axes[0, 1]
    ax.bar(["raw", "classifier good"], [stats["n_units_raw"], stats["n_units_good"]])
    ax.set_title("Classifier QC")
    ax.set_ylabel("Units")

    ax = axes[1, 0]
    nshow = min(60, rates.shape[1])
    image = ax.imshow(
        rates[trial, :nshow], aspect="auto", origin="lower",
        extent=[OFF_START, OFF_END, 0, nshow], interpolation="nearest"
    )
    ax.axvline(0, color="white", linestyle="--", linewidth=1)
    ax.set_title("50-ms firing rates (go = 0)")
    ax.set_ylabel("Curated neuron")
    fig.colorbar(image, ax=ax, label="Hz")

    ax = axes[1, 1]
    ax.plot(x, inputs[trial, 0], label="time from tone onset (s)")
    ax.step(x, inputs[trial, 1], where="mid", label="photostimulation on")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Decoder inputs")
    ax.legend(loc="best")

    ax = axes[2, 0]
    for j, label in enumerate(["choice", "outcome", "early lick", "tongue y class"]):
        ax.step(x, outputs[trial, j] + 4 * j, where="mid", label=label)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Categorical decoder outputs (vertically offset)")
    ax.legend(loc="best")

    ax = axes[2, 1]
    vis = sampled_visible[trial]
    ax.scatter(x[~vis], np.zeros(np.sum(~vis)), s=12, color="gray", label="not visible")
    ax.scatter(x[vis], sampled_tongue_y[trial, vis], s=12, color="tab:blue", label="visible y")
    ax.axhline(q40, color="tab:orange", linestyle="--", label="session q40")
    ax.axhline(q60, color="tab:red", linestyle="--", label="session q60")
    ax.set_title("Tongue visibility and session discretization")
    ax.legend(loc="best")

    ax = axes[3, 0]
    region_counts = Counter(annotations.tolist())
    common = region_counts.most_common(12)
    ax.barh([x[0] for x in common][::-1], [x[1] for x in common][::-1])
    ax.set_title("Fine CCF regions (top 12)")
    ax.set_xlabel("Curated units")

    ax = axes[3, 1]
    mean_rate = rates.mean(axis=(0, 1))
    ax.plot(x, mean_rate, color="black")
    ax.axvline(0, color="tab:red", linestyle="--", label="go cue")
    ax.axvline(stats["example_tone_relative_s"], color="tab:blue", linestyle=":", label="tone onset")
    ax.set_title("Population mean alignment check")
    ax.set_xlabel("Seconds from go cue")
    ax.set_ylabel("Hz")
    ax.legend(loc="best")

    fig.suptitle(session_id)
    out = Path("/app") / f"processing_{session_id}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def convert_session(
    path: str,
    subject_lookup: dict[str, int],
    region_lookup: dict[str, int],
    show_processing: bool,
) -> dict | None:
    """Convert one NWB file, returning None for the all-NaN-QC file."""
    started = time.perf_counter()
    sid = session_id_from_path(path)
    with h5py.File(path, "r") as nwb:
        units = nwb["units"]
        classification = decode_array(units["classification"][:])
        good_indices = np.flatnonzero(classification == "good")
        if len(good_indices) == 0:
            print(f"SKIP {sid}: no classifier-curated units", flush=True)
            return None

        annotations = decode_array(units["anno_name"][:])[good_indices]
        n_units_raw = len(classification)
        n_units_good = len(good_indices)

        trial_table = nwb["intervals/trials"]
        n_trials_table = len(trial_table["id"])
        stability_all = units["is_good_trials"][:]
        stability = stability_all[good_indices]
        n_trials_recorded = stability.shape[1]
        if n_trials_recorded > n_trials_table:
            raise ValueError(f"Neural trials exceed trial table in {path}")

        # The ragged observation-interval count independently verifies how many
        # leading trials contain ephys for each curated unit.
        obs_ends = units["obs_intervals_index"][:]
        obs_lengths = np.diff(np.r_[0, obs_ends])
        if not np.all(obs_lengths[good_indices] == n_trials_recorded):
            raise ValueError(f"Observation intervals disagree with stability table in {path}")

        stable_mask = np.all(stability, axis=0)
        go_all = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:n_trials_recorded]
        trial_starts = trial_table["start_time"][:n_trials_recorded]
        sample_starts = nwb["acquisition/BehavioralEvents/sample_start_times/timestamps"][:]
        tone_all = first_tone_onsets(trial_starts, go_all, sample_starts)

        tongue_group = nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
        tongue_t = tongue_group["timestamps"][:]
        tongue_data = tongue_group["data"][:]
        if tongue_data.ndim != 2 or tongue_data.shape[1] < 3:
            raise ValueError(f"Unexpected tongue data shape in {path}: {tongue_data.shape}")
        tongue_y = tongue_data[:, 1]
        tongue_likelihood = tongue_data[:, 2]
        session_visible = (
            np.isfinite(tongue_y)
            & np.isfinite(tongue_likelihood)
            & (tongue_likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
        )
        if np.sum(session_visible) < 2:
            raise ValueError(f"Insufficient visible tongue samples in {path}")
        q40, q60 = np.percentile(tongue_y[session_visible], [40, 60])

        centers_all = go_all[:, None] + BIN_CENTERS[None, :]
        frame_idx_all = np.searchsorted(tongue_t, centers_all, side="right") - 1
        safe_frame_idx = np.clip(frame_idx_all, 0, len(tongue_t) - 1)
        frame_lag = centers_all - tongue_t[safe_frame_idx]
        # 10 ms is about three 300-Hz frames and robust to the documented 3.4-ms
        # nominal spacing. Some sessions store video only during trial segments,
        # leaving seconds-long inter-trial gaps. A missing contemporaneous frame is
        # represented by the explicitly requested "not visible" class rather than
        # deleting an otherwise valid neural trial.
        frame_is_current_all = (frame_lag >= -1e-9) & (frame_lag <= 0.010)
        video_has_data = np.any(frame_is_current_all, axis=1)
        # An entirely missing video window is still a valid observation for this
        # task: the requested tongue class 3 explicitly means "not visible".
        coverage_mask = np.isfinite(tone_all)
        keep = stable_mask & coverage_mask
        keep_idx = np.flatnonzero(keep)
        if len(keep_idx) < 2:
            raise ValueError(f"Fewer than two valid trials in {path}")

        go = go_all[keep]
        tone = tone_all[keep]
        centers = centers_all[keep]
        frame_idx = safe_frame_idx[keep]
        frame_is_current = frame_is_current_all[keep]
        n_trials = len(keep_idx)
        n_trials_window_coverage = n_trials

        # Binning is vectorized over all trials for each unit. searchsorted at all
        # edges avoids a Python trial loop and exactly implements [left,right).
        absolute_edges = go[:, None] + BIN_EDGES[None, :]
        spike_ends = units["spike_times_index"][:]
        spike_values = units["spike_times"]
        rates = np.empty((n_trials, n_units_good, N_TIME), dtype=np.float32)
        for out_unit, source_unit in enumerate(good_indices):
            start = 0 if source_unit == 0 else int(spike_ends[source_unit - 1])
            stop = int(spike_ends[source_unit])
            spikes = spike_values[start:stop]
            edge_positions = np.searchsorted(spikes, absolute_edges.ravel(), side="left")
            counts = np.diff(edge_positions.reshape(n_trials, N_TIME + 1), axis=1)
            rates[:, out_unit, :] = counts / BIN_WIDTH_S

        # A small number of source files have an off-by-one terminal observation
        # interval after every unit's spike train has ended. With hundreds of units,
        # a completely silent four-second window is a reliable missing-ephys marker.
        neural_present = np.any(rates != 0, axis=(1, 2))
        if not np.all(neural_present):
            print(
                f"  {sid}: excluding {np.sum(~neural_present)} all-zero "
                "population trial(s) beyond spike coverage",
                flush=True,
            )
            rates = rates[neural_present]
            keep_idx = keep_idx[neural_present]
            go = go[neural_present]
            tone = tone[neural_present]
            centers = centers[neural_present]
            frame_idx = frame_idx[neural_present]
            frame_is_current = frame_is_current[neural_present]
            n_trials = len(keep_idx)
            if n_trials < 2:
                raise ValueError(f"Fewer than two trials with neural spikes in {path}")

        # Decoder inputs.
        inputs = np.empty((n_trials, 2, N_TIME), dtype=np.float32)
        inputs[:, 0, :] = centers - tone[:, None]
        onset = numeric_or_nan(trial_table["photostim_onset"][:n_trials_recorded])[keep_idx]
        duration = numeric_or_nan(trial_table["photostim_duration"][:n_trials_recorded])[keep_idx]
        kept_trial_starts = trial_starts[keep_idx]
        stim_start = kept_trial_starts + onset
        stim_stop = stim_start + duration
        inputs[:, 1, :] = (
            np.isfinite(stim_start[:, None])
            & np.isfinite(stim_stop[:, None])
            & (centers >= stim_start[:, None])
            & (centers < stim_stop[:, None])
        ).astype(np.float32)

        # Decoder outputs. Per-trial outputs are broadcast across time to coexist
        # in the same array with the time-varying tongue output.
        outcomes = decode_array(trial_table["outcome"][:n_trials_recorded])[keep_idx]
        instructions = decode_array(trial_table["trial_instruction"][:n_trials_recorded])[keep_idx]
        early_labels = decode_array(trial_table["early_lick"][:n_trials_recorded])[keep_idx]
        choices = choice_codes(outcomes, instructions)
        outcome_lookup = {name: i for i, name in enumerate(OUTCOME_VALUES)}
        outcome_codes = np.asarray([outcome_lookup[x] for x in outcomes], dtype=np.int8)
        if not set(np.unique(early_labels)).issubset({"no early", "early"}):
            raise ValueError(f"Unknown early-lick labels in {path}: {np.unique(early_labels)}")
        early_codes = (early_labels == "early").astype(np.int8)

        sampled_y = tongue_y[frame_idx]
        sampled_likelihood = tongue_likelihood[frame_idx]
        sampled_visible = (
            np.isfinite(sampled_y)
            & np.isfinite(sampled_likelihood)
            & (sampled_likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
            & frame_is_current
        )
        tongue_codes = np.full((n_trials, N_TIME), 3, dtype=np.int8)
        tongue_codes[sampled_visible & (sampled_y < q40)] = 0
        tongue_codes[sampled_visible & (sampled_y >= q40) & (sampled_y <= q60)] = 1
        tongue_codes[sampled_visible & (sampled_y > q60)] = 2

        outputs = np.empty((n_trials, 4, N_TIME), dtype=np.int8)
        outputs[:, 0, :] = choices[:, None]
        outputs[:, 1, :] = outcome_codes[:, None]
        outputs[:, 2, :] = early_codes[:, None]
        outputs[:, 3, :] = tongue_codes

        # Session-level invariants catch alignment/dtype mistakes immediately.
        if rates.shape != (n_trials, n_units_good, N_TIME):
            raise AssertionError(f"Bad neural shape in {path}: {rates.shape}")
        if inputs.shape != (n_trials, 2, N_TIME) or outputs.shape != (n_trials, 4, N_TIME):
            raise AssertionError(f"Bad input/output shape in {path}")
        if not (np.isfinite(rates).all() and np.isfinite(inputs).all() and np.isfinite(outputs).all()):
            raise AssertionError(f"Non-finite converted value in {path}")
        if np.any(rates < 0) or not np.allclose(rates / 20.0, np.round(rates / 20.0)):
            raise AssertionError(f"Firing rates are not valid 50-ms counts in {path}")

        subject = decode_scalar(nwb["general/subject/subject_id"][()])
        brain_idx = np.asarray([region_lookup[x] for x in annotations], dtype=np.int32)
        auto_water = trial_table["auto_water"][:n_trials_recorded][keep_idx]
        free_water = trial_table["free_water"][:n_trials_recorded][keep_idx]
        session_stats = {
            "session_id": sid,
            "source_file": str(Path(path).relative_to(DATA_ROOT)),
            "subject": subject,
            "n_trials_table": int(n_trials_table),
            "n_trials_recorded": int(n_trials_recorded),
            "n_trials_stable": int(np.sum(stable_mask)),
            "n_trials_window_coverage": int(n_trials_window_coverage),
            "n_trials_with_any_video": int(np.sum(video_has_data[keep_idx])),
            "n_trials_coverage": int(n_trials),
            "n_trials_converted": int(n_trials),
            "n_units_raw": int(n_units_raw),
            "n_units_good": int(n_units_good),
            "tongue_q40": float(q40),
            "tongue_q60": float(q60),
            "auto_water_trials_converted": int(np.sum(auto_water != 0)),
            "free_water_trials_converted": int(np.sum(free_water != 0)),
            "example_tone_relative_s": float(tone[0] - go[0]),
        }

        if show_processing:
            make_processing_plot(
                sid, session_stats, rates, inputs, outputs, sampled_y,
                sampled_visible, float(q40), float(q60), annotations
            )

    elapsed = time.perf_counter() - started
    mib = rates.nbytes / (1024 ** 2)
    print(
        f"DONE {sid}: {n_trials} trials, {n_units_good} neurons, "
        f"{mib:.1f} MiB neural, {elapsed:.2f} s",
        flush=True,
    )
    return {
        # Views are already C-contiguous per trial and share one session buffer in
        # memory. Pickle serializes each required trial ndarray normally.
        "neural": [rates[i] for i in range(n_trials)],
        "input": [inputs[i] for i in range(n_trials)],
        "output": [outputs[i] for i in range(n_trials)],
        "subject_idx": subject_lookup[subject],
        "brain_region_idx": brain_idx,
        "session_info": session_stats,
    }


def validate_final(data: dict) -> None:
    """Lightweight conversion-side validation before writing the pickle."""
    nsessions = len(data["neural"])
    if nsessions == 0:
        raise ValueError("No sessions converted")
    if not (len(data["input"]) == len(data["output"]) == nsessions):
        raise AssertionError("Session list lengths differ")
    if len(data["subject_idx"]) != nsessions or len(data["brain_region_idx"]) != nsessions:
        raise AssertionError("Session metadata lengths differ")
    for s in range(nsessions):
        nt = len(data["neural"][s])
        if nt < 2 or len(data["input"][s]) != nt or len(data["output"][s]) != nt:
            raise AssertionError(f"Invalid trial lists for session {s}")
        nn = data["neural"][s][0].shape[0]
        if len(data["brain_region_idx"][s]) != nn:
            raise AssertionError(f"Brain-region length mismatch in session {s}")
        for trial in (0, nt - 1):
            if data["neural"][s][trial].shape != (nn, N_TIME):
                raise AssertionError(f"Neural shape mismatch in session {s}, trial {trial}")
            if data["input"][s][trial].shape != (2, N_TIME):
                raise AssertionError(f"Input shape mismatch in session {s}, trial {trial}")
            if data["output"][s][trial].shape != (4, N_TIME):
                raise AssertionError(f"Output shape mismatch in session {s}, trial {trial}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", help="Destination pickle path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="Process first two usable sessions")
    parser.add_argument(
        "--show-processing", action="store_true",
        help="Write processing_<session_id>.png for up to two sessions",
    )
    args = parser.parse_args()

    overall_start = time.perf_counter()
    paths = sorted(glob.glob(str(DATA_ROOT / "sub-*" / "*.nwb")))
    if not paths:
        raise FileNotFoundError(f"No NWB files under {DATA_ROOT}")
    print(f"Discovered {len(paths)} NWB files", flush=True)

    inventory_start = time.perf_counter()
    subjects, brain_regions = discover_inventory(paths)
    subject_lookup = {name: i for i, name in enumerate(subjects)}
    region_lookup = {name: i for i, name in enumerate(brain_regions)}
    print(
        f"Inventory: {len(subjects)} subjects, {len(brain_regions)} fine CCF regions "
        f"({time.perf_counter() - inventory_start:.2f} s)", flush=True
    )

    converted = []
    for path in paths:
        result = convert_session(
            path,
            subject_lookup,
            region_lookup,
            show_processing=args.show_processing and len(converted) < 2,
        )
        if result is None:
            continue
        converted.append(result)
        if args.sample and len(converted) == 2:
            break

    data = {
        "neural": [x["neural"] for x in converted],
        "input": [x["input"] for x in converted],
        "output": [x["output"] for x in converted],
        "subjects": subjects,
        "subject_idx": np.asarray([x["subject_idx"] for x in converted], dtype=np.int32),
        "brain_regions": brain_regions,
        "brain_region_idx": [x["brain_region_idx"] for x in converted],
        "input_names": ["time from tone onset (s)", "photostimulation on"],
        "output_names": ["lick direction choice", "outcome", "early lick", "tongue y-position"],
        "output_values": [CHOICE_VALUES, OUTCOME_VALUES, EARLY_VALUES, TONGUE_VALUES],
        "metadata": {
            "task_description": (
                "Auditory delayed-response directional licking task; decode reported choice, "
                "outcome, early licking, and session-discretized tongue y-position from "
                "classifier-curated Neuropixels activity."
            ),
            "time_bin_size": 50.0,
            "temporal_alignment_event": "go cue onset",
            "off_start": OFF_START,
            "off_end": OFF_END,
            "n_timepoints": N_TIME,
            "time_bin_centers_seconds": BIN_CENTERS.astype(float).tolist(),
            "neural_measurement": "firing rate (Hz) from non-overlapping 50-ms spike-count bins",
            "spike_bin_interval_convention": "half-open [left, right)",
            "unit_filter": "NWB units.classification == 'good' (published regional classifier QC)",
            "trial_filter": (
                "leading trials with ephys observation intervals; all curated units stable; "
                "finite first tone onset; missing or noncontemporaneous Camera0 frames are "
                "tongue class 'not visible'"
            ),
            "tone_onset_definition": "first sample_start_times event between trial start and go cue",
            "tongue_visibility_likelihood_threshold": TONGUE_LIKELIHOOD_THRESHOLD,
            "tongue_percentile_scope": "all visible Camera0 side-view frames within each session",
            "session_info": [x["session_info"] for x in converted],
            "source": "DANDI 000363 version 0.230822.0128",
        },
    }
    validate_final(data)

    output_path = Path(args.outpicklefile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_start = time.perf_counter()
    with output_path.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_time = time.perf_counter() - write_start
    total_trials = sum(len(x) for x in data["neural"])
    total_neurons = sum(x[0].shape[0] for x in data["neural"])
    size_gib = output_path.stat().st_size / (1024 ** 3)
    print(
        f"WROTE {output_path}: {len(converted)} sessions, {total_trials} trials, "
        f"{total_neurons} session-units, {size_gib:.3f} GiB; "
        f"pickle write {write_time:.2f} s; total {time.perf_counter() - overall_start:.2f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
